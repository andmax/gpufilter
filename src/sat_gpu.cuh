/**
 *  @file sat_gpu.cuh
 *  @brief Algorithm SAT (Summed-area Tables) in the GPU with borders
 *  @author Andre Maximo
 *  @date Feb, 2012
 *  @copyright The MIT License
 */

#ifndef SAT_GPU_CUH
#define SAT_GPU_CUH

#define MST == // measure step time: no: == ; yes: >=
#define LDG // uncomment to use __ldg

//== NAMESPACES ================================================================

namespace gpufilter {

//== IMPLEMENTATION ============================================================

/**
 *  @ingroup gpu
 *  @brief Algorithm SAT step 1
 *
 *  This function computes the algorithm step SAT.1 (corresponding to
 *  the step S.1 in [NehabEtAl:2011]) following:
 *
 *  \li In parallel for all \f$m\f$ and \f$n\f$, load block
 *  \f$B_{m,n}(X)\f$ then compute and store block perimeters
 *  \f$P_{m,n}(Y)\f$ and \f$P^T_{m,n}(V)\f$.
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @see [NehabEtAl:2011] cited in alg5()
 *  @param[out] g_pybar All \f$P_{m,n}(Y)\f$
 *  @param[out] g_ptvhat All \f$P^T_{m,n}(V)\f$
 *  @param[in] inv_width Image width inversed (1/w)
 *  @param[in] inv_height Image height inversed (1/h)
 *  @param[in] m_size The big M (number of row blocks)
 *  @param[in] n_size The big N (number of column blocks)
 *  @tparam BORDER Flag to consider border input padding
 *  @tparam R Filter order
 */
template <bool BORDER, int R>
__global__ __launch_bounds__(WS*NWC, NBCW)
void sat_step1( Matrix<float,R,WS> *g_pybar,
                Matrix<float,R,WS> *g_ptvhat,
                float inv_width, float inv_height,
                int m_size, int n_size ) {

    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n = blockIdx.y;

    __shared__ Matrix<float,WS,WS+1> block;
    if (BORDER) // read considering borders
        read_block<NWW>(block, m-c_border, n-c_border, inv_width, inv_height);
    else
        read_block<NWW>(block, m, n, inv_width, inv_height);
    __syncthreads();

    if (ty==0) {

        Vector<float,R> p = zeros<float,R>();

#pragma unroll // calculate pybar, scan left -> right
        for (int j=0; j<WS; ++j)
            block[tx][j] = fwdI(p, block[tx][j], c_weights);

        g_pybar[m*(n_size+1)+n+1].set_col(tx, p);

        p = zeros<float,R>();

#pragma unroll // calculate ptvhat, scan top -> bottom
        for (int j=0; j<WS; ++j)
            block[j][tx] = fwdI(p, block[j][tx], c_weights);

        g_ptvhat[n*(m_size+1)+m+1].set_col(tx, p);

    }

}
    
/**
 *  @ingroup gpu
 *  @brief Algorithm SAT step 2
 *
 *  This function computes the algorithm step SAT.2 (corresponding to
 *  the step S.2 in [NehabEtAl:2011]) following:
 *
 *  \li In parallel for all \f$m\f$, sequentially for each \f$n\f$,
 *  compute and store all feedbacks \f$P_{m,n-1}(Y)\f$ according to
 *  equation (44).  Compute and store the sum of \f$P_{m,n}(Y)\f$.
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @see [NehabEtAl:2011] cited in alg5()
 *  @param[in,out] g_pybar All \f$P_{m,n}(Y)\f$
 *  @param[out] g_pysum All \f$s(P_{m,n}(Y))\f$
 *  @param[in] n_size The big N (number of column blocks)
 *  @tparam R Filter order
 */
template <int R>
__global__ __launch_bounds__(WS*NWA, NBA)
void sat_step2( Matrix<float,R,WS> *g_pybar,
                Matrix<float,R,R> *g_pysum,
                int n_size ) {

    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n;
    Matrix<float,R,WS> *gpybar;
    Vector<float,R> py, pybar;
    __shared__ Matrix<float,R,WS> spybar[NWA];
    __shared__ Matrix<float,R,R> spysum[NWA];

    // P(ybar) -> P(y) processing ----------------------------------------------
    n = 0;
    gpybar = (Matrix<float,R,WS> *)&g_pybar[m*(n_size+1)+n+ty+1][0][tx];
    py = zeros<float,R>();
    g_pysum += m*(n_size+1)+n+ty+1;

    for (; n < n_size; n += NWA) { // for all column blocks

        if (n+ty < n_size) { // using smem as cache
            spybar[ty].set_col(tx, gpybar->col(0));
        }

        __syncthreads(); // wait to load smem

        if (ty == 0) {
#pragma unroll // adjust pybar left -> right
            for (int w = 0; w < NWA; ++w) {

                pybar = spybar[w].col(tx);

                py = pybar + py * c_AbF_T;

#pragma unroll // computing corner rows south east
                for (int i = 0; i < R; ++i) {
#pragma unroll // computing corner cols south east
                    for (int j = 0; j < R; ++j) {
                        float v = c_TAFB[i][tx] * py[j];
#pragma unroll // recursive doubling by shuffle
                        for (int k = 1; k < WS; k *= 2) {
                            float p = __shfl_up(v, k);
                            if (tx >= k) {
                                v += p;
                            }
                        }
                        if (tx == WS-1) {
                            spysum[w][j][i] = v;
                        }
                    }
                }

                spybar[w].set_col(tx, py);

            }
        }

        __syncthreads(); // wait to store result in gmem

        if (n+ty < n_size && tx == WS-1) {
            *g_pysum = spysum[ty];
        }

        if (n+ty < n_size) {
            gpybar->set_col(0, spybar[ty].col(tx));
        }

        gpybar += NWA;
        g_pysum += NWA;

    }

}

/**
 *  @ingroup gpu
 *  @brief Algorithm SAT step 3
 *
 *  This function computes the algorithm step SAT.3 (corresponding to
 *  the step S.3 in [NehabEtAl:2011]) following:
 *
 *  \li In parallel for all \f$n\f$, sequentially for each \f$m\f$,
 *  compute and store feedbacks \f$P^T_{m-1,n}(V)\f$ according to
 *  equation (45).
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @see [NehabEtAl:2011] cited in alg5()
 *  @param[in,out] g_ptvhat All \f$P^T_{m,n}(V)\f$
 *  @param[in] g_pysum All \f$s(P_{m,n}(Y))\f$
 *  @param[in] m_size The big M (number of row blocks)
 *  @param[in] n_size The big N (number of column blocks)
 *  @tparam R Filter order
 */
template <int R>
__global__ __launch_bounds__(WS*NWA, NBA)
void sat_step3( Matrix<float,R,WS> *g_ptvhat,
                const Matrix<float,R,R> *g_pysum,
                int m_size, int n_size ) {

    int tx = threadIdx.x, ty = threadIdx.y, m, n = blockIdx.y;
    Matrix<float,R,WS> *gptvhat;
    Vector<float,R> ptv, ptvhat;
    __shared__ Matrix<float,R,WS> sptvhat[NWA];
    __shared__ Matrix<float,R,R> spysum[NWA];

    // Pt(vhat) -> Pt(v) processing --------------------------------------------
    m = 0;
    gptvhat = (Matrix<float,R,WS> *)&g_ptvhat[n*(m_size+1)+m+ty+1][0][tx];
    ptv = zeros<float,R>();
    g_pysum += (m+ty)*(n_size+1)+n+0;

    for (; m < m_size; m += NWA) {

        if (m+ty < m_size) { // using smem as cache
            sptvhat[ty].set_col(tx, gptvhat->col(0));
            if (tx == 0) {
#ifdef LDG
#pragma unroll // loading corners rows south east
                for (int i = 0; i < R; ++i) {
#pragma unroll // loading corners cols south east
                    for (int j = 0; j < R; ++j) {
                        spysum[ty][i][j] = __ldg((const float *)&g_pysum[i][j]);
                    }
                }
#else
                spysum[ty] = *g_pysum;
#endif
            }
        }

        __syncthreads(); // wait to load smem

        if (ty == 0) {
#pragma unroll // adjust ptvhat top -> bottom
            for (int w = 0; w < NWA; ++w) {

                ptvhat = sptvhat[w].col(tx);

                ptv = ptvhat + ptv * c_AbF_T;

                ptv += c_AFP_T.col(tx) * spysum[w];

                sptvhat[w].set_col(tx, ptv);

            }
        }

        __syncthreads(); // wait to store result in gmem

        if (m+ty < m_size) {
            gptvhat->set_col(0, sptvhat[ty].col(tx));
        }

        gptvhat += NWA;
        g_pysum += NWA*(n_size+1);

    }

}

/**
 *  @ingroup gpu
 *  @brief Algorithm SAT step 4
 *
 *  This function computes the algorithm step SAT.4 (corresponding to
 *  the step S.4 in [NehabEtAl:2011]) following:
 *
 *  \li In parallel for all \f$m\f$ and \f$n\f$, load input block
 *  \f$B_{m,n}(X)\f$ and all its block feedbacks \f$P_{m,n-1}(Y)\f$,
 *  \f$P^T_{m-1,n}(V)\f$.  Compute and store \f$B_{m,n}(V)\f$.
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @see [NehabEtAl:2011] cited in alg5()
 *  @param[out] g_out The output 2D image
 *  @param[in] g_py All \f$P_{m,n}(Y)\f$
 *  @param[in] g_ptv All \f$P^T_{m,n}(V)\f$
 *  @param[in] inv_width Image width inversed (1/w)
 *  @param[in] inv_height Image height inversed (1/h)
 *  @param[in] m_size The big M (number of row blocks)
 *  @param[in] n_size The big N (number of column blocks)
 *  @param[in] out_stride Image output stride for memory width alignment
 *  @tparam BORDER Flag to consider border input padding
 *  @tparam R Filter order
 */
template <bool BORDER, int R>
__global__ __launch_bounds__(WS*NWW, NBCW)
void sat_step4( float *g_out,
                const Matrix<float,R,WS> *g_py,
                const Matrix<float,R,WS> *g_ptv,
                float inv_width, float inv_height,
                int m_size, int n_size,
                int out_stride ) {

    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n = blockIdx.y;

    __shared__ Matrix<float,WS,WS+1> block;
    if (BORDER) // read considering borders
        read_block<NWW>(block, m-c_border, n-c_border, inv_width, inv_height);
    else
        read_block<NWW>(block, m, n, inv_width, inv_height);
    __syncthreads();

    if (ty==0) {

        Vector<float,R> p;

#ifdef LDG
#pragma unroll
        for (int r=0; r<R; ++r)
            p[r] = __ldg((const float *)&g_py[m*(n_size+1)+n][r][tx]);
#else
        p = ((Matrix<float,R,WS>*)&g_py[m*(n_size+1)+n][0][tx])->col(0);
#endif

#pragma unroll // calculate block, scan left -> right
        for (int j=0; j<WS; ++j)
            block[tx][j] = fwdI(p, block[tx][j], c_weights);

#ifdef LDG
#pragma unroll
        for (int r=0; r<R; ++r)
            p[r] = __ldg((const float *)&g_ptv[n*(m_size+1)+m][r][tx]);
#else
        p = ((Matrix<float,R,WS>*)&g_ptv[n*(m_size+1)+m][0][tx])->col(0);
#endif

#pragma unroll // calculate block, scan top -> bottom
        for (int j=0; j<WS; ++j)
            block[j][tx] = fwdI(p, block[j][tx], c_weights);

        if (BORDER) {
            if ((m >= c_border) && (m < m_size-c_border) && (n >= c_border) && (n < n_size-c_border)) {
                g_out += ((m-c_border+1)*WS-1)*out_stride + (n-c_border)*WS+tx;
#pragma unroll // write block inside valid image
                for (int i=0; i<WS; ++i, g_out-=out_stride) {
                    *g_out = block[WS-1-i][tx];
                }
            }
        } else {
            g_out += ((m+1)*WS-1)*out_stride + n*WS+tx;
#pragma unroll // write block
            for (int i=0; i<WS; ++i, g_out-=out_stride) {
                *g_out = block[WS-1-i][tx];
            }
        }

    }

}

/**
 *  @ingroup api_gpu
 *  @brief Compute algorithm SAT in the GPU
 *
 *  A SAT (Summed-Area Table) is also known as an Integral Image.
 *
 *  @see [NehabEtAl:2011] cited in alg5()
 *  @param[in,out] h_img The in(out)put 2D image to filter in host memory
 *  @param[in] width Image width
 *  @param[in] height Image height
 *  @param[in] runtimes Number of run times (1 for debug and 1000 for performance measurements)
 *  @param[in] w Filter weights (feedforward and feedforward coefficients)
 *  @param[in] border Number of border blocks (32x32) outside image
 *  @param[in] btype Border type (either zero, clamp, repeat or reflect)
 *  @tparam BORDER Flag to consider border input padding
 *  @tparam R Filter order
 */
template <bool BORDER, int R>
__host__
void sat_gpu( float *h_img,
              int width, int height, int runtimes,
              const Vector<float, R+1> &w,
              int border=0,
              BorderType border_type=CLAMP_TO_ZERO ) {

    const int B = WS;

    // pre-compute basic alg3 matrices
    Matrix<float,R,R> Ir = identity<float,R,R>();
    Matrix<float,B,R> Zbr = zeros<float,B,R>();
    Matrix<float,R,B> Zrb = zeros<float,R,B>();
    Matrix<float,B,B> Ib = identity<float,B,B>();

    Matrix<float,R,B> AFP_T = fwd(Ir, Zrb, w);
    Matrix<float,B,B> AFB_T = fwd(Zbr, Ib, w);

    Matrix<float,R,R> AbF_T = tail<R>(AFP_T);

    Matrix<float,R,B> TAFB = transp(tail<R>(AFB_T));

    int m_size = (height+WS-1)/WS, n_size = (width+WS-1)/WS;

    if (BORDER) {
        int border_left, border_top, border_right, border_bottom;
        calc_borders(&border_left, &border_top, &border_right, &border_bottom, 
                     width, height, border);
        int ewidth = width+border_left+border_right,
            eheight = height+border_top+border_bottom;

        m_size = (eheight+WS-1)/WS;
        n_size = (ewidth+WS-1)/WS;

        copy_to_symbol(c_border, border);
    }

    // upload to the GPU
    copy_to_symbol(c_weights, w);

    copy_to_symbol(c_AbF_T, AbF_T);
    copy_to_symbol(c_AFP_T, AFP_T);
    copy_to_symbol(c_TAFB, TAFB);

    float inv_width = 1.f/width, inv_height = 1.f/height;

    cudaArray *a_in;
    cudaChannelFormatDesc ccd = cudaCreateChannelDesc<float>();
    cudaMallocArray(&a_in, &ccd, width, height);
    cudaMemcpyToArray(a_in, 0, 0, h_img, width*height*sizeof(float),
                      cudaMemcpyHostToDevice);

    t_in.normalized = true;
    t_in.filterMode = cudaFilterModePoint;
    t_in.addressMode[0] = t_in.addressMode[1] = cudaAddressModeBorder;

    if (BORDER) {
        switch(border_type) {
        case CLAMP_TO_ZERO:
            t_in.addressMode[0] = t_in.addressMode[1] = cudaAddressModeBorder;
            break;
        case CLAMP_TO_EDGE:
            t_in.addressMode[0] = t_in.addressMode[1] = cudaAddressModeClamp;
            break;
        case REPEAT:
            t_in.addressMode[0] = t_in.addressMode[1] = cudaAddressModeWrap;
            break;
        case REFLECT:
            t_in.addressMode[0] = t_in.addressMode[1] = cudaAddressModeMirror;
            break;
        }
    }

    int stride_img = width+WS;
    if (BORDER) stride_img = width+WS*border+WS;

    dvector<float> d_img(height*stride_img);

    // +1 padding is important even in zero-border to avoid if's in kernels
    dvector< Matrix<float,R,B> >
        d_pybar(m_size*(n_size+1)), d_ptvhat(n_size*(m_size+1));
    d_pybar.fillzero();
    d_ptvhat.fillzero();

    dvector< Matrix<float,R,R> > d_pysum(m_size*(n_size+1));
    d_pysum.fillzero();

    cudaFuncSetCacheConfig(sat_step1<BORDER,R>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(sat_step4<BORDER,R>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(sat_step3<R>, cudaFuncCachePreferShared);

    if (R == 1)
        cudaFuncSetCacheConfig(sat_step2<R>, cudaFuncCachePreferL1);
    else if (R == 2)
        cudaFuncSetCacheConfig(sat_step2<R>, cudaFuncCachePreferEqual);
    else if (R >= 3)
        cudaFuncSetCacheConfig(sat_step2<R>, cudaFuncCachePreferShared);

    double te[4] = {0, 0, 0, 0}; // time elapsed for the four steps
    base_timer *timer[4];
    for (int i = 0; i < 4; ++i)
        timer[i] = new gpu_timer(0, "", false);

    base_timer &timer_total = timers.gpu_add("sat_gpu", width*height, "iP");

    for(int r = 0; r < runtimes; ++r) {

        if (runtimes MST 1) { timer[0]->start(); }

        cudaBindTextureToArray(t_in, a_in);

        sat_step1<BORDER><<< dim3(m_size, n_size), dim3(WS, NWC) >>>
            ( &d_pybar, &d_ptvhat, inv_width, inv_height, m_size, n_size );

        if (runtimes MST 1) { timer[0]->stop(); te[0] += timer[0]->elapsed(); timer[1]->start(); }

        sat_step2<<< dim3(m_size, 1), dim3(WS, NWA) >>>
            ( &d_pybar, &d_pysum, n_size );

        if (runtimes MST 1) { timer[1]->stop(); te[1] += timer[1]->elapsed(); timer[2]->start(); }

        sat_step3<<< dim3(1, n_size), dim3(WS, NWA) >>>
            ( &d_ptvhat, &d_pysum, m_size, n_size );
        
        if (runtimes MST 1) { timer[2]->stop(); te[2] += timer[2]->elapsed(); timer[3]->start(); }

        sat_step4<BORDER><<< dim3(m_size, n_size), dim3(WS, NWW) >>>
            ( d_img, &d_pybar, &d_ptvhat, inv_width, inv_height,
              m_size, n_size, stride_img );

        cudaUnbindTexture(t_in);

        if (runtimes MST 1) { timer[3]->stop(); te[3] += timer[3]->elapsed(); }

    }

    timer_total.stop();

    if (runtimes > 1) {

        if (runtimes MST 1) {
            for (int i = 0; i < 4; ++i)
                std::cout << std::fixed << " " << te[i]/(double)runtimes << std::flush;
        } else {
            std::cout << std::fixed << (timer_total.data_size()*runtimes)/(double)(timer_total.elapsed()*1024*1024) << std::flush;
        }

    } else {

        timers.gpu_add("step 1", timer[0]);
        timers.gpu_add("step 2", timer[1]);
        timers.gpu_add("step 3", timer[2]);
        timers.gpu_add("step 4", timer[3]);
        timers.flush();

    }

    cudaMemcpy2D(h_img, width*sizeof(float), d_img, stride_img*sizeof(float), width*sizeof(float), height, cudaMemcpyDeviceToHost);

}

//==============================================================================
} // namespace gpufilter
//==============================================================================
#endif // SAT_GPU_CUH
//==============================================================================
