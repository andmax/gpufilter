/**
 *  @file alg6_clamp.cuh
 *  @brief Algorithm 6 with clamp boundary condition
 *  @author Andre Maximo
 *  @date Nov, 2012
 *  @copyright The MIT License
 */

#ifndef ALG6_CLAMP_CUH
#define ALG6_CLAMP_CUH

//== NAMESPACES ================================================================

namespace gpufilter {

//=== IMPLEMENTATION ===========================================================

/**
 *  @ingroup gpu
 *  @brief Algorithm 6 stage 1 for clamp
 *
 *  This function computes the algorithm stages \f$6^c.1\f$ and
 *  \f$6^c.2\f$ (corresponding to the base algorithm 6 stage 1).
 *
 *  @see The base algorithm description in alg6_stage1()
 *
 *  @see [NehabMaximo:2016] cited in alg6()
 *  @param[out] g_pybar All \f$P_{m,n}(Y)\f$
 *  @param[out] g_ezhat All \f$E_{m,n}(Z)\f$
 *  @param[out] g_ptucheck All \f$P^T_{m,n}(U)\f$
 *  @param[out] g_etvtilde All \f$E^T_{m,n}(V)\f$
 *  @param[in] inv_width Image width inversed (1/w)
 *  @param[in] inv_height Image height inversed (1/h)
 *  @param[in] m_size The big M (number of row blocks)
 *  @param[in] n_size The big N (number of column blocks)
 *  @tparam R Filter order
 */
template <int R>
__global__ __launch_bounds__(WS*NWC, NBCW)
void alg6_clamp_stage1( Matrix<float,R,WS> *g_pybar, 
                        Matrix<float,R,WS> *g_ezhat,
                        Matrix<float,R,WS> *g_ptucheck,
                        Matrix<float,R,WS> *g_etvtilde,
                        float inv_width, float inv_height,
                        int m_size, int n_size ) {

    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n = blockIdx.y;

    __shared__ Matrix<float,WS,WS+1> block;
    read_block<NWC>(block, m, n, inv_width, inv_height);
    __syncthreads();

    float x[32]; // 32 regs

    if (ty==0) {

#pragma unroll
        for (int i=0; i<32; ++i)
            x[i] = block[tx][i];

        if (m == 0) { // store p(x) in pybar_0
#pragma unroll
            for (int j=0; j<R; ++j)
                g_pybar[n*(m_size+1)+0][j][tx] = x[0];
        } else if (m == m_size-1) { // store e(x) in ezhat_M
#pragma unroll
            for (int j=0; j<R; ++j)
                g_ezhat[n*(m_size+1)+m_size][j][tx] = x[WS-1];
        }

        Vector<float,R> p = zeros<float,R>();

#pragma unroll // calculate pybar, scan left -> right
        for (int j=0; j<WS; ++j)
            x[j] = fwdI(p, x[j], c_weights);

        g_pybar[n*(m_size+1)+m+1].set_col(tx, p);
        
        Vector<float,R> e = zeros<float,R>();

#pragma unroll // calculate ezhat, scan right -> left
        for (int j=WS-1; j>=0; --j)
            x[j] = revI(x[j], e, c_weights);

        g_ezhat[n*(m_size+1)+m].set_col(tx, e);

#pragma unroll // transpose regs part-1
        for (int i=0; i<32; ++i)
            block[tx][i] = x[i];
#pragma unroll // transpose regs part-2
        for (int i=0; i<32; ++i)
            x[i] = block[i][tx];

        if (n == 0) { // store pt(zhat) in ptucheck_0
#pragma unroll
            for (int j=0; j<R; ++j)
                g_ptucheck[m*(n_size+1)+0][j][tx] = x[0];
        } else if (n == n_size-1) { // store et(zhat) in etvtilde_M
#pragma unroll
            for (int j=0; j<R; ++j)
                g_etvtilde[m*(n_size+1)+n_size][j][tx] = x[WS-1];
        }
            
        p = zeros<float,R>();

#pragma unroll // calculate ptucheck, scan top -> bottom
        for (int j=0; j<WS; ++j)
            x[j] = fwdI(p, x[j], c_weights);

        g_ptucheck[m*(n_size+1)+n+1].set_col(tx, p);

        e = zeros<float,R>();

#pragma unroll // calculate etvtilde, scan bottom -> top
        for (int j=WS-1; j>=0; --j)
            revI(x[j], e, c_weights);

        g_etvtilde[m*(n_size+1)+n].set_col(tx, e);

    }

}

/**
 *  @ingroup gpu
 *  @brief Algorithm 6 stage 2 or 4 for clamp
 *
 *  This function computes the algorithm stages \f$6^c.3\f$ and
 *  \f$6^c.4\f$ (corresponding to the base algorithm 6 stage 2) or
 *  \f$6^c.7\f$ and \f$6^c.8\f$ (corresponding to the base algorithm 6
 *  stage 4).
 * 
 *  @see The base algorithm description in alg6_stage2v4()
 *
 *  @see [NehabMaximo:2016] cited in alg6()
 *  @param[in,out] g_pybar All \f$P_{m,n}(Y)\f$ or \f$P^T_{m,n}(U)\f$
 *  @param[in,out] g_ezhat All \f$E_{m,n}(Z)\f$ or \f$E^T_{m,n}(V)\f$
 *  @param[in] m_size The big M or N
 *  @tparam R Filter order
 */
template <int R>
__global__ __launch_bounds__(WS*NWA, NBA)
void alg6_clamp_stage2v4( Matrix<float,R,WS> *g_pybar,
                          Matrix<float,R,WS> *g_ezhat,
                          int m_size ) {

    int tx = threadIdx.x, ty = threadIdx.y, m, n = blockIdx.y;
    Matrix<float,R,WS> *gpybar, *gezhat;
    Vector<float,R> py, ez, pybar, ezhat;
    __shared__ Matrix<float,R,WS> spybar[NWA], sezhat[NWA];

    // Compute P_{-1}(ycla) ----------------------------------------------------
    m = 0;

    if (ty == 0) {

        gpybar = (Matrix<float,R,WS> *)&g_pybar[n*(m_size+1)+m][0][tx];
        py = gpybar->col(0);
        py = py * c_AbarFIArF_T;
        gpybar->set_col(0, py);

    }

    // P(ybar) -> P(ycla) processing -------------------------------------------
    gpybar = (Matrix<float,R,WS> *)&g_pybar[n*(m_size+1)+m+ty+1][0][tx];

    for (; m < m_size; m += NWA) { // for all image blocks

        if (m+ty < m_size) spybar[ty].set_col(tx, gpybar->col(0));

        __syncthreads(); // wait to load smem

        if (ty == 0) {
#pragma unroll // adjust pybar left -> right
            for (int w = 0; w < NWA; ++w) {

                if (m+w >= m_size) break;

                pybar = spybar[w].col(tx);

                py = pybar + py * c_AbF_T;

                spybar[w].set_col(tx, py);

            }
        }

        __syncthreads(); // wait to store result in gmem

        if (m+ty < m_size) gpybar->set_col(0, spybar[ty].col(tx));

        gpybar += NWA;

    }

    // Compute E_{M}(zcla) -----------------------------------------------------
    m = m_size-1;

    if (ty == 0) {

        gezhat = (Matrix<float,R,WS> *)&g_ezhat[n*(m_size+1)+m+1][0][tx];
        ez = gezhat->col(0);
        ez = py * c_ArFSRRF_T + ez * c_AbarFIArFAbarRIArRArFSRRF_T;
        gezhat->set_col(0, ez);

    }

    // E(zhat) -> E(zcla) processing -------------------------------------------
    gezhat = (Matrix<float,R,WS> *)&g_ezhat[n*(m_size+1)+m-ty][0][tx];
    gpybar = (Matrix<float,R,WS> *)&g_pybar[n*(m_size+1)+m-ty][0][tx];

    for (; m >= 0; m -= NWA) { // for all image blocks

        if (m-ty >= 0) { // using smem as cache
            sezhat[ty].set_col(tx, gezhat->col(0));
            spybar[ty].set_col(tx, gpybar->col(0));
        }

        __syncthreads(); // wait to load smem

        if (ty == 0) {
#pragma unroll // adjust ezhat right -> left
            for (int w = 0; w < NWA; ++w) {

                ezhat = sezhat[w].col(tx);
                py = spybar[w].col(tx);

                ez = ezhat + py * c_HARB_AFP_T + ez * c_AbR_T;

                sezhat[w].set_col(tx, ez);

            }
        }

        __syncthreads(); // wait to store result in gmem

        if (m-ty >= 0) gezhat->set_col(0, sezhat[ty].col(tx));

        gezhat -= NWA;
        gpybar -= NWA;

    }

}

/**
 *  @ingroup gpu
 *  @brief Algorithm 6 stage 3 for clamp
 *
 *  This function computes the algorithm stage \f$6^c.5\f$ and 
 *  \f$6^c.6\f$ (corresponding to the base algorithm 6 stage 3).
 *
 *  @see The base algorithm description in alg6_stage3()
 *
 *  @see [NehabMaximo:2016] cited in alg6()
 *  @param[in,out] g_ptucheck All \f$P^T_{m,n}(U)\f$
 *  @param[in,out] g_etvtilde All \f$E^T_{m,n}(V)\f$
 *  @param[in] g_py All \f$P_{m,n}(Y)\f$
 *  @param[in] g_ez All \f$E_{m,n}(Z)\f$
 *  @param[in] g_cmat Constant pre-computed matrices
 *  @param[in] m_size The big M (number of row blocks)
 *  @param[in] n_size The big N (number of column blocks)
 *  @tparam R Filter order
 */
template <int R>
__global__ __launch_bounds__(WS*NWARC, NBARC)
void alg6_clamp_stage3( Matrix<float,R,WS> *g_ptucheck,
                        Matrix<float,R,WS> *g_etvtilde,
                        const Matrix<float,R,WS> *g_py,
                        const Matrix<float,R,WS> *g_ez,
                        const Matrix<float,R,WS> *g_cmat,
                        int m_size, int n_size ) {

    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n = blockIdx.y;
    Matrix<float,R,WS> *gptuetv;
    Vector<float,R> py, ez, ptuetv;
    __shared__ Matrix<float,R,WS> spy, sez;
    Vector<float,R> cmat[3];

    if (ty == 0) {
#pragma unroll
        for (int r=0; r<R; ++r) {
            cmat[0][r] = __ldg((const float *)&g_cmat[0][r][tx]);
            cmat[1][r] = __ldg((const float *)&g_cmat[1][r][tx]);
            cmat[2][r] = __ldg((const float *)&g_cmat[2][r][tx]);
        }
        gptuetv = (Matrix<float,R,WS> *)&g_ptucheck[m*(n_size+1)+n+1][0][tx];
    } else if (ty == 1) {
#pragma unroll
        for (int r=0; r<R; ++r) {
            cmat[0][r] = __ldg((const float *)&g_cmat[0][r][tx]);
            cmat[1][r] = __ldg((const float *)&g_cmat[1][r][tx]);
            cmat[2][r] = __ldg((const float *)&g_cmat[3][r][tx]);
        }
        gptuetv = (Matrix<float,R,WS> *)&g_etvtilde[m*(n_size+1)+n][0][tx];
    } else if (ty == 2) {
#pragma unroll
        for (int r=0; r<R; ++r)
            spy[r][tx] = __ldg((const float *)&g_py[(n)*(m_size+1)+m+0][r][tx]);
    } else if (ty == 3) {
#pragma unroll
        for (int r=0; r<R; ++r)
            sez[r][tx] = __ldg((const float *)&g_ez[(n)*(m_size+1)+m+1][r][tx]);
    }

    __syncthreads();

    if (ty == 0) {

        if (n == 0) {

#pragma unroll
            for (int r=0; r<R; ++r)
                ptuetv[r] = g_ptucheck[m*(n_size+1)+0][r][tx];

            py = spy.col(0);
            ez = sez.col(0);

            mad_clamp(ptuetv, ez, cmat[0]);
            mad_clamp(ptuetv, py, cmat[1]);

#pragma unroll
            for (int r=0; r<R; ++r)
                g_ptucheck[m*(n_size+1)+0][r][tx] = ptuetv[r];

        }

#pragma unroll
        for (int r=0; r<R; ++r)
            ptuetv[r] = g_ptucheck[m*(n_size+1)+n+1][r][tx];

    } else if (ty == 1) {

        if (n == n_size-1) {

#pragma unroll
            for (int r=0; r<R; ++r)
                ptuetv[r] = g_etvtilde[m*(n_size+1)+n_size][r][tx];

            py = spy.col(WS-1);
            ez = sez.col(WS-1);

            mad_clamp(ptuetv, ez, cmat[0]);
            mad_clamp(ptuetv, py, cmat[1]);

#pragma unroll
            for (int r=0; r<R; ++r)
                g_etvtilde[m*(n_size+1)+n_size][r][tx] = ptuetv[r];

        }

#pragma unroll
        for (int r=0; r<R; ++r)
            ptuetv[r] = g_etvtilde[m*(n_size+1)+n][r][tx];

    }

    if (ty < 2) {

        py = spy.col(tx);
        ez = sez.col(tx);

        fixpet(ptuetv, cmat[2], cmat[0], ez);
        fixpet(ptuetv, cmat[2], cmat[1], py);

        gptuetv->set_col(0, ptuetv);

    }

}

/**
 *  @ingroup gpu
 *  @brief Algorithm 6 stage 5 for clamp
 *
 *  This function computes the algorithm stage \f$6^c.9\f$
 *  (corresponding to the base algorithm 6 stage 5).
 *
 *  @see The base algorithm description in alg6_stage5()
 *
 *  @see [NehabMaximo:2016] cited in alg6()
 *  @param[out] g_out The output 2D image
 *  @param[in] g_py All \f$P_{m,n}(Y)\f$
 *  @param[in] g_ez All \f$E_{m,n}(Z)\f$
 *  @param[in] g_ptu All \f$P^T_{m,n}(U)\f$
 *  @param[in] g_etv All \f$E^T_{m,n}(V)\f$
 *  @param[in] inv_width Image width inversed (1/w)
 *  @param[in] inv_height Image height inversed (1/h)
 *  @param[in] m_size The big M (number of row blocks)
 *  @param[in] n_size The big N (number of column blocks)
 *  @param[in] out_stride Image output stride for memory width alignment
 *  @tparam R Filter order
 */
template <int R>
__global__ __launch_bounds__(WS*NWW, NBCW)
void alg6_clamp_stage5( float *g_out,
                        const Matrix<float,R,WS> *g_py,
                        const Matrix<float,R,WS> *g_ez,
                        const Matrix<float,R,WS> *g_ptu,
                        const Matrix<float,R,WS> *g_etv,
                        float inv_width, float inv_height,
                        int m_size, int n_size,
                        int out_stride ) {

    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n = blockIdx.y;

    __shared__ Matrix<float,WS,WS+1> block;
    read_block<NWW>(block, m, n, inv_width, inv_height);
    __syncthreads();

    float x[32];

    if (ty==0) {

#pragma unroll
        for (int i=0; i<32; ++i)
            x[i] = block[tx][i];

        Vector<float,R> p, e;

#pragma unroll
        for (int r=0; r<R; ++r)
            p[r] = __ldg((const float *)&g_py[n*(m_size+1)+m][r][tx]);

#pragma unroll // calculate block, scan left -> right
        for (int j=0; j<WS; ++j)
            x[j] = fwdI(p, x[j], c_weights);

        for (int r=0; r<R; ++r)
            e[r] = __ldg((const float *)&g_ez[n*(m_size+1)+m+1][r][tx]);

#pragma unroll // calculate block, scan right -> left
        for (int j=WS-1; j>=0; --j)
            x[j] = revI(x[j], e, c_weights);

#pragma unroll // tranpose regs part-1
        for (int i=0; i<32; ++i)
            block[tx][i] = x[i];
#pragma unroll // transpose regs part-2
        for (int i=0; i<32; ++i)
            x[i] = block[i][tx];

#pragma unroll
        for (int r=0; r<R; ++r)
            p[r] = __ldg((const float *)&g_ptu[m*(n_size+1)+n][r][tx]);

#pragma unroll // calculate block, scan top -> bottom
        for (int j=0; j<WS; ++j)
            x[j] = fwdI(p, x[j], c_weights);

#pragma unroll
        for (int r=0; r<R; ++r)
            e[r] = __ldg((const float *)&g_etv[m*(n_size+1)+n+1][r][tx]);

#pragma unroll // calculate block, scan bottom -> top
        for (int j=WS-1; j>=0; --j)
            x[j] = revI(x[j], e, c_weights);

        g_out += ((n+1)*WS-1)*out_stride + m*WS+tx;
#pragma unroll // write block
        for (int i=0; i<WS; ++i, g_out-=out_stride)
            *g_out = x[WS-1-i];

    }

}

/**
 *  @ingroup api_gpu
 *  @brief Compute algorithm 6 for clamp in the GPU
 *
 *  @see [NehabMaximo:2016] cited in alg6()
 *  @param[in,out] h_img The in/output 2D image to compute recursive filtering in host memory
 *  @param[in] width Image width
 *  @param[in] height Image height
 *  @param[in] runtimes Number of run times (1 for debug and 1000 for performance measurements)
 *  @param[in] w Filter weights (feedforward and feedback coefficients)
 *  @tparam R Filter order
 */
template <int R>
__host__
void alg6_clamp( float *h_img,
                 const int& width, const int& height, const int& runtimes,
                 const Vector<float, R+1>& w ) {

    const int B = WS;

    // pre-compute basic alg5 matrices
    Matrix<float,R,R> Ir = identity<float,R,R>();
    Matrix<float,B,R> Zbr = zeros<float,B,R>();
    Matrix<float,R,B> Zrb = zeros<float,R,B>();
    Matrix<float,B,B> Ib = identity<float,B,B>();

    Matrix<float,R,B> AFP_T = fwd(Ir, Zrb, w), ARE_T = rev(Zrb, Ir, w);
    Matrix<float,B,B> AFB_T = fwd(Zbr, Ib, w), ARB_T = rev(Ib, Zbr, w);
    Matrix<float,R,R> AbF_T = tail<R>(AFP_T), AbR_T = head<R>(ARE_T);
    Matrix<float,R,R> AbF = transp(AbF_T), AbR = transp(AbR_T);
    Matrix<float,R,R> HARB_AFP_T = AFP_T*head<R>(ARB_T);
    Matrix<float,R,R> HARB_AFP = transp(HARB_AFP_T);
    Matrix<float,R,B> ARB_AFP_T = AFP_T*ARB_T, TAFB = transp(tail<R>(AFB_T));
    Matrix<float,R,B> HARB_AFB = transp(AFB_T*head<R>(ARB_T));

    const int m_size = (width+WS-1)/WS, n_size = (height+WS-1)/WS;

    // pre-compute clamp matrices
    Matrix<float,R,R> ArF_T = head<R>(AFP_T), AbarF_T;
    
    for (int i=0; i<R; ++i) {
        int j;
        for (j=0; j<i; ++j) AbarF_T[i][j] = 0;
        AbarF_T[i][j] = w[0];
        for (++j; j<R; ++j) AbarF_T[i][j] = w[0]*ArF_T[R-1][j-i-1];
    }

    Matrix<float,R,R> ArR_T = flip(ArF_T), AbarR_T = flip(AbarF_T);

    Matrix<float,R,R> IArF_T = inv(Ir - ArF_T);
    Matrix<float,R,R> IArR_T = inv(Ir - ArR_T);

    // clamp SRRF matrix computation: solve for S^{\infty}_{RRF}
    Matrix<float,R,R> SRRF_T;

    { // thx rod :)
        Matrix<float,R*R,R*R> sys = zeros<float,R*R,R*R>();
        Vector<float,R*R> b;
        for (int i=0; i<R; ++i) {
            for (int j=0; j<R; ++j) {
                int k = R*i+j;
                sys[k][k] = 1;
                for (int q=0; q<R; ++q)
                    for (int p=0; p<R; ++p)
                        sys[k][R*p+q] -= ArF_T[i][p] * ArR_T[q][j];
                b[k] = AbarR_T[i][j];
            }
        }

        solve_inplace(sys, b);

        for (int i=0; i<R; ++i)
            for (int j=0; j<R; ++j)
                SRRF_T[i][j] = b[R*i+j];
    }

    Matrix<float,R,R> AbarFIArF_T = AbarF_T * IArF_T;
    Matrix<float,R,R> ArFSRRF_T = ArF_T * SRRF_T;
    Matrix<float,R,R> AbarFIArFAbarRIArRArFSRRF_T = AbarF_T * IArF_T * ( AbarR_T * IArR_T - ArF_T * SRRF_T );

    // upload to the GPU
    copy_to_symbol(c_weights, w);

    copy_to_symbol(c_AbF_T, AbF_T);
    copy_to_symbol(c_AbR_T, AbR_T);
    copy_to_symbol(c_HARB_AFP_T, HARB_AFP_T);

    copy_to_symbol(c_ARE_T, ARE_T);
    copy_to_symbol(c_ARB_AFP_T, ARB_AFP_T);
    copy_to_symbol(c_TAFB, TAFB);
    copy_to_symbol(c_HARB_AFB, HARB_AFB);

    copy_to_symbol(c_AbarFIArF_T, AbarFIArF_T);
    copy_to_symbol(c_ArFSRRF_T,  ArFSRRF_T);
    copy_to_symbol(c_AbarFIArFAbarRIArRArFSRRF_T, AbarFIArFAbarRIArRArFSRRF_T);

    float inv_width = 1.f/width, inv_height = 1.f/height;
    cudaArray *a_in;
    cudaChannelFormatDesc ccd = cudaCreateChannelDesc<float>();
    cudaMallocArray(&a_in, &ccd, width, height);
    cudaMemcpyToArray(a_in, 0, 0, h_img, width*height*sizeof(float),
                      cudaMemcpyHostToDevice);

    t_in.normalized = true;
    t_in.filterMode = cudaFilterModePoint;
    t_in.addressMode[0] = t_in.addressMode[1] = cudaAddressModeClamp;

    int stride_img = width+WS;
    dvector<float> d_img(height*stride_img);

    dvector< Matrix<float,R,B> > d_pybar((m_size+1)*n_size), d_ezhat((m_size+1)*n_size);
    dvector< Matrix<float,R,B> > d_ptucheck((n_size+1)*m_size), d_etvtilde((n_size+1)*m_size);
    d_pybar.fillzero();
    d_ezhat.fillzero();
    d_ptucheck.fillzero();
    d_etvtilde.fillzero();

    Matrix<float,R,B> h_cmat[4] = { ARE_T, ARB_AFP_T, TAFB, HARB_AFB };
    dvector< Matrix<float,R,B> > d_cmat(h_cmat, 4);

    cudaFuncSetCacheConfig(alg6_clamp_stage1<R>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(alg6_clamp_stage5<R>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(alg6_clamp_stage3<R>, cudaFuncCachePreferL1);

    if (R == 1)
        cudaFuncSetCacheConfig(alg6_clamp_stage2v4<R>, cudaFuncCachePreferL1);
    else if (R == 2)
        cudaFuncSetCacheConfig(alg6_clamp_stage2v4<R>, cudaFuncCachePreferEqual);
    else if (R >= 3)
        cudaFuncSetCacheConfig(alg6_clamp_stage2v4<R>, cudaFuncCachePreferShared);

    double te[5] = {0, 0, 0, 0, 0}; // time elapsed for the five stages
    base_timer *timer[5];
    for (int i = 0; i < 5; ++i)
        timer[i] = new gpu_timer(0, "", false);

    base_timer &timer_total = timers.gpu_add("alg6_clamp", width*height, "iP");

    for(int r = 0; r < runtimes; ++r) {

        if (runtimes == 1) { timer[0]->start(); }

        cudaBindTextureToArray(t_in, a_in);

        alg6_clamp_stage1<<< dim3(m_size, n_size), dim3(WS, NWC) >>>
            ( &d_pybar, &d_ezhat, &d_ptucheck, &d_etvtilde, inv_width, inv_height, m_size, n_size );

        if (runtimes == 1) { timer[0]->stop(); te[0] += timer[0]->elapsed(); timer[1]->start(); }

        alg6_clamp_stage2v4<<< dim3(1, n_size), dim3(WS, NWA) >>>
            ( &d_pybar, &d_ezhat, m_size );

        if (runtimes == 1) { timer[1]->stop(); te[1] += timer[1]->elapsed(); timer[2]->start(); }

        alg6_clamp_stage3<<< dim3(m_size, n_size), dim3(WS, NWARC) >>>
            ( &d_ptucheck, &d_etvtilde, &d_pybar, &d_ezhat, &d_cmat, m_size, n_size );

        if (runtimes == 1) { timer[2]->stop(); te[2] += timer[2]->elapsed(); timer[3]->start(); }

        alg6_clamp_stage2v4<<< dim3(1, m_size), dim3(WS, NWA) >>>
            ( &d_ptucheck, &d_etvtilde, n_size );

        if (runtimes == 1) { timer[3]->stop(); te[3] += timer[3]->elapsed(); timer[4]->start(); }

        alg6_clamp_stage5<<< dim3(m_size, n_size), dim3(WS, NWW) >>>
            ( d_img, &d_pybar, &d_ezhat, &d_ptucheck, &d_etvtilde, inv_width, inv_height, m_size, n_size, stride_img );

        cudaUnbindTexture(t_in);

        if (runtimes == 1) { timer[4]->stop(); te[4] += timer[4]->elapsed(); }

    }

    timer_total.stop();

    if (runtimes > 1) {

        std::cout << std::fixed << (timer_total.data_size()*runtimes)/(double)(timer_total.elapsed()*1024*1024) << std::flush;

    } else {

        timers.gpu_add("stage 1", timer[0]);
        timers.gpu_add("stage 2", timer[1]);
        timers.gpu_add("stage 3", timer[2]);
        timers.gpu_add("stage 4", timer[3]);
        timers.gpu_add("stage 5", timer[4]);
        timers.flush();

    }

    cudaMemcpy2D(h_img, width*sizeof(float), d_img, stride_img*sizeof(float), width*sizeof(float), height, cudaMemcpyDeviceToHost);

}

//==============================================================================
} // namespace gpufilter
//==============================================================================
#endif // ALG6_CLAMP_CUH
//==============================================================================
