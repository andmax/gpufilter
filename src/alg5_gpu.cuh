/**
 *  @file alg5_gpu.cuh
 *  @brief Algorithm 5 in the GPU with borders
 *  @author Andre Maximo
 *  @date Nov, 2012
 *  @copyright The MIT License
 */

#ifndef ALG5_GPU_CUH
#define ALG5_GPU_CUH

#define MST == // measure step time: no: == ; yes: >=
#define LDG // uncomment to use __ldg
#if ORDER==1 || ORDER==2 || ORDER==4
#define REGS // uncomment to use registers
#define GMAT // uncomment to use global constant matrices
#endif

//== INCLUDES ==================================================================

#include "alg4v5v6_gpu.cuh"

//== NAMESPACES ================================================================

namespace gpufilter {

//== IMPLEMENTATION ============================================================

/**
 *  @ingroup gpu
 *  @brief Algorithm 5 step 3
 *
 *  This function computes the algorithm step 5.3 (corresponding to
 *  the steps 5.4 and 5.5 in [NehabEtAl:2011]) following:
 *
 *  \li In parallel for all \f$m\f$, sequentially for each \f$n\f$,
 *  compute and store feedbacks \f$P^T_{m,n-1}(U)\f$ and
 *  \f$E^T_{m,n+1}(V)\f$ according to equations (26) and (28).
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @note This follows the improved base-line implementation in [NehabMaximo:2016]
 *  @see [NehabEtAl:2011] cited in alg5() and [NehabMaximo:2016] cited in alg6()
 *  @param[in,out] g_ptucheck All \f$P^T_{m,n}(U)\f$
 *  @param[in,out] g_etvtilde All \f$E^T_{m,n}(V)\f$
 *  @param[in] g_py All \f$P_{m,n}(Y)\f$
 *  @param[in] g_ez All \f$E_{m,n}(Z)\f$
 *  @param[in] g_cmat Constant pre-computed matrices on equations (27) and (29)
 *  @param[in] m_size The big M (number of row blocks)
 *  @param[in] n_size The big N (number of column blocks)
 *  @tparam R Filter order
 */
template <int R>
__global__ __launch_bounds__(WS*NWAC, NBA)
void alg5_step3( Matrix<float,R,WS> *g_ptucheck,
                 Matrix<float,R,WS> *g_etvtilde,
                 const Matrix<float,R,WS> *g_py,
                 const Matrix<float,R,WS> *g_ez,
#ifdef GMAT
                  const Matrix<float,R,WS> *g_cmat,
#endif
                  int m_size, int n_size ) {

    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n;
    Matrix<float,R,WS> *gptucheck, *getvtilde;
    Vector<float,R> ptu, etv, ptucheck, etvtilde;
    Vector<float,R> py, ez;
    __shared__ Matrix<float,R,WS> sptucheck[NWAC], setvtilde[NWAC];
    __shared__ Matrix<float,R,WS> spy[NWAC], sez[NWAC];

#ifdef GMAT
    Vector<float,R> cmat[3];

    if (ty == 0) {
#pragma unroll
        for (int r=0; r<R; ++r) {
#ifdef LDG
            cmat[0][r] = __ldg((const float *)&g_cmat[0][r][tx]);
            cmat[1][r] = __ldg((const float *)&g_cmat[1][r][tx]);
            cmat[2][r] = __ldg((const float *)&g_cmat[2][r][tx]);
#else
            cmat[0][r] = g_cmat[0][r][tx];
            cmat[1][r] = g_cmat[1][r][tx];
            cmat[2][r] = g_cmat[2][r][tx];
#endif
        }
    }
#endif

    // Pt(ucheck) -> Pt(u) processing ------------------------------------------
    n = 0;
    gptucheck = (Matrix<float,R,WS> *)&g_ptucheck[m*(n_size+1)+n+ty+1][0][tx];
    ptu = zeros<float,R>();

    for (; n < n_size; n += NWAC) { // for all image blocks

        if (n+ty < n_size) { // using smem as cache
            sptucheck[ty].set_col(tx, gptucheck->col(0));
#ifdef LDG
#pragma unroll
            for (int r=0; r<R; ++r) {
                spy[ty][r][tx] = __ldg((const float *)&g_py[(n+ty)*(m_size+1)+m+0][r][tx]);
                sez[ty][r][tx] = __ldg((const float *)&g_ez[(n+ty)*(m_size+1)+m+1][r][tx]);
            }
#else
            spy[ty].set_col(tx, ((Matrix<float,R,WS> *)&g_py[(n+ty)*(m_size+1)+m+0][0][tx])->col(0));
            sez[ty].set_col(tx, ((Matrix<float,R,WS> *)&g_ez[(n+ty)*(m_size+1)+m+1][0][tx])->col(0));
#endif
        }

        __syncthreads(); // wait to load smem

        if (ty == 0) {
#pragma unroll // adjust ptucheck top -> bottom
            for (int w = 0; w < NWAC; ++w) {

                ptucheck = sptucheck[w].col(tx);
                py = spy[w].col(tx);
                ez = sez[w].col(tx);

                ptu = ptucheck + ptu * c_AbF_T;

#ifdef GMAT
                fixpet(ptu, cmat[2], cmat[0], ez);
                fixpet(ptu, cmat[2], cmat[1], py);
#else
                fixpet(ptu, c_TAFB, c_ARE_T, ez);
                fixpet(ptu, c_TAFB, c_ARB_AFP_T, py);
#endif
                sptucheck[w].set_col(tx, ptu);

            }
        }

        __syncthreads(); // wait to store result in gmem

        if (n+ty < n_size)
            gptucheck->set_col(0, sptucheck[ty].col(tx));

        gptucheck += NWAC;

    }

#ifdef GMAT
    if (ty == 0) {
#pragma unroll
        for (int r=0; r<R; ++r) {
#ifdef LDG
            cmat[2][r] = __ldg((const float *)&g_cmat[3][r][tx]);
#else
            cmat[2][r] = g_cmat[3][r][tx];
#endif
        }
    }
#endif

    // Et(vtilde) -> Et(v) processing ------------------------------------------
    n = n_size-1;
    getvtilde = (Matrix<float,R,WS> *)&g_etvtilde[m*(n_size+1)+n-ty][0][tx];
    gptucheck = (Matrix<float,R,WS> *)&g_ptucheck[m*(n_size+1)+n-ty][0][tx];
    etv = zeros<float,R>();

    for (; n >= 0; n -= NWAC) { // for all image blocks

        if (n-ty >= 0) { // using smem as cache
            setvtilde[ty].set_col(tx, getvtilde->col(0));
            sptucheck[ty].set_col(tx, gptucheck->col(0));
#ifdef LDG
#pragma unroll
            for (int r=0; r<R; ++r) {
                spy[ty][r][tx] = __ldg((const float *)&g_py[(n-ty)*(m_size+1)+m+0][r][tx]);
                sez[ty][r][tx] = __ldg((const float *)&g_ez[(n-ty)*(m_size+1)+m+1][r][tx]);
            }
#else
            spy[ty].set_col(tx, ((Matrix<float,R,WS> *)&g_py[(n-ty)*(m_size+1)+m+0][0][tx])->col(0));
            sez[ty].set_col(tx, ((Matrix<float,R,WS> *)&g_ez[(n-ty)*(m_size+1)+m+1][0][tx])->col(0));
#endif
        }

        __syncthreads(); // wait to load smem

        if (ty == 0) {
#pragma unroll // adjust etvtilde bottom -> top
            for (int w = 0; w < NWAC; ++w) {

                etvtilde = setvtilde[w].col(tx);
                ptu = sptucheck[w].col(tx);
                py = spy[w].col(tx);
                ez = sez[w].col(tx);

                etv = etvtilde + etv * c_AbR_T + ptu * c_HARB_AFP_T;

#ifdef GMAT
                fixpet(etv, cmat[2], cmat[0], ez);
                fixpet(etv, cmat[2], cmat[1], py);
#else
                fixpet(etv, c_HARB_AFB, c_ARE_T, ez);
                fixpet(etv, c_HARB_AFB, c_ARB_AFP_T, py);
#endif

                setvtilde[w].set_col(tx, etv);

            }
        }

        __syncthreads(); // wait to store result in gmem

        if (n-ty >= 0)
            getvtilde->set_col(0, setvtilde[ty].col(tx));

        getvtilde -= NWAC;
        gptucheck -= NWAC;

    }

}

/**
 *  @ingroup api_gpu
 *  @brief Compute algorithm 5 in the GPU
 *
 *  @see [NehabEtAl:2011] cited in alg5() and [NehabMaximo:2016] cited in alg6()
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
void alg5_gpu( float *h_img,
               int width, int height, int runtimes,
               const Vector<float, R+1>& w,
               int border=0,
               BorderType border_type=CLAMP_TO_ZERO ) {

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

    int m_size = (width+WS-1)/WS, n_size = (height+WS-1)/WS;

    if (BORDER) {
        int border_left, border_top, border_right, border_bottom;
        calc_borders(&border_left, &border_top, &border_right, &border_bottom, 
                     width, height, border);
        int ewidth = width+border_left+border_right,
            eheight = height+border_top+border_bottom;

        m_size = (ewidth+WS-1)/WS;
        n_size = (eheight+WS-1)/WS;

        copy_to_symbol(c_border, border);
    }

    // upload to the GPU
    copy_to_symbol(c_weights, w);

    copy_to_symbol(c_AbF_T, AbF_T);
    copy_to_symbol(c_AbR_T, AbR_T);
    copy_to_symbol(c_HARB_AFP_T, HARB_AFP_T);

    copy_to_symbol(c_ARE_T, ARE_T);
    copy_to_symbol(c_ARB_AFP_T, ARB_AFP_T);
    copy_to_symbol(c_TAFB, TAFB);
    copy_to_symbol(c_HARB_AFB, HARB_AFB);

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
    dvector< Matrix<float,R,B> > d_pybar((m_size+1)*n_size), d_ezhat((m_size+1)*n_size);
    dvector< Matrix<float,R,B> > d_ptucheck((n_size+1)*m_size), d_etvtilde((n_size+1)*m_size);
    d_pybar.fillzero();
    d_ezhat.fillzero();
    d_ptucheck.fillzero();
    d_etvtilde.fillzero();

    // the following matrices can not be read from constant memory as one warp
    // will access each column by a different thread, serializing access and
    // hurting performance, the solution is to store them in global memory
    // and manage to have them in L1 cache as soon as possible;
    // constant r x b matrices: ARE_T, ARB_AFP_T, TAFB, HARB_AFB
    Matrix<float,R,B> h_cmat[4] = { ARE_T, ARB_AFP_T, TAFB, HARB_AFB };
    dvector< Matrix<float,R,B> > d_cmat(h_cmat, 4);

    cudaFuncSetCacheConfig(alg5v6_step1<BORDER,R>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(alg5v6_step4v5<BORDER,R>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(alg5_step3<R>, cudaFuncCachePreferShared);

    if (R == 1)
        cudaFuncSetCacheConfig(alg4v5v6_step2v4<R>, cudaFuncCachePreferL1);
    else if (R == 2)
        cudaFuncSetCacheConfig(alg4v5v6_step2v4<R>, cudaFuncCachePreferEqual);
    else if (R >= 3)
        cudaFuncSetCacheConfig(alg4v5v6_step2v4<R>, cudaFuncCachePreferShared);

    double te[4] = {0, 0, 0, 0}; // time elapsed for the four steps
    base_timer *timer[4];
    for (int i = 0; i < 4; ++i)
        timer[i] = new gpu_timer(0, "", false);

    base_timer &timer_total = timers.gpu_add("alg5_gpu", width*height, "iP");

    for(int r = 0; r < runtimes; ++r) {

        if (runtimes MST 1) { timer[0]->start(); }

        cudaBindTextureToArray(t_in, a_in);

        alg5v6_step1<BORDER><<< dim3(m_size, n_size), dim3(WS, NWC) >>>
            ( &d_pybar, &d_ezhat, &d_ptucheck, &d_etvtilde, inv_width, inv_height, m_size, n_size );

        if (runtimes MST 1) { timer[0]->stop(); te[0] += timer[0]->elapsed(); timer[1]->start(); }

        alg4v5v6_step2v4<<< dim3(1, n_size), dim3(WS, NWA) >>>
            ( &d_pybar, &d_ezhat, m_size );

        if (runtimes MST 1) { timer[1]->stop(); te[1] += timer[1]->elapsed(); timer[2]->start(); }

        alg5_step3<<< dim3(m_size, 1), dim3(WS, NWAC) >>>
            ( &d_ptucheck, &d_etvtilde, &d_pybar, &d_ezhat,
#ifdef GMAT
              &d_cmat,
#endif
              m_size, n_size );

        if (runtimes MST 1) { timer[2]->stop(); te[2] += timer[2]->elapsed(); timer[3]->start(); }

        alg5v6_step4v5<BORDER><<< dim3(m_size, n_size), dim3(WS, NWW) >>>
            ( d_img, &d_pybar, &d_ezhat, &d_ptucheck, &d_etvtilde, inv_width, inv_height, m_size, n_size, stride_img );

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
#endif // ALG5_GPU_CUH
//==============================================================================
