/**
 *  @file alg5varc_gpu.cuh
 *  @brief Algorithm 5 in the GPU using variable coefficients
 *  @author Andre Maximo
 *  @date Jan, 2013
 *  @copyright The MIT License
 */

#ifndef ALG5VARC_GPU_CUH
#define ALG5VARC_GPU_CUH

//== NAMESPACES ================================================================

namespace gpufilter {

// basics
__constant__ float c_b0f, c_b0r;

//== IMPLEMENTATION ============================================================

/**
 *  @ingroup gpu
 *  @brief Algorithm 5 varying coefficients stage 1 for border blocks
 *  @note This follows the improved base-line implementation in [NehabMaximo:2016]
 *  @see alg5_stage1()
 *  @see [NehabHoppe:2011] cited in nehab_hoppe_tr2011_sec6()
 *  @param[out] g_pybar All \f$P_{m,n}(Y)\f$
 *  @param[out] g_ezhat All \f$E_{m,n}(Z)\f$
 *  @param[out] g_ptucheck All \f$P^T_{m,n}(U)\f$
 *  @param[out] g_etvtilde All \f$E^T_{m,n}(V)\f$
 *  @param[in] inv_width Image width inversed (1/w)
 *  @param[in] inv_height Image height inversed (1/h)
 *  @param[in] m_size The big M (number of row blocks)
 *  @param[in] n_size The big N (number of column blocks)
 *  @param[in] g_aw The feedback coefficients in width direction
 *  @param[in] g_ah The feedback coefficients in height direction
 *  @tparam R Filter order
 */
template <int R>
__global__ __launch_bounds__(WS*NWC, NBCW)
void alg5varc_stage1_bor( Matrix<float,R,WS> *g_pybar, 
                          Matrix<float,R,WS> *g_ezhat,
                          Matrix<float,R,WS> *g_ptucheck,
                          Matrix<float,R,WS> *g_etvtilde,
                          float inv_width, float inv_height,
                          int m_size, int n_size,
                          const float *g_aw, const float *g_ah ) {

    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n = blockIdx.y;

    if ((m >= c_border && m <= m_size-1-c_border) &&
        (n >= c_border && n <= n_size-1-c_border)) // at the middle
        return;

    __shared__ Matrix<float,WS,WS+1> block;
    read_block<NWC>(block, m, n, inv_width, inv_height);
    __syncthreads();

    __shared__ float s_aw[WS+1], s_ah[WS+1];
    g_aw+=m*WS+tx+1;
    g_ah+=n*WS+tx+1;

    float x[32]; // 32 regs

    if (ty==0) {

#pragma unroll
        for (int i=0; i<32; ++i)
            x[i] = block[tx][i];

        s_aw[tx+1] = __ldg(g_aw);
        s_ah[tx+1] = __ldg(g_ah);

        Vector<float,R> p = zeros<float,R>();

#pragma unroll // calculate pybar, scan left -> right
        for (int j=0; j<WS; ++j)
            x[j] = fwd(p, x[j], c_b0f, s_aw+j);

        g_pybar[n*(m_size+1)+m+1].set_col(tx, p);

        Vector<float,R> e = zeros<float,R>();

#pragma unroll // calculate ezhat, scan right -> left
        for (int j=WS-1; j>=0; --j)
            x[j] = rev(x[j], e, c_b0r, s_aw+j+1);

        g_ezhat[n*(m_size+1)+m].set_col(tx, e);

#pragma unroll // transpose regs part-1
        for (int i=0; i<32; ++i)
            block[tx][i] = x[i];
#pragma unroll // transpose regs part-2
        for (int i=0; i<32; ++i)
            x[i] = block[i][tx];

        p = zeros<float,R>();

#pragma unroll // calculate ptucheck, scan top -> bottom
        for (int j=0; j<WS; ++j)
            x[j] = fwd(p, x[j], c_b0f, s_ah+j);

        g_ptucheck[m*(n_size+1)+n+1].set_col(tx, p);

        e = zeros<float,R>();

#pragma unroll // calculate etvtilde, scan bottom -> top
        for (int j=WS-1; j>=0; --j)
            rev(x[j], e, c_b0r, s_ah+j+1);

        g_etvtilde[m*(n_size+1)+n].set_col(tx, e);

    }

}

/**
 *  @ingroup gpu
 *  @brief Algorithm 5 varying coefficients stage 1 for middle blocks
 *  @note This follows the improved base-line implementation in [NehabMaximo:2016]
 *  @see alg5_stage1()
 *  @see [NehabHoppe:2011] cited in nehab_hoppe_tr2011_sec6()
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
void alg5varc_stage1_mid( Matrix<float,R,WS> *g_pybar,
                          Matrix<float,R,WS> *g_ezhat,
                          Matrix<float,R,WS> *g_ptucheck,
                          Matrix<float,R,WS> *g_etvtilde,
                          float inv_width, float inv_height,
                          int m_size, int n_size ) {

    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n = blockIdx.y;

    if ((m < c_border || m > m_size-1-c_border) ||
        (n < c_border || n > n_size-1-c_border)) // at borders
        return;

    __shared__ Matrix<float,WS,WS+1> block;
    read_block<NWC>(block, m, n, inv_width, inv_height);
    __syncthreads();

    float x[32]; // 32 regs

    if (ty==0) {

#pragma unroll
        for (int i=0; i<32; ++i)
            x[i] = block[tx][i];

        Vector<float,R> p = zeros<float,R>();

#pragma unroll // calculate pybar, scan left -> right
        for (int j=0; j<WS; ++j)
            x[j] = fwd(p, x[j], c_b0f, &c_weights[1]);

        g_pybar[n*(m_size+1)+m+1].set_col(tx, p);
        
        Vector<float,R> e = zeros<float,R>();

#pragma unroll // calculate ezhat, scan right -> left
        for (int j=WS-1; j>=0; --j)
            x[j] = rev(x[j], e, c_b0r, &c_weights[1]);

        g_ezhat[n*(m_size+1)+m].set_col(tx, e);

#pragma unroll // transpose regs part-1
        for (int i=0; i<32; ++i)
            block[tx][i] = x[i];
#pragma unroll // transpose regs part-2
        for (int i=0; i<32; ++i)
            x[i] = block[i][tx];
            
        p = zeros<float,R>();

#pragma unroll // calculate ptucheck, scan top -> bottom
        for (int j=0; j<WS; ++j)
            x[j] = fwd(p, x[j], c_b0f, &c_weights[1]);

        g_ptucheck[m*(n_size+1)+n+1].set_col(tx, p);

        e = zeros<float,R>();

#pragma unroll // calculate etvtilde, scan bottom -> top
        for (int j=WS-1; j>=0; --j)
            rev(x[j], e, c_b0r, &c_weights[1]);

        g_etvtilde[m*(n_size+1)+n].set_col(tx, e);

    }

}

/**
 *  @ingroup gpu
 *  @brief Algorithm 5 varying coefficients stage 2 for border blocks
 *  @note This follows the improved base-line implementation in [NehabMaximo:2016]
 *  @see alg5_stage2()
 *  @see [NehabHoppe:2011] cited in nehab_hoppe_tr2011_sec6()
 *  @param[in,out] g_pybar All \f$P_{m,n}(Y)\f$
 *  @param[in,out] g_ezhat All \f$E_{m,n}(Z)\f$
 *  @param[in] m_size The big M (number of row blocks)
 *  @param[in] n_size The big N (number of column blocks)
 *  @param[in] g_AbF_T The AbF_T varying matrix per block
 *  @param[in] g_AbR_T The AbR_T varying matrix per block
 *  @param[in] g_HARB_AFP_T The HARB_AFP_T varying matrix per block
 *  @tparam R Filter order
 */
template <int R>
__global__ __launch_bounds__(WS*NWA, NBA)
void alg5varc_stage2_bor( Matrix<float,R,WS> *g_pybar,
                          Matrix<float,R,WS> *g_ezhat,
                          int m_size, int n_size,
                          const Matrix<float,R,R> *g_AbF_T,
                          const Matrix<float,R,R> *g_AbR_T,
                          const Matrix<float,R,R> *g_HARB_AFP_T ) {

    int tx = threadIdx.x, ty = threadIdx.y, m, n = blockIdx.y;

    if (n >= c_border && n <= n_size-1-c_border) // at the middle
        return;

    Matrix<float,R,WS> *gpybar, *gezhat;
    Vector<float,R> py, ez, pybar, ezhat;
    __shared__ Matrix<float,R,WS> spybar[NWA], sezhat[NWA];

    // P(ybar) -> P(y) processing ----------------------------------------------
    m = 0;
    gpybar = (Matrix<float,R,WS> *)&g_pybar[n*(m_size+1)+m+ty+1][0][tx];
    py = zeros<float,R>();

    for (; m < m_size; m += NWA) { // for all image blocks

        if (m+ty < m_size) spybar[ty].set_col(tx, gpybar->col(0)); // using smem as cache

        __syncthreads(); // wait to load smem

        if (ty == 0) {
#pragma unroll // adjust pybar left -> right
            for (int w = 0; w < NWA; ++w) {

                pybar = spybar[w].col(tx);

                py = pybar + py * g_AbF_T[m+w];

                spybar[w].set_col(tx, py);

            }
        }

        __syncthreads(); // wait to store result in gmem

        if (m+ty < m_size) gpybar->set_col(0, spybar[ty].col(tx));

        gpybar += NWA;

    }

    // E(zhat) -> E(z) processing ----------------------------------------------
    m = m_size-1;
    gezhat = (Matrix<float,R,WS> *)&g_ezhat[n*(m_size+1)+m-ty][0][tx];
    gpybar = (Matrix<float,R,WS> *)&g_pybar[n*(m_size+1)+m-ty][0][tx];
    ez = zeros<float,R>();

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

                ez = ezhat + py * g_HARB_AFP_T[m-w] + ez * g_AbR_T[m-w];

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
 *  @brief Algorithm 5 varying coefficients stage 2 for middle blocks
 *  @note This follows the improved base-line implementation in [NehabMaximo:2016]
 *  @see alg5_stage2()
 *  @see [NehabHoppe:2011] cited in nehab_hoppe_tr2011_sec6()
 *  @param[in,out] g_pybar All \f$P_{m,n}(Y)\f$
 *  @param[in,out] g_ezhat All \f$E_{m,n}(Z)\f$
 *  @param[in] m_size The big M (number of row blocks)
 *  @param[in] n_size The big N (number of column blocks)
 *  @tparam R Filter order
 */
template <int R>
__global__ __launch_bounds__(WS*NWA, NBA)
void alg5varc_stage2_mid( Matrix<float,R,WS> *g_pybar,
                          Matrix<float,R,WS> *g_ezhat,
                          int m_size, int n_size ) {

    int tx = threadIdx.x, ty = threadIdx.y, m, n = blockIdx.y;

    if (n < c_border || n > n_size-1-c_border) // at borders
        return;

    Matrix<float,R,WS> *gpybar, *gezhat;
    Vector<float,R> py, ez, pybar, ezhat;
    __shared__ Matrix<float,R,WS> spybar[NWA], sezhat[NWA];

    // P(ybar) -> P(y) processing ----------------------------------------------
    m = 0;
    gpybar = (Matrix<float,R,WS> *)&g_pybar[n*(m_size+1)+m+ty+1][0][tx];
    py = zeros<float,R>();

    for (; m < m_size; m += NWA) { // for all image blocks

        if (m+ty < m_size) spybar[ty].set_col(tx, gpybar->col(0)); // using smem as cache

        __syncthreads(); // wait to load smem

        if (ty == 0) {
#pragma unroll // adjust pybar left -> right
            for (int w = 0; w < NWA; ++w) {

                pybar = spybar[w].col(tx);

                py = pybar + py * c_AbF_T;

                spybar[w].set_col(tx, py);

            }
        }

        __syncthreads(); // wait to store result in gmem

        if (m+ty < m_size) gpybar->set_col(0, spybar[ty].col(tx));

        gpybar += NWA;

    }

    // E(zhat) -> E(z) processing ----------------------------------------------
    m = m_size-1;
    gezhat = (Matrix<float,R,WS> *)&g_ezhat[n*(m_size+1)+m-ty][0][tx];
    gpybar = (Matrix<float,R,WS> *)&g_pybar[n*(m_size+1)+m-ty][0][tx];
    ez = zeros<float,R>();

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
 *  @brief Algorithm 5 varying coefficients stage 3 for border blocks
 *  @note This follows the improved base-line implementation in [NehabMaximo:2016]
 *  @see alg5_stage3()
 *  @see [NehabHoppe:2011] cited in nehab_hoppe_tr2011_sec6()
 *  @param[in,out] g_ptucheck All \f$P^T_{m,n}(U)\f$
 *  @param[in,out] g_etvtilde All \f$E^T_{m,n}(V)\f$
 *  @param[in] g_pybar All \f$P_{m,n}(Y)\f$
 *  @param[in] g_ezhat All \f$E_{m,n}(Z)\f$
 *  @param[in] m_size The big M (number of row blocks)
 *  @param[in] n_size The big N (number of column blocks)
 *  @param[in] g_AbF The AbF varying matrix per block
 *  @param[in] g_TAFB The TAFB varying matrix per block
 *  @param[in] g_ARE_T The ARE_T varying matrix per block
 *  @param[in] g_ARB_AFP_T The ARB_AFP_T varying matrix per block
 *  @param[in] g_AbR The AbR varying matrix per block
 *  @param[in] g_HARB_AFP The HARB_AFP varying matrix per block
 *  @param[in] g_HARB_AFB The HARB_AFB varying matrix per block
 *  @tparam R Filter order
 */
template <int R>
__global__ __launch_bounds__(WS*NWA, NBA)
void alg5varc_stage3_bor( Matrix<float,R,WS> *g_ptucheck, 
                          Matrix<float,R,WS> *g_etvtilde,
                          Matrix<float,R,WS> *g_py,
                          Matrix<float,R,WS> *g_ez,
                          int m_size, int n_size,
                          const Matrix<float,R,R> *g_AbF,
                          const Matrix<float,R,WS> *g_TAFB,
                          const Matrix<float,R,WS> *g_ARE_T,
                          const Matrix<float,R,WS> *g_ARB_AFP_T,
                          const Matrix<float,R,R> *g_AbR,
                          const Matrix<float,R,R> *g_HARB_AFP,
                          const Matrix<float,R,WS> *g_HARB_AFB ) {

    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n;

    if (m >= c_border && m <= m_size-1-c_border) // at the middle
        return;

    Matrix<float,R,WS> *gptucheck, *getvtilde;
    Vector<float,R> ptu, etv, ptucheck, etvtilde;
    Matrix<float,R,WS> *gpy, *gez;
    Vector<float,R> py, ez;
    __shared__ Matrix<float,R,WS> sptucheck[NWA], setvtilde[NWA];
    __shared__ Matrix<float,R,WS> spy[NWA], sez[NWA];

    // Pt(ucheck) -> Pt(u) processing ------------------------------------------
    n = 0;
    gptucheck = (Matrix<float,R,WS> *)&g_ptucheck[m*(n_size+1)+n+ty+1][0][tx];
    gpy = (Matrix<float,R,WS> *)&g_py[(n+ty)*(m_size+1)+m+0][0][tx];
    gez = (Matrix<float,R,WS> *)&g_ez[(n+ty)*(m_size+1)+m+1][0][tx];
    ptu = zeros<float,R>();

    for (; n < n_size; n += NWA) { // for all image blocks

        if (n+ty < n_size) { // using smem as cache
            sptucheck[ty].set_col(tx, gptucheck->col(0));
            spy[ty].set_col(tx, gpy->col(0));
            sez[ty].set_col(tx, gez->col(0));
        }

        __syncthreads(); // wait to load smem

        if (ty == 0) {
#pragma unroll // adjust ptucheck top -> bottom
            for (int w = 0; w < NWA; ++w) {

                ptucheck = sptucheck[w].col(tx);
                py = spy[w].col(tx);
                ez = sez[w].col(tx);

                ptu = ptucheck + g_AbF[n+w] * ptu;

                fixpet(ptu, g_TAFB[n+w], g_ARE_T[m], ez);
                fixpet(ptu, g_TAFB[n+w], g_ARB_AFP_T[m], py);

                sptucheck[w].set_col(tx, ptu);

            }
        }

        __syncthreads(); // wait to store result in gmem

        if (n+ty < n_size) gptucheck->set_col(0, sptucheck[ty].col(tx));

        gptucheck += NWA;
        gpy += NWA*(m_size+1);
        gez += NWA*(m_size+1);

    }

    // Et(vtilde) -> Et(v) processing ------------------------------------------
    n = n_size-1;
    getvtilde = (Matrix<float,R,WS> *)&g_etvtilde[m*(n_size+1)+n-ty][0][tx];
    gptucheck = (Matrix<float,R,WS> *)&g_ptucheck[m*(n_size+1)+n-ty][0][tx];
    gpy = (Matrix<float,R,WS> *)&g_py[(n-ty)*(m_size+1)+m][0][tx];
    gez = (Matrix<float,R,WS> *)&g_ez[(n-ty)*(m_size+1)+m+1][0][tx];
    etv = zeros<float,R>();

    for (; n >= 0; n -= NWA) { // for all image blocks

        if (n-ty >= 0) { // using smem as cache
            setvtilde[ty].set_col(tx, getvtilde->col(0));
            sptucheck[ty].set_col(tx, gptucheck->col(0));
            spy[ty].set_col(tx, gpy->col(0));
            sez[ty].set_col(tx, gez->col(0));
        }

        __syncthreads(); // wait to load smem

        if (ty == 0) {
#pragma unroll // adjust etvtilde bottom -> top
            for (int w = 0; w < NWA; ++w) {

                etvtilde = setvtilde[w].col(tx);
                ptu = sptucheck[w].col(tx);
                py = spy[w].col(tx);
                ez = sez[w].col(tx);

                etv = etvtilde + g_AbR[n-w] * etv + g_HARB_AFP[n-w] * ptu;

                fixpet(etv, g_HARB_AFB[n-w], g_ARE_T[m], ez);
                fixpet(etv, g_HARB_AFB[n-w], g_ARB_AFP_T[m], py);

                setvtilde[w].set_col(tx, etv);

            }
        }

        __syncthreads(); // wait to store result in gmem

        if (n-ty >= 0) getvtilde->set_col(0, setvtilde[ty].col(tx));

        getvtilde -= NWA;
        gptucheck -= NWA;
        gpy -= NWA*(m_size+1);
        gez -= NWA*(m_size+1);

    }

}

/**
 *  @ingroup gpu
 *  @brief Algorithm 5 varying coefficients stage 3 for middle blocks
 *  @note This follows the improved base-line implementation in [NehabMaximo:2016]
 *  @see alg5_stage3()
 *  @see [NehabHoppe:2011] cited in nehab_hoppe_tr2011_sec6()
 *  @param[in,out] g_ptucheck All \f$P^T_{m,n}(U)\f$
 *  @param[in,out] g_etvtilde All \f$E^T_{m,n}(V)\f$
 *  @param[in] g_pybar All \f$P_{m,n}(Y)\f$
 *  @param[in] g_ezhat All \f$E_{m,n}(Z)\f$
 *  @param[in] m_size The big M (number of row blocks)
 *  @param[in] n_size The big N (number of column blocks)
 *  @tparam R Filter order
 */
template <int R>
__global__ __launch_bounds__(WS*NWA, NBA)
void alg5varc_stage3_mid( Matrix<float,R,WS> *g_ptucheck, 
                          Matrix<float,R,WS> *g_etvtilde,
                          const Matrix<float,R,WS> *g_py,
                          const Matrix<float,R,WS> *g_ez,
                          int m_size, int n_size ) {

    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n;

    if (m < c_border || m > m_size-1-c_border) // at borders
        return;

    Matrix<float,R,WS> *gptucheck, *getvtilde;
    Vector<float,R> ptu, etv, ptucheck, etvtilde;
    Vector<float,R> py, ez;
    __shared__ Matrix<float,R,WS> sptucheck[NWA], setvtilde[NWA];
    __shared__ Matrix<float,R,WS> spy[NWA], sez[NWA];

    // Pt(ucheck) -> Pt(u) processing ------------------------------------------
    n = 0;
    gptucheck = (Matrix<float,R,WS> *)&g_ptucheck[m*(n_size+1)+n+ty+1][0][tx];
    ptu = zeros<float,R>();

    for (; n < n_size; n += NWA) { // for all image blocks

        if (n+ty < n_size) { // using smem as cache
            sptucheck[ty].set_col(tx, gptucheck->col(0));
            for (int r=0; r<R; ++r) {
                spy[ty][r][tx] = __ldg((const float *)&g_py[(n+ty)*(m_size+1)+m+0][r][tx]);
                sez[ty][r][tx] = __ldg((const float *)&g_ez[(n+ty)*(m_size+1)+m+1][r][tx]);
            }
        }

        __syncthreads(); // wait to load smem

        if (ty == 0) {
#pragma unroll // adjust ptucheck top -> bottom
            for (int w = 0; w < NWA; ++w) {

                ptucheck = sptucheck[w].col(tx);
                py = spy[w].col(tx);
                ez = sez[w].col(tx);

                ptu = ptucheck + ptu * c_AbF_T;

                fixpet(ptu, c_TAFB, c_ARE_T, ez);
                fixpet(ptu, c_TAFB, c_ARB_AFP_T, py);

                sptucheck[w].set_col(tx, ptu);

            }
        }

        __syncthreads(); // wait to store result in gmem

        if (n+ty < n_size) gptucheck->set_col(0, sptucheck[ty].col(tx));

        gptucheck += NWA;

    }

    // Et(vtilde) -> Et(v) processing ------------------------------------------
    n = n_size-1;
    getvtilde = (Matrix<float,R,WS> *)&g_etvtilde[m*(n_size+1)+n-ty][0][tx];
    gptucheck = (Matrix<float,R,WS> *)&g_ptucheck[m*(n_size+1)+n-ty][0][tx];
    etv = zeros<float,R>();

    for (; n >= 0; n -= NWA) { // for all image blocks

        if (n-ty >= 0) { // using smem as cache
            setvtilde[ty].set_col(tx, getvtilde->col(0));
            sptucheck[ty].set_col(tx, gptucheck->col(0));
            for (int r=0; r<R; ++r) {
                spy[ty][r][tx] = __ldg((const float *)&g_py[(n-ty)*(m_size+1)+m+0][r][tx]);
                sez[ty][r][tx] = __ldg((const float *)&g_ez[(n-ty)*(m_size+1)+m+1][r][tx]);
            }
        }

        __syncthreads(); // wait to load smem

        if (ty == 0) {
#pragma unroll // adjust etvtilde bottom -> top
            for (int w = 0; w < NWA; ++w) {

                etvtilde = setvtilde[w].col(tx);
                ptu = sptucheck[w].col(tx);
                py = spy[w].col(tx);
                ez = sez[w].col(tx);

                etv = etvtilde + etv * c_AbR_T + ptu * c_HARB_AFP_T;

                fixpet(etv, c_HARB_AFB, c_ARE_T, ez);
                fixpet(etv, c_HARB_AFB, c_ARB_AFP_T, py);

                setvtilde[w].set_col(tx, etv);

            }
        }

        __syncthreads(); // wait to store result in gmem

        if (n-ty >= 0) getvtilde->set_col(0, setvtilde[ty].col(tx));

        getvtilde -= NWA;
        gptucheck -= NWA;

    }

}

/**
 *  @ingroup gpu
 *  @brief Algorithm 5 varying coefficients stage 4 for border blocks
 *  @note This follows the improved base-line implementation in [NehabMaximo:2016]
 *  @see alg5_stage4()
 *  @see [NehabHoppe:2011] cited in nehab_hoppe_tr2011_sec6()
 *  @param[out] g_out The output 2D image
 *  @param[in] g_pybar All \f$P_{m,n}(Y)\f$
 *  @param[in] g_ezhat All \f$E_{m,n}(Z)\f$
 *  @param[in] g_ptucheck All \f$P^T_{m,n}(U)\f$
 *  @param[in] g_etvtilde All \f$E^T_{m,n}(V)\f$
 *  @param[in] inv_width Image width inversed (1/w)
 *  @param[in] inv_height Image height inversed (1/h)
 *  @param[in] m_size The big M (number of row blocks)
 *  @param[in] n_size The big N (number of column blocks)
 *  @param[in] out_stride Image output stride for memory width alignment
 *  @param[in] g_aw The feedback coefficients in width direction
 *  @param[in] g_ah The feedback coefficients in height direction
 *  @tparam R Filter order
 */
template <int R>
__global__ __launch_bounds__(WS*NWW, NBCW)
void alg5varc_stage4_bor( float *g_out,
                          const Matrix<float,R,WS> *g_py,
                          const Matrix<float,R,WS> *g_ez,
                          const Matrix<float,R,WS> *g_ptu,
                          const Matrix<float,R,WS> *g_etv,
                          float inv_width, float inv_height,
                          int m_size, int n_size,
                          int out_stride,
                          const float *g_aw, const float *g_ah ) {

    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n = blockIdx.y;

    if ((m >= c_border && m <= m_size-1-c_border) &&
        (n >= c_border && n <= n_size-1-c_border)) // at the middle
        return;

    __shared__ float s_aw[WS+1], s_ah[WS+1];

    __shared__ Matrix<float,WS,WS+1> block;
    read_block<NWW>(block, m, n, inv_width, inv_height);
    __syncthreads();

    float x[32];

    if (ty==0) {

#pragma unroll
        for (int i=0; i<32; ++i)
            x[i] = block[tx][i];

        g_aw+=m*WS; g_ah+=n*WS;
        if (tx==0) s_aw[0]=*g_aw;
        if (tx==1) s_ah[0]=*g_ah;
        g_aw+=tx+1; g_ah+=tx+1;

        s_aw[tx+1] = __ldg(g_aw);
        s_ah[tx+1] = __ldg(g_ah);

        Vector<float,R> p, e;

        for (int r=0; r<R; ++r)
            p[r] = __ldg((const float *)&g_py[n*(m_size+1)+m][r][tx]);

#pragma unroll // calculate block, scan left -> right
        for (int j=0; j<WS; ++j)
            x[j] = fwd(p, x[j], c_b0f, s_aw+j);

        for (int r=0; r<R; ++r)
            e[r] = __ldg((const float *)&g_ez[n*(m_size+1)+m+1][r][tx]);

#pragma unroll // calculate block, scan right -> left
        for (int j=WS-1; j>=0; --j)
            x[j] = rev(x[j], e, c_b0r, s_aw+j+1);

#pragma unroll // tranpose regs part-1
        for (int i=0; i<32; ++i)
            block[tx][i] = x[i];
#pragma unroll // transpose regs part-2
        for (int i=0; i<32; ++i)
            x[i] = block[i][tx];

        for (int r=0; r<R; ++r)
            p[r] = __ldg((const float *)&g_ptu[m*(n_size+1)+n][r][tx]);

#pragma unroll // calculate block, scan top -> bottom
        for (int j=0; j<WS; ++j)
            x[j] = fwd(p, x[j], c_b0f, s_ah+j);

        for (int r=0; r<R; ++r)
            e[r] = __ldg((float *)&g_etv[m*(n_size+1)+n+1][r][tx]);

#pragma unroll // calculate block, scan bottom -> top
        for (int j=WS-1; j>=0; --j)
            x[j] = rev(x[j], e, c_b0r, s_ah+j+1);

        g_out += ((n+1)*WS-1)*out_stride + m*WS+tx;
#pragma unroll // write block
        for (int i=0; i<WS; ++i, g_out-=out_stride)
            *g_out = x[WS-1-i];

    }

}

/**
 *  @ingroup gpu
 *  @brief Algorithm 5 varying coefficients stage 4 for middle blocks
 *  @note This follows the improved base-line implementation in [NehabMaximo:2016]
 *  @see alg5_stage4()
 *  @see [NehabHoppe:2011] cited in nehab_hoppe_tr2011_sec6()
 *  @param[out] g_out The output 2D image
 *  @param[in] g_pybar All \f$P_{m,n}(Y)\f$
 *  @param[in] g_ezhat All \f$E_{m,n}(Z)\f$
 *  @param[in] g_ptucheck All \f$P^T_{m,n}(U)\f$
 *  @param[in] g_etvtilde All \f$E^T_{m,n}(V)\f$
 *  @param[in] inv_width Image width inversed (1/w)
 *  @param[in] inv_height Image height inversed (1/h)
 *  @param[in] m_size The big M (number of row blocks)
 *  @param[in] n_size The big N (number of column blocks)
 *  @param[in] out_stride Image output stride for memory width alignment
 *  @tparam R Filter order
 */
template <int R>
__global__ __launch_bounds__(WS*NWW, NBCW)
void alg5varc_stage4_mid( float *g_out,
                          const Matrix<float,R,WS> *g_py,
                          const Matrix<float,R,WS> *g_ez,
                          const Matrix<float,R,WS> *g_ptu,
                          const Matrix<float,R,WS> *g_etv,
                          float inv_width, float inv_height,
                          int m_size, int n_size,
                          int out_stride ) {

    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n = blockIdx.y;

    if ((m < c_border || m > m_size-1-c_border) ||
        (n < c_border || n > n_size-1-c_border)) // at borders
        return;

    __shared__ Matrix<float,WS,WS+1> block;
    read_block<NWW>(block, m, n, inv_width, inv_height);
    __syncthreads();

    float x[32]; // 32 regs

    if (ty==0) {

#pragma unroll
        for (int i=0; i<32; ++i)
            x[i] = block[tx][i];

        Vector<float,R> p, e;

        for (int r=0; r<R; ++r)
            p[r] = __ldg((const float *)&g_py[n*(m_size+1)+m][r][tx]);

#pragma unroll // calculate block, scan left -> right
        for (int j=0; j<WS; ++j)
            x[j] = fwd(p, x[j], c_b0f, &c_weights[1]);

        for (int r=0; r<R; ++r)
            e[r] = __ldg((const float *)&g_ez[n*(m_size+1)+m+1][r][tx]);

#pragma unroll // calculate block, scan right -> left
        for (int j=WS-1; j>=0; --j)
            x[j] = rev(x[j], e, c_b0r, &c_weights[1]);

#pragma unroll // tranpose regs part-1
        for (int i=0; i<32; ++i)
            block[tx][i] = x[i];
#pragma unroll // transpose regs part-2
        for (int i=0; i<32; ++i)
            x[i] = block[i][tx];

        for (int r=0; r<R; ++r)
            p[r] = __ldg((const float *)&g_ptu[m*(n_size+1)+n][r][tx]);

#pragma unroll // calculate block, scan top -> bottom
        for (int j=0; j<WS; ++j)
            x[j] = fwd(p, x[j], c_b0f, &c_weights[1]);

        for (int r=0; r<R; ++r)
            e[r] = __ldg((float *)&g_etv[m*(n_size+1)+n+1][r][tx]);

#pragma unroll // calculate block, scan bottom -> top
        for (int j=WS-1; j>=0; --j)
            x[j] = rev(x[j], e, c_b0r, &c_weights[1]);

        g_out += ((n+1)*WS-1)*out_stride + m*WS+tx;
#pragma unroll // write block
        for (int i=0; i<WS; ++i, g_out-=out_stride)
            *g_out = x[WS-1-i];

    }

}

/**
 *  @ingroup api_gpu
 *  @brief Compute algorithm 5 with varying coefficients in the GPU
 *  @see alg5_gpu()
 *  @param[in,out] h_img The in(out)put 2D image to filter in host memory
 *  @param[in] width Image width
 *  @param[in] height Image height
 *  @param[in] runtimes Number of run times (1 for debug and 1000 for performance measurements)
 *  @param[in] border Number of border blocks (32x32) outside image
 *  @tparam R Filter order
 */
template <int R>
void alg5varc_gpu( float *h_img,
                   int width, int height, int runtimes,
                   int border=1 ) {
    if (R!=1) return;
    const int B = WS;
    const int m_size = (width+B-1)/B, n_size = (height+B-1)/B;

    float *aw=0, *ah=0;
    if (!build_coefficients(aw, width)) { std::cerr << "Error building variable coefficients!\n"; return; }
    if (!build_coefficients(ah, height)) { std::cerr << "Error building variable coefficients!\n"; return; }

    Vector<float, R+1> w;
    w[0] = 0.f; // will not be used
    w[1] = spline::linf;

    // pre-compute basic alg5 matrices
    Matrix<float,R,R> Ir = identity<float,R,R>();
    Matrix<float,B,R> Zbr = zeros<float,B,R>();
    Matrix<float,R,B> Zrb = zeros<float,R,B>();
    Matrix<float,B,B> Ib = identity<float,B,B>();

    // matrices in fwd/rev different than fwdT/revT
    std::vector< Matrix<float,R,B> > AFP_T(m_size), ARE_T(m_size);
    std::vector< Matrix<float,R,R> > AbF_T(m_size), AbR_T(m_size);
    std::vector< Matrix<float,B,B> > AFB_T(m_size), ARB_T(m_size);
    std::vector< Matrix<float,R,R> > HARB_AFP_T(m_size);
    std::vector< Matrix<float,R,B> > ARB_AFP_T(m_size);
    for (int m=0; m<m_size; ++m) {
        AFP_T[m] = fwd(Ir, Zrb, b0f, aw+m*B);
        AbF_T[m] = tail<R>(AFP_T[m]);
        ARE_T[m] = rev(Zrb, Ir, b0r, aw+1+m*B);
        AbR_T[m] = head<R>(ARE_T[m]);
        AFB_T[m] = fwd(Zbr, Ib, b0f, aw+m*B);
        ARB_T[m] = rev(Ib, Zbr, b0r, aw+1+m*B);
        HARB_AFP_T[m] = AFP_T[m]*head<R>(ARB_T[m]);
        ARB_AFP_T[m] = AFP_T[m]*ARB_T[m];
    }

    std::vector< Matrix<float,B,R> > AFP(n_size), ARE(n_size);
    std::vector< Matrix<float,R,R> > AbF(n_size), AbR(n_size);
    std::vector< Matrix<float,B,B> > AFB(n_size), ARB(n_size);
    std::vector< Matrix<float,R,R> > HARB_AFP(n_size);
    std::vector< Matrix<float,R,B> > TAFB(n_size), HARB_AFB(n_size);
    for (int n=0; n<n_size; ++n) {
        AFP[n] = fwdT(Ir, Zbr, b0f, ah+n*B);
        AbF[n] = tailT<R>(AFP[n]);
        ARE[n] = revT(Zbr, Ir, b0r, ah+1+n*B);
        AbR[n] = headT<R>(ARE[n]);
        AFB[n] = fwdT(Zrb, Ib, b0f, ah+n*B);
        ARB[n] = revT(Ib, Zrb, b0r, ah+1+n*B);
        HARB_AFP[n] = headT<R>(ARB[n])*AFP[n];
        TAFB[n] = tailT<R>(AFB[n]);
        HARB_AFB[n] = headT<R>(ARB[n])*AFB[n];
    }

    float inv_width = 1.f/width, inv_height = 1.f/height;

    // upload to the GPU
    copy_to_symbol(c_b0f, b0f);
    copy_to_symbol(c_b0r, b0r);

    copy_to_symbol(c_border, border);

    copy_to_symbol(c_weights, w);

    copy_to_symbol(c_AbF_T, AbF_T[border+1]);
    copy_to_symbol(c_AbR_T, AbR_T[border+1]);
    copy_to_symbol(c_HARB_AFP_T, HARB_AFP_T[border+1]);

    copy_to_symbol(c_ARE_T, ARE_T[border+1]);
    copy_to_symbol(c_ARB_AFP_T, ARB_AFP_T[border+1]);
    copy_to_symbol(c_TAFB, TAFB[border+1]);
    copy_to_symbol(c_HARB_AFB, HARB_AFB[border+1]);

    dvector<float> d_aw(aw, width+1), d_ah(ah, height+1);

    dvector< Matrix<float,R,R> > d_AbF_T(AbF_T), d_AbR_T(AbR_T),
        d_HARB_AFP_T(HARB_AFP_T), d_AbF(AbF), d_AbR(AbR), d_HARB_AFP(HARB_AFP);

    dvector< Matrix<float,R,B> > d_TAFB(TAFB), d_ARE_T(ARE_T),
        d_ARB_AFP_T(ARB_AFP_T), d_HARB_AFB(HARB_AFB);

    cudaArray *a_in;
    cudaChannelFormatDesc ccd = cudaCreateChannelDesc<float>();
    cudaMallocArray(&a_in, &ccd, width, height);
    cudaMemcpyToArray(a_in, 0, 0, h_img, width*height*sizeof(float),
                      cudaMemcpyHostToDevice);

    t_in.normalized = true;
    t_in.filterMode = cudaFilterModePoint;
    t_in.addressMode[0] = t_in.addressMode[1] = cudaAddressModeBorder;

    int stride_img = width+WS;

    dvector<float> d_img(height*stride_img);

    // +1 padding is important even in zero-border to avoid if's in kernels
    dvector< Matrix<float,R,B> > d_pybar((m_size+1)*n_size), d_ezhat((m_size+1)*n_size);
    dvector< Matrix<float,R,B> > d_ptucheck((n_size+1)*m_size), d_etvtilde((n_size+1)*m_size);

    d_pybar.fillzero();
    d_ezhat.fillzero();
    d_ptucheck.fillzero();
    d_etvtilde.fillzero();

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    double te[4] = {0, 0, 0, 0}; // time elapsed for four stages
    base_timer *timer[4];
    for (int i = 0; i < 4; ++i)
        timer[i] = new gpu_timer(0, "", false);

    base_timer &timer_total = timers.gpu_add("alg5varc_gpu", width*height, "iP");

    for(int r = 0; r < runtimes; ++r) {

        if (runtimes == 1) { timer[0]->start(); }

        cudaBindTextureToArray(t_in, a_in);

        alg5varc_stage1_bor<<< dim3(m_size, n_size), dim3(WS, NWC), 0, stream1 >>>
            ( &d_pybar, &d_ezhat, &d_ptucheck, &d_etvtilde,
              inv_width, inv_height, m_size, n_size, &d_aw, &d_ah );
        alg5varc_stage1_mid<<< dim3(m_size, n_size), dim3(WS, NWC), 0, stream2 >>>
            ( &d_pybar, &d_ezhat, &d_ptucheck, &d_etvtilde,
              inv_width, inv_height, m_size, n_size );

        cudaDeviceSynchronize();

        if (runtimes == 1) { timer[0]->stop(); te[0] += timer[0]->elapsed(); timer[1]->start(); }

        alg5varc_stage2_bor<<< dim3(1, n_size), dim3(WS, NWA), 0, stream1 >>>
            ( &d_pybar, &d_ezhat, m_size, n_size, &d_AbF_T,
              &d_AbR_T, &d_HARB_AFP_T );
        alg5varc_stage2_mid<<< dim3(1, n_size), dim3(WS, NWA), 0, stream2 >>>
            ( &d_pybar, &d_ezhat, m_size, n_size );

        cudaDeviceSynchronize();

        if (runtimes == 1) { timer[1]->stop(); te[1] += timer[1]->elapsed(); timer[2]->start(); }

        alg5varc_stage3_bor<<< dim3(m_size, 1), dim3(WS, NWA), 0, stream1 >>>
            ( &d_ptucheck, &d_etvtilde, &d_pybar, &d_ezhat, m_size, n_size,
              &d_AbF, &d_TAFB, &d_ARE_T, &d_ARB_AFP_T, &d_AbR, &d_HARB_AFP,
              &d_HARB_AFB );
        alg5varc_stage3_mid<<< dim3(m_size, 1), dim3(WS, NWA), 0, stream2 >>>
            ( &d_ptucheck, &d_etvtilde, &d_pybar, &d_ezhat, m_size, n_size );

        cudaDeviceSynchronize();

        if (runtimes == 1) { timer[2]->stop(); te[2] += timer[2]->elapsed(); timer[3]->start(); }

        alg5varc_stage4_bor<<< dim3(m_size, n_size), dim3(WS, NWW), 0, stream1 >>>
            ( d_img, &d_pybar, &d_ezhat, &d_ptucheck, &d_etvtilde,
              inv_width, inv_height, m_size, n_size, stride_img,
              &d_aw, &d_ah );
        alg5varc_stage4_mid<<< dim3(m_size, n_size), dim3(WS, NWW), 0, stream2 >>>
            ( d_img, &d_pybar, &d_ezhat, &d_ptucheck, &d_etvtilde,
              inv_width, inv_height, m_size, n_size, stride_img );

        cudaDeviceSynchronize();

        cudaUnbindTexture(t_in);

        if (runtimes == 1) { timer[3]->stop(); te[3] += timer[3]->elapsed(); }

    }

    timer_total.stop();

    if (runtimes > 1) {

        std::cout << std::fixed << (timer_total.data_size()*runtimes)/(double)(timer_total.elapsed()*1024*1024) << std::flush;

    } else {

        timers.gpu_add("stage 1", timer[0]);
        timers.gpu_add("stage 2", timer[1]);
        timers.gpu_add("stage 3", timer[2]);
        timers.gpu_add("stage 4", timer[3]);
        timers.flush();

    }

    cudaMemcpy2D(h_img, width*sizeof(float), d_img, stride_img*sizeof(float), width*sizeof(float), height, cudaMemcpyDeviceToHost);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    delete [] aw;
    delete [] ah;
}

//==============================================================================
} // namespace gpufilter
//==============================================================================
#endif // ALG5VARC_GPU_CUH
//==============================================================================
