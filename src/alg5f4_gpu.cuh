/**
 *  @file alg5f4_gpu.cuh
 *  @brief Algorithm 5_1 fusioned with algorithm 4_2 in the GPU
 *  @author Andre Maximo
 *  @date Jan, 2014
 *  @copyright The MIT License
 */

#ifndef ALG5F4_GPU_CUH
#define ALG5F4_GPU_CUH

//== GLOBAL-SCOPE DEFINITIONS ==================================================

// there is no template option for CUDA constant variables
const int R1 = 1, R2 = 2;

//== NAMESPACES ================================================================

namespace gpufilter {

//== CONSTANTS =================================================================

// basics
__constant__ Vector<float,R1+1> c_weights1;
__constant__ Vector<float,R2+1> c_weights2;

// alg4
__constant__ Matrix<float,R2,R2> c_AbF_T2, c_AbR_T2, c_HARB_AFP_T2;

// alg5
__constant__ Matrix<float,R1,R1> c_AbF_T1, c_AbR_T1, c_HARB_AFP_T1;
__constant__ Matrix<float,R1,WS> c_ARE_T1, c_ARB_AFP_T1, c_TAFB1, c_HARB_AFB1;

//== IMPLEMENTATION ============================================================

/**
 *  @ingroup gpu
 *  @brief Algorithm 5 stage 1 for order r1=1
 *  @note This follows the original implementation in [NehabEtAl:2011]
 *  @see alg5_stage1()
 *  @param[out] g_pybar All \f$P_{m,n}(Y)\f$
 *  @param[out] g_ezhat All \f$E_{m,n}(Z)\f$
 *  @param[out] g_ptucheck All \f$P^T_{m,n}(U)\f$
 *  @param[out] g_etvtilde All \f$E^T_{m,n}(V)\f$
 *  @param[in] inv_width Image width inversed (1/w)
 *  @param[in] inv_height Image height inversed (1/h)
 *  @param[in] m_size The big M (number of row blocks)
 *  @param[in] n_size The big N (number of column blocks)
 *  @tparam BORDER Flag to consider border input padding
 */
template <bool BORDER>
__global__ __launch_bounds__(WS*NWC, NBCW)
void alg5_stage1_r1( Matrix<float,R1,WS> *g_pybar, 
                     Matrix<float,R1,WS> *g_ezhat,
                     Matrix<float,R1,WS> *g_ptucheck,
                     Matrix<float,R1,WS> *g_etvtilde,
                     float inv_width, float inv_height,
                     int m_size, int n_size ) {

    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n = blockIdx.y;

    __shared__ Matrix<float,WS,WS+1> block;
    if (BORDER) // read considering borders
        read_block<NWC>(block, m-c_border, n-c_border, inv_width, inv_height);
    else
        read_block<NWC>(block, m, n, inv_width, inv_height);
    __syncthreads();

    if (ty==0) {

        Vector<float,R1> p = zeros<float,R1>();

#pragma unroll // calculate pybar, scan left -> right
        for (int j=0; j<WS; ++j)
            block[tx][j] = fwdI(p, block[tx][j], c_weights1);

        g_pybar[n*(m_size+1)+m+1].set_col(tx, p);
        
        Vector<float,R1> e = zeros<float,R1>();

#pragma unroll // calculate ezhat, scan right -> left
        for (int j=WS-1; j>=0; --j)
            block[tx][j] = revI(block[tx][j], e, c_weights1);

        g_ezhat[n*(m_size+1)+m].set_col(tx, e);
            
        p = zeros<float,R1>();

#pragma unroll // calculate ptucheck, scan top -> bottom
        for (int j=0; j<WS; ++j)
            block[j][tx] = fwdI(p, block[j][tx], c_weights1);

        g_ptucheck[m*(n_size+1)+n+1].set_col(tx, p);

        e = zeros<float,R1>();

#pragma unroll // calculate etvtilde, scan bottom -> top
        for (int j=WS-1; j>=0; --j)
            revI(block[j][tx], e, c_weights1);

        g_etvtilde[m*(n_size+1)+n].set_col(tx, e);

    }

}

/**
 *  @ingroup gpu
 *  @brief Algorithm 5 stage 2 for order r1=1
 *  @note This follows the original implementation in [NehabEtAl:2011]
 *  @see alg5_stage2()
 *  @param[in,out] g_pybar All \f$P_{m,n}(Y)\f$
 *  @param[in,out] g_ezhat All \f$E_{m,n}(Z)\f$
 *  @param[in] m_size The big M (number of row blocks)
 */
__global__ __launch_bounds__(WS*NWA, NBA)
void alg5_stage2_r1( Matrix<float,R1,WS> *g_pybar,
                     Matrix<float,R1,WS> *g_ezhat,
                     int m_size ) {

    int tx = threadIdx.x, ty = threadIdx.y, m, n = blockIdx.y;
    Matrix<float,R1,WS> *gpybar, *gezhat;
    Vector<float,R1> py, ez, pybar, ezhat;
    __shared__ Matrix<float,R1,WS> spybar[NWA], sezhat[NWA];

    // P(ybar) -> P(y) processing ---------------------------------------------
    m = 0;
    gpybar = (Matrix<float,R1,WS> *)&g_pybar[n*(m_size+1)+m+ty+1][0][tx];
    py = zeros<float,R1>();

    for (; m < m_size; m += NWA) { // for all image blocks

        if (m+ty < m_size) spybar[ty].set_col(tx, gpybar->col(0)); // using smem as cache

        __syncthreads(); // wait to load smem

        if (ty == 0) {
#pragma unroll // adjust pybar left -> right
            for (int w = 0; w < NWA; ++w) {

                pybar = spybar[w].col(tx);

                py = pybar + py * c_AbF_T1;

                spybar[w].set_col(tx, py);

            }
        }

        __syncthreads(); // wait to store result in gmem

        if (m+ty < m_size) gpybar->set_col(0, spybar[ty].col(tx));

        gpybar += NWA;

    }

    // E(zhat) -> E(z) processing ----------------------------------------------
    m = m_size-1;
    gezhat = (Matrix<float,R1,WS> *)&g_ezhat[n*(m_size+1)+m-ty][0][tx];
    gpybar = (Matrix<float,R1,WS> *)&g_pybar[n*(m_size+1)+m-ty][0][tx];
    ez = zeros<float,R1>();

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

                ez = ezhat + py * c_HARB_AFP_T1 + ez * c_AbR_T1;

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
 *  @brief Algorithm 5 stage 3 for order r1=1
 *  @note This follows the original implementation in [NehabEtAl:2011]
 *  @see alg5_stage3()
 *  @param[in,out] g_ptucheck All \f$P^T_{m,n}(U)\f$
 *  @param[in,out] g_etvtilde All \f$E^T_{m,n}(V)\f$
 *  @param[in] g_py All \f$P_{m,n}(Y)\f$
 *  @param[in] g_ez All \f$E_{m,n}(Z)\f$
 *  @param[in] m_size The big M (number of row blocks)
 *  @param[in] n_size The big N (number of column blocks)
 */
__global__ __launch_bounds__(WS*NWA, NBA)
void alg5_stage3_r1( Matrix<float,R1,WS> *g_ptucheck, 
                     Matrix<float,R1,WS> *g_etvtilde,
                     Matrix<float,R1,WS> *g_py,
                     Matrix<float,R1,WS> *g_ez,
                     int m_size, int n_size ) {

    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n;
    Matrix<float,R1,WS> *gptucheck, *getvtilde;
    Vector<float,R1> ptu, etv, ptucheck, etvtilde;
    Matrix<float,R1,WS> *gpy, *gez;
    Vector<float,R1> py, ez;
    __shared__ Matrix<float,R1,WS> sptucheck[NWA], setvtilde[NWA];
    __shared__ Matrix<float,R1,WS> spy[NWA], sez[NWA];

    // Pt(ucheck) -> Pt(u) processing ------------------------------------------
    n = 0;
    gptucheck = (Matrix<float,R1,WS> *)&g_ptucheck[m*(n_size+1)+n+ty+1][0][tx];
    gpy = (Matrix<float,R1,WS> *)&g_py[(n+ty)*(m_size+1)+m+0][0][tx];
    gez = (Matrix<float,R1,WS> *)&g_ez[(n+ty)*(m_size+1)+m+1][0][tx];
    ptu = zeros<float,R1>();

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

                ptu = ptucheck + ptu * c_AbF_T1;

                fixpet(ptu, c_TAFB1, c_ARE_T1, ez);
                fixpet(ptu, c_TAFB1, c_ARB_AFP_T1, py);

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
    getvtilde = (Matrix<float,R1,WS> *)&g_etvtilde[m*(n_size+1)+n-ty][0][tx];
    gptucheck = (Matrix<float,R1,WS> *)&g_ptucheck[m*(n_size+1)+n-ty][0][tx];
    gpy = (Matrix<float,R1,WS> *)&g_py[(n-ty)*(m_size+1)+m][0][tx];
    gez = (Matrix<float,R1,WS> *)&g_ez[(n-ty)*(m_size+1)+m+1][0][tx];
    etv = zeros<float,R1>();

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

                etv = etvtilde + etv * c_AbR_T1 + ptu * c_HARB_AFP_T1;

                fixpet(etv, c_HARB_AFB1, c_ARE_T1, ez);
                fixpet(etv, c_HARB_AFB1, c_ARB_AFP_T1, py);

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
 *  @brief Algorithm 5 stage 4 fusioned with algorithm 4 stage 1
 *  @note This follows the original implementation in [NehabEtAl:2011]
 *  @see alg5_stage4()
 *  @see alg4_stage1()
 *  @param[out] g_out The output 2D image
 *  @param[in] g_py All \f$P_{m,n}(Y)\f$
 *  @param[in] g_ez All \f$E_{m,n}(Z)\f$
 *  @param[in] g_ptu All \f$P^T_{m,n}(U)\f$
 *  @param[in] g_etv All \f$E^T_{m,n}(V)\f$
 *  @param[out] g_pybar All \f$P_{m,n}(Y)\f$
 *  @param[out] g_ezhat All \f$E_{m,n}(Z)\f$
 *  @param[in] inv_width Image width inversed (1/w)
 *  @param[in] inv_height Image height inversed (1/h)
 *  @param[in] m_size The big M (number of row blocks)
 *  @param[in] n_size The big N (number of column blocks)
 *  @param[in] out_stride Image output stride for memory width alignment
 *  @tparam BORDER Flag to consider border input padding
 */
template <bool BORDER>
__global__ __launch_bounds__(WS*NWW, NBCW)
void alg5f4_r1r2( float *g_out,
                  const Matrix<float,R1,WS> *g_py,
                  const Matrix<float,R1,WS> *g_ez,
                  const Matrix<float,R1,WS> *g_ptu,
                  const Matrix<float,R1,WS> *g_etv,
                  Matrix<float,R2,WS> *g_pybar,
                  Matrix<float,R2,WS> *g_ezhat,
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

    Matrix<float,R1,WS> 
        *pym1 = (Matrix<float,R1,WS>*)&g_py[n*(m_size+1)+m][0][tx],
        *ezm1 = (Matrix<float,R1,WS>*)&g_ez[n*(m_size+1)+m+1][0][tx],
        *ptum1 = (Matrix<float,R1,WS>*)&g_ptu[m*(n_size+1)+n][0][tx],
        *etvm1 = (Matrix<float,R1,WS>*)&g_etv[m*(n_size+1)+n+1][0][tx];

    if (ty==0) {

        Vector<float,R1> p1, e1;

        p1 = pym1->col(0); // load py

#pragma unroll // calculate block, scan left -> right
        for (int j=0; j<WS; ++j)
            block[tx][j] = fwdI(p1, block[tx][j], c_weights1);

        e1 = ezm1->col(0); // load ez

#pragma unroll // calculate block, scan right -> left
        for (int j=WS-1; j>=0; --j)
            block[tx][j] = revI(block[tx][j], e1, c_weights1);

        p1 = ptum1->col(0); // load ptu

#pragma unroll // calculate block, scan top -> bottom
        for (int j=0; j<WS; ++j)
            block[j][tx] = fwdI(p1, block[j][tx], c_weights1);

        e1 = etvm1->col(0); // load etv

#pragma unroll // calculate block, scan bottom -> top
        for (int j=WS-1; j>=0; --j)
            block[j][tx] = revI(block[j][tx], e1, c_weights1);

        if (BORDER) {
            if ((m >= c_border) && (m < m_size-c_border) && (n >= c_border) && (n < n_size-c_border)) {
                g_out += ((n-c_border+1)*WS-1)*out_stride + (m-c_border)*WS+tx;
#pragma unroll // write block inside valid image
                for (int i=0; i<WS; ++i, g_out-=out_stride) {
                    *g_out = block[WS-1-i][tx];
                }
            }
        } else {
            g_out += ((n+1)*WS-1)*out_stride + m*WS+tx;
#pragma unroll // write block
            for (int i=0; i<WS; ++i, g_out-=out_stride) {
                *g_out = block[WS-1-i][tx];
            }
        }

        // fusion with alg4 collect carries

        Vector<float,R2> p2 = zeros<float,R2>();

#pragma unroll // calculate block, scan left -> right
        for (int j=0; j<WS; ++j)
            block[tx][j] = fwdI(p2, block[tx][j], c_weights2);

        g_pybar[n*(m_size+1)+m+1].set_col(tx, p2);

        Vector<float,R2> e2 = zeros<float,R2>();

#pragma unroll // calculate block, scan right -> left
        for (int j=WS-1; j>=0; --j)
            block[tx][j] = revI(block[tx][j], e2, c_weights2);

        g_ezhat[n*(m_size+1)+m].set_col(tx, e2);

    }

}

/**
 *  @ingroup gpu
 *  @brief Algorithm 4 stage 1 for order r2=2
 *  @note This follows the original implementation in [NehabEtAl:2011]
 *  @see alg4_stage1()
 *  @param[out] g_pybar All \f$P_{m,n}(Y)\f$
 *  @param[out] g_ezhat All \f$E_{m,n}(Z)\f$
 *  @param[in] inv_width Image width inversed (1/w)
 *  @param[in] inv_height Image height inversed (1/h)
 *  @param[in] m_size The big M (number of row blocks)
 *  @tparam BORDER Flag to consider border input padding
 */
template <bool BORDER>
__global__ __launch_bounds__(WS*NWC, NBCW)
void alg4_stage1_r2( Matrix<float,R2,WS> *g_pybar, 
                     Matrix<float,R2,WS> *g_ezhat,
                     float inv_width, float inv_height,
                     int m_size ) {

    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n = blockIdx.y;

    __shared__ Matrix<float,WS,WS+1> block;
    if (BORDER) // read considering borders
        read_block<NWC>(block, m-c_border, n-c_border, inv_width, inv_height);
    else
        read_block<NWC>(block, m, n, inv_width, inv_height);
    __syncthreads();

    if (ty==0) {

        Vector<float,R2> p = zeros<float,R2>();

#pragma unroll // calculate pybar, scan left -> right
        for (int j=0; j<WS; ++j)
            block[tx][j] = fwdI(p, block[tx][j], c_weights2);

        g_pybar[n*(m_size+1)+m+1].set_col(tx, p);
        
        Vector<float,R2> e = zeros<float,R2>();

#pragma unroll // calculate ezhat, scan right -> left
        for (int j=WS-1; j>=0; --j)
            block[tx][j] = revI(block[tx][j], e, c_weights2);

        g_ezhat[n*(m_size+1)+m].set_col(tx, e);

    }

}

/**
 *  @ingroup gpu
 *  @brief Algorithm 4 stage 2 or 4 for order r2=2
 *  @note This follows the original implementation in [NehabEtAl:2011]
 *  @see alg4_stage2v4()
 *  @param[in,out] g_pybar All \f$P_{m,n}(Y)\f$ or \f$P^T_{m,n}(U)\f$
 *  @param[in,out] g_ezhat All \f$E_{m,n}(Z)\f$ or \f$E^T_{m,n}(V)\f$
 *  @param[in] m_size The big M or N (number of row or column blocks)
 */
__global__ __launch_bounds__(WS*NWA, NBA)
void alg4_stage2v4_r2( Matrix<float,R2,WS> *g_pybar,
                       Matrix<float,R2,WS> *g_ezhat,
                       int m_size ) {

    int tx = threadIdx.x, ty = threadIdx.y, m, n = blockIdx.y;
    Matrix<float,R2,WS> *gpybar, *gezhat;
    Vector<float,R2> py, ez, pybar, ezhat;
    __shared__ Matrix<float,R2,WS> spybar[NWA], sezhat[NWA];

    // P(ybar) -> P(y) processing ----------------------------------------------
    m = 0;
    gpybar = (Matrix<float,R2,WS> *)&g_pybar[n*(m_size+1)+m+ty+1][0][tx];
    py = zeros<float,R2>();

    for (; m < m_size; m += NWA) { // for all image blocks

        if (m+ty < m_size) spybar[ty].set_col(tx, gpybar->col(0)); // using smem as cache

        __syncthreads(); // wait to load smem

        if (ty == 0) {
#pragma unroll // adjust pybar left -> right
            for (int w = 0; w < NWA; ++w) {

                pybar = spybar[w].col(tx);

                py = pybar + py * c_AbF_T2;

                spybar[w].set_col(tx, py);

            }
        }

        __syncthreads(); // wait to store result in gmem

        if (m+ty < m_size) gpybar->set_col(0, spybar[ty].col(tx));

        gpybar += NWA;

    }

    // E(zhat) -> E(z) processing ----------------------------------------------
    m = m_size-1;
    gezhat = (Matrix<float,R2,WS> *)&g_ezhat[n*(m_size+1)+m-ty][0][tx];
    gpybar = (Matrix<float,R2,WS> *)&g_pybar[n*(m_size+1)+m-ty][0][tx];
    ez = zeros<float,R2>();

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

                ez = ezhat + py * c_HARB_AFP_T2 + ez * c_AbR_T2;

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
 *  @brief Algorithm 4 stage 3 or 5 for order r2=2
 *  @note This follows the original implementation in [NehabEtAl:2011]
 *  @see alg4_stage3v5()
 *  @param[out] g_transp_out The output transposed 2D image
 *  @param[in] g_rows_py All \f$P_{m,n}(Y)\f$ or \f$P^T_{m,n}(U)\f$ 
 *  @param[in] g_rows_ez All \f$E_{m,n}(Z)\f$ or \f$E^T_{m,n}(V)\f$ 
 *  @param[in] g_cols_py All \f$P^T_{m,n}(U)\f$ (for 4.3 fusion)
 *  @param[in] g_cols_ez All \f$E^T_{m,n}(V)\f$ (for 4.3 fusion)
 *  @param[in] inv_width Image width inversed (1/w)
 *  @param[in] inv_height Image height inversed (1/h)
 *  @param[in] m_size The big M (number of row blocks)
 *  @param[in] n_size The big N (number of column blocks)
 *  @param[in] out_stride Image output stride for memory width alignment
 *  @tparam FUSION Flag for 4.3 fusion to compute feedbacks in other direction
 *  @tparam BORDER Flag to consider border input padding
 */
template <bool FUSION, bool BORDER>
__global__ __launch_bounds__(WS*NWW, NBCW)
void alg4_stage3v5_r2( float *g_transp_out,
                       const Matrix<float,R2,WS> *g_rows_py,
                       const Matrix<float,R2,WS> *g_rows_ez,
                       Matrix<float,R2,WS> *g_cols_py,
                       Matrix<float,R2,WS> *g_cols_ez,
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

    Matrix<float,R2,WS> 
        *pym1 = (Matrix<float,R2,WS>*)&g_rows_py[n*(m_size+1)+m][0][tx],
        *ezm1 = (Matrix<float,R2,WS>*)&g_rows_ez[n*(m_size+1)+m+1][0][tx];

    if (ty==0) {

        Vector<float,R2> p = pym1->col(0); // load py

#pragma unroll // calculate block, scan left -> right
        for (int j=0; j<WS; ++j)
            block[tx][j] = fwdI(p, block[tx][j], c_weights2);

        Vector<float,R2> e = ezm1->col(0); // load ez

#pragma unroll // calculate block, scan right -> left
        for (int j=WS-1; j>=0; --j)
            block[tx][j] = revI(block[tx][j], e, c_weights2);

        if (BORDER) {
            if ((m >= c_border) && (m < m_size-c_border) && (n >= c_border) && (n < n_size-c_border)) {
                g_transp_out += (m-c_border)*WS*out_stride + (n-c_border)*WS + tx;
#pragma unroll // write block inside valid transpose image
                for (int i=0; i<WS; ++i, g_transp_out += out_stride) {
                    *g_transp_out = block[tx][i];
                }
            }
        } else {
            g_transp_out += m*WS*out_stride + n*WS + tx;
#pragma unroll // write block inside valid transpose image
            for (int i=0; i<WS; ++i, g_transp_out += out_stride) {
                *g_transp_out = block[tx][i];
            }
        }

        if (FUSION) {

            Matrix<float,R2,WS>
                &pybar = (Matrix<float,R2,WS>&)g_cols_py[m*(n_size+1)+n+1][0][tx],
                &ezhat = (Matrix<float,R2,WS>&)g_cols_ez[m*(n_size+1)+n][0][tx];

            p = zeros<float,R2>();

#pragma unroll // calculate pybar cols, scan left -> right
            for (int j=0; j<WS; ++j)
                block[j][tx] = fwdI(p, block[j][tx], c_weights2);

            pybar.set_col(0, p); // store pybar cols

            e = zeros<float,R2>();

#pragma unroll // calculate ezhat cols, scan right -> left
            for (int j=WS-1; j>=0; --j)
                block[j][tx] = revI(block[j][tx], e, c_weights2);

            ezhat.set_col(0, e); // store ezhat cols

        }

    }

}

/**
 *  @ingroup api_gpu
 *  @brief Compute algorithm 5 fusioned with algorithm 4 in the GPU
 *  @see alg5_gpu()
 *  @see alg4_gpu()
 *  @param[in,out] h_img The in(out)put 2D image to filter in host memory
 *  @param[in] width Image width
 *  @param[in] height Image height
 *  @param[in] runtimes Number of run times (1 for debug and 1000 for performance measurements)
 *  @param[in] w1 Filter weights for order r1=1
 *  @param[in] w2 Filter weights for order r2=2
 *  @param[in] border Number of border blocks (32x32) outside image
 *  @param[in] btype Border type (either zero, clamp, repeat or reflect)
 *  @tparam BORDER Flag to consider border input padding
 */
template <bool BORDER>
void alg5f4_gpu( float *h_img,
                 int width, int height, int runtimes,
                 const Vector<float, R1+1>& w1,
                 const Vector<float, R2+1>& w2,
                 int border=0,
                 BorderType border_type=CLAMP_TO_ZERO ) {

    const int B = WS;

    // pre-compute basic alg5 matrices
    Matrix<float,B,B> Ib = identity<float,B,B>();

    // for order r1
    Matrix<float,R1,R1> Ir1 = identity<float,R1,R1>();
    Matrix<float,B,R1> Zbr1 = zeros<float,B,R1>();
    Matrix<float,R1,B> Zrb1 = zeros<float,R1,B>();

    Matrix<float,R1,B> AFP_T1 = fwd(Ir1, Zrb1, w1), ARE_T1 = rev(Zrb1, Ir1, w1);
    Matrix<float,B,B> AFB_T1 = fwd(Zbr1, Ib, w1), ARB_T1 = rev(Ib, Zbr1, w1);
    Matrix<float,R1,R1> AbF_T1 = tail<R1>(AFP_T1), AbR_T1 = head<R1>(ARE_T1);
    Matrix<float,R1,R1> AbF1 = transp(AbF_T1), AbR1 = transp(AbR_T1);
    Matrix<float,R1,R1> HARB_AFP_T1 = AFP_T1*head<R1>(ARB_T1);
    Matrix<float,R1,R1> HARB_AFP1 = transp(HARB_AFP_T1);
    Matrix<float,R1,B> ARB_AFP_T1 = AFP_T1*ARB_T1, TAFB1 = transp(tail<R1>(AFB_T1));
    Matrix<float,R1,B> HARB_AFB1 = transp(AFB_T1*head<R1>(ARB_T1));

    // for order r2
    Matrix<float,R2,R2> Ir2 = identity<float,R2,R2>();
    Matrix<float,B,R2> Zbr2 = zeros<float,B,R2>();
    Matrix<float,R2,B> Zrb2 = zeros<float,R2,B>();

    Matrix<float,R2,B> AFP_T2 = fwd(Ir2, Zrb2, w2), ARE_T2 = rev(Zrb2, Ir2, w2);
    Matrix<float,B,B> AFB_T2 = fwd(Zbr2, Ib, w2), ARB_T2 = rev(Ib, Zbr2, w2);
    Matrix<float,R2,R2> AbF_T2 = tail<R2>(AFP_T2), AbR_T2 = head<R2>(ARE_T2);
    Matrix<float,R2,R2> AbF2 = transp(AbF_T2), AbR2 = transp(AbR_T2);
    Matrix<float,R2,R2> HARB_AFP_T2 = AFP_T2*head<R2>(ARB_T2);
    Matrix<float,R2,R2> HARB_AFP2 = transp(HARB_AFP_T2);
    Matrix<float,R2,B> ARB_AFP_T2 = AFP_T2*ARB_T2, TAFB2 = transp(tail<R2>(AFB_T2));
    Matrix<float,R2,B> HARB_AFB2 = transp(AFB_T2*head<R2>(ARB_T2));

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

    // order r1
    copy_to_symbol(c_weights1, w1);

    copy_to_symbol(c_AbF_T1, AbF_T1);
    copy_to_symbol(c_AbR_T1, AbR_T1);
    copy_to_symbol(c_HARB_AFP_T1, HARB_AFP_T1);

    copy_to_symbol(c_ARE_T1, ARE_T1);
    copy_to_symbol(c_ARB_AFP_T1, ARB_AFP_T1);
    copy_to_symbol(c_TAFB1, TAFB1);
    copy_to_symbol(c_HARB_AFB1, HARB_AFB1);

    // order r2
    copy_to_symbol(c_weights2, w2);

    copy_to_symbol(c_AbF_T2, AbF_T2);
    copy_to_symbol(c_AbR_T2, AbR_T2);
    copy_to_symbol(c_HARB_AFP_T2, HARB_AFP_T2);

    float inv_width = 1.f/width, inv_height = 1.f/height;
    size_t offset;
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

    int stride_img = width+WS, stride_transp_img = width+WS;
    if (BORDER) stride_img = stride_transp_img = width+WS*border+WS;

    dvector<float> d_img(height*stride_img), d_transp_img(width*stride_transp_img);

    // +1 padding is important even in zero-border to avoid if's in kernels
    // order r1
    dvector< Matrix<float,R1,B> > d_pybar1((m_size+1)*n_size), d_ezhat1((m_size+1)*n_size);
    dvector< Matrix<float,R1,B> > d_ptucheck1((n_size+1)*m_size), d_etvtilde1((n_size+1)*m_size);
    d_pybar1.fillzero();
    d_ezhat1.fillzero();
    d_ptucheck1.fillzero();
    d_etvtilde1.fillzero();

    // order r2
    dvector< Matrix<float,R2,B> > d_pybar2((m_size+1)*n_size), d_ezhat2((m_size+1)*n_size);
    dvector< Matrix<float,R2,B> > d_pubar2((n_size+1)*m_size), d_evhat2((n_size+1)*m_size);
    d_pybar2.fillzero();
    d_ezhat2.fillzero();
    d_pubar2.fillzero();
    d_evhat2.fillzero();

    double te[8] = {0, 0, 0, 0, 0, 0, 0, 0}; // time elapsed for eight stages
    base_timer *timer[8];
    for (int i = 0; i < 8; ++i)
        timer[i] = new gpu_timer(0, "", false);

    base_timer &timer_total = timers.gpu_add("alg5f4_gpu", width*height, "iP");

    for(int r = 0; r < runtimes; ++r) {

        if (runtimes == 1) { timer[0]->start(); }

        cudaBindTextureToArray(t_in, a_in);

        alg5_stage1_r1<BORDER><<< dim3(m_size, n_size), dim3(WS, NWC) >>>
            ( &d_pybar1, &d_ezhat1, &d_ptucheck1, &d_etvtilde1, inv_width, inv_height, m_size, n_size );

        if (runtimes == 1) { timer[0]->stop(); te[0] += timer[0]->elapsed(); timer[1]->start(); }

        alg5_stage2_r1<<< dim3(1, n_size), dim3(WS, NWA) >>>
            ( &d_pybar1, &d_ezhat1, m_size );

        if (runtimes == 1) { timer[1]->stop(); te[1] += timer[1]->elapsed(); timer[2]->start(); }

        alg5_stage3_r1<<< dim3(m_size, 1), dim3(WS, NWA) >>>
            ( &d_ptucheck1, &d_etvtilde1, &d_pybar1, &d_ezhat1, m_size, n_size );

        if (runtimes == 1) { timer[2]->stop(); te[2] += timer[2]->elapsed(); timer[3]->start(); }

        alg5f4_r1r2<BORDER><<< dim3(m_size, n_size), dim3(WS, NWW) >>>
            ( d_img, &d_pybar1, &d_ezhat1, &d_ptucheck1, &d_etvtilde1,
              &d_pybar2, &d_ezhat2, inv_width, inv_height,
              m_size, n_size, stride_img );

        cudaUnbindTexture(t_in);
        cudaBindTexture2D( &offset, t_in, d_img, height, width, stride_img*sizeof(float) );

        if (runtimes == 1) { timer[3]->stop(); te[3] += timer[3]->elapsed(); timer[4]->start(); }

        alg4_stage2v4_r2<<< dim3(1, n_size), dim3(WS, NWA) >>>
            ( &d_pybar2, &d_ezhat2, m_size );

        if (runtimes == 1) { timer[4]->stop(); te[4] += timer[4]->elapsed(); timer[5]->start(); }

        alg4_stage3v5_r2<true, BORDER><<< dim3(m_size, n_size), dim3(WS, NWW) >>>
            ( d_transp_img, &d_pybar2, &d_ezhat2, &d_pubar2, &d_evhat2,
              inv_width, inv_height, m_size, n_size, stride_transp_img );

        cudaUnbindTexture(t_in);
        cudaBindTexture2D(&offset, t_in, d_transp_img, height, width, stride_transp_img*sizeof(float));

        if (runtimes == 1) { timer[5]->stop(); te[5] += timer[5]->elapsed(); timer[6]->start(); }

        alg4_stage2v4_r2<<< dim3(1, m_size), dim3(WS, NWA) >>>
            ( &d_pubar2, &d_evhat2, n_size );

        if (runtimes == 1) { timer[6]->stop(); te[6] += timer[6]->elapsed(); timer[7]->start(); }

        alg4_stage3v5_r2<false, BORDER><<< dim3(n_size, m_size), dim3(WS, NWW) >>>
            ( d_img, &d_pubar2, &d_evhat2, &d_pybar2, &d_ezhat2,
              inv_height, inv_width, n_size, m_size, stride_img );

        cudaUnbindTexture(t_in);

        if (runtimes == 1) { timer[7]->stop(); te[7] += timer[7]->elapsed(); }

    }

    timer_total.stop();

    if (runtimes > 1) {

        std::cout << std::fixed << (timer_total.data_size()*runtimes)/(double)(timer_total.elapsed()*1024*1024) << std::flush;

    } else {

        timers.gpu_add("alg5 stage 1", timer[0]);
        timers.gpu_add("alg5 stage 2", timer[1]);
        timers.gpu_add("alg5 stage 3", timer[2]);
        timers.gpu_add("5.4 + 4.1", timer[3]);
        timers.gpu_add("alg4 stage 2", timer[4]);
        timers.gpu_add("alg4 stage 3", timer[5]);
        timers.gpu_add("alg4 stage 4", timer[6]);
        timers.gpu_add("alg4 stage 5", timer[7]);
        timers.flush();

    }

    cudaMemcpy2D(h_img, width*sizeof(float), d_img, stride_img*sizeof(float), width*sizeof(float), height, cudaMemcpyDeviceToHost);

}

//==============================================================================
} // namespace gpufilter
//==============================================================================
#endif // ALG5F4_GPU_CUH
//==============================================================================
