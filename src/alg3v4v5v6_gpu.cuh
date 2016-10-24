/**
 *  @file alg3v4v5v6_gpu.cuh
 *  @brief Algorithm 3 or 4 or 5 or 6 in the GPU with borders
 *  @author Andre Maximo
 *  @date Nov, 2012
 *  @copyright The MIT License
 */

#ifndef ALG3v4v5v6_GPU_CUH
#define ALG3v4v5v6_GPU_CUH

//== NAMESPACES ================================================================

namespace gpufilter {

//== IMPLEMENTATION ============================================================

/**
 *  @ingroup gpu
 *  @brief Algorithm 3 or 4 step 1
 *
 *  This function computes the algorithm step 3.1 or 4.1 following:
 *
 *  \li In parallel for all \f$m\f$ and \f$n\f$, load block
 *  \f$B_{m,n}(X)\f$ then compute and store block perimeters
 *  \f$P_{m,n}(Y)\f$ and \f$E_{m,n}(Z)\f$.
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @note This follows the improved base-line implementation in [NehabMaximo:2016]
 *  @see [NehabEtAl:2011] cited in alg5() and [NehabMaximo:2016] cited in alg6()
 *  @param[out] g_pybar All \f$P_{m,n}(Y)\f$
 *  @param[out] g_ezhat All \f$E_{m,n}(Z)\f$
 *  @param[in] inv_width Image width inversed (1/w)
 *  @param[in] inv_height Image height inversed (1/h)
 *  @param[in] m_size The big M (number of row blocks)
 *  @tparam BORDER Flag to consider border input padding
 *  @tparam R Filter order
 */
template <bool BORDER, int R>
__global__ __launch_bounds__(WS*NWC, NBCW)
void alg3v4_step1( Matrix<float,R,WS> *g_pybar, 
                   Matrix<float,R,WS> *g_ezhat,
                   float inv_width, float inv_height,
                   int m_size ) {

    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n = blockIdx.y;

    __shared__ Matrix<float,WS,WS+1> block;
    if (BORDER) // read considering borders
        read_block<NWC>(block, m-c_border, n-c_border, inv_width, inv_height);
    else
        read_block<NWC>(block, m, n, inv_width, inv_height);
    __syncthreads();

#ifdef REGS
    float x[32]; // 32 regs
#endif

    if (ty==0) {

#ifdef REGS
#pragma unroll
        for (int i=0; i<32; ++i)
            x[i] = block[tx][i];
#endif

        Vector<float,R> p = zeros<float,R>();

#pragma unroll // calculate pybar, scan left -> right
        for (int j=0; j<WS; ++j)
#ifdef REGS
            x[j] = fwdI(p, x[j], c_weights);
#else
            block[tx][j] = fwdI(p, block[tx][j], c_weights);
#endif

        g_pybar[n*(m_size+1)+m+1].set_col(tx, p);
        
        Vector<float,R> e = zeros<float,R>();

#pragma unroll // calculate ezhat, scan right -> left
        for (int j=WS-1; j>=0; --j)
#ifdef REGS
            revI(x[j], e, c_weights);
#else
            revI(block[tx][j], e, c_weights);
#endif

        g_ezhat[n*(m_size+1)+m].set_col(tx, e);

    }

}

/**
 *  @ingroup gpu
 *  @brief Algorithm 3 or 4 or 6 step 2 or 4 or algorithm 5 step 2
 *
 *  This function computes the algorithm step 3.2 or 4.2 or 4.4 or 6.2
 *  or 6.4 or 5.4 (corresponding to the steps 3.2 and 3.3 or 4.2 and
 *  4.3 or 4.5 and 4.6 or 5.2 and 5.3 in [NehabEtAl:2011]) following:
 *
 *  \li In parallel for all \f$n\f$, sequentially for each \f$m\f$,
 *  compute and store all feedbacks \f$P_{m-1,n}(Y)\f$ and
 *  \f$E_{m+1,n}(Z)\f$ according to equations (24) and (25).
 *
 *  \li In parallel for all \f$m\f$, sequentially for each \f$n\f$,
 *  compute and store all feedbacks \f$P^T_{m,n-1}(U)\f$ and
 *  \f$E^T_{m,n+1}(V)\f$ according to equations (32) and (33).
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @note This follows the improved base-line implementation in [NehabMaximo:2016]
 *  @see [NehabEtAl:2011] cited in alg5() and [NehabMaximo:2016] cited in alg6()
 *  @param[in,out] g_pybar All \f$P_{m,n}(Y)\f$ or \f$P^T_{m,n}(U)\f$
 *  @param[in,out] g_ezhat All \f$E_{m,n}(Z)\f$ or \f$E^T_{m,n}(V)\f$
 *  @param[in] m_size The big M or N (number of row or column blocks)
 *  @tparam R Filter order
 */
template <int R>
__global__ __launch_bounds__(WS*NWA, NBA)
void alg3v4v5v6_step2v4( Matrix<float,R,WS> *g_pybar,
                         Matrix<float,R,WS> *g_ezhat,
                         int m_size ) {

    int tx = threadIdx.x, ty = threadIdx.y, m, n = blockIdx.y;
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
 *  @brief Algorithm 5 step 1 or algorithm 6 step 1
 *
 *  This function computes the algorithm step 5.1 or 6.1 following:
 *
 *  \li In parallel for all \f$m\f$ and \f$n\f$, load block
 *  \f$B_{m,n}(X)\f$ then compute and store block perimeters
 *  \f$P_{m,n}(Y)\f$, \f$E_{m,n}(Z)\f$, \f$P^T_{m,n}(U)\f$ and
 *  \f$E^T_{m,n}(V)\f$.
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @note This follows the improved base-line implementation in [NehabMaximo:2016]
 *  @see [NehabEtAl:2011] cited in alg5() and [NehabMaximo:2016] cited in alg6()
 *  @param[out] g_pybar All \f$P_{m,n}(Y)\f$
 *  @param[out] g_ezhat All \f$E_{m,n}(Z)\f$
 *  @param[out] g_ptucheck All \f$P^T_{m,n}(U)\f$
 *  @param[out] g_etvtilde All \f$E^T_{m,n}(V)\f$
 *  @param[in] inv_width Image width inversed (1/w)
 *  @param[in] inv_height Image height inversed (1/h)
 *  @param[in] m_size The big M (number of row blocks)
 *  @param[in] n_size The big N (number of column blocks)
 *  @tparam BORDER Flag to consider border input padding
 *  @tparam R Filter order
 */
template <bool BORDER, int R>
__global__ __launch_bounds__(WS*NWC, NBCW)
void alg5v6_step1( Matrix<float,R,WS> *g_pybar, 
                   Matrix<float,R,WS> *g_ezhat,
                   Matrix<float,R,WS> *g_ptucheck,
                   Matrix<float,R,WS> *g_etvtilde,
                   float inv_width, float inv_height,
                   int m_size, int n_size ) {

    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n = blockIdx.y;

    __shared__ Matrix<float,WS,WS+1> block;
    if (BORDER) // read considering borders
        read_block<NWC>(block, m-c_border, n-c_border, inv_width, inv_height);
    else
        read_block<NWC>(block, m, n, inv_width, inv_height);
    __syncthreads();

#ifdef REGS
    float x[32]; // 32 regs
#endif

    if (ty==0) {

#ifdef REGS
#pragma unroll
        for (int i=0; i<32; ++i)
            x[i] = block[tx][i];
#endif

        Vector<float,R> p = zeros<float,R>();

#pragma unroll // calculate pybar, scan left -> right
        for (int j=0; j<WS; ++j)
#ifdef REGS
            x[j] = fwdI(p, x[j], c_weights);
#else
            block[tx][j] = fwdI(p, block[tx][j], c_weights);
#endif

        g_pybar[n*(m_size+1)+m+1].set_col(tx, p);
        
        Vector<float,R> e = zeros<float,R>();

#pragma unroll // calculate ezhat, scan right -> left
        for (int j=WS-1; j>=0; --j)
#ifdef REGS
            x[j] = revI(x[j], e, c_weights);
#else
            block[tx][j] = revI(block[tx][j], e, c_weights);
#endif

        g_ezhat[n*(m_size+1)+m].set_col(tx, e);

#ifdef REGS
#pragma unroll // transpose regs part-1
        for (int i=0; i<32; ++i)
            block[tx][i] = x[i];
#pragma unroll // transpose regs part-2
        for (int i=0; i<32; ++i)
            x[i] = block[i][tx];
#endif

        p = zeros<float,R>();

#pragma unroll // calculate ptucheck, scan top -> bottom
        for (int j=0; j<WS; ++j)
#ifdef REGS
            x[j] = fwdI(p, x[j], c_weights);
#else
            block[j][tx] = fwdI(p, block[j][tx], c_weights);
#endif

        g_ptucheck[m*(n_size+1)+n+1].set_col(tx, p);

        e = zeros<float,R>();

#pragma unroll // calculate etvtilde, scan bottom -> top
        for (int j=WS-1; j>=0; --j)
#ifdef REGS
            revI(x[j], e, c_weights);
#else
            revI(block[j][tx], e, c_weights);
#endif

        g_etvtilde[m*(n_size+1)+n].set_col(tx, e);

    }

}

/**
 *  @ingroup gpu
 *  @brief Algorithm 5 step 4 or algorithm 6 step 5
 *
 *  This function computes the algorithm step 5.4 (corresponding to
 *  the step 5.6 in [NehabEtAl:2011]) or algorithm step 6.5 following:
 *
 *  \li In parallel for all \f$m\f$ and \f$n\f$, load input block
 *  \f$B_{m,n}(X)\f$ and all its block feedbacks \f$P_{m-1,n}(Y)\f$,
 *  \f$E_{m+1,n}(Z)\f$, \f$P^T_{m,n-1}(U)\f$, and
 *  \f$E^T_{m,n+1}(V)\f$.  Compute and store \f$B_{m,n}(V)\f$.
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @note This follows the improved base-line implementation in [NehabMaximo:2016]
 *  @see [NehabEtAl:2011] cited in alg5() and [NehabMaximo:2016] cited in alg6()
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
 *  @tparam BORDER Flag to consider border input padding
 *  @tparam R Filter order
 */
template <bool BORDER, int R>
__global__ __launch_bounds__(WS*NWW, NBCW)
void alg5v6_step4v5( float *g_out,
                     const Matrix<float,R,WS> *g_py,
                     const Matrix<float,R,WS> *g_ez,
                     const Matrix<float,R,WS> *g_ptu,
                     const Matrix<float,R,WS> *g_etv,
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

#ifdef REGS
    float x[32]; // 32 regs
#endif

    if (ty==0) {

#ifdef REGS
#pragma unroll
        for (int i=0; i<32; ++i)
            x[i] = block[tx][i];
#endif

        Vector<float,R> p, e;

#ifdef LDG
#pragma unroll
        for (int r=0; r<R; ++r)
            p[r] = __ldg((const float *)&g_py[n*(m_size+1)+m][r][tx]);
#else
        p = ((Matrix<float,R,WS>*)&g_py[n*(m_size+1)+m][0][tx])->col(0);
#endif

#pragma unroll // calculate block, scan left -> right
        for (int j=0; j<WS; ++j)
#ifdef REGS
            x[j] = fwdI(p, x[j], c_weights);
#else
            block[tx][j] = fwdI(p, block[tx][j], c_weights);
#endif

#ifdef LDG
#pragma unroll
        for (int r=0; r<R; ++r)
            e[r] = __ldg((const float *)&g_ez[n*(m_size+1)+m+1][r][tx]);
#else
        e = ((Matrix<float,R,WS>*)&g_ez[n*(m_size+1)+m+1][0][tx])->col(0);
#endif

#pragma unroll // calculate block, scan right -> left
        for (int j=WS-1; j>=0; --j)
#ifdef REGS
            x[j] = revI(x[j], e, c_weights);
#else
            block[tx][j] = revI(block[tx][j], e, c_weights);
#endif

#ifdef REGS
#pragma unroll // tranpose regs part-1
        for (int i=0; i<32; ++i)
            block[tx][i] = x[i];
#pragma unroll // transpose regs part-2
        for (int i=0; i<32; ++i)
            x[i] = block[i][tx];
#endif

#ifdef LDG
#pragma unroll
        for (int r=0; r<R; ++r)
            p[r] = __ldg((const float *)&g_ptu[m*(n_size+1)+n][r][tx]);
#else
        p = ((Matrix<float,R,WS>*)&g_ptu[m*(n_size+1)+n][0][tx])->col(0);
#endif

#pragma unroll // calculate block, scan top -> bottom
        for (int j=0; j<WS; ++j)
#ifdef REGS
            x[j] = fwdI(p, x[j], c_weights);
#else
            block[j][tx] = fwdI(p, block[j][tx], c_weights);
#endif

#ifdef LDG
#pragma unroll
        for (int r=0; r<R; ++r)
            e[r] = __ldg((float *)&g_etv[m*(n_size+1)+n+1][r][tx]);
#else
        e = ((Matrix<float,R,WS>*)&g_etv[m*(n_size+1)+n+1][0][tx])->col(0);
#endif

#pragma unroll // calculate block, scan bottom -> top
        for (int j=WS-1; j>=0; --j)
#ifdef REGS
            x[j] = revI(x[j], e, c_weights);
#else
            block[j][tx] = revI(block[j][tx], e, c_weights);
#endif

        if (BORDER) {
            if ((m >= c_border) && (m < m_size-c_border) && (n >= c_border) && (n < n_size-c_border)) {
                g_out += ((n-c_border+1)*WS-1)*out_stride + (m-c_border)*WS+tx;
#pragma unroll // write block inside valid image
                for (int i=0; i<WS; ++i, g_out-=out_stride) {
#ifdef REGS
                    *g_out = x[WS-1-i];
#else
                    *g_out = block[WS-1-i][tx];
#endif
                }
            }
        } else {
            g_out += ((n+1)*WS-1)*out_stride + m*WS+tx;
#pragma unroll // write block
            for (int i=0; i<WS; ++i, g_out-=out_stride) {
#ifdef REGS
                *g_out = x[WS-1-i];
#else
                *g_out = block[WS-1-i][tx];
#endif
            }
        }

    }

}

//==============================================================================
} // namespace gpufilter
//==============================================================================
#endif // ALG3v4v5v6_GPU_CUH
//==============================================================================
