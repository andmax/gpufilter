/**
 *  @file alg4d_gpu.cuh
 *  @brief Algorithm 4 in the GPU (Deriche version) with borders
 *  @author Andre Maximo
 *  @date Jun, 2014
 */

#ifndef ALG4D_GPU_CUH
#define ALG4D_GPU_CUH

#define USE_FUSION true

//=== IMPLEMENTATION ============================================================

// Alg4 (Deriche) GPU -----------------------------------------------------------

// Collect carries --------------------------------------------------------------

template <int R>
__global__ __launch_bounds__(WS*NWC, NBCW)
void alg4d_collect_carries( Matrix<float,R,WS> *g_py1bar, 
                            Matrix<float,R,WS> *g_ey2bar,
                            Matrix<float,R,WS> *g_px,
                            Matrix<float,R,WS> *g_ex,
                            float inv_width, float inv_height,
                            int m_size ) {

    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n = blockIdx.y;

    __shared__ Matrix<float,WS,WS+1> block;
    read_block<NWC>(block, m, n, inv_width, inv_height);
    __syncthreads();

    float x[32]; // 32 regs per thread

    if (ty==0) { // 1 warp forward computing y1

#pragma unroll
        for (int i=0; i<32; ++i)
            x[i] = block[tx][i];

        float xp = 0.f; // previous x
        Vector<float,R> py1; // current P(Y_1)
        Vector<float,R> px; // this block P(X)

        if (m==0) { // 1st block
#if CLAMPTOEDGE
            xp = x[0]; // clamp to edge
            py1[0] = c_coefs[4]*xp;
            px[0] = xp;
#pragma unroll
            for (int i=1; i<R; ++i) {
                py1[i] = py1[0];
                px[i] = px[0];
            }
            g_py1bar[n*(m_size+1)+0].set_col(tx, py1);
            g_px[n*(m_size+1)+0].set_col(tx, px);
#endif
        }

        py1 = zeros<float,R>();

#pragma unroll
        for (int i=0; i<R; ++i)
            px[i] = x[WS-1-i];

#pragma unroll // calculate py1bar, scan left -> right
        for (int j=0; j<WS; ++j) {
            float xc = x[j]; // current x
            x[j] = c_coefs[0]*xc + c_coefs[1]*xp;
            x[j] = fwdI(py1, x[j], c_weights);
            xp = xc;
        }

        g_py1bar[n*(m_size+1)+m+1].set_col(tx, py1);
        g_px[n*(m_size+1)+m+1].set_col(tx, px);

    } else if (ty==1) { // 1 warp reverse computing y2

#pragma unroll
        for (int i=0; i<32; ++i)
            x[i] = block[tx][i];
        
        float xn = 0.f, xa = 0.f; // next and after next x
        Vector<float,R> ey2; // current E(Y_2)
        Vector<float,R> ex; // this block E(X)

        if (m==m_size-1) { // last block
#if CLAMPTOEDGE
            xn = xa = x[WS-1]; // clamp to edge
            ey2[0] = c_coefs[5]*xn;
            ex[0] = xn;
#pragma unroll
            for (int i=1; i<R; ++i) {
                ey2[i] = ey2[0];
                ex[i] = ex[0];
            }
            g_ey2bar[n*(m_size+1)+m_size].set_col(tx, ey2);
            g_ex[n*(m_size+1)+m_size].set_col(tx, ex);
#endif
        }

        ey2 = zeros<float,R>();

#pragma unroll
        for (int i=0; i<R; ++i)
            ex[i] = x[i];

#pragma unroll // calculate ey2bar, scan right -> left
        for (int j=WS-1; j>=0; --j) {
            float xc = x[j]; // current x
            x[j] = c_coefs[2]*xn + c_coefs[3]*xa;
            x[j] = revI(x[j], ey2, c_weights);
            xa = xn;
            xn = xc;
        }

        g_ey2bar[n*(m_size+1)+m].set_col(tx, ey2);
        g_ex[n*(m_size+1)+m].set_col(tx, ex);

    }

}

// Adjust carries ---------------------------------------------------------------

template <int R>
__global__ __launch_bounds__(WS*NWA, NBA)
void alg4d_adjust_carries( Matrix<float,R,WS> *g_py1bar,
                           Matrix<float,R,WS> *g_ey2bar,
                           Matrix<float,R,WS> *g_px,
                           Matrix<float,R,WS> *g_ex,
                           int m_size ) {

    int tx = threadIdx.x, ty = threadIdx.y, m, n = blockIdx.y;
    Matrix<float,R,WS> *gpy1bar, *gey2bar;
    Vector<float,R> py1, ey2, py1bar, ey2bar;
    __shared__ Matrix<float,R,WS> spy1ey2bar[NWA];

    Matrix<float,R,WS> *gpx, *gex;
    Vector<float,R> px, ex;
    __shared__ Matrix<float,R,WS> spxex[NWA];

    // P(y1bar) -> P(y1) processing ---------------------------------------------
    m = 0;
    gpy1bar = (Matrix<float,R,WS> *)&g_py1bar[n*(m_size+1)+m+ty+1][0][tx];
    gpx = (Matrix<float,R,WS> *)&g_px[n*(m_size+1)+m+ty][0][tx];
    if (ty==0) // for boundary condition
        py1 = g_py1bar[n*(m_size+1)+0].col(tx);

    // more parallelism here is not important
    for (; m < m_size; m += NWA) { // for all image blocks

        if (m+ty < m_size) { // using smem as cache
            spy1ey2bar[ty].set_col(tx, gpy1bar->col(0));
            spxex[ty].set_col(tx, gpx->col(0));
        }

        __syncthreads(); // wait to load smem

        if (ty==0) {
#pragma unroll // adjust py1bar left -> right
            for (int w = 0; w < NWA; ++w) {

                py1bar = spy1ey2bar[w].col(tx);
                px = spxex[w].col(tx);

                py1 = py1bar + py1 * c_AbF_T + px * c_TAFB_AC1P_T;

                spy1ey2bar[w].set_col(tx, py1);

            }
        }

        __syncthreads(); // wait to store result in gmem

        if (m+ty < m_size) gpy1bar->set_col(0, spy1ey2bar[ty].col(tx));

        gpy1bar += NWA;
        gpx += NWA;

    }

    // E(y2bar) -> E(y2) processing -----------------------------------------------
    m = m_size-1;
    gey2bar = (Matrix<float,R,WS> *)&g_ey2bar[n*(m_size+1)+m-ty][0][tx];
    gex = (Matrix<float,R,WS> *)&g_ex[n*(m_size+1)+m+1-ty][0][tx];
    if (ty==0) // for boundary condition
        ey2 = g_ey2bar[n*(m_size+1)+m_size].col(tx);

    for (; m >= 0; m -= NWA) { // for all image blocks

        // using smem as cache
        if (m-ty >= 0) {
            spy1ey2bar[ty].set_col(tx, gey2bar->col(0));
            spxex[ty].set_col(tx, gex->col(0));
        }

        __syncthreads(); // wait to load smem

        if (ty==0) {
#pragma unroll // adjust ey2bar right -> left
            for (int w = 0; w < NWA; ++w) {

                ey2bar = spy1ey2bar[w].col(tx);
                ex = spxex[w].col(tx);

                ey2 = ey2bar + ey2 * c_AbR_T + ex * c_HARB_AC2E_T;

                spy1ey2bar[w].set_col(tx, ey2);

            }
        }

        __syncthreads(); // wait to store result in gmem

        if (m-ty >= 0) gey2bar->set_col(0, spy1ey2bar[ty].col(tx));

        gey2bar -= NWA;
        gex -= NWA;

    }

}

// Write results transpose ------------------------------------------------------

template <bool FUSION, int R>
__global__ __launch_bounds__(WS*NWW, NBCW)
void alg4d_write_results_transp( float *g_transp_out,
                                 const Matrix<float,R,WS> *g_rows_py1,
                                 const Matrix<float,R,WS> *g_rows_ey2,
                                 Matrix<float,R,WS> *g_cols_py1,
                                 Matrix<float,R,WS> *g_cols_ey2,
                                 const Matrix<float,R,WS> *g_rows_px,
                                 const Matrix<float,R,WS> *g_rows_ex,
                                 Matrix<float,R,WS> *g_cols_pz,
                                 Matrix<float,R,WS> *g_cols_ez,
                                 float inv_width, float inv_height,
                                 int m_size, int n_size,
                                 int out_stride ) {

    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n = blockIdx.y;

    __shared__ Matrix<float,WS,WS+1> block;
    read_block<NWW>(block, m, n, inv_width, inv_height);
    __syncthreads();

    float x[32]; // 32 regs per thread

    Matrix<float,R,WS>
        *py1m1 = (Matrix<float,R,WS>*)&g_rows_py1[n*(m_size+1)+m][0][tx],
        *ey2m1 = (Matrix<float,R,WS>*)&g_rows_ey2[n*(m_size+1)+m+1][0][tx];

    Matrix<float,R,WS>
        *pxm1 = (Matrix<float,R,WS>*)&g_rows_px[n*(m_size+1)+m][0][tx],
        *exm1 = (Matrix<float,R,WS>*)&g_rows_ex[n*(m_size+1)+m+1][0][tx];

    if (ty==0) { // 1 warp forward computing y1

#pragma unroll
        for (int i=0; i<32; ++i)
            x[i] = block[tx][i];

        float xp; // previous x
        Vector<float,R> py1 = py1m1->col(0); // current P(Y_1)
        Vector<float,R> px = pxm1->col(0); // previous block P(X)

        xp = px[0];

#pragma unroll // calculate block, scan left -> right
        for (int j=0; j<WS; ++j) {
            float xc = x[j]; // current x
            x[j] = c_coefs[0]*xc + c_coefs[1]*xp;
            x[j] = fwdI(py1, x[j], c_weights);
            xp = xc;
        }

    } else if (ty==1) { // 1 warp reverse computing y2

#pragma unroll
        for (int i=0; i<32; ++i)
            x[i] = block[tx][i];

        float xn, xa; // next and after next x
        Vector<float,R> ey2 = ey2m1->col(0); // current E(Y_2)
        Vector<float,R> ex = exm1->col(0); // next block E(X)

        xn = ex[0];
        if (R==1) xa = xn;
        else xa = ex[1];

#pragma unroll // calculate block, scan right -> left
        for (int j=WS-1; j>=0; --j) {
            float xc = x[j]; // current x
            x[j] = c_coefs[2]*xn + c_coefs[3]*xa;
            x[j] = revI(x[j], ey2, c_weights);
            xa = xn;
            xn = xc;
        }

    }

    // preparing for adding y1 and y2

    __syncthreads(); // guarantee no smem usage

    if (ty==1) {

#pragma unroll
        for (int i=0; i<32; ++i)
            block[tx][i] = x[i]; // save partial result in smem

    }

    __syncthreads(); // guarantee warp 0 can read smem

    if (ty==0) { // now add

#pragma unroll
        for (int i=0; i<32; ++i)
            x[i] += block[tx][i]; // x now has z

        g_transp_out += m*WS*out_stride + n*WS + tx;
#pragma unroll // write block inside valid transpose image
        for (int i=0; i<WS; ++i, g_transp_out += out_stride) {
            *g_transp_out = x[i];
        }

    }

    if (FUSION) {

        if (ty==0) {
#pragma unroll // transpose regs part-1
            for (int i=0; i<32; ++i)
                block[tx][i] = x[i];

        }

        __syncthreads();

        if (ty==2) { // 1 warp forward computing u1

#pragma unroll // transpose regs part-2
            for (int i=0; i<32; ++i)
                x[i] = block[i][tx];

            Matrix<float,R,WS>
                &pu1bar = (Matrix<float,R,WS>&)g_cols_py1[m*(n_size+1)+n+1][0][tx];

            float zp = 0.f; // previous z
            Vector<float,R> pu1; // current P(U_1)
            Vector<float,R> pz; // this block P(Z)

            if (n==0) { // 1st block
#if CLAMPTOEDGE
                zp = x[0]; // clamp to edge
                pu1[0] = c_coefs[4]*zp;
                pz[0] = zp;
#pragma unroll
                for (int i=1; i<R; ++i) {
                    pu1[i] = pu1[0];
                    pz[i] = pz[0];
                }
                g_cols_py1[m*(n_size+1)+0].set_col(tx, pu1);
                g_cols_pz[m*(n_size+1)+0].set_col(tx, pz);
#endif
            }

            pu1 = zeros<float,R>();

#pragma unroll
            for (int i=0; i<R; ++i)
                pz[i] = x[WS-1-i];

#pragma unroll // calculate py1bar cols, scan left -> right
            for (int j=0; j<WS; ++j) {
                float zc = x[j]; // current z
                x[j] = c_coefs[0]*zc + c_coefs[1]*zp;
                x[j] = fwdI(pu1, x[j], c_weights);
                zp = zc;
            }

            pu1bar.set_col(0, pu1); // store pu1bar cols
            g_cols_pz[m*(n_size+1)+n+1].set_col(tx, pz);

        } else if (ty==3) { // 1 warp reverse computing u2

#pragma unroll // transpose regs part-2
            for (int i=0; i<32; ++i)
                x[i] = block[i][tx];

            Matrix<float,R,WS>
                &eu2bar = (Matrix<float,R,WS>&)g_cols_ey2[m*(n_size+1)+n][0][tx];

            float zn = 0.f, za = 0.f; // next and after next z
            Vector<float,R> eu2; // current E(U_2)
            Vector<float,R> ez; // this block E(Z)

            if (n==n_size-1) { // last block
#if CLAMPTOEDGE
                zn = za = x[WS-1]; // clamp to edge
                eu2[0] = c_coefs[5]*zn;
                ez[0] = zn;
#pragma unroll
                for (int i=1; i<R; ++i) {
                    eu2[i] = eu2[0];
                    ez[i] = ez[0];
                }
                g_cols_ey2[m*(n_size+1)+n_size].set_col(tx, eu2);
                g_cols_ez[m*(n_size+1)+n_size].set_col(tx, ez);
#endif
            }

            eu2 = zeros<float,R>();

#pragma unroll
            for (int i=0; i<R; ++i)
                ez[i] = x[i];

#pragma unroll // calculate ey2bar cols, scan right -> left
            for (int j=WS-1; j>=0; --j) {
                float zc = x[j]; // current z
                x[j] = c_coefs[2]*zn + c_coefs[3]*za;
                x[j] = revI(x[j], eu2, c_weights);
                za = zn;
                zn = zc;
            }

            eu2bar.set_col(0, eu2); // store eu2bar cols
            g_cols_ez[m*(n_size+1)+n].set_col(tx, ez);

        }

    } // if fusion

}

// Alg4 Deriche in the GPU ------------------------------------------------------

__host__
void alg4d_gpu( float *h_img, int width, int height,
                float sigma, int order ) {

    const int B = WS;
    if (R != 2) { std::cout << "R must be 2!\n"; return; }

    Vector<float, 6> coefs; // [a0-a3, coefp, coefn]
    Vector<float, R+1> w; // weights
    w[0] = 1.f;
    
    computeFilterParams(coefs[0], coefs[1], coefs[2], coefs[3],
                        w[1], w[2], coefs[4], coefs[5],
                        sigma, order);

    Vector<float, R+1> ck1; // convolution kernel 1 (plus)
    ck1[0] = coefs[0]; ck1[1] = coefs[1]; ck1[2] = 0.f;
    Vector<float, R+1> ck2; // convolution kernel 2 (minus)
    ck2[0] = 0.f; ck2[1] = coefs[2]; ck2[2] = coefs[3];

    // pre-compute basic alg4 Deriche matrices
    Matrix<float,R,R> Ir = identity<float,R,R>();
    Matrix<float,R,B> Zrb = zeros<float,R,B>();
    Matrix<float,B,R> Zbr = zeros<float,B,R>();
    Matrix<float,B,B> Ib = identity<float,B,B>();

    Matrix<float,R,B> AFP_T = fwd(Ir, Zrb, w), ARE_T = rev(Zrb, Ir, w);
    Matrix<float,B,B> AFB_T = fwd(Zbr, Ib, w), ARB_T = rev(Ib, Zbr, w);
    Matrix<float,R,R> AbF_T = tail<R>(AFP_T), AbR_T = head<R>(ARE_T);
    Matrix<float,B,R> TAFB_T = tail<R>(AFB_T), HARB_T = head<R>(ARB_T);

    Matrix<float,R,B> AC1P_T = transp(conv1p<float,B,R>(ck1));
    Matrix<float,R,B> AC2E_T = transp(conv2e<float,B,R>(ck2));

    Matrix<float,R,R> TAFB_AC1P_T = AC1P_T*TAFB_T;
    Matrix<float,R,R> HARB_AC2E_T = AC2E_T*HARB_T;

    int m_size = (width+WS-1)/WS, n_size = (height+WS-1)/WS;

    // upload to the GPU
    copy_to_symbol(c_weights, w);
    copy_to_symbol(c_coefs, coefs);

    copy_to_symbol(c_AbF_T, AbF_T);
    copy_to_symbol(c_AbR_T, AbR_T);

    copy_to_symbol(c_TAFB_AC1P_T, TAFB_AC1P_T);
    copy_to_symbol(c_HARB_AC2E_T, HARB_AC2E_T);

    float inv_width = 1.f/width, inv_height = 1.f/height;

    cudaArray *a_in;
    size_t offset;
    cudaChannelFormatDesc ccd = cudaCreateChannelDesc<float>();
    cudaMallocArray(&a_in, &ccd, width, height);
    cudaMemcpyToArray(a_in, 0, 0, h_img, width*height*sizeof(float),
                      cudaMemcpyHostToDevice);

    t_in.normalized = true;
    t_in.filterMode = cudaFilterModePoint;
    t_in.addressMode[0] = t_in.addressMode[1] = cudaAddressModeBorder;

    // naming convention: stride in elements and pitch in bytes
    // to achieve at the same time coalescing and L2-cache colision avoidance
    // the transposed and regular output images have stride of maximum size
    // (4Ki) + 1 block (WS), considering up to 4096x4096 input images
    int stride_img = MAXSIZE+WS, stride_transp_img = MAXSIZE+WS;
    dvector<float> d_img(height*stride_img), d_transp_img(width*stride_transp_img);

    // +1 padding is important even in zero-border to avoid if's in kernels
    dvector< Matrix<float,R,B> >
        d_rows_py1bar((m_size+1)*n_size), d_rows_ey2bar((m_size+1)*n_size),
        d_cols_py1bar((n_size+1)*m_size), d_cols_ey2bar((n_size+1)*m_size);

    dvector< Matrix<float,R,B> >
        d_rows_px((m_size+1)*n_size), d_rows_ex((m_size+1)*n_size),
        d_cols_pz((n_size+1)*m_size), d_cols_ez((n_size+1)*m_size);

    d_rows_px.fillzero();
    d_rows_ex.fillzero();
    d_cols_pz.fillzero();
    d_cols_ez.fillzero();

    d_rows_py1bar.fillzero();
    d_rows_ey2bar.fillzero();
    d_cols_py1bar.fillzero();
    d_cols_ey2bar.fillzero();

#if RUNTIMES>1
    base_timer &timer_total = timers.gpu_add("alg4d_gpu", width*height*RUNTIMES, "iP");

    for(int r=0; r<RUNTIMES; ++r) {

        cudaBindTextureToArray(t_in, a_in);

        alg4d_collect_carries<<< dim3(m_size, n_size), dim3(WS, NWC) >>>
            ( &d_rows_py1bar, &d_rows_ey2bar, &d_rows_px, &d_rows_ex, inv_width, inv_height, m_size );

        alg4d_adjust_carries<<< dim3(1, n_size), dim3(WS, NWA) >>>
            ( &d_rows_py1bar, &d_rows_ey2bar, &d_rows_px, &d_rows_ex, m_size );

        alg4d_write_results_transp<true><<< dim3(m_size, n_size), dim3(WS, NWW) >>>
            ( d_transp_img, &d_rows_py1bar, &d_rows_ey2bar, &d_cols_py1bar, &d_cols_ey2bar, &d_rows_px, &d_rows_ex, &d_cols_pz, &d_cols_ez,
              inv_width, inv_height, m_size, n_size, stride_transp_img );
    
        cudaUnbindTexture(t_in);
        cudaBindTexture2D(&offset, t_in, d_transp_img, height, width, stride_transp_img*sizeof(float));

        alg4d_adjust_carries<<< dim3(1, m_size), dim3(WS, NWA) >>>
            ( &d_cols_py1bar, &d_cols_ey2bar, &d_cols_pz, &d_cols_ez, n_size );

        alg4d_write_results_transp<false><<< dim3(n_size, m_size), dim3(WS, NWW) >>>
            ( d_img, &d_cols_py1bar, &d_cols_ey2bar, &d_rows_py1bar, &d_cols_ey2bar, &d_cols_pz, &d_cols_ez, &d_rows_px, &d_rows_ex,
              inv_height, inv_width, n_size, m_size, stride_img );

        cudaUnbindTexture(t_in);

    }

    timer_total.stop();

    std::cout << std::fixed << timer_total.data_size()/(timer_total.elapsed()*1024*1024) << " " << std::flush;
#else
    base_timer &timer_total = timers.gpu_add("alg4d_gpu", width*height, "iP");
    base_timer *timer;

    cudaBindTextureToArray(t_in, a_in);

    timer = &timers.gpu_add("collect-carries");

    alg4d_collect_carries<<< dim3(m_size, n_size), dim3(WS, NWC) >>>
        ( &d_rows_py1bar, &d_rows_ey2bar,
          &d_rows_px, &d_rows_ex,
          inv_width, inv_height, m_size );

    timer->stop(); timer = &timers.gpu_add("adjust-carries");

    alg4d_adjust_carries<<< dim3(1, n_size), dim3(WS, NWA) >>>
        ( &d_rows_py1bar, &d_rows_ey2bar,
          &d_rows_px, &d_rows_ex,
          m_size );

    timer->stop(); timer = &timers.gpu_add("write-block-collect-carries");

    alg4d_write_results_transp<USE_FUSION><<< dim3(m_size, n_size), dim3(WS, NWW) >>>
        ( d_transp_img, &d_rows_py1bar, &d_rows_ey2bar, &d_cols_py1bar, &d_cols_ey2bar,
          &d_rows_px, &d_rows_ex, &d_cols_pz, &d_cols_ez,
          inv_width, inv_height, m_size, n_size, stride_transp_img );

    timer->stop(); timer = &timers.gpu_add("adjust-carries");

    cudaUnbindTexture(t_in);
    cudaBindTexture2D(&offset, t_in, d_transp_img, height, width, stride_transp_img*sizeof(float));

    if (!USE_FUSION) { // it seems faster to not use FUSION - not quite
        alg4d_collect_carries<<< dim3(n_size, m_size), dim3(WS, NWC) >>>
            ( &d_cols_py1bar, &d_cols_ey2bar,
              &d_cols_pz, &d_cols_ez,
              inv_height, inv_width, n_size );
    }

    alg4d_adjust_carries<<< dim3(1, m_size), dim3(WS, NWA) >>>
        ( &d_cols_py1bar, &d_cols_ey2bar,
          &d_cols_pz, &d_cols_ez,
          n_size );

    timer->stop(); timer = &timers.gpu_add("write-block");

    alg4d_write_results_transp<false><<< dim3(n_size, m_size), dim3(WS, NWW) >>>
        ( d_img, &d_cols_py1bar, &d_cols_ey2bar, &d_rows_py1bar, &d_rows_ey2bar,
          &d_cols_pz, &d_cols_ez, &d_rows_px, &d_rows_ex,
          inv_height, inv_width, n_size, m_size, stride_img );

    timer->stop();

    cudaUnbindTexture(t_in);

    timer_total.stop();
    timers.flush();
#endif
    cudaMemcpy2D(h_img, width*sizeof(float), d_img, stride_img*sizeof(float), width*sizeof(float), height, cudaMemcpyDeviceToHost);

}

#endif // ALG4D_GPU_CUH

/**
    //checkings
    Matrix<float,B,R> pm1nx, em1nx;
    Matrix<float,B,B> bmnx;

    srand(time(NULL));
    for (int i=0; i<B; ++i)
        for (int j=0; j<R; ++j) {
            pm1nx[i][j] = (rand()%255)/255.f;
            em1nx[i][j] = (rand()%255)/255.f;
        }
    for (int i=0; i<B; ++i)
        for (int j=0; j<B; ++j)
            bmnx[i][j] = (rand()%255)/255.f;

    //checking 1
    Matrix<float,B,B> bmny1a = conv1(pm1nx, bmnx, ck1);
    Matrix<float,B,B> bmny1b = pm1nx * AC1P_T + bmnx * AC1B_T;

    //checking 2
    Matrix<float,B,B> bmny2a = conv2(bmnx, em1nx, ck2);
    Matrix<float,B,B> bmny2b = em1nx * AC2E_T + bmnx * AC2B_T;

    //checking 3
    Matrix<float,B,R> pmny1bara = tail<R>(fwd(Zbr, bmny1a, w));
    Matrix<float,B,R> pmny1barb = pm1nx * TAFB_AC1P_T + bmnx * TAFB_AC1B_T;

    //checking 4
    Matrix<float,B,R> emny2bara = head<R>(rev(bmny2a, Zbr, w));
    Matrix<float,B,R> emny2barb = em1nx * HARB_AC2E_T + bmnx * HARB_AC2B_T;

    float a, b, r, me1, mre1, me2, mre2;
    mre2 = me2 = mre1 = me1 = 0;

    for (int i=0; i<B; ++i) {
        for (int j=0; j<R; ++j) {
            a = pmny1bara[i][j] - pmny1barb[i][j];
            a = a < 0 ? -a : a;
            if (pmny1bara[i][j] != 0) {
                r = pmny1bara[i][j] < 0 ? -pmny1bara[i][j] : pmny1bara[i][j];
                b = a / r;
                mre1 = b > mre1 ? b : mre1;
            }
            me1 = a > me1 ? a : me1;
            a = emny2bara[i][j] - emny2barb[i][j];
            a = a < 0 ? -a : a;
            if (emny2bara[i][j] != 0) {
                r = emny2bara[i][j] < 0 ? -emny2bara[i][j] : emny2bara[i][j];
                b = a / r;
                mre2 = b > mre2 ? b : mre2;
            }
            me2 = a > me2 ? a : me2;
        }
    }

    //std::cout << "bmny1a x bmny1b: me = " << me1 << " mre = " << mre1 << "\n";
    //std::cout << "bmny2a x bmny2b: me = " << me2 << " mre = " << mre2 << "\n";
    std::cout << "pmny1bara x pmny1barb: me = " << me1 << " mre = " << mre1 << "\n";
    std::cout << "emny2bara x emny2barb: me = " << me2 << " mre = " << mre2 << "\n";

*/
