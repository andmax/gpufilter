/**
 *  @file alg4pd_gpu.cuh
 *  @brief Algorithm 4 plic in the GPU (Deriche version) with borders
 *  @author Andre Maximo
 *  @date Jun, 2014
 */

#ifndef ALG4PD_GPU_CUH
#define ALG4PD_GPU_CUH

#define USE_FUSION true

//=== IMPLEMENTATION ============================================================

// TODO LIST:
// * MAKE ADJUST CARRIES MORE PARALLEL

// Alg4p (Deriche) GPU ----------------------------------------------------------

// Collect carries --------------------------------------------------------------

template <int R>
__global__ __launch_bounds__(WS*NWC, NBCW)
void alg4pd_gpu_collect_carries( Matrix<float,R,WS> *g_py1bar, 
                                 Matrix<float,R,WS> *g_ey2bar,
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

        float px = 0.f; // previous x
        Vector<float,R> py1; // current P(Y_1)

        if (m==0) { // 1st block
#if CLAMPTOEDGE
            px = x[0]; // clamp to edge
            py1[0] = c_coefs[4]*px;
#pragma unroll
            for (int i=1; i<R; ++i)
                py1[i] = py1[0];
            g_py1bar[n*(m_size+1)+0].set_col(tx, py1);
#endif
        } else { // other blocks
            float tu = ((m-1)*WS+(WS-1)+.5f)*inv_width; // last column of previous block
            float tv = (n*WS+tx+.5f)*inv_height; // one element per row
            px = tex2D(t_in, tu, tv);
        }

        py1 = zeros<float,R>();

#pragma unroll // calculate pybar, scan left -> right
        for (int j=0; j<WS; ++j) {
            float cx = x[j]; // current x
            x[j] = c_coefs[0]*cx + c_coefs[1]*px;
            x[j] = fwdI(py1, x[j], c_weights);
            px = cx;
        }

        g_py1bar[n*(m_size+1)+m+1].set_col(tx, py1);

    } else if (ty==1) { // 1 warp reverse computing y2

#pragma unroll
        for (int i=0; i<32; ++i)
            x[i] = block[tx][i];
        
        float nx = 0.f, nnx = 0.f; // next and next-next x
        Vector<float,R> ey2; // current E(Y_2)

        if (m==m_size-1) { // last block
#if CLAMPTOEDGE
            nx = nnx = x[WS-1]; // clamp to edge
            ey2[0] = c_coefs[5]*nx;
#pragma unroll
            for (int i=1; i<R; ++i)
                ey2[i] = ey2[0];
            g_ey2bar[n*(m_size+1)+m_size].set_col(tx, ey2);
#endif
        } else { // other blocks
            float tu = ((m+1)*WS+.5f)*inv_width; // first column of next block
            float tv = (n*WS+tx+.5f)*inv_height; // one element per row
            nx = tex2D(t_in, tu, tv);
            nnx = tex2D(t_in, tu+inv_width, tv); // second column of next block
        }

        ey2 = zeros<float,R>();

#pragma unroll // calculate ezhat, scan right -> left
        for (int j=WS-1; j>=0; --j) {
            float cx = x[j]; // current x
            x[j] = c_coefs[2]*nx + c_coefs[3]*nnx;
            x[j] = revI(x[j], ey2, c_weights);
            nnx = nx;
            nx = cx;
        }

        g_ey2bar[n*(m_size+1)+m].set_col(tx, ey2);

    }

}

// Adjust carries ---------------------------------------------------------------

template <int R>
__global__ __launch_bounds__(WS*NWA, NBA)
void alg4pd_gpu_adjust_carries( Matrix<float,R,WS> *g_py1bar,
                                Matrix<float,R,WS> *g_ey2bar,
                                int m_size ) {

    int tx = threadIdx.x, ty = threadIdx.y, m, n = blockIdx.y;
    Matrix<float,R,WS> *gpy1bar, *gey2bar;
    Vector<float,R> py1, ey2, py1bar, ey2bar;
    __shared__ Matrix<float,R,WS> spy1ey2bar[NWA];

    // P(y1bar) -> P(y1) processing -----------------------------------------------
    m = 0;
    gpy1bar = (Matrix<float,R,WS> *)&g_py1bar[n*(m_size+1)+m+ty+1][0][tx];
    if (ty==0) py1 = g_py1bar[n*(m_size+1)+0].col(tx);

    for (; m < m_size; m += NWA) { // for all image blocks

        // using smem as cache
        if (m+ty < m_size) spy1ey2bar[ty].set_col(tx, gpy1bar->col(0));

        __syncthreads(); // wait to load smem

        if (ty==0) {
#pragma unroll // adjust py1bar left -> right
            for (int w = 0; w < NWA; ++w) {

                py1bar = spy1ey2bar[w].col(tx);

                py1 = py1bar + py1 * c_AbF_T;

                spy1ey2bar[w].set_col(tx, py1);

            }
        }

        __syncthreads(); // wait to store result in gmem

        if (m+ty < m_size) gpy1bar->set_col(0, spy1ey2bar[ty].col(tx));

        gpy1bar += NWA;

    }

    // E(zhat) -> E(z) processing -----------------------------------------------
    m = m_size-1;
    gey2bar = (Matrix<float,R,WS> *)&g_ey2bar[n*(m_size+1)+m-ty][0][tx];
    if (ty==0) ey2 = g_ey2bar[n*(m_size+1)+m_size].col(tx);

    for (; m >= 0; m -= NWA) { // for all image blocks

        // using smem as cache
        if (m-ty >= 0) spy1ey2bar[ty].set_col(tx, gey2bar->col(0));

        __syncthreads(); // wait to load smem

        if (ty==0) {
#pragma unroll // adjust ey2bar right -> left
            for (int w = 0; w < NWA; ++w) {

                ey2bar = spy1ey2bar[w].col(tx);

                ey2 = ey2bar + ey2 * c_AbR_T;

                spy1ey2bar[w].set_col(tx, ey2);

            }
        }

        __syncthreads(); // wait to store result in gmem

        if (m-ty >= 0) gey2bar->set_col(0, spy1ey2bar[ty].col(tx));

        gey2bar -= NWA;

    }

}

// Write results transpose ------------------------------------------------------

template <bool FUSION, int R>
__global__ __launch_bounds__(WS*NWW, NBCW)
void alg4pd_gpu_write_results_transp( float *g_transp_out,
                                      const Matrix<float,R,WS> *g_rows_py1,
                                      const Matrix<float,R,WS> *g_rows_ey2,
                                      Matrix<float,R,WS> *g_cols_py1,
                                      Matrix<float,R,WS> *g_cols_ey2,
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

    if (ty==0) { // 1 warp forward computing y1

#pragma unroll
        for (int i=0; i<32; ++i)
            x[i] = block[tx][i];

        float px = 0.f; // previous x
        Vector<float,R> py1 = py1m1->col(0); // current P(Y_1)

        if (m==0) { // 1st block
#if CLAMPTOEDGE
            px = x[0]; // clamp to edge
#endif
        } else { // other blocks
            float tu = ((m-1)*WS+(WS-1)+.5f)*inv_width; // last column of previous block
            float tv = (n*WS+tx+.5f)*inv_height; // one element per row
            px = tex2D(t_in, tu, tv);
        }

#pragma unroll // calculate block, scan left -> right
        for (int j=0; j<WS; ++j) {
            float cx = x[j]; // current x
            x[j] = c_coefs[0]*cx + c_coefs[1]*px;
            x[j] = fwdI(py1, x[j], c_weights);
            px = cx;
        }

    } else if (ty==1) { // 1 warp reverse computing y2

#pragma unroll
        for (int i=0; i<32; ++i)
            x[i] = block[tx][i];

        float nx = 0.f, nnx = 0.f; // next and next-next x
        Vector<float,R> ey2 = ey2m1->col(0); // current E(Y_2)

        if (m==m_size-1) { // last block
#if CLAMPTOEDGE
            nx = nnx = x[WS-1]; // clamp to edge
#endif
        } else { // other blocks
            float tu = ((m+1)*WS+.5f)*inv_width; // first column of next block
            float tv = (n*WS+tx+.5f)*inv_height; // one element per row
            nx = tex2D(t_in, tu, tv);
            nnx = tex2D(t_in, tu+inv_width, tv); // second column of next block
        }

#pragma unroll // calculate block, scan right -> left
        for (int j=WS-1; j>=0; --j) {
            float cx = x[j]; // current x
            x[j] = c_coefs[2]*nx + c_coefs[3]*nnx;
            x[j] = revI(x[j], ey2, c_weights);
            nnx = nx;
            nx = cx;
        }

    }

    __syncthreads(); // guarantee no smem usage

    if (ty==1) {

#pragma unroll
        for (int i=0; i<32; ++i)
            block[tx][i] = x[i]; // save partial result in smem

    }

    __syncthreads(); // guarantee warp 0 can read smem

    if (ty==0) {

#pragma unroll
        for (int i=0; i<32; ++i)
            x[i] += block[tx][i]; // x now has z or v

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

            float pz = 0.f; // previous z
            Vector<float,R> pu1; // current P(U_1)

            if (n==0) { // 1st block
#if CLAMPTOEDGE
                pz = x[0]; // clamp to edge
                pu1[1] = pu1[0] = c_coefs[4]*pz;
                g_cols_py1[m*(n_size+1)+0].set_col(tx, pu1);
#endif
            } else { // other blocks
                g_transp_out += (m*WS+tx)*out_stride + (n-1)*WS + (WS-1);
                pz = *g_transp_out;
            }

            pu1 = zeros<float,R>();

#pragma unroll // calculate pybar cols, scan left -> right
            for (int j=0; j<WS; ++j) {
                float cz = x[j]; // current z
                x[j] = c_coefs[0]*cz + c_coefs[1]*pz;
                x[j] = fwdI(pu1, x[j], c_weights);
                pz = cz;
            }

            pu1bar.set_col(0, pu1); // store pu1bar cols

        } else if (ty==3) { // 1 warp reverse computing u2

#pragma unroll // transpose regs part-2
            for (int i=0; i<32; ++i)
                x[i] = block[i][tx];

            Matrix<float,R,WS>
                &eu2bar = (Matrix<float,R,WS>&)g_cols_ey2[m*(n_size+1)+n][0][tx];

            float nz = 0.f, nnz = 0.f; // next and next-next z
            Vector<float,R> eu2; // current E(U_2)

            if (n==n_size-1) { // last block
#if CLAMPTOEDGE
                nz = nnz = x[WS-1]; // clamp to edge
                eu2[1] = eu2[0] = c_coefs[5]*nz;
                g_cols_ey2[m*(n_size+1)+n_size].set_col(tx, eu2);
#endif
            } else { // other blocks
                g_transp_out += (m*WS+tx)*out_stride + (n+1)*WS;
                nz = *g_transp_out;
                g_transp_out += 1;
                nnz = *g_transp_out;
            }

            eu2 = zeros<float,R>();

#pragma unroll // calculate ezhat cols, scan right -> left
            for (int j=WS-1; j>=0; --j) {
                float cz = x[j]; // current z
                x[j] = c_coefs[2]*nz + c_coefs[3]*nnz;
                x[j] = revI(x[j], eu2, c_weights);
                nnz = nz;
                nz = cz;
            }

            eu2bar.set_col(0, eu2); // store eu2bar cols

        }

    } // if fusion

}

// Alg4p Deriche in the GPU -----------------------------------------------------

__host__
void alg4pd_gpu( float *h_img, int width, int height,
                 float sigma, int order ) {

    const int B = WS;
    if (R != 2) { std::cout << "R must be 2!\n"; return; }

    Vector<float, 6> coefs; // [a0-a3, coefp, coefn]
    Vector<float, R+1> w; // weights
    w[0] = 1.f;
    
    computeFilterParams(coefs[0], coefs[1], coefs[2], coefs[3],
                        w[1], w[2], coefs[4], coefs[5],
                        sigma, order);

    // pre-compute basic alg4p Deriche matrices
    Matrix<float,R,R> Ir = identity<float,R,R>();
    Matrix<float,R,B> Zrb = zeros<float,R,B>();

    Matrix<float,R,B> AFP_T = fwd(Ir, Zrb, w),
                      ARE_T = rev(Zrb, Ir, w);

    Matrix<float,R,R> AbF_T = tail<R>(AFP_T),
                      AbR_T = head<R>(ARE_T);

    int m_size = (width+WS-1)/WS, n_size = (height+WS-1)/WS;

    // upload to the GPU
    copy_to_symbol(c_weights, w);
    copy_to_symbol(c_coefs, coefs);

    copy_to_symbol(c_AbF_T, AbF_T);
    copy_to_symbol(c_AbR_T, AbR_T);

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

    d_rows_py1bar.fillzero();
    d_rows_ey2bar.fillzero();
    d_cols_py1bar.fillzero();
    d_cols_ey2bar.fillzero();

#if RUNTIMES>1
    base_timer &timer_total = timers.gpu_add("alg4pd_gpu", width*height*RUNTIMES, "iP");

    for(int r=0; r<RUNTIMES; ++r) {

        cudaBindTextureToArray(t_in, a_in);

        alg4pd_gpu_collect_carries<<< dim3(m_size, n_size), dim3(WS, NWC) >>>
            ( &d_rows_py1bar, &d_rows_ey2bar, inv_width, inv_height, m_size );

        alg4pd_gpu_adjust_carries<<< dim3(1, n_size), dim3(WS, NWA) >>>
            ( &d_rows_py1bar, &d_rows_ey2bar, m_size );

        alg4pd_gpu_write_results_transp<USE_FUSION><<< dim3(m_size, n_size), dim3(WS, NWW) >>>
            ( d_transp_img, &d_rows_py1bar, &d_rows_ey2bar, &d_cols_py1bar, &d_cols_ey2bar,
              inv_width, inv_height, m_size, n_size, stride_transp_img );

        if (!USE_FUSION) {
            alg4pd_gpu_collect_carries<<< dim3(n_size, m_size), dim3(WS, NWC) >>>
                ( &d_cols_py1bar, &d_cols_ey2bar,
                  inv_height, inv_width, n_size );
        }
    
        cudaUnbindTexture(t_in);
        cudaBindTexture2D(&offset, t_in, d_transp_img, height, width, stride_transp_img*sizeof(float));

        alg4pd_gpu_adjust_carries<<< dim3(1, m_size), dim3(WS, NWA) >>>
            ( &d_cols_py1bar, &d_cols_ey2bar, n_size );

        alg4pd_gpu_write_results_transp<false><<< dim3(n_size, m_size), dim3(WS, NWW) >>>
            ( d_img, &d_cols_py1bar, &d_cols_ey2bar, &d_rows_py1bar, &d_cols_ey2bar,
              inv_height, inv_width, n_size, m_size, stride_img );

        cudaUnbindTexture(t_in);

    }

    timer_total.stop();

    std::cout << std::fixed << timer_total.data_size()/(timer_total.elapsed()*1024*1024) << " " << std::flush;
#else
    base_timer &timer_total = timers.gpu_add("alg4pd_gpu", width*height, "iP");
    base_timer *timer;

    cudaBindTextureToArray(t_in, a_in);

    timer = &timers.gpu_add("collect-carries");

    alg4pd_gpu_collect_carries<<< dim3(m_size, n_size), dim3(WS, NWC) >>>
        ( &d_rows_py1bar, &d_rows_ey2bar,
          inv_width, inv_height, m_size );

    timer->stop(); timer = &timers.gpu_add("adjust-carries");

    alg4pd_gpu_adjust_carries<<< dim3(1, n_size), dim3(WS, NWA) >>>
        ( &d_rows_py1bar, &d_rows_ey2bar,
          m_size );

    timer->stop(); timer = &timers.gpu_add("write-block-collect-carries");

    alg4pd_gpu_write_results_transp<USE_FUSION><<< dim3(m_size, n_size), dim3(WS, NWW) >>>
        ( d_transp_img, &d_rows_py1bar, &d_rows_ey2bar, &d_cols_py1bar, &d_cols_ey2bar,
          inv_width, inv_height, m_size, n_size, stride_transp_img );

    timer->stop(); timer = &timers.gpu_add("adjust-carries");

    cudaUnbindTexture(t_in);
    cudaBindTexture2D(&offset, t_in, d_transp_img, height, width, stride_transp_img*sizeof(float));

    if (!USE_FUSION) { // it seems faster to not use FUSION - not quite
        alg4pd_gpu_collect_carries<<< dim3(n_size, m_size), dim3(WS, NWC) >>>
            ( &d_cols_py1bar, &d_cols_ey2bar,
              inv_height, inv_width, n_size );
    }

    alg4pd_gpu_adjust_carries<<< dim3(1, m_size), dim3(WS, NWA) >>>
        ( &d_cols_py1bar, &d_cols_ey2bar,
          n_size );

    timer->stop(); timer = &timers.gpu_add("write-block");

    alg4pd_gpu_write_results_transp<false><<< dim3(n_size, m_size), dim3(WS, NWW) >>>
        ( d_img, &d_cols_py1bar, &d_cols_ey2bar, &d_rows_py1bar, &d_rows_ey2bar,
          inv_height, inv_width, n_size, m_size, stride_img );

    timer->stop();

    cudaUnbindTexture(t_in);

    timer_total.stop();
    timers.flush();
#endif
    cudaMemcpy2D(h_img, width*sizeof(float), d_img, stride_img*sizeof(float), width*sizeof(float), height, cudaMemcpyDeviceToHost);

}

#endif // ALG4PD_GPU_CUH
