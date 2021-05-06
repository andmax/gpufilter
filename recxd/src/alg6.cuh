/**
 *  @file alg6.cuh
 *  @brief Algorithm 6 improved in the GPU
 *  @author Andre Maximo
 *  @date Sep, 2019
 *  @copyright The MIT License
 */

#ifndef ALG6I_GPU_CUH
#define ALG6I_GPU_CUH

#define LDG // uncomment to use __ldg
//#if ORDER==1 || ORDER==2 || ORDER==4
#define REGS // uncomment to use registers
//#endif
#define GMAT // uncomment to use global constant matrices

//== INCLUDES ==================================================================

#include "alg3v4v5v6_gpu.cuh"

#include "alg6_gpu.cuh"

//== NAMESPACES ================================================================

namespace gpufilter {

// Launch bounds here does not make any difference
template <int R>
__global__
void alg6i_step2v4( Matrix<float,R,WS> *g_ezhat,
                    const Matrix<float,R,WS> *g_pybar,
                    int m_size, int n_size ) {

    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n = blockIdx.y;
    Vector<float,R> py, ez;

#pragma unroll
    for (int r=0; r<R; ++r) {
        ez[r] = __ldg((const float *)&g_ezhat[n*(m_size+1)+m][r][tx]);
        py[r] = __ldg((const float *)&g_pybar[n*(m_size+1)+m][r][tx]);
    }

    ez = ez + py * c_HARB_AFP_T;

    g_ezhat[n*(m_size+1)+m].set_col(tx, ez);

}

// THE KERNEL BELOW IS NOT WORKING PROPERLY !!!
template <int R>
__global__ __launch_bounds__(WS*NWARC, NBARC)
void alg6i_step3( Matrix<float,R,WS> *g_ptucheck,
                  Matrix<float,R,WS> *g_etvtilde,
                  const Matrix<float,R,WS> *g_py,
                  Matrix<float,R,WS> *g_ez,
#ifdef GMAT
                  const Matrix<float,R,WS> *g_cmat,
#endif
                  int m_size, int n_size ) {

    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n = blockIdx.y;
    Matrix<float,R,WS> *gptuetv;
    Vector<float,R> py, ez, ptuetv;
    __shared__ Matrix<float,R,WS> spy, sez;
#ifdef GMAT
    Vector<float,R> cmat[3];
#endif

    if (ty == 0) {
#pragma unroll
        for (int r=0; r<R; ++r) {
#ifdef GMAT
#ifdef LDG
            cmat[0][r] = __ldg((const float *)&g_cmat[0][r][tx]);
            cmat[1][r] = __ldg((const float *)&g_cmat[1][r][tx]);
            cmat[2][r] = __ldg((const float *)&g_cmat[2][r][tx]);
#else
            cmat[0][r] = g_cmat[0][r][tx];
            cmat[1][r] = g_cmat[1][r][tx];
            cmat[2][r] = g_cmat[2][r][tx];
#endif
#endif
            ptuetv[r] = g_ptucheck[m*(n_size+1)+n+1][r][tx];
        }
        gptuetv = (Matrix<float,R,WS> *)&g_ptucheck[m*(n_size+1)+n+1][0][tx];
    } else if (ty == 1) {
#pragma unroll
        for (int r=0; r<R; ++r) {
#ifdef GMAT
#ifdef LDG
            cmat[0][r] = __ldg((const float *)&g_cmat[0][r][tx]);
            cmat[1][r] = __ldg((const float *)&g_cmat[1][r][tx]);
            cmat[2][r] = __ldg((const float *)&g_cmat[3][r][tx]);
#else
            cmat[0][r] = g_cmat[0][r][tx];
            cmat[1][r] = g_cmat[1][r][tx];
            cmat[2][r] = g_cmat[3][r][tx];
#endif
#endif
            ptuetv[r] = g_etvtilde[m*(n_size+1)+n][r][tx];
        }
        gptuetv = (Matrix<float,R,WS> *)&g_etvtilde[m*(n_size+1)+n][0][tx];
    } else if (ty == 2) {
#ifdef LDG
#pragma unroll
        for (int r=0; r<R; ++r)
            spy[r][tx] = __ldg((const float *)&g_py[n*(m_size+1)+m][r][tx]);
#else
        spy.set_col(tx, ((Matrix<float,R,WS>*)&g_py[n*(m_size+1)+m][0][tx])->col(0));
#endif
    } else if (ty == 3) {
#pragma unroll
        for (int r=0; r<R; ++r) {
            ez[r] = __ldg((const float *)&g_ez[n*(m_size+1)+m][r][tx]);
            py[r] = __ldg((const float *)&g_py[n*(m_size+1)+m][r][tx]);
        }

        ez = ez + py * c_HARB_AFP_T;

        __syncwarp();

        g_ez[n*(m_size+1)+m].set_col(tx, ez);

        if (m < m_size-1) {
#pragma unroll
            for (int r=0; r<R; ++r) {
                ez[r] = __ldg((const float *)&g_ez[n*(m_size+1)+m+1][r][tx]);
                py[r] = __ldg((const float *)&g_py[n*(m_size+1)+m+1][r][tx]);
            }

            ez = ez + py * c_HARB_AFP_T;
        } else {
            ez = zeros<float,R>();
        }

        sez.set_col(tx, ez);

    }

    __syncthreads();

    if (ty < 2) {

        py = spy.col(tx);
        ez = sez.col(tx);

#ifdef GMAT
        fixpet(ptuetv, cmat[2], cmat[0], ez);
        fixpet(ptuetv, cmat[2], cmat[1], py);
#else
        if (ty == 0) {
            fixpet(ptuetv, c_TAFB, c_ARE_T, ez);
            fixpet(ptuetv, c_TAFB, c_ARB_AFP_T, py);
        } else { // ty == 1
            fixpet(ptuetv, c_HARB_AFB, c_ARE_T, ez);
            fixpet(ptuetv, c_HARB_AFB, c_ARB_AFP_T, py);
        }
#endif

        gptuetv->set_col(0, ptuetv);

    }

}

template <bool BORDER, int R>
__global__ __launch_bounds__(WS*NWW, NBCW)
void alg6i_step5( float *g_out,
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

        if (n < n_size-1) {
#ifdef LDG
#pragma unroll
            for (int r=0; r<R; ++r)
                e[r] = __ldg((const float *)&g_etv[m*(n_size+1)+n+1][r][tx]);
#else
            e = ((Matrix<float,R,WS>*)&g_etv[m*(n_size+1)+n+1][0][tx])->col(0);
#endif
            e = e + p * c_HARB_AFP_T;
        } else {
            e = zeros<float,R>();
        }

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

//== IMPLEMENTATION ============================================================

template <bool BORDER, int R>
void alg6i_gpu( float *h_img,
                const int& width, const int& height, const int& runtimes,
                const Vector<float, R+1>& w,
                const int& border=0,
                const BorderType& border_type=CLAMP_TO_ZERO ) {

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
    cudaFuncSetCacheConfig(alg6_step3<R>, cudaFuncCachePreferL1);

    if (R == 1)
        cudaFuncSetCacheConfig(alg3v4v5v6_step2v4<R>, cudaFuncCachePreferL1);
    else if (R == 2)
        cudaFuncSetCacheConfig(alg3v4v5v6_step2v4<R>, cudaFuncCachePreferEqual);
    else if (R >= 3)
        cudaFuncSetCacheConfig(alg3v4v5v6_step2v4<R>, cudaFuncCachePreferShared);

    base_timer &timer_total = timers.gpu_add("alg6_gpu", width*height, "iP");

    for(int r = 0; r < runtimes; ++r) {

        cudaBindTextureToArray(t_in, a_in);

        alg5v6_step1<BORDER><<< dim3(m_size, n_size), dim3(WS, NWC) >>>
            ( &d_pybar, &d_ezhat, &d_ptucheck, &d_etvtilde, inv_width, inv_height, m_size, n_size );

#ifdef ALG6I1
        alg6i_step2v4<<< dim3(m_size, n_size), dim3(WS, 1) >>>
            ( &d_ezhat, &d_pybar, m_size, n_size );

        alg6_step3<<< dim3(m_size, n_size), dim3(WS, NWARC) >>>
            ( &d_ptucheck, &d_etvtilde, &d_pybar, &d_ezhat,
#ifdef GMAT
              &d_cmat,
#endif
              m_size, n_size );

        alg6i_step2v4<<< dim3(n_size, m_size), dim3(WS, 1) >>>
            ( &d_etvtilde, &d_ptucheck, n_size, m_size );

        alg5v6_step4v5<BORDER><<< dim3(m_size, n_size), dim3(WS, NWW) >>>
            ( d_img, &d_pybar, &d_ezhat, &d_ptucheck, &d_etvtilde, inv_width, inv_height, m_size, n_size, stride_img );
#endif
#ifdef ALG6I2

        alg6i_step2v4<<< dim3(m_size, n_size), dim3(WS, 1) >>>
            ( &d_ezhat, &d_pybar, m_size, n_size );

        alg6_step3<<< dim3(m_size, n_size), dim3(WS, NWARC) >>>
            ( &d_ptucheck, &d_etvtilde, &d_pybar, &d_ezhat,
#ifdef GMAT
              &d_cmat,
#endif
              m_size, n_size );

/*
        alg6i_step3<<< dim3(m_size, n_size), dim3(WS, NWARC) >>>
            ( &d_ptucheck, &d_etvtilde, &d_pybar, &d_ezhat,
#ifdef GMAT
              &d_cmat,
#endif
              m_size, n_size );
*/
        alg6i_step5<BORDER><<< dim3(m_size, n_size), dim3(WS, NWW) >>>
            ( d_img, &d_pybar, &d_ezhat, &d_ptucheck, &d_etvtilde, inv_width, inv_height, m_size, n_size, stride_img );
#endif
        
        cudaUnbindTexture(t_in);

    }

    timer_total.stop();

    if (runtimes > 1) {

        std::cout << std::fixed << (timer_total.data_size()*runtimes)/(double)(timer_total.elapsed()*1024*1024) << std::flush;

    } else {

        timers.flush();

    }

    cudaMemcpy2D(h_img, width*sizeof(float), d_img, stride_img*sizeof(float), width*sizeof(float), height, cudaMemcpyDeviceToHost);

}

//==============================================================================
} // namespace gpufilter
//==============================================================================
#endif // ALG6I_GPU_CUH
//==============================================================================
