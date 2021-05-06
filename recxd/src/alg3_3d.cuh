/**
 *  @file alg3_3d.cuh
 *  @brief Header of Recursive filtering algorithm 3 for 3D
 *  @author Andre Maximo
 *  @date Jun, 2019
 *  @copyright The MIT License
 */

#ifndef ORDER
#define ORDER 1 // default filter order r=1
#endif
#define APPNAME "[alg3_3d]"

#include <util/image.h>
#include <gpudefs.h>
#include <alg6_gpu.cuh>

#define NWD 3
//#define NBD 21

namespace gpufilter {

__constant__ int c_depth;

template <int R>
__global__ //__launch_bounds__(WS*NWD, NBD)
void alg3_3d_inplace( float *g_inout,
                      int width,
                      int height ) {

    int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x, by = blockIdx.y;

    extern __shared__ float s_block[];

    float *sp = &s_block[ty*(WS+1)+tx];
    float *gp = &g_inout[ty*width*height+by*width+bx*WS+tx];

#pragma unroll
    for (int i=0; i<c_depth-(c_depth%NWD); i+=NWD) { // all warps reading
        *sp = *gp;
        sp += NWD*(WS+1);
        gp += NWD*width*height;
    }
    if (ty < c_depth%NWD) {
        *sp = *gp;
    }

    __syncthreads();

    if (ty == 0) { // 1 warp computing

        Vector<float,R> py = zeros<float,R>();

#pragma unroll
        for (int i=0; i<c_depth; ++i)
            s_block[i*(WS+1)+tx] = fwdI(py, s_block[i*(WS+1)+tx], c_weights);

        Vector<float,R> ez = zeros<float,R>();

#pragma unroll
        for (int i=c_depth-1; i>=0; --i)
            s_block[i*(WS+1)+tx] = revI(s_block[i*(WS+1)+tx], ez, c_weights);

    }

    __syncthreads();

    sp = &s_block[ty*(WS+1)+tx];
    gp = &g_inout[ty*width*height+by*width+bx*WS+tx];
#pragma unroll
    for (int i=0; i<c_depth-(c_depth%NWD); i+=NWD) { // all warps writing
        *gp = *sp;
        sp += NWD*(WS+1);
        gp += NWD*width*height;
    }
    if (ty < c_depth%NWD) {
        *gp = *sp;
    }

}

template <int R>
void alg3_3d_gpu( float *h_vol,
                  const int& width, const int& height, const int& depth,
                  const int& runtimes,
                  const Vector<float, R+1>& w ) {

    // The alg3 in 3D is in fact alg6 on each 2D slice image (width x height)
    // and then alg3 in the 3rd remaining dimension

    const bool BORDER = false;
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

    cudaArray *a_in[depth];
    cudaChannelFormatDesc ccd = cudaCreateChannelDesc<float>();
    for (int z = 0; z < depth; ++z) {
        cudaMallocArray(&a_in[z], &ccd, width, height);
        cudaMemcpyToArray(a_in[z], 0, 0, &h_vol[z*width*height],
                          width*height*sizeof(float),
                          cudaMemcpyHostToDevice);
    }

    t_in.normalized = true;
    t_in.filterMode = cudaFilterModePoint;

    dvector<float> d_vol(depth*height*width);

    // +1 padding is important even in zero-border to avoid if's in kernels
    dvector< Matrix<float,R,B> >
        d_pybar((m_size+1)*n_size),
        d_ezhat((m_size+1)*n_size),
        d_ptucheck((n_size+1)*m_size),
        d_etvtilde((n_size+1)*m_size);
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

    // 3d setup
    copy_to_symbol(c_depth, depth); // upload depth value to the GPU
    size_t smem_size = (WS+1)*depth*sizeof(float); // set shared memory size

    // first run to warm the GPU up

    for (int z = 0; z < depth; ++z) {

        cudaBindTextureToArray(t_in, a_in[z]);

        alg5v6_step1<BORDER><<< dim3(m_size, n_size), dim3(WS, NWC) >>>
            ( &d_pybar, &d_ezhat, &d_ptucheck, &d_etvtilde,
              inv_width, inv_height, m_size, n_size );

        alg3v4v5v6_step2v4<<< dim3(1, n_size), dim3(WS, NWA) >>>
            ( &d_pybar, &d_ezhat, m_size );

        alg6_step3<<< dim3(m_size, n_size), dim3(WS, NWARC) >>>
            ( &d_ptucheck, &d_etvtilde, &d_pybar, &d_ezhat,
              &d_cmat, m_size, n_size );

        alg3v4v5v6_step2v4<<< dim3(1, m_size), dim3(WS, NWA) >>>
            ( &d_ptucheck, &d_etvtilde, n_size );

        alg5v6_step4v5<BORDER><<< dim3(m_size, n_size), dim3(WS, NWW) >>>
            ( &d_vol+z*width*height,
              &d_pybar, &d_ezhat, &d_ptucheck, &d_etvtilde,
              inv_width, inv_height, m_size, n_size, width );

        cudaUnbindTexture(t_in);

    }
        
    // after all slice images are done in volume using alg6
    // go thru depth using alg3, i.e. across slice images

    alg3_3d_inplace<R><<< dim3(m_size, height), dim3(WS, NWD), smem_size >>>
        ( &d_vol, width, height );

    base_timer &timer_total = timers.gpu_add("alg6_gpu", width*height, "iP");

    for (int r = 0; r < runtimes; ++r) {

        for (int z = 0; z < depth; ++z) {

            cudaBindTextureToArray(t_in, a_in[z]);

            alg5v6_step1<BORDER><<< dim3(m_size, n_size), dim3(WS, NWC) >>>
                ( &d_pybar, &d_ezhat, &d_ptucheck, &d_etvtilde,
                  inv_width, inv_height, m_size, n_size );

            alg3v4v5v6_step2v4<<< dim3(1, n_size), dim3(WS, NWA) >>>
                ( &d_pybar, &d_ezhat, m_size );

            alg6_step3<<< dim3(m_size, n_size), dim3(WS, NWARC) >>>
                ( &d_ptucheck, &d_etvtilde, &d_pybar, &d_ezhat,
                  &d_cmat, m_size, n_size );

            alg3v4v5v6_step2v4<<< dim3(1, m_size), dim3(WS, NWA) >>>
                ( &d_ptucheck, &d_etvtilde, n_size );

            alg5v6_step4v5<BORDER><<< dim3(m_size, n_size), dim3(WS, NWW) >>>
                ( &d_vol+z*width*height,
                  &d_pybar, &d_ezhat, &d_ptucheck, &d_etvtilde,
                  inv_width, inv_height, m_size, n_size, width );

            cudaUnbindTexture(t_in);

        }
        
        // after all slice images are done in volume using alg6
        // go thru depth using alg3, i.e. across slice images

        alg3_3d_inplace<R><<< dim3(m_size, height), dim3(WS, NWD), smem_size >>>
            ( &d_vol, width, height );

    }

    timer_total.stop();

    if (runtimes > 1) {

        std::cout << std::fixed << (timer_total.data_size()*runtimes)/
            (double)(timer_total.elapsed()*1024*1024) << std::flush;

    } else {

        timers.flush();

    }

    for (int z = 0; z < depth; ++z)
        cudaMemcpy2D(&h_vol[z*width*height], width*sizeof(float),
                     &d_vol+z*width*height, width*sizeof(float),
                     width*sizeof(float), height, cudaMemcpyDeviceToHost);

}

} // gpufilter namespace
