/**
 *  @file alg_rd.cuh
 *  @brief Header of algorithm recursive doubling
 *  @author Andre Maximo
 *  @date Aug, 2019
 *  @copyright The MIT License
 */

#ifndef ORDER
#define ORDER 1 // default filter order r=1
#endif
#define APPNAME "[alg3_1d]"

#define USE_REGS
#define USE_LDG

#define SAMPLETYPE float
#define WARPSIZE 32
#define NUMWARPS 8
#define NUMBLOCKS 8
#define FULL_MASK 0xffffffff

namespace gpufilter {

__constant__ Vector<SAMPLETYPE,ORDER+1> c_weights;

__constant__ Vector<SAMPLETYPE,ORDER> c_w2, c_w4, c_w8, c_w16;


template <typename T>
__device__
void compute_block_rd_shuffle( T& x,
                               const int& tx ) {
    T xprev;

    x = x * c_weights[0];

    xprev = __shfl_up_sync(FULL_MASK, x, 1);
    if (tx >= 1)
        x = x - xprev * c_weights[1];
        
    xprev = __shfl_up_sync(FULL_MASK, x, 2);
    if (tx >= 2)
        x = x - xprev * c_w2[0];

    xprev = __shfl_up_sync(FULL_MASK, x, 4);
    if (tx >= 4)
        x = x - xprev * c_w4[0];

    xprev = __shfl_up_sync(FULL_MASK, x, 8);
    if (tx >= 8)
        x = x - xprev * c_w8[0];

    xprev = __shfl_up_sync(FULL_MASK, x, 16);
    if (tx >= 16)
        x = x - xprev * c_w16[0];

}

template <typename T>
__global__ __launch_bounds__(WARPSIZE*NUMWARPS, NUMBLOCKS)
void alg_rd( T *g_out,
             const T *g_in ) {

    const int tx = threadIdx.x, ty = threadIdx.y, bi = blockIdx.x;

#ifdef USE_SMEM
    __shared__ Matrix<T,NUMWARPS,WARPSIZE+1> s_block;
    s_block[ty][tx] = g_in[bi*WARPSIZE*NUMWARPS+ty*WARPSIZE+tx];
#else
    T x = g_in[bi*WARPSIZE*NUMWARPS+ty*WARPSIZE+tx];
#endif

    __syncwarp();

#ifdef USE_SMEM
    s_block[ty][tx] = s_block[ty][tx] * c_weights[0];

    __syncwarp();

    T w = c_weights[1];
    
    for (int i=1; i<WARPSIZE; i*=2) {
        if (tx >= i) {
            s_block[ty][tx] = s_block[ty][tx] - s_block[ty][tx-i] * w;
            w *= -w;
            __syncwarp();
        }
    }
#else
    compute_block_rd_shuffle(x, tx);
#endif
    
    __syncwarp();

#ifdef USE_SMEM
    g_out[bi*WARPSIZE*NUMWARPS+ty*WARPSIZE+tx] = s_block[ty][tx];
#else
    g_out[bi*WARPSIZE*NUMWARPS+ty*WARPSIZE+tx] = x;
#endif

}


template <typename T, int R>
__host__
void alg_rd_gpu( T *h_in,
                 const long int& num_samples,
                 const long int& num_repeats,
                 const Vector<T, R+1> &w ) {

    // upload to the GPU
    copy_to_symbol(c_weights, w);

    Vector<T, R> wgt;

    for (int r=0; r<R; ++r) {
        wgt[r] = w[1+r];
        wgt[r] *= -wgt[r];
    }

    copy_to_symbol(c_w2, wgt);

    for (int r=0; r<R; ++r)
        wgt[r] *= -wgt[r];

    copy_to_symbol(c_w4, wgt);

    for (int r=0; r<R; ++r)
        wgt[r] *= -wgt[r];

    copy_to_symbol(c_w8, wgt);

    for (int r=0; r<R; ++r)
        wgt[r] *= -wgt[r];

    copy_to_symbol(c_w16, wgt);

    dvector<T> d_in(h_in, num_samples), d_out(num_samples);

    long int num_blocks = num_samples/(WARPSIZE*NUMWARPS);
    
    dim3 grid(num_blocks);
    
    dim3 block(WARPSIZE, NUMWARPS);

    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    // first run to warm the GPU up

    alg_rd<<< grid, block >>>( &d_out, &d_in );

    base_timer &timer_total = timers.gpu_add("alg_rd_gpu", num_samples, "is");

    for (int r = 0; r < num_repeats; ++r) {

        alg_rd<<< grid, block >>>( &d_out, &d_in );

    }

    timer_total.stop();

    if (num_repeats > 1) {

        std::size_t proc_samples = timer_total.data_size()*num_repeats;
        double time_sec_inv_mebi = timer_total.elapsed()*1024*1024;
        std::cout << std::fixed << proc_samples/time_sec_inv_mebi << std::flush;

    } else { // running for debugging

        timers.flush();

    }

    d_out.copy_to(h_in, num_samples);

}

} // gpufilter namespace
