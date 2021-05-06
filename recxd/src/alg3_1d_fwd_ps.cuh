/**
 *  @file alg3_1d_fwd.cuh
 *  @brief Header of Prefix Sum algorithm 3 for 1D (forward only)
 *  @note DO NOT USE !!! THIS IS NOT BETTER THAN WITHOUT _PS
 *  @Author Andre Maximo
 *  @date Jan, 2018
 *  @copyright The MIT License
 */

#ifndef ORDER
#define ORDER 1 // prefix sum order should be 1
#endif
#define APPNAME "[alg3_1d_fwd_ps]"

#define USE_REGS

#define SAMPLETYPE float
#define WARPSIZE 32
#define NUMWARPS 3
#define NUMBLOCKS 21
#define FULL_MASK 0xffffffff


namespace gpufilter {

template<typename T>
__device__
void read_block( Matrix<T,WARPSIZE,WARPSIZE+1>& block,
                 const T *input,
                 const int& tx, const int& ty, const int& bi ) {
    // read block from global memory at input pointer to shared memory
    // set block (b) and input (i) current pointers (cp)
    T *bcp = &block[ty][tx];
    const T *icp = &input[bi*WARPSIZE*WARPSIZE+ty*WARPSIZE+tx];
#pragma unroll
    for (int i=0; i<WARPSIZE-(WARPSIZE%NUMWARPS); i+=NUMWARPS) {
        *bcp = *icp;
        bcp += NUMWARPS*(WARPSIZE+1);
        icp += NUMWARPS*WARPSIZE;
    }
    if (ty < WARPSIZE%NUMWARPS) {
        *bcp = *icp;
    }
}

template<typename T>
__device__
void write_block( T *output,
                  Matrix<T,WARPSIZE,WARPSIZE+1>& block,
                  const int& tx, const int& ty, const int& bi ) {
    // write block from shared to global memory at output pointer
    // set block (b) and output (o) current pointers (cp)
    T *bcp = &block[ty][tx];
    T *ocp = &output[bi*WARPSIZE*WARPSIZE+ty*WARPSIZE+tx];
#pragma unroll
    for (int i=0; i<WARPSIZE-(WARPSIZE%NUMWARPS); i+=NUMWARPS) {
        *ocp = *bcp;
        bcp += NUMWARPS*(WARPSIZE+1);
        ocp += NUMWARPS*WARPSIZE;
    }
    if (ty < WARPSIZE%NUMWARPS) {
        *ocp = *bcp;
    }
}

template<typename T>
__device__
void write_read_block( T *output,
                       const T *input,
                       Matrix<T,WARPSIZE,WARPSIZE+1>& block,
                       const int& tx, const int& ty,
                       const int& biw, const int& bir ) {
    // write block from shared to global memory at output pointer
    // read block from global memory at input pointer to shared memory
    // set block (b), output (o) and input (i) current pointers (cp)
    T *bcp = &block[ty][tx];
    T *ocp = &output[biw*WARPSIZE*WARPSIZE+ty*WARPSIZE+tx];
    const T *icp = &input[bir*WARPSIZE*WARPSIZE+ty*WARPSIZE+tx];
    for (int i=0; i<WARPSIZE-(WARPSIZE%NUMWARPS); i+=NUMWARPS) {
        *ocp = *bcp;
        *bcp = *icp;
        bcp += NUMWARPS*(WARPSIZE+1);
        ocp += NUMWARPS*WARPSIZE;
        icp += NUMWARPS*WARPSIZE;
    }
    if (ty < WARPSIZE%NUMWARPS) {
        *ocp = *bcp;
        *bcp = *icp;
    }
}

template <typename T>
__device__
void compute_block( T& py,
                    Matrix<T,WARPSIZE,WARPSIZE+1>& block,
                    const int& tx ) {

#ifdef USE_REGS
    T x[WARPSIZE];

#pragma unroll
    for (int i=0; i<WARPSIZE; ++i)
        x[i] = block[tx][i];
#endif

#pragma unroll
    for (int i=0; i<WARPSIZE; ++i) {
#ifdef USE_REGS
        py = py + x[i];
        block[tx][i] = py;
#else
        py = py + block[tx][i];
        block[tx][i] = py;
#endif
    }

}

template <typename T>
__device__
void compute_py( T& py,
                 Matrix<T,WARPSIZE,WARPSIZE+1>& block,
                 const int& tx ) {

#ifdef USE_REGS
    T x[WARPSIZE];
    
#pragma unroll
    for (int i=0; i<WARPSIZE; ++i)
        x[i] = block[tx][i];
#endif
    
#pragma unroll
    for (int i=0; i<WARPSIZE; ++i) {
#ifdef USE_REGS
        py = py + x[i];
#else
        py = py + block[tx][i];
#endif
    }

}

template <typename T>
__device__
void fix_py( T& py,
             const int& tx ) {
    // fix py on each thread register of the same warp
    T pyprev;

    pyprev = __shfl_up_sync(FULL_MASK, py, 1);
    if (tx >= 1)
        py = py + pyprev;
        
    pyprev = __shfl_up_sync(FULL_MASK, py, 2);
    if (tx >= 2)
        py = py + pyprev;

    pyprev = __shfl_up_sync(FULL_MASK, py, 4);
    if (tx >= 4)
        py = py + pyprev;

    pyprev = __shfl_up_sync(FULL_MASK, py, 8);
    if (tx >= 8)
        py = py + pyprev;

    pyprev = __shfl_up_sync(FULL_MASK, py, 16);
    if (tx >= 16)
        py = py + pyprev;

}

//
// step 1
//

template <typename T>
__global__ __launch_bounds__(WARPSIZE*NUMWARPS, NUMBLOCKS)
void alg3_step1( T *g_pybar, 
                 const T *g_in ) {

    const int tx = threadIdx.x, ty = threadIdx.y, bi = blockIdx.x;

    __shared__ Matrix<T,WARPSIZE,WARPSIZE+1> s_block;
    read_block(s_block, g_in, tx, ty, bi);

    __syncthreads();
    
    if (ty == 0) {

        T py = 0;

        compute_py(py, s_block, tx); // step 1

        fix_py(py, tx); // inner step 2

        if (tx == WARPSIZE-1)
            g_pybar[bi+1] = py;
        
    }

}

//
// step 2
//

template <typename T>
__global__
void alg3_step2( T *g_py,
                 const T* g_pybar,
                 long int num_blocks ) {

    const int tx = threadIdx.x, ty = threadIdx.y, nw = blockDim.y;
    const int pyi = ty*WARPSIZE + tx + 1;

    T py = 0;

    if (pyi < num_blocks+1)
        py = g_pybar[pyi];
   
    __syncthreads();

    fix_py(py, tx);

    __syncthreads();

    if (nw > 1) {
        
        extern __shared__ T s_py[];

        if ((tx == WARPSIZE-1) and (ty < nw-1))
            s_py[ty] = py;

        if (ty > 0) {
            for (int ti = 0; ti < ty; ++ti) {
                py += s_py[ti];
            }
        }
        
    }

    __syncthreads();

    if (pyi < num_blocks+1)
        g_py[pyi] = py;

}

//
// step 3
//

template <typename T>
__global__ __launch_bounds__(WARPSIZE*NUMWARPS, NUMBLOCKS)
void alg3_step3( T *g_out,
                 const T *g_py,
                 const T *g_in ) {

    const int tx = threadIdx.x, ty = threadIdx.y, bi = blockIdx.x;

    __shared__ Matrix<T,WARPSIZE,WARPSIZE+1> s_block;
    read_block(s_block, g_in, tx, ty, bi);

    __syncthreads();

    if (ty == 0) {

        T py = 0;

        if (tx == 0) {
            
            py = g_py[bi]; // first py is from previous block
            
        } else { // Shift run ignoring last row
            
            compute_py(py, s_block, tx-1); // step 1
            
        }

        fix_py(py, tx); // inner step 2

        compute_block(py, s_block, tx); // step 3

    }

    __syncthreads();

    write_block(g_out, s_block, tx, ty, bi);

}


template <typename T, int R>
__host__
void alg3_1d_gpu( T *h_in,
                  const long int& num_samples,
                  const long int& num_repeats,
                  const Vector<T, 2> &w ) {

    dvector<T> d_in(h_in, num_samples), d_out(num_samples);

    long int num_blocks = num_samples/(WARPSIZE*WARPSIZE);
    
    dim3 grid(num_blocks);
    
    dim3 block(WARPSIZE, NUMWARPS);

    dvector<T> d_pybar(num_blocks+1);
    d_pybar.fillzero();

    dvector<T> d_py(num_blocks+1);
    d_py.fillzero();

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                  &d_pybar, &d_py, num_blocks);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    long int num_warps2 = (num_blocks+WARPSIZE-1)/WARPSIZE;
    size_t smem_size = (num_warps2-1)*sizeof(T);
    dim3 grid2(1), block2(WARPSIZE, num_warps2);

    base_timer &timer_total = timers.gpu_add("alg1d_gpu", num_samples, "iP");

    for (int r = 0; r < num_repeats; ++r) {

        alg3_step1<<< grid, block >>>( &d_pybar, &d_in );

        if (num_blocks < 1024) {

            alg3_step2<<< grid2, block2, smem_size >>>(
                &d_py, &d_pybar, num_blocks );

        } else {

            cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                          &d_pybar, &d_py, num_blocks);

        }

        alg3_step3<<< grid, block >>>( &d_out, &d_py, &d_in );

    }

    timer_total.stop();

    if (num_repeats > 1) {

        std::size_t proc_samples = timer_total.data_size()*num_repeats;
        double time_sec_inv_mebi = timer_total.elapsed()*1024*1024;
        std::cout << std::fixed << proc_samples/time_sec_inv_mebi << std::flush;

    } else { // running for debugging

        timers.flush();

        T *py = new T[d_py.size()];
        d_py.copy_to(py, d_py.size());

        std::cout << std::fixed << "d_py size: " << d_py.size() << " :: ";

        gpufilter::print_array(py, 32, "d_py [:32]:");
        
        gpufilter::print_array(py+d_py.size()-32, 32, "d_py [-32:]:");

    }

    d_out.copy_to(h_in, num_samples);

}

} // gpufilter namespace

