/**
 *  @file alg3_1d_fwd.cuh
 *  @brief Header of Recursive filtering algorithm 3 for 1D (forward only)
 *  @Author Andre Maximo
 *  @date Jan, 2018
 *  @copyright The MIT License
 */

/**
 * @links
 * + http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
 * + http://www.icl.utk.edu/~mgates3/docs/cuda.html
 *
 * @todo
 * + Bigger blocks (v4) did not help!!
 * + Save each inter-block py in fix_py that is not saved (only border py is saved)
 * + REPLACE rd_py BY A non-recursive-doubling strategy to fix the inter-warp py
 * + load block into regs directly to avoid using shared memory
 *
 * + cannot use texture: 
 * ++ Maximum width for a 1D texture reference bound to linear memory: 2^27
 *
 * + low-level improvements:
 * ++ remove asserts in gpufilter::Vector to improve device usage
 */

// From: https://en.wikipedia.org/wiki/CUDA
// Compute capability 6.1 (Pascal Titan X) has:
// Max. number of resident warps per MP: 64
// Max. number of resident blocks per MP: 32
// Max. number of resident threads per MP: 2Ki
// Max. amount of shared memory per MP: 96KiB
// Max. amount of shared memory per thread block: 48KiB
// Number of warp schedulers: 4
// Strangely, the previous 5 warps and 11 blocks setup
//   is not the most optimized one anymore...
//   better was 3 warps and 21 blocks...

#ifndef ORDER
#define ORDER 1 // default filter order r=1
#endif
#define APPNAME "[alg3_1d_fwd]"

#define USE_REGS
#define USE_LDG

#define SAMPLETYPE long int
#define WARPSIZE 32
#define NUMWARPS 3
#define NUMBLOCKS 21
#define FULL_MASK 0xffffffff

namespace gpufilter {

__constant__ Vector<SAMPLETYPE,ORDER+1> c_weights;

__constant__ Vector<SAMPLETYPE,ORDER> c_w2, c_w4, c_w8, c_w16;

__constant__ Matrix<SAMPLETYPE,ORDER,ORDER> c_AbF_T;

__constant__ Matrix<SAMPLETYPE,ORDER,ORDER> c_Ab2F_T, c_Ab4F_T,
    c_Ab8F_T, c_Ab16F_T, c_Ab32F_T, c_Ab64F_T, c_Ab128F_T,
    c_Ab256F_T, c_Ab512F_T;

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

template <typename T, int R>
__device__
void compute_block( Vector<T,R>& py,
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
        block[tx][i] = fwdI(py, x[i], c_weights);
#else
        block[tx][i] = fwdI(py, block[tx][i], c_weights);
#endif
    }

}

template <typename T, int R>
__device__
void compute_py( Vector<T,R>& py,
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
        fwdI(py, x[i], c_weights);
#else
        fwdI(py, block[tx][i], c_weights);
#endif
    }

}

template <typename T, int R>
__device__
void compute_block_rd( Vector<T,R>& py,
                       T& x,
                       const int& tx ) {
    // compute blcok on each thread registers using recursive doubling
    T xprev;

    if (tx == 0)
        x = fwdI(py, x, c_weights);
    else
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

template <typename T, int R>
__device__
void fix_py( Vector<T,R>& py,
             const int& tx ) {
    // fix py on each thread register of the same warp
    Vector<T,R> pyprev;

#pragma unroll
    for (int r=0; r<R; ++r)
        pyprev[r] = __shfl_up_sync(FULL_MASK, py[r], 1);
    if (tx >= 1)
        py = py + pyprev * c_AbF_T;
        
#pragma unroll
    for (int r=0; r<R; ++r)
        pyprev[r] = __shfl_up_sync(FULL_MASK, py[r], 2);
    if (tx >= 2)
        py = py + pyprev * c_Ab2F_T;

#pragma unroll
    for (int r=0; r<R; ++r)
        pyprev[r] = __shfl_up_sync(FULL_MASK, py[r], 4);
    if (tx >= 4)
        py = py + pyprev * c_Ab4F_T;

#pragma unroll
    for (int r=0; r<R; ++r)
        pyprev[r] = __shfl_up_sync(FULL_MASK, py[r], 8);
    if (tx >= 8)
        py = py + pyprev * c_Ab8F_T;

#pragma unroll
    for (int r=0; r<R; ++r)
        pyprev[r] = __shfl_up_sync(FULL_MASK, py[r], 16);
    if (tx >= 16)
        py = py + pyprev * c_Ab16F_T;
}

template <typename T, int R>
__device__
void fix_py2( Vector<T,R>& py,
              const int& tx ) {
    // fix for inter-block py i.e. across blocks
    // fix py on each thread register of the same warp
    Vector<T,R> pyprev;

#pragma unroll
    for (int r=0; r<R; ++r)
        pyprev[r] = __shfl_up_sync(FULL_MASK, py[r], 1);
    if (tx >= 1)
        py = py + pyprev * c_Ab32F_T;
        
#pragma unroll
    for (int r=0; r<R; ++r)
        pyprev[r] = __shfl_up_sync(FULL_MASK, py[r], 2);
    if (tx >= 2)
        py = py + pyprev * c_Ab64F_T;

#pragma unroll
    for (int r=0; r<R; ++r)
        pyprev[r] = __shfl_up_sync(FULL_MASK, py[r], 4);
    if (tx >= 4)
        py = py + pyprev * c_Ab128F_T;

#pragma unroll
    for (int r=0; r<R; ++r)
        pyprev[r] = __shfl_up_sync(FULL_MASK, py[r], 8);
    if (tx >= 8)
        py = py + pyprev * c_Ab256F_T;

#pragma unroll
    for (int r=0; r<R; ++r)
        pyprev[r] = __shfl_up_sync(FULL_MASK, py[r], 16);
    if (tx >= 16)
        py = py + pyprev * c_Ab512F_T;
}

//
// step 1
//

template <typename T, int R>
__global__ __launch_bounds__(WARPSIZE*2, 32)
void alg3_step1v5( Vector<T,R> *g_pybar, 
                   const T *g_in ) {

    const int tx = threadIdx.x, ty = threadIdx.y, bi = blockIdx.x;

    T x;
    Vector<T,R> py = zeros<T,R>();

#pragma unroll
    for (int i=0; i<WARPSIZE; ++i) {
        //x[i] = g_in[bi*WARPSIZE*WARPSIZE+tx*WARPSIZE+i]; // do not coalesce
        x = g_in[(bi*2+ty)*WARPSIZE*WARPSIZE+i*WARPSIZE+tx];

        compute_block_rd(py, x, tx);
            
#pragma unroll
        for (int r=0; r<R; ++r)
            py[r] = __shfl_sync(FULL_MASK, x, WARPSIZE-1-r);

    }
    
    if (tx == 0)
        g_pybar[bi*2+ty+1] = py;
    
}

template <typename T, int R>
__global__ __launch_bounds__(WARPSIZE*NUMWARPS, NUMBLOCKS)
void alg3_step1v2v3( Vector<T,R> *g_pybar, 
                     const T *g_in ) {

    const int tx = threadIdx.x, ty = threadIdx.y, bi = blockIdx.x;

    __shared__ Matrix<T,WARPSIZE,WARPSIZE+1> s_block;
    read_block(s_block, g_in, tx, ty, bi);

    __syncthreads();
    
    if (ty == 0) {

        Vector<T,R> py = zeros<T,R>();

        compute_py(py, s_block, tx); // step 1

        fix_py(py, tx); // inner step 2

        if (tx == WARPSIZE-1)
            g_pybar[bi+1] = py;
        
    }

}

template <typename T, int R>
__global__ __launch_bounds__(WARPSIZE*NUMWARPS, NUMBLOCKS)
void alg3_step1v4( Vector<T,R> *g_pybar, 
                   const T *g_in ) {

    const int tx = threadIdx.x, ty = threadIdx.y, bi = blockIdx.x;

    __shared__ Matrix<T,WARPSIZE,WARPSIZE+1> s_block;
    read_block(s_block, g_in, tx, ty, bi*2+0);

    __syncthreads();

    Vector<T,R> py;

    if (ty == 0) {

        py = zeros<T,R>();

        compute_py(py, s_block, tx); // step 1 for 1st block

        fix_py(py, tx); // inner step 2 for 1st block

    }

    __syncthreads();

    read_block(s_block, g_in, tx, ty, bi*2+1);

    __syncthreads();

    if (ty == 0) {

#pragma unroll
        for (int r=0; r<R; ++r)
            py[r] = __shfl_sync(FULL_MASK, py[r], WARPSIZE-1);

        if (tx > 0) // only first thread will hold correct py
            py = zeros<T,R>();

        compute_py(py, s_block, tx);

        fix_py(py, tx); // inner step 2 for 2nd block
        
        if (tx == WARPSIZE-1)
            g_pybar[bi+1] = py;
        
    }

}

template <typename T, int R>
__global__ __launch_bounds__(WARPSIZE*NUMWARPS, NUMBLOCKS)
void alg3_step3v5( T *g_out,
                   const Vector<T,R> *g_py,
                   const T *g_in ) {

    const int tx = threadIdx.x, ty = threadIdx.y, bi = blockIdx.x;

    T x;
    Vector<T,R> py;

#ifdef USE_LDG
#pragma unroll
    for (int r=0; r<R; ++r)
        py[r] = __ldg((const T*)&g_py[bi][r]);
#else
    py = g_py[bi];
#endif
    
#pragma unroll
    for (int i=0; i<WARPSIZE; ++i) {
        x = g_in[(bi*2+ty)*WARPSIZE*WARPSIZE+i*WARPSIZE+tx];

        // NEED TO DO RD HERE

        x += py[0];

        g_out[(bi*2+ty)*WARPSIZE*WARPSIZE+i*WARPSIZE+tx] = x;
    }

}


// outer step 2 may not be done in versions bellow for stable recursive filters
// it will incurs in neglible errors;
// however for prefix sums, outer step 2 must be done

//
// step 2
//

template <typename T, int R>
__global__
void alg3_step2v1( Vector<T,R> *g_py,
                   int num_blocks ) {

    const int tx = threadIdx.x, nx = blockDim.x;

    extern __shared__ Vector<T, R> s_pybar[];

    for (int i = 0; i < num_blocks; i += nx) {

        __syncthreads();

        if (i+tx+1 < num_blocks+1)
            s_pybar[tx] = g_py[i+tx+1];

        __syncthreads();
               
        if (tx == 0) {

            Vector<T, R> py = zeros<T, R>();

            for (int j = 0; j < nx; ++j) {

                py = s_pybar[j] + py * c_Ab32F_T;
                s_pybar[j] = py;

            }

        }

        __syncthreads();

        if (i+tx+1 < num_blocks+1)
            g_py[i+tx+1] = s_pybar[tx];

    }
    
}

template <typename T, int R>
__global__ __launch_bounds__(WARPSIZE, 1)
void alg3_step2v2( Vector<T,R> *g_py,
                   int num_blocks ) {

    const int tx = threadIdx.x;

    Vector<T, R> py = zeros<T, R>();

    for (int i = 0; i < num_blocks+1; i += WARPSIZE) {
        
        if (i+tx < num_blocks+1)
            py = g_py[i+tx+1] + py * c_Ab32F_T;

        fix_py2(py, tx);

        if (i+tx < num_blocks+1)
            g_py[i+tx+1] = py;

#pragma unroll
        for (int r=0; r<R; ++r)
            py[r] = __shfl_down_sync(FULL_MASK, py[r], WARPSIZE-1);
        if (tx > 0)
            py = zeros<T, R>();

    }
    
}

//
// step 3
//

template <typename T, int R>
__global__ __launch_bounds__(WARPSIZE*NUMWARPS, NUMBLOCKS)
void alg3_step3v3( T *g_out,
                   const Vector<T,R> *g_py,
                   const T *g_in ) {

    const int tx = threadIdx.x, ty = threadIdx.y, bi = blockIdx.x;

    __shared__ Matrix<T,WARPSIZE,WARPSIZE+1> s_block;
    read_block(s_block, g_in, tx, ty, bi);

    __syncthreads();

    if (ty == 0) {

        Vector<T,R> py = zeros<T,R>();

        if (tx == 0) { // first py is from previous block
#ifdef USE_LDG
#pragma unroll
            for (int r=0; r<R; ++r)
                py[r] = __ldg((const T*)&g_py[bi][r]);
#else
            py = g_py[bi];
#endif
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
__global__ __launch_bounds__(WARPSIZE*NUMWARPS, NUMBLOCKS)
void alg3_step3v4( T *g_out,
                   const Vector<T,R> *g_py,
                   const T *g_in ) {

    const int tx = threadIdx.x, ty = threadIdx.y, bi = blockIdx.x;

    __shared__ Matrix<T,WARPSIZE,WARPSIZE+1> s_block;
    read_block(s_block, g_in, tx, ty, bi*2+0);

    __syncthreads();

    Vector<T,R> py;

    if (ty == 0) {

        if (tx == 0) { // first py is from previous block
#ifdef USE_LDG
#pragma unroll
            for (int r=0; r<R; ++r)
                py[r] = __ldg((const T*)&g_py[bi][r]);
#else
            py = g_py[bi];
#endif
        } else { // Shift run ignoring last row
            py = zeros<T,R>();
            compute_py(py, s_block, tx-1); // step 1
        }

        fix_py(py, tx); // inner step 2 for 1st block

        compute_block(py, s_block, tx); // step 3 for 1st block

    }

    __syncthreads();

    write_read_block(g_out, g_in, s_block, tx, ty, bi*2+0, bi*2+1);

    __syncthreads();

    if (ty == 0) {

#pragma unroll
        for (int r=0; r<R; ++r)
            py[r] = __shfl_sync(FULL_MASK, py[r], WARPSIZE-1);

        if (tx > 0) { // Shift run ignoring last row
            py = zeros<T,R>();
            compute_py(py, s_block, tx-1);
        }

        fix_py(py, tx); // inner step 2 for 2nd block

        compute_block(py, s_block, tx); // step 3 for 2nd block
        
    }
    
    __syncthreads();

    write_block(g_out, s_block, tx, ty, bi*2+1);

}


template <typename T, int R>
__host__
void alg3_1d_gpu( T *h_in,
                  const long int& num_samples,
                  const long int& num_repeats,
                  const Vector<T, R+1> &w ) {

    const int B = WARPSIZE;

    // pre-compute basic alg1d matrices
    Matrix<T,R,R> Ir = identity<T,R,R>();
    Matrix<T,R,B> Zrb = zeros<T,R,B>();
    Matrix<T,B,R> Zbr = zeros<T,B,R>();
    Matrix<T,B,B> Ib = identity<T,B,B>();

    Matrix<T,R,B> AFP_T = fwd(Ir, Zrb, w);
    
    Matrix<T,R,R> AbF_T = tail<R>(AFP_T);

    Matrix<T,R,R> Ab2F_T = AbF_T * AbF_T, Ab4F_T = Ab2F_T * Ab2F_T;
    Matrix<T,R,R> Ab8F_T = Ab4F_T * Ab4F_T, Ab16F_T = Ab8F_T * Ab8F_T;
    Matrix<T,R,R> Ab32F_T = Ab16F_T * Ab16F_T, Ab64F_T = Ab32F_T * Ab32F_T;
    Matrix<T,R,R> Ab128F_T = Ab64F_T * Ab64F_T, Ab256F_T = Ab128F_T * Ab128F_T;
    Matrix<T,R,R> Ab512F_T = Ab256F_T * Ab256F_T;

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

    copy_to_symbol(c_AbF_T, AbF_T);

    copy_to_symbol(c_Ab2F_T, Ab2F_T); copy_to_symbol(c_Ab4F_T, Ab4F_T);
    copy_to_symbol(c_Ab8F_T, Ab8F_T); copy_to_symbol(c_Ab16F_T, Ab16F_T);
    copy_to_symbol(c_Ab32F_T, Ab32F_T); copy_to_symbol(c_Ab64F_T, Ab64F_T);
    copy_to_symbol(c_Ab128F_T, Ab128F_T); copy_to_symbol(c_Ab256F_T, Ab256F_T);
    copy_to_symbol(c_Ab512F_T, Ab512F_T);

    dvector<T> d_in(h_in, num_samples), d_out(num_samples);

    long int num_blocks = num_samples/(WARPSIZE*WARPSIZE);
    
    //long int num_blocks = num_samples/(WARPSIZE*WARPSIZE*2); // V4

    dim3 grid(num_blocks);
    
    //dim3 block(WARPSIZE*WARPSIZE); // V1
    
    dim3 block(WARPSIZE, NUMWARPS);

    dvector< Vector<T,R> > d_pybar(num_blocks+1);
    d_pybar.fillzero();
   
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    //long int block2_width = std::min((long int)1024, num_blocks);
    //dim3 block2(block2_width); // for step 2 v1
    //size_t smem_size = block2_width*sizeof(Vector<T, R>); // set shared memory size
    dim3 grid2(1); // for step 2
    dim3 block2(WARPSIZE); // for step 2 v2

    // first run to warm the GPU up

    alg3_step1v2v3<<< grid, block >>>( &d_pybar, &d_in );

    alg3_step2v2<<< grid2, block2 >>>( &d_pybar, num_blocks );

    alg3_step3v3<<< grid, block >>>( &d_out, &d_pybar, &d_in );

    base_timer &timer_total = timers.gpu_add("alg1d_gpu", num_samples, "iP");

    for (int r = 0; r < num_repeats; ++r) {

        //alg3_step1v1<<< grid, block >>>( &d_pybar, &d_in );
        
        alg3_step1v2v3<<< grid, block >>>( &d_pybar, &d_in );

        //alg3_step1v4<<< grid, block >>>( &d_pybar, &d_in );

        //alg3_step1v5<<< dim3(num_blocks/2), dim3(WARPSIZE, 2) >>>( &d_pybar, &d_in );

        //alg3_step2v1<<< grid2, block2, smem_size >>>( &d_pybar, num_blocks );
        
        alg3_step2v2<<< grid2, block2 >>>( &d_pybar, num_blocks );

        //alg3_step3v1<<< grid, block >>>( &d_out, &d_pybar, &d_in );

        //alg3_step3v2<<< grid, block >>>( &d_out, &d_pybar, &d_in );

        alg3_step3v3<<< grid, block >>>( &d_out, &d_pybar, &d_in );

        //alg3_step3v4<<< grid, block >>>( &d_out, &d_pybar, &d_in );

        //alg3_step3v5<<< grid, block >>>( &d_out, &d_pybar, &d_in );

    }

    timer_total.stop();

    if (num_repeats > 1) {

        std::size_t proc_samples = timer_total.data_size()*num_repeats;
        double time_sec_inv_mebi = timer_total.elapsed()*1024*1024;
        std::cout << std::fixed << proc_samples/time_sec_inv_mebi << std::flush;

    } else { // running for debugging

        timers.flush();

        Vector<T, R> *pybar = new Vector<T, R>[d_pybar.size()];
        d_pybar.copy_to(pybar, d_pybar.size());
        
        std::cout << std::fixed << std::flush;
        gpufilter::print_array(pybar, 32, "d_pybar [:32]:");

    }

    d_out.copy_to(h_in, num_samples);

}

} // gpufilter namespace

/**
 *  grave:

//
// ----
//

template <typename T, int R>
__global__
void alg3_step1v1( Vector<T,R> *g_pybar,
                   const T *g_in ) {
    
    const int tx = threadIdx.x, bi = blockIdx.x;

    __shared__ Vector<T,R> s_pybar[WARPSIZE+1];

    T x = g_in[bi*WARPSIZE*WARPSIZE+tx];

    x = c_weights[0] * x;
    
    Vector<T,R> py = zeros<T,R>();

    Vector<T,R+1> w = c_weights;
    w[0] = (T)1;
    
    for (int k = 1; k < WARPSIZE; k *= 2) {
        // BIG Recursive Doubling

        for (int r=0; r<R; ++r)
            py[r] = __shfl_up_sync(0xffffffff, x, k+r);

        if (tx%WARPSIZE >= k)
            x = fwdI(py, x, w);

        for (int r=0; r<R; ++r)
            w[1+r] *= -w[1+r];

    }
    
    // inner step 2 fix py inside step 1, going from reg
    // to smem to reg back for another RD
    if (tx%WARPSIZE == WARPSIZE-1)
        s_pybar[tx/WARPSIZE] = py;

    __syncthreads();
        
    if (tx < WARPSIZE) {
            
        py = s_pybar[tx];
            
        fix_py(py, tx);
            
        if (tx == WARPSIZE-1)
            g_pybar[bi+1] = py;
    }
        
}

template <typename T, int R>
__global__
void alg3_step3v1( T *g_out,
                   const Vector<T,R> *g_py,
                   const T *g_in ) {
    
    const int tx = threadIdx.x, bi = blockIdx.x;

    __shared__ Vector<T,WARPSIZE*WARPSIZE+1> x;
    
    x[tx] = g_in[bi*WARPSIZE*WARPSIZE+tx];

    x[tx] *= c_weights[0];
    
    Vector<T,R> py = zeros<T,R>();

    Vector<T,R+1> w = c_weights;
    w[0] = (T)1;
    
    for (int k = 1; k < WARPSIZE*WARPSIZE; k *= 2) {
        // BIG Recursive Doubling

        if (tx >= k-1) {

            if (tx-k == -1)
                py = g_py[bi];
            else
                for (int r=0; r<R; ++r)
                    py[r] = x[tx-k-r];

            x[tx] = fwdI(py, x[tx], w);

            for (int r=0; r<R; ++r)
                w[1+r] *= -w[1+r];
        }

        __syncthreads();

    }

    g_out[bi*WARPSIZE*WARPSIZE+tx] = x[tx];
        
}

template <typename T, int R>
__global__ __launch_bounds__(WARPSIZE*NUMWARPS, NUMBLOCKS)
void alg3_step3v2( T *g_out,
                   const Vector<T,R> *g_py,
                   const T *g_in ) {

    const int tx = threadIdx.x, ty = threadIdx.y, bi = blockIdx.x;

    __shared__ Matrix<T,WARPSIZE,WARPSIZE+1> s_block;
    read_block(s_block, g_in);

    __syncthreads();

    if (ty == 0) {

        Vector<T,R> py = zeros<T,R>();

        for (int i=0; i<WARPSIZE; ++i)
            fwdI(py, s_block[tx][i], c_weights);

        for (int r=0; r<R; ++r)
            s_block[tx][WARPSIZE-1-r] = py[R-1-r];

        if (tx == 0) { // sequential inner py fixing

            Vector<T,R> gpy = g_py[bi];
            py = gpy;
            
            for (int i=0; i<WARPSIZE; ++i) {

                Vector<T,R> pybar;

                for (int r=0; r<R; ++r)
                    pybar[R-1-r] = s_block[i][WARPSIZE-1-r];

                // each inner py was not saved, only the py
                // at the border of each BIG block
                py = pybar + py * c_AbF_T;

                for (int r=0; r<R; ++r)
                    s_block[i][WARPSIZE-1-r] = py[R-1-r];
            }
            
            py = gpy;

        }

        __syncwarp();

        if (tx > 0) {
            for (int r=0; r<R; ++r)
                py[R-1-r] = s_block[tx-1][WARPSIZE-1-r];
        }

        for (int i=0; i<WARPSIZE-R; ++i)
            s_block[tx][i] = fwdI(py, s_block[tx][i], c_weights);

    }

    __syncthreads();

    write_block(g_out, s_block);

}


// V4 is not implemented, only described below

// The problem is that up to 256Ki and 512Ki samples the performance
// increases steadily and then in 1024Ki forward it decreases, never
// reaching the performance achieved at 256Ki samples.

// The idea is to have two 32x32 blocks side-by-side, i.e. 32x64,
// continuing to have 1 'vertical' warp computing, where each thread
// computes one row, but computing twice more per row (64).

// A minor problem with this idea is that it will restrict the number
// of data blocks of (4KiB+padding) to an even number resident in a MP,
// impacting the odd-number strategy of having 3 warps (previously 5)
// and 21 data blocks (previously 11) to fit the 64 warps and 96KiB
// (previously 48KiB) of shared memory per multiprocessor (MP).

// A major problem with this idea is how to avoid bank conflicts in
// shared memory with padding, since loading an entire row (64 threads)
// will cause conflicts even with padding at the end of the row.

// A possible solution to the major problem is to load only half of
// each row (32 threads) at a time, applying the padding at the end
// of the row (64+1 columns), finishing the first 4KiB data block
// half of the shared memory and then the second half at right.

// Another solutioin to the major problem is to leverage 64-bits banks
// (or 8-Bytes) instead of 4-Bytes as in:
//     cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);

void alg3_step1( 
                   Vector<float,R> *g_ezhat,

        for (int r=0; r<R; ++r)
            s_block[tx][WARPSIZE-1-r] = py[R-1-r];

        if (tx==0) {

            for (int i=1; i<WARPSIZE; ++i) {
                
                Vector<float,R> pybar;
                
                for (int r=0; r<R; ++r)
                    pybar[R-1-r] = s_block[i][WARPSIZE-1-r];

                py = pybar + py * c_AbF_T;
                
            }

            g_pybar[bi+1] = py;

        }

        Vector<float,R> e = zeros<float,R>();

        for (int i=WARPSIZE-1; i>=0; --i)
            s_block[tx][i] = revI(s_block[tx][i], e, c_weights);

        g_ezhat[bi*WARPSIZE+tx] = p;


void alg3v1_step3( 
                   const Vector<float,R> *g_ez,

        Vector<float,R> py = g_py[bi*WARPSIZE+tx];

        for (int i=0; i<WARPSIZE; ++i)
            s_block[tx][i] = fwdI(py, s_block[tx][i], c_weights);

        for (int i=0; i<WARPSIZE; ++i) {
            if (tx==0)
                s_block[tx][i] = fwdI(py, s_block[tx][i], c_weights);
            else
                fwdI(py, s_block[tx][i], c_weights);
        }

        rd_py(py, tx);

        Vector<float,R> pyprev;
        for (int r=0; r<R; ++r)
            pyprev[r] = __shfl_up_sync(0xffffffff, py[r], 1);
        if (tx >= 1) {
            py = pyprev;
            for (int i=0; i<WARPSIZE; ++i)
                s_block[tx][i] = fwdI(py, s_block[tx][i], c_weights);
        }

        Vector<float,R> e = g_ez[bi*WARPSIZE+tx];

        for (int i=WARPSIZE-1; i>=0; --i)
            s_block[tx][i] = revI(s_block[tx][i], e, c_weights);

    dvector< Vector<float,R> > d_ezhat(num_blocks+1);
    d_ezhat.fillzero();

*/
