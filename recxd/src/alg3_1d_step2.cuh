/**
 *  @file alg3_1d.cuh
 *  @brief Header of Recursive filtering algorithm 3 for 1D
 *  @author Andre Maximo
 *  @date Jan, 2018
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
#define NUMWARPS 3
#define NUMBLOCKS 21
#define FULL_MASK 0xffffffff

#include <cub/device/device_scan.cuh>


template <typename T>
struct RecOp
{
    const T w0, w1;
    
    RecOp(const T& _w0, const T& _w1) : w0(_w0), w1(_w1) { }
    
    __host__ __device__  __forceinline__
    T operator()(const T &a, const T &x) const {
        return w0 * x - w1 * a;
    }
};


namespace gpufilter {

__constant__ Vector<SAMPLETYPE,ORDER+1> c_weights;

__constant__ Matrix<SAMPLETYPE,ORDER,ORDER> c_AbF_T, c_AbR_T, c_HARB_AFP_T;

__constant__ Matrix<SAMPLETYPE,ORDER,ORDER> c_Ab2F_T, c_Ab4F_T,
    c_Ab8F_T, c_Ab16F_T, c_Ab2R_T, c_Ab4R_T, c_Ab8R_T, c_Ab16R_T;


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

template <typename T, int R>
__device__
void compute_block_py( Vector<T,R>& py,
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
void compute_block_ez( Vector<T,R>& ez,
                       Matrix<T,WARPSIZE,WARPSIZE+1>& block,
                       const int& tx ) {

#ifdef USE_REGS
    T x[WARPSIZE];
    
#pragma unroll
    for (int i=0; i<WARPSIZE; ++i)
        x[i] = block[tx][i];
#endif
    
#pragma unroll
    for (int i=WARPSIZE-1; i>=0; --i) {
#ifdef USE_REGS
        block[tx][i] = revI(x[i], ez, c_weights);
#else
        block[tx][i] = revI(block[tx][i], ez, c_weights);
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
void compute_ez( Vector<T,R>& ez,
                 Matrix<T,WARPSIZE,WARPSIZE+1>& block,
                 const int& tx ) {

#ifdef USE_REGS
    T x[WARPSIZE];
    
#pragma unroll
    for (int i=0; i<WARPSIZE; ++i)
        x[i] = block[tx][i];
#endif
    
#pragma unroll
    for (int i=WARPSIZE-1; i>=0; --i) {
#ifdef USE_REGS
        revI(x[i], ez, c_weights);
#else
        revI(block[tx][i], ez, c_weights);
#endif
    }

}

template <typename T, int R>
__device__
void fix_py( Vector<T,R>& py,
             const int& tx ) {
    // fix py on each thread register of the same warp
    // using rd: recursive block by shuffling
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
void fix_ez( Vector<T,R>& ez,
             const int& tx ) {
    // fix ez on each thread register of the same warp
    // using rd: recursive block by shuffling
    Vector<T,R> eznext;

#pragma unroll
    for (int r=0; r<R; ++r)
        eznext[r] = __shfl_down_sync(FULL_MASK, ez[r], 1);
    if (tx < WARPSIZE-1)
        ez = ez + eznext * c_AbR_T;
        
#pragma unroll
    for (int r=0; r<R; ++r)
        eznext[r] = __shfl_down_sync(FULL_MASK, ez[r], 2);
    if (tx < WARPSIZE-2)
        ez = ez + eznext * c_Ab2R_T;

#pragma unroll
    for (int r=0; r<R; ++r)
        eznext[r] = __shfl_down_sync(FULL_MASK, ez[r], 4);
    if (tx < WARPSIZE-4)
        ez = ez + eznext * c_Ab4R_T;

#pragma unroll
    for (int r=0; r<R; ++r)
        eznext[r] = __shfl_down_sync(FULL_MASK, ez[r], 8);
    if (tx < WARPSIZE-8)
        ez = ez + eznext * c_Ab8R_T;

#pragma unroll
    for (int r=0; r<R; ++r)
        eznext[r] = __shfl_down_sync(FULL_MASK, ez[r], 16);
    if (tx < WARPSIZE-16)
        ez = ez + eznext * c_Ab16R_T;
}

template <typename T, int R>
__global__ __launch_bounds__(WARPSIZE*NUMWARPS, NUMBLOCKS)
void alg3_step1( T *g_pybar,
                 T *g_ezhat,
                 const T *g_in,
                 int num_blocks ) {

    const int tx = threadIdx.x, ty = threadIdx.y, bi = blockIdx.x;

    __shared__ Matrix<T,WARPSIZE,WARPSIZE+1> s_block;
    read_block(s_block, g_in, tx, ty, bi);

    __syncthreads();

    if (ty == 0) {

        Vector<T,R> py = zeros<T,R>();

        compute_py(py, s_block, tx);

        fix_py(py, tx);

        if (tx == WARPSIZE-1)
            g_pybar[bi+1] = py[0];

        // push one thread up the py for them to match
        // direction when computing the entire block
        
#pragma unroll
        for (int r=0; r<R; ++r)
            py[r] = __shfl_up_sync(FULL_MASK, py[r], 1);
        if (tx == 0)
            py = zeros<T,R>();

        __syncwarp();
        
        compute_block_py(py, s_block, tx);

        // with the entire block locally correct,
        // compute ez from it and fix ez using rd

        Vector<T,R> ez = zeros<T,R>();

        compute_ez(ez, s_block, tx);

        fix_ez(ez, tx);

        if (tx == 0)
            g_ezhat[num_blocks-bi] = ez[0];
        
    }

}

// step 2 should fix py and ez across blocks using constant matrices:
// c_AbF_T, c_AbR_T and c_HARB_AFP_T

template <typename T, int R>
__global__ __launch_bounds__(WARPSIZE*NUMWARPS, NUMBLOCKS)
void alg3_step3( T *g_out,
                 const T *g_py,
                 const T *g_ez,
                 const T *g_in,
                 int num_blocks ) {

    const int tx = threadIdx.x, ty = threadIdx.y, bi = blockIdx.x;

    __shared__ Matrix<T,WARPSIZE,WARPSIZE+1> s_block;
    read_block(s_block, g_in, tx, ty, bi);

    __syncthreads();

    if (ty == 0) {

        Vector<T,R> py = zeros<T,R>();

        if (tx == 0) { // first py is from previous block
            py[0] = g_py[bi];
        } else { // Shift run ignoring last row
            compute_py(py, s_block, tx-1); // step 1 fwd
        }

        fix_py(py, tx); // inner step 2 fwd

        compute_block_py(py, s_block, tx); // step 3 fwd

        Vector<T,R> ez = zeros<T,R>();

        if (tx == WARPSIZE-1) { // last ez is from next block
            ez[0] = g_ez[num_blocks-bi-1];
            // fixing next ez by this py as in equation (34)
            if (bi < gridDim.x-1) // next ez in last block is zero
                ez = ez + py * c_HARB_AFP_T;
        } else { // shift run ignoring first row
            compute_ez(ez, s_block, tx+1); // step 1 rev
        }

        __syncwarp();

        fix_ez(ez, tx); // inner step 2 rev

        compute_block_ez(ez, s_block, tx); // step 3 rev

    }

    __syncthreads();

    write_block(g_out, s_block, tx, ty, bi);

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

    Matrix<T,R,B> AFP_T = fwd(Ir, Zrb, w), ARE_T = rev(Zrb, Ir, w);
    Matrix<T,B,B> ARB_T = rev(Ib, Zbr, w);
    
    Matrix<T,R,R> AbF_T = tail<R>(AFP_T), AbR_T = head<R>(ARE_T);
    Matrix<T,R,R> HARB_AFP_T = AFP_T*head<R>(ARB_T);

    Matrix<T,R,R> Ab2F_T = AbF_T * AbF_T, Ab4F_T = Ab2F_T * Ab2F_T;
    Matrix<T,R,R> Ab8F_T = Ab4F_T * Ab4F_T, Ab16F_T = Ab8F_T * Ab8F_T;

    Matrix<T,R,R> Ab2R_T = AbR_T * AbR_T, Ab4R_T = Ab2R_T * Ab2R_T;
    Matrix<T,R,R> Ab8R_T = Ab4R_T * Ab4R_T, Ab16R_T = Ab8R_T * Ab8R_T;

    // ***
    // * how to compute HARB_AFP_T in 2, 4, 8, 16 ?
    // * only necessary if not fixing block by py before computing ez
    // ***

    // upload to the GPU
    copy_to_symbol(c_weights, w);

    copy_to_symbol(c_AbF_T, AbF_T);
    copy_to_symbol(c_AbR_T, AbR_T);
    copy_to_symbol(c_HARB_AFP_T, HARB_AFP_T);

    copy_to_symbol(c_Ab2F_T, Ab2F_T); copy_to_symbol(c_Ab4F_T, Ab4F_T);
    copy_to_symbol(c_Ab8F_T, Ab8F_T); copy_to_symbol(c_Ab16F_T, Ab16F_T);

    copy_to_symbol(c_Ab2R_T, Ab2R_T); copy_to_symbol(c_Ab4R_T, Ab4R_T);
    copy_to_symbol(c_Ab8R_T, Ab8R_T); copy_to_symbol(c_Ab16R_T, Ab16R_T);

    dvector<T> d_in(h_in, num_samples), d_out(num_samples);

    long int num_blocks = num_samples/(WARPSIZE*WARPSIZE);
    
    dim3 grid(num_blocks);
    
    dim3 block(WARPSIZE, NUMWARPS);

    dvector<T> d_pybar(num_blocks+1), d_ezhat(num_blocks+1);
    d_pybar.fillzero();
    d_ezhat.fillzero();

    // Preparing to use CUB in step 2

    dvector<T> d_pybar_out(num_blocks+1), d_ezhat_out(num_blocks+1);
    d_pybar_out.fillzero();
    d_ezhat_out.fillzero();

    Matrix<T,R,R> Ab32F_T = Ab16F_T * Ab16F_T;
    Matrix<T,R,R> Ab32R_T = Ab16R_T * Ab16R_T;

    RecOp<float> rec_op_fwd(1., Ab32F_T[0][0]);
    RecOp<float> rec_op_rev(1., Ab32R_T[0][0]);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes,
                                   &d_pybar, &d_pybar_out, rec_op_fwd,
                                   num_blocks+1);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes,
                                   &d_pybar, &d_pybar_out, rec_op_fwd,
                                   num_blocks+1);
   
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    // first run to warm the GPU up

    alg3_step1<T, R><<< grid, block >>>( &d_pybar, &d_ezhat, &d_in,
                                         num_blocks );

    alg3_step3<T, R><<< grid, block >>>( &d_out, &d_pybar, &d_ezhat, &d_in,
                                         num_blocks );

    base_timer &timer_total = timers.gpu_add("alg1d_gpu", num_samples, "iP");

    for (int r = 0; r < num_repeats; ++r) {

        alg3_step1<T, R><<< grid, block >>>( &d_pybar, &d_ezhat, &d_in,
                                             num_blocks );

        cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes,
                                       &d_pybar, &d_pybar_out, rec_op_fwd,
                                       num_blocks+1);

        d_pybar = d_pybar_out;

        cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes,
                                       &d_ezhat, &d_ezhat_out, rec_op_rev,
                                       num_blocks+1);

        d_ezhat = d_ezhat_out;

        alg3_step3<T, R><<< grid, block >>>( &d_out, &d_pybar, &d_ezhat,
                                             &d_in, num_blocks );

    }

    timer_total.stop();

    if (num_repeats > 1) {

        std::size_t proc_samples = timer_total.data_size()*num_repeats;
        double time_sec_inv_mebi = timer_total.elapsed()*1024*1024;
        std::cout << std::fixed << proc_samples/time_sec_inv_mebi << std::flush;

    } else { // running for debugging

        timers.flush();

        T *pybar = new T[d_pybar.size()];
        d_pybar.copy_to(pybar, d_pybar.size());
        
        std::cout << std::fixed << std::flush;
        gpufilter::print_array(pybar, 32, "d_pybar [:32]:");

    }

    d_out.copy_to(h_in, num_samples);

}

} // gpufilter namespace
