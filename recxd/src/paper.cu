/**
 *  @file paper.cu
 *  @brief Source code for paper
 *  @author Andre Maximo
 *  @date Dec, 2019
 *  @copyright The MIT License
 */

#include <cstdlib>

#include <algorithm>
#include <iostream>
#include <fstream>

#include <util/util.h>
#include <util/timer.h>
#include <util/symbol.h>
#include <util/dvector.h>
#include <util/gaussian.h>
#include <util/alg0_xd_cpu.h>

#define FTYPE float // filter float type
#define FORDER 1 // filter order
#define FM 0xffffffff // shuffle's full mask
#define WS 32 // warp size
#define NW 3 // number of warps in a block
#define NB 21 // number of blocks in a grid

#include <util/linalg.h>
#include <util/recfilter.h>

using namespace gpufilter;

__constant__
Vector<FTYPE, FORDER+1> c_weights;

__constant__
Vector<Matrix<FTYPE, FORDER, FORDER>, 10>
c_AbF_T, c_AbR_T;

__constant__
Matrix<FTYPE,FORDER,FORDER> c_HARB_AFP_T;

template<typename T>
__device__
void read_block(
  Matrix<T, WS, WS+1>& block,
  const T *input,
  const int& tx, const int& ty, const int& bi) {
  T *bcp = &block[ty][tx];
  const T *icp = &input[bi*WS*WS+ty*WS+tx];
  for (int i = 0; i < WS - (WS % NW); i += NW) {
    *bcp = *icp;
    bcp += NW*(WS+1);
    icp += NW*WS;
  }
  if (ty < WS % NW) {
    *bcp = *icp;
  }
}

template<typename T>
__device__
void write_block(
  T *output,
  Matrix<T, WS, WS+1>& block,
  const int& tx, const int& ty, const int& bi) {
  T *bcp = &block[ty][tx];
  T *ocp = &output[bi*WS*WS+ty*WS+tx];
  for (int i = 0; i < WS - (WS % NW); i += NW) {
    *ocp = *bcp;
    bcp += NW*(WS+1);
    ocp += NW*WS;
  }
  if (ty < WS % NW) {
    *ocp = *bcp;
  }
}

template <typename T, int R>
__device__
void compute_py(
  Vector<T, R>& py,
  Matrix<T, WS, WS+1>& block,
  const int& tx, const bool& save_in_block) {
  T x[WS];
  for (int i = 0; i < WS; ++i)
    x[i] = block[tx][i];
  for (int i = 0; i < WS; ++i) {
    if (save_in_block)
      block[tx][i] = fwdI(py, x[i], c_weights);
    else
      fwdI(py, x[i], c_weights);
  }
}

template <typename T, int R>
__device__
void compute_ez(
  Vector<T, R>& ez,
  Matrix<T, WS, WS+1>& block,
  const int& tx, const bool& save_in_block) {
  T x[WS];
  for (int i = 0; i < WS; ++i)
    x[i] = block[tx][i];
  for (int i = WS-1; i >= 0; --i) {
    if (save_in_block)
      block[tx][i] = revI(x[i], ez, c_weights);
    else
      revI(x[i], ez, c_weights);
  }
}

template <typename T, int R>
__device__
void fix_py(
    Vector<T, R>& py,
    const int& tx,
    const int& ci=0) {
  Vector<T, R> pyprev;
  for (int i = 0; i < 5; ++i) {
    int k = 1 << i;
    for (int r = 0; r < R; ++r) {
      pyprev[r] = __shfl_up_sync(FM, py[r], k);
    }
    if (tx >= k) {
      py = py + pyprev * c_AbF_T[ci+i];
    }
  }
}

template <typename T, int R>
__device__
void fix_ez(
    Vector<T, R>& ez,
    const int& tx,
    const int& ci=0) {
  Vector<T, R> eznext;
  for (int i = 0; i < 5; ++i) {
    int k = 1 << i;
    for (int r = 0; r < R; ++r) {
      eznext[r] = __shfl_down_sync(FM, ez[r], k);
    }
    if (tx < WS - k) {
      ez = ez + eznext * c_AbR_T[ci+i];
    }
  }
}

template <typename T, int R>
__global__ __launch_bounds__(WS*NW, NB)
void alg3_step1(
  Vector<T, R> *g_py,
  Vector<T, R> *g_ez,
  const T *g_in) {
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int bi = blockIdx.x;
  __shared__ Matrix<T, WS, WS+1> s_block;
  read_block(s_block, g_in, tx, ty, bi);
  __syncthreads();
  if (ty == 0) {
    Vector<T, R> py = zeros<T, R>();
    compute_py(py, s_block, tx, false);
    fix_py(py, tx);
    if (tx == WS-1)
      g_py[bi+1] = py;
    for (int r = 0; r < R; ++r)
        py[r] = __shfl_up_sync(FM, py[r], 1);
    if (tx == 0)
      py = zeros<T, R>();
    __syncwarp();
    compute_py(py, s_block, tx, true);
    Vector<T, R> ez = zeros<T, R>();
    compute_ez(ez, s_block, tx, false);
    fix_ez(ez, tx);
    if (tx == 0)
      g_ez[bi] = ez;        
  }
}

template <typename T, int R>
__global__ __launch_bounds__(WS, 2)
void alg3_step2(
  Vector<T, R> *g_py,
  Vector<T, R> *g_ez,
  int num_blocks) {
  const int tx = threadIdx.x, ty = threadIdx.y;
  Vector<T, R> pe;
  for (int i = 0; i < num_blocks+1; i += WS) {
    if (ty == 0) {
      if (i > 0 && tx == 0)
        pe = g_py[i+tx+1] + pe * c_AbF_T[5];
      else if (i+tx < num_blocks+1)
        pe = g_py[i+tx+1];
      fix_py(pe, tx, 5);
      if (i+tx < num_blocks+1)
        g_py[i+tx+1] = pe;
      for (int r = 0; r < R; ++r)
        pe[r] = __shfl_down_sync(FM, pe[r], WS-1);
    } else if (ty == 1) {
      i = num_blocks+1 - i;
      if (i < num_blocks+1 && tx == WS-1)
        pe = g_ez[i+tx+1] + pe * c_AbR_T[5];
      else if (i+tx >= 0)
        pe = g_ez[i+tx+1];
      fix_ez(pe, tx, 5);
      if (i+tx >= 0)
        g_ez[i+tx+1] = pe;
      for (int r = 0; r < R; ++r)
        pe[r] = __shfl_up_sync(FM, pe[r], WS-1);
    }
  }
}

template <typename T, int R>
__global__ __launch_bounds__(WS*NW, NB)
void alg3_step3(
  T *g_out,
  const Vector<T, R> *g_py,
  const Vector<T, R> *g_ez,
  const T *g_in ) {
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int bi = blockIdx.x;
  __shared__ Matrix<T, WS, WS+1> s_block;
  read_block(s_block, g_in, tx, ty, bi);
  __syncthreads();
  if (ty == 0) {
    Vector<T, R> py = zeros<T, R>();
    if (tx == 0) {
      for (int r = 0; r < R; ++r)
        py[r] = __ldg((const T*)&g_py[bi][r]);
    } else {
      compute_py(py, s_block, tx-1, false); 
    }
    fix_py(py, tx);
    compute_py(py, s_block, tx, true);
    Vector<T, R> ez = zeros<T, R>();
    if (tx == WS-1) {
      for (int r = 0; r < R; ++r)
        ez[r] = __ldg((const T*)&g_ez[bi+1][r]);
      if (bi < gridDim.x-1)
        ez = ez + py * c_HARB_AFP_T;
    } else {
      compute_ez(ez, s_block, tx+1, false);
    }
    __syncwarp();
    fix_ez(ez, tx);
    compute_ez(ez, s_block, tx, true);
  }
  __syncthreads();
  write_block(g_out, s_block, tx, ty, bi);
}

template <typename T, int R>
__host__
void oa1d_gpu(
  T *h_in,
  const long int& num_samples,
  const long int& num_repeats,
  const Vector<T, R+1> &w ) {

  const int B = WS;

  // pre-compute basic alg1d matrices
  Matrix<T,R,B> Zrb = zeros<T,R,B>();
  Matrix<T,B,R> Zbr = zeros<T,B,R>();
  Matrix<T,R,R> Ir = identity<T,R,R>();
  Matrix<T,B,B> Ib = identity<T,B,B>();

  Matrix<T,R,B> AFP_T = fwd(Ir, Zrb, w);
  Matrix<T,R,B> ARE_T = rev(Zrb, Ir, w);
  Matrix<T,B,B> ARB_T = rev(Ib, Zbr, w);
  
  Matrix<T,R,R> AbF_T = tail<R>(AFP_T);
  Matrix<T,R,R> AbR_T = head<R>(ARE_T);
  Matrix<T,R,R> HARB_AFP_T = AFP_T*head<R>(ARB_T);

  Vector<Matrix<FTYPE, FORDER, FORDER>, 10> v_AbF_T;
  Vector<Matrix<FTYPE, FORDER, FORDER>, 10> v_AbR_T;

  v_AbF_T[0] = AbF_T;
  v_AbR_T[0] = AbR_T;
  for (int i = 1; i < 10; ++i) {
    v_AbF_T[i] = v_AbF_T[i-1] * v_AbF_T[i-1];
    v_AbR_T[i] = v_AbR_T[i-1] * v_AbR_T[i-1];
  }

  // upload to the GPU
  copy_to_symbol(c_weights, w);

  copy_to_symbol(c_AbF_T, v_AbF_T);
  copy_to_symbol(c_AbR_T, v_AbR_T);
  copy_to_symbol(c_HARB_AFP_T, HARB_AFP_T);

  dvector<T> d_in(h_in, num_samples), d_out(num_samples);

  long int num_blocks = num_samples/(B*B);
  
  dim3 grid(num_blocks);
  
  dim3 block(WS, NW);

  dvector< Vector<T, R> > d_pybar(num_blocks+1);
  dvector< Vector<T, R> > d_ezhat(num_blocks+1);
  d_pybar.fillzero();
  d_ezhat.fillzero();
 
  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
  cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

  base_timer &timer_total = timers.gpu_add("paper_code", num_samples, "iP");

  // first run to warm the GPU up

  alg3_step1<<< grid, block >>>( &d_pybar, &d_ezhat, &d_in );

  alg3_step2<<< dim3(2), dim3(WS) >>>( &d_pybar, &d_ezhat, num_blocks );

  alg3_step3<<< grid, block >>>( &d_out, &d_pybar, &d_ezhat, &d_in );

  for (int r = 0; r < num_repeats; ++r) {

    alg3_step1<<< grid, block >>>( &d_pybar, &d_ezhat, &d_in );

    alg3_step2<<< dim3(2), dim3(WS) >>>( &d_pybar, &d_ezhat, num_blocks );

    alg3_step3<<< grid, block >>>( &d_out, &d_pybar, &d_ezhat, &d_in );

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
    print_array(pybar, 32, "d_pybar [:32]:");

  }

  d_out.copy_to(h_in, num_samples);

}


int main(int argc, char** argv) {

  long int num_samples = 1 << 23, num_repeats = 1; // defaults
  char array_bin_fn[200] = "../bin/random_array.bin";
  
  if ((argc != 1 && argc != 4)
    || (argc==4 && (sscanf(argv[1], "%ld", &num_samples) != 1 ||
                    sscanf(argv[2], "%ld", &num_repeats) != 1 ||
                    sscanf(argv[3], "%s", array_bin_fn) != 1))) {
    std::cerr << " Bad arguments!\n";
    std::cerr << " Usage: " << argv[0]
              << " [num_samples num_repeats array_bin_fn] ->"
              << " Output: Mis/s MAE MRE\n";
    std::cerr << " Where: num_samples = number of samples "
              << "in the 1D array to run this on (up to 1Gi)\n";
    std::cerr << " Where: num_repeats = number of repetitions "
              << "to measure the run timing performance\n";
    std::cerr << " Where: array_bin_fn = array of inputs in "
              << "binary to read 1D input data from\n";
    std::cerr << " Where: Mis/s = Mebi samples per second; "
              << "MAE = max. abs. error; MRE = max. rel. error\n";
    return EXIT_FAILURE;
  }

  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
  if (num_repeats == 1) { // running for debugging
    std::cout << get_cuda_device_properties();
  }

  Vector<FTYPE, FORDER+1> iir_weights;
  FTYPE gaussian_sigma = 4.0;
  weights(gaussian_sigma, iir_weights);

  FTYPE *cpu_arr = new FTYPE[num_samples];
  FTYPE *gpu_arr = new FTYPE[num_samples];

  std::ifstream in_file(array_bin_fn, std::ios::binary);
  in_file.read(reinterpret_cast<char*>(cpu_arr),
               sizeof(FTYPE)*num_samples);
  in_file.close();

  memcpy(gpu_arr, cpu_arr, sizeof(FTYPE) * num_samples);

  recursive_1d<0,true,FORDER>(cpu_arr, num_samples, iir_weights);
  recursive_1d<0,false,FORDER>(cpu_arr, num_samples, iir_weights);

  oa1d_gpu<FTYPE,FORDER>(
      gpu_arr, num_samples, num_repeats, iir_weights);

  FTYPE max_abs_err, max_rel_err;
  check_cpu_reference(cpu_arr, gpu_arr, num_samples,
    max_abs_err, max_rel_err);

  if (num_repeats == 1) // running for debugging
    std::cout << " [max-absolute-error] [max-relative-error]:";

  std::cout << " " << std::scientific << max_abs_err << " "
            << std::scientific << max_rel_err << "\n";

  if (cpu_arr) delete [] cpu_arr;
  if (gpu_arr) delete [] gpu_arr;

  return EXIT_SUCCESS;

}
