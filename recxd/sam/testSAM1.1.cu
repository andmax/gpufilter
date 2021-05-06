/*
SAM demo v1.1: This code illustrates how to call and use SAM, a fast prefix-scan
template written in CUDA that supports higher orders and/or tuple values as
described in http://cs.txstate.edu/~burtscher/papers/pldi16.pdf.

Copyright (c) 2016, Texas State University. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted for academic, research, experimental, or personal use provided
that the following conditions are met:

   * Redistributions of source code must retain the above copyright notice,
     this list of conditions, and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions, and the following disclaimer in the documentation
     and/or other materials provided with the distribution.
   * Neither the name of Texas State University nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

For all other uses, please contact the Office for Commercialization and Industry
Relations at Texas State University <http://www.txstate.edu/ocir/>.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Version 1.1 (2016/3/13):
 - uses simplified SAM interface
 - improved output
 - improved timing

Authors: Sepideh Maleki and Martin Burtscher
*/

#include <cstdio>
#include <cassert>
#include "sam.h"

template <typename T>
__host__ __device__ T sum(T a, T b)
{
  return a + b;
}

template <typename T>
__host__ __device__ T maximum(T a, T b)
{
  return max(a, b);
}

template <typename T>
static int compare(T* cpuout, T* gpuout, int items)
{
  for (int i = 0; i < items; i++) {
    if (cpuout[i] != gpuout[i]) {
      return i + 1;
    }
  }
  return 0;
}

template <typename T, int dim, int order, T (*op)(T, T)>
static void cpusolve(T* input, T* cpuout, int items)
{
  for (int i = 0; i < items; i++) {
    cpuout[i] = input[i];
  }
  for (int j = 0; j < order; j++) {
    T inclusive[dim];
    for (int k = 0; k < dim; k++) {
      inclusive[k] = 0;
    }
    for (int i = 0; i < items; i++) {
      inclusive[i % dim] = op(inclusive[i % dim], cpuout[i]);
      cpuout[i] = inclusive[i % dim];
    }
  }
}

struct GPUTimer
{
  cudaEvent_t beg, end;

  GPUTimer()
  {
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
  }

  ~GPUTimer()
  {
    cudaEventDestroy(beg);
    cudaEventDestroy(end);
  }

  void start()
  {
    cudaEventRecord(beg, 0);
  }

  double stop()
  {
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float ms;
    cudaEventElapsedTime(&ms, beg, end);
    return 0.001 * ms;
  }
};

template< typename T1, typename T2, typename T3 >
void check_cpu_reference(const T1 *ref,
                         const T2 *res,
                         const long int& ne,
                         T3& me, T3& mre) {
    mre = me = 0.0;
    for (long int i = 0; i < ne; i++)
    {
        double a = (double)(res[i] - ref[i]);
        if( a < 0.0 ) a = -a;
        if( ref[i] != (T1)0 )
        {
            T1 r = (ref[i] < (T1)0) ? -ref[i] : ref[i];
            double b = a / (double)r;
            mre = b > mre ? b : mre;
        }
        me = a > me ? a : me;
    }
}

template <typename T,int dim, int order, T (*op)(T, T)>
static void demo(const long int items, const long int repetitions)
{
  const size_t size = items * sizeof(T);

  // allocate CPU memory
  T *input, *gpuout, *cpuout;
  input = (T *)malloc(size);  assert(input != NULL);
  gpuout = (T *)malloc(size);  assert(gpuout != NULL);
  cpuout = (T *)malloc(size);  assert(cpuout != NULL);

  // initialize input with some random data
  for (int i = 0; i < items; i++) {
    input[i] = i;
  }

  // solve on the CPU for later comparison
  cpusolve<T, dim, order, op>(input, cpuout, items);

  // allocate GPU memory
  T *ginput, *goutput;
  cudaMalloc(&ginput, size);
  cudaMalloc(&goutput, size);

  // copy input to GPU
  assert(cudaSuccess == cudaMemcpy(ginput, input, size, cudaMemcpyHostToDevice));

  SAM<T, dim, order, op>(ginput, goutput, items);

  // timed code section
  GPUTimer timer;
  timer.start();
  for (long i = 0; i < repetitions; i++) {  // repeat a few times for more accurate timing
    SAM<T, dim, order, op>(ginput, goutput, items);
  }
  double runtime = timer.stop();

  // output performance results
/*
  double throughput = 0.000000001 * repetitions * items / runtime;
  printf("%.3f ms\n", 1000.0 * runtime / repetitions);
  printf("%.3f Giga-items/s\n", throughput);
*/

  // copy output from GPU
  assert(cudaSuccess == cudaMemcpy(gpuout, goutput, size, cudaMemcpyDeviceToHost));

  // compare GPU result to CPU result
  int cmp = compare(cpuout, gpuout, items);
  if (cmp) {
    printf("ERROR: %lu != %lu at pos %d\n", gpuout[cmp - 1], cpuout[cmp - 1], cmp - 1);
    exit(-1);
  }
//  printf("test passed\n");

  double mebissec = (repetitions * items) / (runtime*1024*1024); // Mis/s
  double max_abs_err, max_rel_err;
  check_cpu_reference(cpuout, gpuout, items, max_abs_err, max_rel_err);
  printf("%7.7f %e %e\n", mebissec, max_abs_err, max_rel_err);

  free(input);  free(gpuout);  free(cpuout);
  cudaFree(ginput);  cudaFree(goutput);
}

static void checkGPU()
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
    fprintf(stderr, "ERROR: There is no CUDA capable device.\n");
    exit(-1);
  }
  if (deviceProp.major < 3) {
    fprintf(stderr, "ERROR: Need at least compute capability 3.0.\n");
    exit(-1);
  }
  if (SMs != deviceProp.multiProcessorCount) {
    fprintf(stderr, "ERROR: Please set SMs to %d in the header file and recompile code.\n", deviceProp.multiProcessorCount);
    exit(-1);
  }
//  printf("using %s\n", deviceProp.name);
}

int main(int argc, char *argv[])
{
/*
  printf("SAM Prefix Scan (%s)\n", __FILE__);
  printf("Copyright (c) 2016 Texas State University\n");
*/
  // run some checks
  if (argc != 3) {fprintf(stderr, "usage: %s number_of_items repeats\n", argv[0]); exit(-1);}
  long int items = atol(argv[1]);
  long int repeats = atol(argv[2]);
  if (items < 1) {fprintf(stderr, "ERROR: items must be at least 1\n"); exit(-1);}

  checkGPU();

  // change the following info if another data type, tuple size, order, and/or operator is needed
  typedef long int mytype;
  const int dim = 1;
  const int order = 1;
//  printf("dim = %d  order = %d  %d-byte type  items = %d\n", dim, order, sizeof(mytype), items);
  demo<mytype, dim, order, sum<mytype> >(items, repeats);

  return 0;
}
