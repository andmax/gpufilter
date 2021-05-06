/*
SAM v1.1: SAM is a fast prefix-scan template written in CUDA that
supports higher orders and/or tuple values as described in
http://cs.txstate.edu/~burtscher/papers/pldi16.pdf.

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
 - reduction in auxiliary memory usage
 - simplified interface
 - small performance optimizations

Authors: Sepideh Maleki, Annie Yang, and Martin Burtscher
*/


static const int SMs = 28;  // this value must match the used GPU
static const int MOD = 256;  // do not change
static const int MM1 = MOD - 1;  // do not change
static const int Max_Dim = 32;  // do not increase

template <typename T, int factor, int dim, int order, T (*op)(T, T)>
static __global__ __launch_bounds__(1024, 2)
void kSAM(const T * const __restrict__ ginput, T * const __restrict__ goutput, const int items, volatile T * const __restrict__ gcarry, volatile int * const __restrict__ gwait)
{
/*
  // The following assertions need to hold but are commented out for performance reasons.
  assert(1024 == blockDim.x);
  assert(SMs * 2 == gridDim.x);
  assert(64 >= gridDim.x);
  assert(Max_Dim >= dim);
*/
  const int chunks = (items + (1024 * factor - 1)) / (1024 * factor);
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int corr = 1024 % dim;

  __shared__ T globcarry[dim][order];
  __shared__ T tempcarry[dim];
  __shared__ T sbuf[factor][32 * dim];

  for (int i = tid; i < dim * order; i += 1024) {
    globcarry[i / order][i % order] = 0;
  }

  int pos = 0;
  for (int chunk = blockIdx.x; chunk < chunks; chunk += SMs * 2) {
    const int offs = tid + chunk * (1024 * factor);
    const int firstid = offs % dim;
    const int lastid = (offs + 1024 * (factor - 1)) % dim;

    T val[factor];
    if (chunk < chunks - 1) {
      for (int i = 0; i < factor; i++) {
        val[i] = ginput[offs + 1024 * i];
      }
    } else {
      for (int i = 0; i < factor; i++) {
        val[i] = 0;
        if (offs + 1024 * i < items) {
          val[i] = ginput[offs + 1024 * i];
        }
      }
    }

    for (int round = 0; round < order; round++) {
      for (int i = 0; i < factor; i++) {
        for (int d = dim; d < 32; d *= 2) {
          T tmp = __shfl_up(val[i], d);
          if (lane >= d) val[i] = op(val[i], tmp);
        }
      }

      if (lane >= (32 - dim)) {
        const int tix = warp * dim;
        int id = firstid;
        for (int i = 0; i < factor; i++) {
          sbuf[i][tix + id] = val[i];
          id += corr;
          if (id >= dim) id -= dim;
        }
      }

      __syncthreads();
      if (warp < dim) {
        const int idx = (lane * dim) + warp;
        for (int i = 0; i < factor; i++){
          T v = sbuf[i][idx];
          for (int d = 1; d < 32; d *= 2) {
            T tmp = __shfl_up(v, d);
            if (lane >= d) v = op(v, tmp);
          }
          sbuf[i][idx] = v;
        }
      }

      __syncthreads();
      if (warp > 0) {
        const int tix = warp * dim - dim;
        int id = firstid;
        for (int i = 0; i < factor; i++) {
          val[i] = op(val[i], sbuf[i][tix + id]);
          id += corr;
          if (id >= dim) id -= dim;
        }
      }

      T carry[dim];
      for (int d = 0; d < dim; d++) {
        carry[d] = 0;
      }
      int id = firstid;
      for (int i = 1; i < factor; i++) {
        for (int d = 0; d < dim; d++) {
          carry[d] = op(carry[d], sbuf[i - 1][31 * dim + d]);
        }
        id += corr;
        if (id >= dim) id -= dim;
        val[i] = op(val[i], carry[id]);
      }

      int wait = round + 1;
      if (tid > 1023 - dim) {
        gcarry[lastid * (order * MOD) + round * MOD + (chunk & MM1)] = val[factor - 1];
        gwait[round * MOD + ((chunk + (MOD - 4 * SMs)) & MM1)] = 0;
        __threadfence();
        if (tid == 1023) {
          gwait[round * MOD + (chunk & MM1)] = wait;
        }
      }

      const int tidx = pos + tid;
      if (tidx < chunk) {
        wait = gwait[round * MOD + (tidx & MM1)];
      }
      while (__syncthreads_count(wait <= round) != 0) {
        if (wait <= round) {
          wait = gwait[round * MOD + (tidx & MM1)];
        }
      }

      if (warp < dim) {
        int posx = pos + lane;
        T carry = 0;
        if (posx < chunk) {
          carry = gcarry[warp * (order * MOD) + round * MOD + (posx & MM1)];
        }
        if (SMs > 16) {
          posx += 32;
          if (posx < chunk) {
            carry = op(carry, gcarry[warp * (order * MOD) + round * MOD + (posx & MM1)]);
          }
        }
        for (int d = 1; d < 32; d *= 2) {
          carry = op(carry, __shfl_up(carry, d));
        }
        if (lane == 31) {
          T temp = op(globcarry[warp][round], carry);
          tempcarry[warp] = globcarry[warp][round] = temp;
        }
      }

      __syncthreads();
      if (tid > 1023 - dim) {
        globcarry[lastid][round] = op(globcarry[lastid][round], val[factor - 1]);
      }

      id = firstid;
      for (int i = 0; i < factor; i++) {
        val[i] = op(val[i], tempcarry[id]);
        id += corr;
        if (id >= dim) id -= dim;
      }
    } // round

    if (chunk < chunks - 1) {
      for (int i = 0; i < factor; i++) {
        goutput[offs + 1024 * i] = val[i];
      }
    } else {
      for (int i = 0; i < factor; i++) {
        if (offs + 1024 * i < items) {
          goutput[offs + 1024 * i] = val[i];
        }
      }
    }
    pos = chunk + 1;
  } // chunk
}

template <typename T, int factor, int dim, int order, T (*op)(T, T)>
static void rSAM(const T * const __restrict__ ginput, T * const __restrict__ goutput, const int items)
{
  static int* aux = NULL;
  if (aux == NULL) {
    cudaMalloc(&aux, order * MOD * sizeof(int) + dim * order * MOD * sizeof(T));
  }
  cudaMemsetAsync(aux, 0, order * MOD * sizeof(int));
  kSAM<T, factor, dim, order, op><<<SMs * 2, 1024>>>(ginput, goutput, items, (T *)&aux[order * MOD], aux);
}

/*
Below is the SAM function that should be called like this:

  SAM<T, dim, order, op>(input, output, items);

Template parameters
-------------------
T: type of the elements (the code has only been tested with "int" and "long")
dim: tuple size (1 through 32, where 1 is a normal prefix scan)
order: requested order (at least 1, where 1 is a normal prefix scan)
op: associative operator (e.g., sum, max, xor)

For example, the sum operator would be specified as follows:

  template <typename T> 
  static __host__ __device__ T sum(T a, T b)
  {
    return a + b;
  }

Inputs
------
items: number of elements in input and output arrays
input: array of values over which to perform the prefix scan (input must have "items" elements)

Output
------
output: result of the prefix scan (output must have capacity for "items" elements)
*/
