/*
PLR - Parallelized Linear Recurrences [float]
Copyright (c) 2018 Texas State University. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted for academic, research, experimental, or personal use provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
* Neither the name of Texas State University nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

For all other uses, please contact the Office for Commercialization and Industry Relations at Texas State University http://www.txstate.edu/ocir/.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Sepideh Maleki and Martin Burtscher


non-recursive coefficients: (0.280990)
recursive coefficients: (0.719010)

*/

#include <cstdio>
#include <cassert>
#include <cuda.h>

typedef float T;
static const int device = 0;
static const int order = 1;
static const int warp_size = 32;
static const int block_size = 1024;

static __device__ const T facA[264] = {7.190100e-01f, 5.169754e-01f, 3.717105e-01f, 2.672636e-01f, 1.921652e-01f, 1.381687e-01f, 9.934466e-02f, 7.142980e-02f, 5.135874e-02f, 3.692745e-02f, 2.655121e-02f, 1.909058e-02f, 1.372632e-02f, 9.869360e-03f, 7.096169e-03f, 5.102216e-03f, 3.668545e-03f, 2.637720e-03f, 1.896547e-03f, 1.363636e-03f, 9.804681e-04f, 7.049664e-04f, 5.068778e-04f, 3.644502e-04f, 2.620434e-04f, 1.884118e-04f, 1.354700e-04f, 9.740427e-05f, 7.003464e-05f, 5.035561e-05f, 3.620618e-05f, 2.603261e-05f, 1.871771e-05f, 1.345822e-05f, 9.676593e-06f, 6.957568e-06f, 5.002561e-06f, 3.596891e-06f, 2.586201e-06f, 1.859504e-06f, 1.337002e-06f, 9.613179e-07f, 6.911972e-07f, 4.969777e-07f, 3.573319e-07f, 2.569252e-07f, 1.847318e-07f, 1.328240e-07f, 9.550178e-08f, 6.866674e-08f, 4.937207e-08f, 3.549901e-08f, 2.552414e-08f, 1.835211e-08f, 1.319535e-08f, 9.487591e-09f, 6.821673e-09f, 4.904851e-09f, 3.526637e-09f, 2.535687e-09f, 1.823184e-09f, 1.310888e-09f, 9.425415e-10f, 6.776967e-10f, 4.872707e-10f, 3.503525e-10f, 2.519070e-10f, 1.811236e-10f, 1.302297e-10f, 9.363646e-11f, 6.732555e-11f, 4.840774e-11f, 3.480565e-11f, 2.502561e-11f, 1.799367e-11f, 1.293763e-11f, 9.302282e-12f, 6.688434e-12f, 4.809051e-12f, 3.457756e-12f, 2.486161e-12f, 1.787574e-12f, 1.285284e-12f, 9.241320e-13f, 6.644601e-13f, 4.777535e-13f, 3.435095e-13f, 2.469868e-13f, 1.775859e-13f, 1.276861e-13f, 9.180756e-14f, 6.601056e-14f, 4.746225e-14f, 3.412583e-14f, 2.453681e-14f, 1.764222e-14f, 1.268493e-14f, 9.120591e-15f, 6.557796e-15f, 4.715121e-15f, 3.390219e-15f, 2.437601e-15f, 1.752660e-15f, 1.260180e-15f, 9.060818e-16f, 6.514819e-16f, 4.684220e-16f, 3.368001e-16f, 2.421626e-16f, 1.741173e-16f, 1.251921e-16f, 9.001438e-17f, 6.472124e-17f, 4.653522e-17f, 3.345928e-17f, 2.405756e-17f, 1.729763e-17f, 1.243717e-17f, 8.942447e-18f, 6.429709e-18f, 4.623025e-18f, 3.324001e-18f, 2.389990e-18f, 1.718427e-18f, 1.235566e-18f, 8.883843e-19f, 6.387572e-19f, 4.592728e-19f, 3.302217e-19f, 2.374327e-19f, 1.707165e-19f, 1.227469e-19f, 8.825622e-20f, 6.345711e-20f, 4.562630e-20f, 3.280576e-20f, 2.358767e-20f, 1.695977e-20f, 1.219425e-20f, 8.767785e-21f, 6.304125e-21f, 4.532729e-21f, 3.259077e-21f, 2.343309e-21f, 1.684863e-21f, 1.211433e-21f, 8.710326e-22f, 6.262811e-22f, 4.503024e-22f, 3.237719e-22f, 2.327953e-22f, 1.673821e-22f, 1.203494e-22f, 8.653243e-23f, 6.221768e-23f, 4.473513e-23f, 3.216501e-23f, 2.312696e-23f, 1.662852e-23f, 1.195607e-23f, 8.596534e-24f, 6.180994e-24f, 4.444197e-24f, 3.195422e-24f, 2.297540e-24f, 1.651954e-24f, 1.187772e-24f, 8.540197e-25f, 6.140487e-25f, 4.415072e-25f, 3.174481e-25f, 2.282483e-25f, 1.641128e-25f, 1.179988e-25f, 8.484229e-26f, 6.100245e-26f, 4.386138e-26f, 3.153677e-26f, 2.267525e-26f, 1.630373e-26f, 1.172255e-26f, 8.428628e-27f, 6.060268e-27f, 4.357393e-27f, 3.133009e-27f, 2.252665e-27f, 1.619689e-27f, 1.164572e-27f, 8.373393e-28f, 6.020553e-28f, 4.328838e-28f, 3.112478e-28f, 2.237902e-28f, 1.609074e-28f, 1.156940e-28f, 8.318518e-29f, 5.981097e-29f, 4.300469e-29f, 3.092080e-29f, 2.223236e-29f, 1.598529e-29f, 1.149359e-29f, 8.264003e-30f, 5.941901e-30f, 4.272286e-30f, 3.071816e-30f, 2.208667e-30f, 1.588053e-30f, 1.141826e-30f, 8.209844e-31f, 5.902960e-31f, 4.244287e-31f, 3.051685e-31f, 2.194192e-31f, 1.577646e-31f, 1.134343e-31f, 8.156041e-32f, 5.864275e-32f, 4.216472e-32f, 3.031686e-32f, 2.179812e-32f, 1.567307e-32f, 1.126909e-32f, 8.102591e-33f, 5.825844e-33f, 4.188840e-33f, 3.011818e-33f, 2.165527e-33f, 1.557036e-33f, 1.119524e-33f, 8.049491e-34f, 5.787664e-34f, 4.161388e-34f, 2.992080e-34f, 2.151335e-34f, 1.546831e-34f, 1.112187e-34f, 7.996737e-35f, 5.749734e-35f, 4.134116e-35f, 2.972471e-35f, 2.137236e-35f, 1.536694e-35f, 1.104899e-35f, 7.944331e-36f, 5.712053e-36f, 4.107023e-36f, 2.952991e-36f, 2.123230e-36f, 1.526623e-36f, 1.097657e-36f, 7.892267e-37f, 5.674619e-37f, 4.080108e-37f, 2.933638e-37f, 2.109315e-37f, 1.516619e-37f, 1.090464e-37f, 7.840545e-38f, 5.637430e-38f, 4.053369e-38f, 2.914413e-38f, 2.095492e-38f, 1.506680e-38f};

// shared memory size is 5256 bytes

static __device__ unsigned int counter = 0;

static __global__ __launch_bounds__(block_size, 2)
void Recurrence1(const int items, const T* const __restrict__ input, T* const __restrict__ output, volatile int* const __restrict__ status, volatile T* const __restrict__ partcarry, volatile T* const __restrict__ fullcarry)
{
  const int valsperthread = 1;
  const int chunk_size = valsperthread * block_size;
  __shared__ T spartc[chunk_size / warp_size * order];
  __shared__ T sfullc[order];
  __shared__ int cid;

  const int tid = threadIdx.x;
  const int warp = tid / warp_size;
  const int lane = tid % warp_size;

  __shared__ T sfacA[block_size];
  if (tid < 264) sfacA[tid] = facA[tid];
  else sfacA[tid] = 0;

  if (tid == 0) {
    cid = atomicInc(&counter, gridDim.x - 1);
  }

  __syncthreads();
  const int chunk_id = cid;
  const int offs = tid + chunk_id * chunk_size;

  T val0;
  if (chunk_id == gridDim.x - 1) {
    val0 = 0;
    if (offs + (0 * block_size) < items) val0 = input[offs + (0 * block_size)];
  } else {
    val0 = input[offs + (0 * block_size)];
  }

  val0 *= 2.809900e-01f;

  const T sfA = sfacA[lane];

  int cond;
  T help, spc;

  help = 7.190100e-01f;
  cond = ((lane & 1) != 0);
  spc = help * __shfl(val0, 0, 2);
  if (cond) val0 += spc;

  help = __shfl(sfA, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 1, 4);
  if (cond) val0 += spc;

  help = __shfl(sfA, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 3, 8);
  if (cond) val0 += spc;

  help = __shfl(sfA, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 7, 16);
  if (cond) val0 += spc;

  help = __shfl(sfA, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 15, 32);
  if (cond) val0 += spc;

  const int delta = block_size / warp_size * order;
  const int clane = lane - (warp_size - order);
  const int clwo = clane + warp * order;

  if (((warp & 1) == 0) && (clane >= 0)) {
    spartc[clwo + 0 * delta] = val0;
  }

  __syncthreads();
  if ((warp & 1) != 0) {
    const int cwarp = ((warp & ~1) | 0) * order;
    const T helpA = sfacA[tid % (warp_size * 1)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    if (((warp & 3) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
    }
  }

  __syncthreads();
  if ((warp & 2) != 0) {
    const int cwarp = ((warp & ~3) | 1) * order;
    const T helpA = sfacA[tid % (warp_size * 2)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    if (((warp & 7) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
    }
  }

  __syncthreads();
  if ((warp & 4) != 0) {
    const int cwarp = ((warp & ~7) | 3) * order;
    const T helpA = sfacA[tid % (warp_size * 4)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    if (((warp & 15) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
    }
  }

  __syncthreads();
  if ((warp & 8) != 0) {
    const int cwarp = ((warp & ~15) | 7) * order;
    const T helpA = sfacA[tid % (warp_size * 8)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    if ((warp == 15) && (clane >= 0)) {
      spartc[clane + (15 * order + 0 * delta)] = val0;
    }
  }

  __syncthreads();
  if ((warp & 16) != 0) {
   if ((warp & 15) < 9) {
    const int cwarp = 15 * order;
    const T helpA = sfacA[tid % (warp_size * 16)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
   }
    if ((warp == 31) && (clane >= 0)) {
      spartc[clane + (31 * order + 0 * delta)] = val0;
    }
  }

  const int idx = tid - (block_size - order);
  if (idx >= 0) {
    fullcarry[chunk_id * order + idx] = val0;
    __threadfence();
    if (idx == 0) {
      status[chunk_id] = 2;
    }
  }

  if (chunk_id > 0) {
    __syncthreads();
    if (warp == 0) {
      const int cidm1 = chunk_id - 1;
      int flag = 1;
      do {
        if ((cidm1 - lane) >= 0) {
          flag = status[cidm1 - lane];
        }
      } while ((__any(flag == 0)) || (__all(flag != 2)));
      int mask = __ballot(flag == 2);
      const int pos = __ffs(mask) - 1;
      T X0 = fullcarry[cidm1 - pos];
      if (lane == 0) {
        sfullc[0] = X0;
      }
    }

    __syncthreads();
    T X0 = sfullc[0];
    val0 += sfacA[tid] * X0;
  }

  if (chunk_id == gridDim.x - 1) {
    if (offs + (0 * block_size) < items) output[offs + (0 * block_size)] = val0;
  } else {
    output[offs + (0 * block_size)] = val0;
  }
}

static __global__ __launch_bounds__(block_size, 2)
void Recurrence2(const int items, const T* const __restrict__ input, T* const __restrict__ output, volatile int* const __restrict__ status, volatile T* const __restrict__ partcarry, volatile T* const __restrict__ fullcarry)
{
  const int valsperthread = 2;
  const int chunk_size = valsperthread * block_size;
  __shared__ T spartc[chunk_size / warp_size * order];
  __shared__ T sfullc[order];
  __shared__ int cid;

  const int tid = threadIdx.x;
  const int warp = tid / warp_size;
  const int lane = tid % warp_size;

  __shared__ T sfacA[block_size];
  if (tid < 264) sfacA[tid] = facA[tid];
  else sfacA[tid] = 0;

  if (tid == 0) {
    cid = atomicInc(&counter, gridDim.x - 1);
  }

  __syncthreads();
  const int chunk_id = cid;
  const int offs = tid + chunk_id * chunk_size;

  T val0, val1;
  if (chunk_id == gridDim.x - 1) {
    val0 = 0;
    if (offs + (0 * block_size) < items) val0 = input[offs + (0 * block_size)];
    val1 = 0;
    if (offs + (1 * block_size) < items) val1 = input[offs + (1 * block_size)];
  } else {
    val0 = input[offs + (0 * block_size)];
    val1 = input[offs + (1 * block_size)];
  }

  val0 *= 2.809900e-01f;
  val1 *= 2.809900e-01f;

  const T sfA = sfacA[lane];

  int cond;
  T help, spc;

  help = 7.190100e-01f;
  cond = ((lane & 1) != 0);
  spc = help * __shfl(val0, 0, 2);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 2);
  if (cond) val1 += spc;

  help = __shfl(sfA, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 1, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 1, 4);
  if (cond) val1 += spc;

  help = __shfl(sfA, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 3, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 3, 8);
  if (cond) val1 += spc;

  help = __shfl(sfA, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 7, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 7, 16);
  if (cond) val1 += spc;

  help = __shfl(sfA, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 15, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 15, 32);
  if (cond) val1 += spc;

  const int delta = block_size / warp_size * order;
  const int clane = lane - (warp_size - order);
  const int clwo = clane + warp * order;

  if (((warp & 1) == 0) && (clane >= 0)) {
    spartc[clwo + 0 * delta] = val0;
    spartc[clwo + 1 * delta] = val1;
  }

  __syncthreads();
  if ((warp & 1) != 0) {
    const int cwarp = ((warp & ~1) | 0) * order;
    const T helpA = sfacA[tid % (warp_size * 1)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    if (((warp & 3) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
    }
  }

  __syncthreads();
  if ((warp & 2) != 0) {
    const int cwarp = ((warp & ~3) | 1) * order;
    const T helpA = sfacA[tid % (warp_size * 2)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    if (((warp & 7) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
    }
  }

  __syncthreads();
  if ((warp & 4) != 0) {
    const int cwarp = ((warp & ~7) | 3) * order;
    const T helpA = sfacA[tid % (warp_size * 4)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    if (((warp & 15) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
    }
  }

  __syncthreads();
  if ((warp & 8) != 0) {
    const int cwarp = ((warp & ~15) | 7) * order;
    const T helpA = sfacA[tid % (warp_size * 8)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    if ((warp == 15) && (clane >= 0)) {
      spartc[clane + (15 * order + 0 * delta)] = val0;
      spartc[clane + (15 * order + 1 * delta)] = val1;
    }
  }

  __syncthreads();
  if ((warp & 16) != 0) {
   if ((warp & 15) < 9) {
    const int cwarp = 15 * order;
    const T helpA = sfacA[tid % (warp_size * 16)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
   }
    if ((warp == 31) && (clane >= 0)) {
      spartc[clane + (31 * order + 0 * delta)] = val0;
    }
  }

  __syncthreads();
 if (warp < 9) {
  val1 += sfacA[tid] * spartc[31 * order + (0 * delta + 0)];
 }

  const int idx = tid - (block_size - order);
  if (idx >= 0) {
    fullcarry[chunk_id * order + idx] = val1;
    __threadfence();
    if (idx == 0) {
      status[chunk_id] = 2;
    }
  }

  if (chunk_id > 0) {
    __syncthreads();
    if (warp == 0) {
      const int cidm1 = chunk_id - 1;
      int flag = 1;
      do {
        if ((cidm1 - lane) >= 0) {
          flag = status[cidm1 - lane];
        }
      } while ((__any(flag == 0)) || (__all(flag != 2)));
      int mask = __ballot(flag == 2);
      const int pos = __ffs(mask) - 1;
      T X0 = fullcarry[cidm1 - pos];
      if (lane == 0) {
        sfullc[0] = X0;
      }
    }

    __syncthreads();
    T X0 = sfullc[0];
    val0 += sfacA[tid] * X0;
  }

  if (chunk_id == gridDim.x - 1) {
    if (offs + (0 * block_size) < items) output[offs + (0 * block_size)] = val0;
    if (offs + (1 * block_size) < items) output[offs + (1 * block_size)] = val1;
  } else {
    output[offs + (0 * block_size)] = val0;
    output[offs + (1 * block_size)] = val1;
  }
}

static __global__ __launch_bounds__(block_size, 2)
void Recurrence3(const int items, const T* const __restrict__ input, T* const __restrict__ output, volatile int* const __restrict__ status, volatile T* const __restrict__ partcarry, volatile T* const __restrict__ fullcarry)
{
  const int valsperthread = 3;
  const int chunk_size = valsperthread * block_size;
  __shared__ T spartc[chunk_size / warp_size * order];
  __shared__ T sfullc[order];
  __shared__ int cid;

  const int tid = threadIdx.x;
  const int warp = tid / warp_size;
  const int lane = tid % warp_size;

  __shared__ T sfacA[block_size];
  if (tid < 264) sfacA[tid] = facA[tid];
  else sfacA[tid] = 0;

  if (tid == 0) {
    cid = atomicInc(&counter, gridDim.x - 1);
  }

  __syncthreads();
  const int chunk_id = cid;
  const int offs = tid + chunk_id * chunk_size;

  T val0, val1, val2;
  if (chunk_id == gridDim.x - 1) {
    val0 = 0;
    if (offs + (0 * block_size) < items) val0 = input[offs + (0 * block_size)];
    val1 = 0;
    if (offs + (1 * block_size) < items) val1 = input[offs + (1 * block_size)];
    val2 = 0;
    if (offs + (2 * block_size) < items) val2 = input[offs + (2 * block_size)];
  } else {
    val0 = input[offs + (0 * block_size)];
    val1 = input[offs + (1 * block_size)];
    val2 = input[offs + (2 * block_size)];
  }

  val0 *= 2.809900e-01f;
  val1 *= 2.809900e-01f;
  val2 *= 2.809900e-01f;

  const T sfA = sfacA[lane];

  int cond;
  T help, spc;

  help = 7.190100e-01f;
  cond = ((lane & 1) != 0);
  spc = help * __shfl(val0, 0, 2);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 2);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 2);
  if (cond) val2 += spc;

  help = __shfl(sfA, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 1, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 1, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 1, 4);
  if (cond) val2 += spc;

  help = __shfl(sfA, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 3, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 3, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 3, 8);
  if (cond) val2 += spc;

  help = __shfl(sfA, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 7, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 7, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 7, 16);
  if (cond) val2 += spc;

  help = __shfl(sfA, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 15, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 15, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 15, 32);
  if (cond) val2 += spc;

  const int delta = block_size / warp_size * order;
  const int clane = lane - (warp_size - order);
  const int clwo = clane + warp * order;

  if (((warp & 1) == 0) && (clane >= 0)) {
    spartc[clwo + 0 * delta] = val0;
    spartc[clwo + 1 * delta] = val1;
    spartc[clwo + 2 * delta] = val2;
  }

  __syncthreads();
  if ((warp & 1) != 0) {
    const int cwarp = ((warp & ~1) | 0) * order;
    const T helpA = sfacA[tid % (warp_size * 1)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    if (((warp & 3) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
    }
  }

  __syncthreads();
  if ((warp & 2) != 0) {
    const int cwarp = ((warp & ~3) | 1) * order;
    const T helpA = sfacA[tid % (warp_size * 2)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    if (((warp & 7) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
    }
  }

  __syncthreads();
  if ((warp & 4) != 0) {
    const int cwarp = ((warp & ~7) | 3) * order;
    const T helpA = sfacA[tid % (warp_size * 4)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    if (((warp & 15) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
    }
  }

  __syncthreads();
  if ((warp & 8) != 0) {
    const int cwarp = ((warp & ~15) | 7) * order;
    const T helpA = sfacA[tid % (warp_size * 8)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    if ((warp == 15) && (clane >= 0)) {
      spartc[clane + (15 * order + 0 * delta)] = val0;
      spartc[clane + (15 * order + 1 * delta)] = val1;
      spartc[clane + (15 * order + 2 * delta)] = val2;
    }
  }

  __syncthreads();
  if ((warp & 16) != 0) {
   if ((warp & 15) < 9) {
    const int cwarp = 15 * order;
    const T helpA = sfacA[tid % (warp_size * 16)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
   }
    if ((warp == 31) && (clane >= 0)) {
      spartc[clane + (31 * order + 0 * delta)] = val0;
      spartc[clane + (31 * order + 2 * delta)] = val2;
    }
  }

  __syncthreads();
 if (warp < 9) {
  val1 += sfacA[tid] * spartc[31 * order + (0 * delta + 0)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 1 * delta)] = val1;
  }

  __syncthreads();
 if (warp < 9) {
  val2 += sfacA[tid] * spartc[31 * order + (1 * delta + 0)];
 }

  const int idx = tid - (block_size - order);
  if (idx >= 0) {
    fullcarry[chunk_id * order + idx] = val2;
    __threadfence();
    if (idx == 0) {
      status[chunk_id] = 2;
    }
  }

  if (chunk_id > 0) {
    __syncthreads();
    if (warp == 0) {
      const int cidm1 = chunk_id - 1;
      int flag = 1;
      do {
        if ((cidm1 - lane) >= 0) {
          flag = status[cidm1 - lane];
        }
      } while ((__any(flag == 0)) || (__all(flag != 2)));
      int mask = __ballot(flag == 2);
      const int pos = __ffs(mask) - 1;
      T X0 = fullcarry[cidm1 - pos];
      if (lane == 0) {
        sfullc[0] = X0;
      }
    }

    __syncthreads();
    T X0 = sfullc[0];
    val0 += sfacA[tid] * X0;
  }

  if (chunk_id == gridDim.x - 1) {
    if (offs + (0 * block_size) < items) output[offs + (0 * block_size)] = val0;
    if (offs + (1 * block_size) < items) output[offs + (1 * block_size)] = val1;
    if (offs + (2 * block_size) < items) output[offs + (2 * block_size)] = val2;
  } else {
    output[offs + (0 * block_size)] = val0;
    output[offs + (1 * block_size)] = val1;
    output[offs + (2 * block_size)] = val2;
  }
}

static __global__ __launch_bounds__(block_size, 2)
void Recurrence4(const int items, const T* const __restrict__ input, T* const __restrict__ output, volatile int* const __restrict__ status, volatile T* const __restrict__ partcarry, volatile T* const __restrict__ fullcarry)
{
  const int valsperthread = 4;
  const int chunk_size = valsperthread * block_size;
  __shared__ T spartc[chunk_size / warp_size * order];
  __shared__ T sfullc[order];
  __shared__ int cid;

  const int tid = threadIdx.x;
  const int warp = tid / warp_size;
  const int lane = tid % warp_size;

  __shared__ T sfacA[block_size];
  if (tid < 264) sfacA[tid] = facA[tid];
  else sfacA[tid] = 0;

  if (tid == 0) {
    cid = atomicInc(&counter, gridDim.x - 1);
  }

  __syncthreads();
  const int chunk_id = cid;
  const int offs = tid + chunk_id * chunk_size;

  T val0, val1, val2, val3;
  if (chunk_id == gridDim.x - 1) {
    val0 = 0;
    if (offs + (0 * block_size) < items) val0 = input[offs + (0 * block_size)];
    val1 = 0;
    if (offs + (1 * block_size) < items) val1 = input[offs + (1 * block_size)];
    val2 = 0;
    if (offs + (2 * block_size) < items) val2 = input[offs + (2 * block_size)];
    val3 = 0;
    if (offs + (3 * block_size) < items) val3 = input[offs + (3 * block_size)];
  } else {
    val0 = input[offs + (0 * block_size)];
    val1 = input[offs + (1 * block_size)];
    val2 = input[offs + (2 * block_size)];
    val3 = input[offs + (3 * block_size)];
  }

  val0 *= 2.809900e-01f;
  val1 *= 2.809900e-01f;
  val2 *= 2.809900e-01f;
  val3 *= 2.809900e-01f;

  const T sfA = sfacA[lane];

  int cond;
  T help, spc;

  help = 7.190100e-01f;
  cond = ((lane & 1) != 0);
  spc = help * __shfl(val0, 0, 2);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 2);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 2);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 0, 2);
  if (cond) val3 += spc;

  help = __shfl(sfA, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 1, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 1, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 1, 4);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 1, 4);
  if (cond) val3 += spc;

  help = __shfl(sfA, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 3, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 3, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 3, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 3, 8);
  if (cond) val3 += spc;

  help = __shfl(sfA, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 7, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 7, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 7, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 7, 16);
  if (cond) val3 += spc;

  help = __shfl(sfA, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 15, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 15, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 15, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 15, 32);
  if (cond) val3 += spc;

  const int delta = block_size / warp_size * order;
  const int clane = lane - (warp_size - order);
  const int clwo = clane + warp * order;

  if (((warp & 1) == 0) && (clane >= 0)) {
    spartc[clwo + 0 * delta] = val0;
    spartc[clwo + 1 * delta] = val1;
    spartc[clwo + 2 * delta] = val2;
    spartc[clwo + 3 * delta] = val3;
  }

  __syncthreads();
  if ((warp & 1) != 0) {
    const int cwarp = ((warp & ~1) | 0) * order;
    const T helpA = sfacA[tid % (warp_size * 1)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    if (((warp & 3) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
    }
  }

  __syncthreads();
  if ((warp & 2) != 0) {
    const int cwarp = ((warp & ~3) | 1) * order;
    const T helpA = sfacA[tid % (warp_size * 2)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    if (((warp & 7) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
    }
  }

  __syncthreads();
  if ((warp & 4) != 0) {
    const int cwarp = ((warp & ~7) | 3) * order;
    const T helpA = sfacA[tid % (warp_size * 4)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    if (((warp & 15) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
    }
  }

  __syncthreads();
  if ((warp & 8) != 0) {
    const int cwarp = ((warp & ~15) | 7) * order;
    const T helpA = sfacA[tid % (warp_size * 8)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    if ((warp == 15) && (clane >= 0)) {
      spartc[clane + (15 * order + 0 * delta)] = val0;
      spartc[clane + (15 * order + 1 * delta)] = val1;
      spartc[clane + (15 * order + 2 * delta)] = val2;
      spartc[clane + (15 * order + 3 * delta)] = val3;
    }
  }

  __syncthreads();
  if ((warp & 16) != 0) {
   if ((warp & 15) < 9) {
    const int cwarp = 15 * order;
    const T helpA = sfacA[tid % (warp_size * 16)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
   }
    if ((warp == 31) && (clane >= 0)) {
      spartc[clane + (31 * order + 0 * delta)] = val0;
      spartc[clane + (31 * order + 2 * delta)] = val2;
    }
  }

  __syncthreads();
 if (warp < 9) {
  val1 += sfacA[tid] * spartc[31 * order + (0 * delta + 0)];
  val3 += sfacA[tid] * spartc[31 * order + (2 * delta + 0)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 1 * delta)] = val1;
  }

  __syncthreads();
 if (warp < 9) {
  val2 += sfacA[tid] * spartc[31 * order + (1 * delta + 0)];
 }

  const int idx = tid - (block_size - order);
  if (idx >= 0) {
    fullcarry[chunk_id * order + idx] = val3;
    __threadfence();
    if (idx == 0) {
      status[chunk_id] = 2;
    }
  }

  if (chunk_id > 0) {
    __syncthreads();
    if (warp == 0) {
      const int cidm1 = chunk_id - 1;
      int flag = 1;
      do {
        if ((cidm1 - lane) >= 0) {
          flag = status[cidm1 - lane];
        }
      } while ((__any(flag == 0)) || (__all(flag != 2)));
      int mask = __ballot(flag == 2);
      const int pos = __ffs(mask) - 1;
      T X0 = fullcarry[cidm1 - pos];
      if (lane == 0) {
        sfullc[0] = X0;
      }
    }

    __syncthreads();
    T X0 = sfullc[0];
    val0 += sfacA[tid] * X0;
  }

  if (chunk_id == gridDim.x - 1) {
    if (offs + (0 * block_size) < items) output[offs + (0 * block_size)] = val0;
    if (offs + (1 * block_size) < items) output[offs + (1 * block_size)] = val1;
    if (offs + (2 * block_size) < items) output[offs + (2 * block_size)] = val2;
    if (offs + (3 * block_size) < items) output[offs + (3 * block_size)] = val3;
  } else {
    output[offs + (0 * block_size)] = val0;
    output[offs + (1 * block_size)] = val1;
    output[offs + (2 * block_size)] = val2;
    output[offs + (3 * block_size)] = val3;
  }
}

static __global__ __launch_bounds__(block_size, 2)
void Recurrence5(const int items, const T* const __restrict__ input, T* const __restrict__ output, volatile int* const __restrict__ status, volatile T* const __restrict__ partcarry, volatile T* const __restrict__ fullcarry)
{
  const int valsperthread = 5;
  const int chunk_size = valsperthread * block_size;
  __shared__ T spartc[chunk_size / warp_size * order];
  __shared__ T sfullc[order];
  __shared__ int cid;

  const int tid = threadIdx.x;
  const int warp = tid / warp_size;
  const int lane = tid % warp_size;

  __shared__ T sfacA[block_size];
  if (tid < 264) sfacA[tid] = facA[tid];
  else sfacA[tid] = 0;

  if (tid == 0) {
    cid = atomicInc(&counter, gridDim.x - 1);
  }

  __syncthreads();
  const int chunk_id = cid;
  const int offs = tid + chunk_id * chunk_size;

  T val0, val1, val2, val3, val4;
  if (chunk_id == gridDim.x - 1) {
    val0 = 0;
    if (offs + (0 * block_size) < items) val0 = input[offs + (0 * block_size)];
    val1 = 0;
    if (offs + (1 * block_size) < items) val1 = input[offs + (1 * block_size)];
    val2 = 0;
    if (offs + (2 * block_size) < items) val2 = input[offs + (2 * block_size)];
    val3 = 0;
    if (offs + (3 * block_size) < items) val3 = input[offs + (3 * block_size)];
    val4 = 0;
    if (offs + (4 * block_size) < items) val4 = input[offs + (4 * block_size)];
  } else {
    val0 = input[offs + (0 * block_size)];
    val1 = input[offs + (1 * block_size)];
    val2 = input[offs + (2 * block_size)];
    val3 = input[offs + (3 * block_size)];
    val4 = input[offs + (4 * block_size)];
  }

  val0 *= 2.809900e-01f;
  val1 *= 2.809900e-01f;
  val2 *= 2.809900e-01f;
  val3 *= 2.809900e-01f;
  val4 *= 2.809900e-01f;

  const T sfA = sfacA[lane];

  int cond;
  T help, spc;

  help = 7.190100e-01f;
  cond = ((lane & 1) != 0);
  spc = help * __shfl(val0, 0, 2);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 2);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 2);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 0, 2);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 0, 2);
  if (cond) val4 += spc;

  help = __shfl(sfA, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 1, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 1, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 1, 4);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 1, 4);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 1, 4);
  if (cond) val4 += spc;

  help = __shfl(sfA, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 3, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 3, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 3, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 3, 8);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 3, 8);
  if (cond) val4 += spc;

  help = __shfl(sfA, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 7, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 7, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 7, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 7, 16);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 7, 16);
  if (cond) val4 += spc;

  help = __shfl(sfA, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 15, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 15, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 15, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 15, 32);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 15, 32);
  if (cond) val4 += spc;

  const int delta = block_size / warp_size * order;
  const int clane = lane - (warp_size - order);
  const int clwo = clane + warp * order;

  if (((warp & 1) == 0) && (clane >= 0)) {
    spartc[clwo + 0 * delta] = val0;
    spartc[clwo + 1 * delta] = val1;
    spartc[clwo + 2 * delta] = val2;
    spartc[clwo + 3 * delta] = val3;
    spartc[clwo + 4 * delta] = val4;
  }

  __syncthreads();
  if ((warp & 1) != 0) {
    const int cwarp = ((warp & ~1) | 0) * order;
    const T helpA = sfacA[tid % (warp_size * 1)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    if (((warp & 3) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
    }
  }

  __syncthreads();
  if ((warp & 2) != 0) {
    const int cwarp = ((warp & ~3) | 1) * order;
    const T helpA = sfacA[tid % (warp_size * 2)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    if (((warp & 7) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
    }
  }

  __syncthreads();
  if ((warp & 4) != 0) {
    const int cwarp = ((warp & ~7) | 3) * order;
    const T helpA = sfacA[tid % (warp_size * 4)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    if (((warp & 15) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
    }
  }

  __syncthreads();
  if ((warp & 8) != 0) {
    const int cwarp = ((warp & ~15) | 7) * order;
    const T helpA = sfacA[tid % (warp_size * 8)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    if ((warp == 15) && (clane >= 0)) {
      spartc[clane + (15 * order + 0 * delta)] = val0;
      spartc[clane + (15 * order + 1 * delta)] = val1;
      spartc[clane + (15 * order + 2 * delta)] = val2;
      spartc[clane + (15 * order + 3 * delta)] = val3;
      spartc[clane + (15 * order + 4 * delta)] = val4;
    }
  }

  __syncthreads();
  if ((warp & 16) != 0) {
   if ((warp & 15) < 9) {
    const int cwarp = 15 * order;
    const T helpA = sfacA[tid % (warp_size * 16)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
   }
    if ((warp == 31) && (clane >= 0)) {
      spartc[clane + (31 * order + 0 * delta)] = val0;
      spartc[clane + (31 * order + 2 * delta)] = val2;
      spartc[clane + (31 * order + 4 * delta)] = val4;
    }
  }

  __syncthreads();
 if (warp < 9) {
  val1 += sfacA[tid] * spartc[31 * order + (0 * delta + 0)];
  val3 += sfacA[tid] * spartc[31 * order + (2 * delta + 0)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 1 * delta)] = val1;
  }

  __syncthreads();
 if (warp < 9) {
  val2 += sfacA[tid] * spartc[31 * order + (1 * delta + 0)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 3 * delta)] = val3;
  }

  __syncthreads();
 if (warp < 9) {
  val4 += sfacA[tid] * spartc[31 * order + (3 * delta + 0)];
 }

  const int idx = tid - (block_size - order);
  if (idx >= 0) {
    fullcarry[chunk_id * order + idx] = val4;
    __threadfence();
    if (idx == 0) {
      status[chunk_id] = 2;
    }
  }

  if (chunk_id > 0) {
    __syncthreads();
    if (warp == 0) {
      const int cidm1 = chunk_id - 1;
      int flag = 1;
      do {
        if ((cidm1 - lane) >= 0) {
          flag = status[cidm1 - lane];
        }
      } while ((__any(flag == 0)) || (__all(flag != 2)));
      int mask = __ballot(flag == 2);
      const int pos = __ffs(mask) - 1;
      T X0 = fullcarry[cidm1 - pos];
      if (lane == 0) {
        sfullc[0] = X0;
      }
    }

    __syncthreads();
    T X0 = sfullc[0];
    val0 += sfacA[tid] * X0;
  }

  if (chunk_id == gridDim.x - 1) {
    if (offs + (0 * block_size) < items) output[offs + (0 * block_size)] = val0;
    if (offs + (1 * block_size) < items) output[offs + (1 * block_size)] = val1;
    if (offs + (2 * block_size) < items) output[offs + (2 * block_size)] = val2;
    if (offs + (3 * block_size) < items) output[offs + (3 * block_size)] = val3;
    if (offs + (4 * block_size) < items) output[offs + (4 * block_size)] = val4;
  } else {
    output[offs + (0 * block_size)] = val0;
    output[offs + (1 * block_size)] = val1;
    output[offs + (2 * block_size)] = val2;
    output[offs + (3 * block_size)] = val3;
    output[offs + (4 * block_size)] = val4;
  }
}

static __global__ __launch_bounds__(block_size, 2)
void Recurrence6(const int items, const T* const __restrict__ input, T* const __restrict__ output, volatile int* const __restrict__ status, volatile T* const __restrict__ partcarry, volatile T* const __restrict__ fullcarry)
{
  const int valsperthread = 6;
  const int chunk_size = valsperthread * block_size;
  __shared__ T spartc[chunk_size / warp_size * order];
  __shared__ T sfullc[order];
  __shared__ int cid;

  const int tid = threadIdx.x;
  const int warp = tid / warp_size;
  const int lane = tid % warp_size;

  __shared__ T sfacA[block_size];
  if (tid < 264) sfacA[tid] = facA[tid];
  else sfacA[tid] = 0;

  if (tid == 0) {
    cid = atomicInc(&counter, gridDim.x - 1);
  }

  __syncthreads();
  const int chunk_id = cid;
  const int offs = tid + chunk_id * chunk_size;

  T val0, val1, val2, val3, val4, val5;
  if (chunk_id == gridDim.x - 1) {
    val0 = 0;
    if (offs + (0 * block_size) < items) val0 = input[offs + (0 * block_size)];
    val1 = 0;
    if (offs + (1 * block_size) < items) val1 = input[offs + (1 * block_size)];
    val2 = 0;
    if (offs + (2 * block_size) < items) val2 = input[offs + (2 * block_size)];
    val3 = 0;
    if (offs + (3 * block_size) < items) val3 = input[offs + (3 * block_size)];
    val4 = 0;
    if (offs + (4 * block_size) < items) val4 = input[offs + (4 * block_size)];
    val5 = 0;
    if (offs + (5 * block_size) < items) val5 = input[offs + (5 * block_size)];
  } else {
    val0 = input[offs + (0 * block_size)];
    val1 = input[offs + (1 * block_size)];
    val2 = input[offs + (2 * block_size)];
    val3 = input[offs + (3 * block_size)];
    val4 = input[offs + (4 * block_size)];
    val5 = input[offs + (5 * block_size)];
  }

  val0 *= 2.809900e-01f;
  val1 *= 2.809900e-01f;
  val2 *= 2.809900e-01f;
  val3 *= 2.809900e-01f;
  val4 *= 2.809900e-01f;
  val5 *= 2.809900e-01f;

  const T sfA = sfacA[lane];

  int cond;
  T help, spc;

  help = 7.190100e-01f;
  cond = ((lane & 1) != 0);
  spc = help * __shfl(val0, 0, 2);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 2);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 2);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 0, 2);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 0, 2);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 0, 2);
  if (cond) val5 += spc;

  help = __shfl(sfA, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 1, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 1, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 1, 4);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 1, 4);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 1, 4);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 1, 4);
  if (cond) val5 += spc;

  help = __shfl(sfA, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 3, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 3, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 3, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 3, 8);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 3, 8);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 3, 8);
  if (cond) val5 += spc;

  help = __shfl(sfA, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 7, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 7, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 7, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 7, 16);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 7, 16);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 7, 16);
  if (cond) val5 += spc;

  help = __shfl(sfA, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 15, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 15, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 15, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 15, 32);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 15, 32);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 15, 32);
  if (cond) val5 += spc;

  const int delta = block_size / warp_size * order;
  const int clane = lane - (warp_size - order);
  const int clwo = clane + warp * order;

  if (((warp & 1) == 0) && (clane >= 0)) {
    spartc[clwo + 0 * delta] = val0;
    spartc[clwo + 1 * delta] = val1;
    spartc[clwo + 2 * delta] = val2;
    spartc[clwo + 3 * delta] = val3;
    spartc[clwo + 4 * delta] = val4;
    spartc[clwo + 5 * delta] = val5;
  }

  __syncthreads();
  if ((warp & 1) != 0) {
    const int cwarp = ((warp & ~1) | 0) * order;
    const T helpA = sfacA[tid % (warp_size * 1)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    if (((warp & 3) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
      spartc[clwo + 5 * delta] = val5;
    }
  }

  __syncthreads();
  if ((warp & 2) != 0) {
    const int cwarp = ((warp & ~3) | 1) * order;
    const T helpA = sfacA[tid % (warp_size * 2)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    if (((warp & 7) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
      spartc[clwo + 5 * delta] = val5;
    }
  }

  __syncthreads();
  if ((warp & 4) != 0) {
    const int cwarp = ((warp & ~7) | 3) * order;
    const T helpA = sfacA[tid % (warp_size * 4)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    if (((warp & 15) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
      spartc[clwo + 5 * delta] = val5;
    }
  }

  __syncthreads();
  if ((warp & 8) != 0) {
    const int cwarp = ((warp & ~15) | 7) * order;
    const T helpA = sfacA[tid % (warp_size * 8)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    if ((warp == 15) && (clane >= 0)) {
      spartc[clane + (15 * order + 0 * delta)] = val0;
      spartc[clane + (15 * order + 1 * delta)] = val1;
      spartc[clane + (15 * order + 2 * delta)] = val2;
      spartc[clane + (15 * order + 3 * delta)] = val3;
      spartc[clane + (15 * order + 4 * delta)] = val4;
      spartc[clane + (15 * order + 5 * delta)] = val5;
    }
  }

  __syncthreads();
  if ((warp & 16) != 0) {
   if ((warp & 15) < 9) {
    const int cwarp = 15 * order;
    const T helpA = sfacA[tid % (warp_size * 16)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
   }
    if ((warp == 31) && (clane >= 0)) {
      spartc[clane + (31 * order + 0 * delta)] = val0;
      spartc[clane + (31 * order + 2 * delta)] = val2;
      spartc[clane + (31 * order + 4 * delta)] = val4;
    }
  }

  __syncthreads();
 if (warp < 9) {
  val1 += sfacA[tid] * spartc[31 * order + (0 * delta + 0)];
  val3 += sfacA[tid] * spartc[31 * order + (2 * delta + 0)];
  val5 += sfacA[tid] * spartc[31 * order + (4 * delta + 0)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 1 * delta)] = val1;
    spartc[clane + (31 * order + 5 * delta)] = val5;
  }

  __syncthreads();
 if (warp < 9) {
  val2 += sfacA[tid] * spartc[31 * order + (1 * delta + 0)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 3 * delta)] = val3;
  }

  __syncthreads();
 if (warp < 9) {
  val4 += sfacA[tid] * spartc[31 * order + (3 * delta + 0)];
 }

  const int idx = tid - (block_size - order);
  if (idx >= 0) {
    fullcarry[chunk_id * order + idx] = val5;
    __threadfence();
    if (idx == 0) {
      status[chunk_id] = 2;
    }
  }

  if (chunk_id > 0) {
    __syncthreads();
    if (warp == 0) {
      const int cidm1 = chunk_id - 1;
      int flag = 1;
      do {
        if ((cidm1 - lane) >= 0) {
          flag = status[cidm1 - lane];
        }
      } while ((__any(flag == 0)) || (__all(flag != 2)));
      int mask = __ballot(flag == 2);
      const int pos = __ffs(mask) - 1;
      T X0 = fullcarry[cidm1 - pos];
      if (lane == 0) {
        sfullc[0] = X0;
      }
    }

    __syncthreads();
    T X0 = sfullc[0];
    val0 += sfacA[tid] * X0;
  }

  if (chunk_id == gridDim.x - 1) {
    if (offs + (0 * block_size) < items) output[offs + (0 * block_size)] = val0;
    if (offs + (1 * block_size) < items) output[offs + (1 * block_size)] = val1;
    if (offs + (2 * block_size) < items) output[offs + (2 * block_size)] = val2;
    if (offs + (3 * block_size) < items) output[offs + (3 * block_size)] = val3;
    if (offs + (4 * block_size) < items) output[offs + (4 * block_size)] = val4;
    if (offs + (5 * block_size) < items) output[offs + (5 * block_size)] = val5;
  } else {
    output[offs + (0 * block_size)] = val0;
    output[offs + (1 * block_size)] = val1;
    output[offs + (2 * block_size)] = val2;
    output[offs + (3 * block_size)] = val3;
    output[offs + (4 * block_size)] = val4;
    output[offs + (5 * block_size)] = val5;
  }
}

static __global__ __launch_bounds__(block_size, 2)
void Recurrence7(const int items, const T* const __restrict__ input, T* const __restrict__ output, volatile int* const __restrict__ status, volatile T* const __restrict__ partcarry, volatile T* const __restrict__ fullcarry)
{
  const int valsperthread = 7;
  const int chunk_size = valsperthread * block_size;
  __shared__ T spartc[chunk_size / warp_size * order];
  __shared__ T sfullc[order];
  __shared__ int cid;

  const int tid = threadIdx.x;
  const int warp = tid / warp_size;
  const int lane = tid % warp_size;

  __shared__ T sfacA[block_size];
  if (tid < 264) sfacA[tid] = facA[tid];
  else sfacA[tid] = 0;

  if (tid == 0) {
    cid = atomicInc(&counter, gridDim.x - 1);
  }

  __syncthreads();
  const int chunk_id = cid;
  const int offs = tid + chunk_id * chunk_size;

  T val0, val1, val2, val3, val4, val5, val6;
  if (chunk_id == gridDim.x - 1) {
    val0 = 0;
    if (offs + (0 * block_size) < items) val0 = input[offs + (0 * block_size)];
    val1 = 0;
    if (offs + (1 * block_size) < items) val1 = input[offs + (1 * block_size)];
    val2 = 0;
    if (offs + (2 * block_size) < items) val2 = input[offs + (2 * block_size)];
    val3 = 0;
    if (offs + (3 * block_size) < items) val3 = input[offs + (3 * block_size)];
    val4 = 0;
    if (offs + (4 * block_size) < items) val4 = input[offs + (4 * block_size)];
    val5 = 0;
    if (offs + (5 * block_size) < items) val5 = input[offs + (5 * block_size)];
    val6 = 0;
    if (offs + (6 * block_size) < items) val6 = input[offs + (6 * block_size)];
  } else {
    val0 = input[offs + (0 * block_size)];
    val1 = input[offs + (1 * block_size)];
    val2 = input[offs + (2 * block_size)];
    val3 = input[offs + (3 * block_size)];
    val4 = input[offs + (4 * block_size)];
    val5 = input[offs + (5 * block_size)];
    val6 = input[offs + (6 * block_size)];
  }

  val0 *= 2.809900e-01f;
  val1 *= 2.809900e-01f;
  val2 *= 2.809900e-01f;
  val3 *= 2.809900e-01f;
  val4 *= 2.809900e-01f;
  val5 *= 2.809900e-01f;
  val6 *= 2.809900e-01f;

  const T sfA = sfacA[lane];

  int cond;
  T help, spc;

  help = 7.190100e-01f;
  cond = ((lane & 1) != 0);
  spc = help * __shfl(val0, 0, 2);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 2);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 2);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 0, 2);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 0, 2);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 0, 2);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 0, 2);
  if (cond) val6 += spc;

  help = __shfl(sfA, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 1, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 1, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 1, 4);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 1, 4);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 1, 4);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 1, 4);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 1, 4);
  if (cond) val6 += spc;

  help = __shfl(sfA, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 3, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 3, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 3, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 3, 8);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 3, 8);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 3, 8);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 3, 8);
  if (cond) val6 += spc;

  help = __shfl(sfA, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 7, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 7, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 7, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 7, 16);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 7, 16);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 7, 16);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 7, 16);
  if (cond) val6 += spc;

  help = __shfl(sfA, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 15, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 15, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 15, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 15, 32);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 15, 32);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 15, 32);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 15, 32);
  if (cond) val6 += spc;

  const int delta = block_size / warp_size * order;
  const int clane = lane - (warp_size - order);
  const int clwo = clane + warp * order;

  if (((warp & 1) == 0) && (clane >= 0)) {
    spartc[clwo + 0 * delta] = val0;
    spartc[clwo + 1 * delta] = val1;
    spartc[clwo + 2 * delta] = val2;
    spartc[clwo + 3 * delta] = val3;
    spartc[clwo + 4 * delta] = val4;
    spartc[clwo + 5 * delta] = val5;
    spartc[clwo + 6 * delta] = val6;
  }

  __syncthreads();
  if ((warp & 1) != 0) {
    const int cwarp = ((warp & ~1) | 0) * order;
    const T helpA = sfacA[tid % (warp_size * 1)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    if (((warp & 3) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
      spartc[clwo + 5 * delta] = val5;
      spartc[clwo + 6 * delta] = val6;
    }
  }

  __syncthreads();
  if ((warp & 2) != 0) {
    const int cwarp = ((warp & ~3) | 1) * order;
    const T helpA = sfacA[tid % (warp_size * 2)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    if (((warp & 7) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
      spartc[clwo + 5 * delta] = val5;
      spartc[clwo + 6 * delta] = val6;
    }
  }

  __syncthreads();
  if ((warp & 4) != 0) {
    const int cwarp = ((warp & ~7) | 3) * order;
    const T helpA = sfacA[tid % (warp_size * 4)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    if (((warp & 15) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
      spartc[clwo + 5 * delta] = val5;
      spartc[clwo + 6 * delta] = val6;
    }
  }

  __syncthreads();
  if ((warp & 8) != 0) {
    const int cwarp = ((warp & ~15) | 7) * order;
    const T helpA = sfacA[tid % (warp_size * 8)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    if ((warp == 15) && (clane >= 0)) {
      spartc[clane + (15 * order + 0 * delta)] = val0;
      spartc[clane + (15 * order + 1 * delta)] = val1;
      spartc[clane + (15 * order + 2 * delta)] = val2;
      spartc[clane + (15 * order + 3 * delta)] = val3;
      spartc[clane + (15 * order + 4 * delta)] = val4;
      spartc[clane + (15 * order + 5 * delta)] = val5;
      spartc[clane + (15 * order + 6 * delta)] = val6;
    }
  }

  __syncthreads();
  if ((warp & 16) != 0) {
   if ((warp & 15) < 9) {
    const int cwarp = 15 * order;
    const T helpA = sfacA[tid % (warp_size * 16)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
   }
    if ((warp == 31) && (clane >= 0)) {
      spartc[clane + (31 * order + 0 * delta)] = val0;
      spartc[clane + (31 * order + 2 * delta)] = val2;
      spartc[clane + (31 * order + 4 * delta)] = val4;
      spartc[clane + (31 * order + 6 * delta)] = val6;
    }
  }

  __syncthreads();
 if (warp < 9) {
  val1 += sfacA[tid] * spartc[31 * order + (0 * delta + 0)];
  val3 += sfacA[tid] * spartc[31 * order + (2 * delta + 0)];
  val5 += sfacA[tid] * spartc[31 * order + (4 * delta + 0)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 1 * delta)] = val1;
    spartc[clane + (31 * order + 5 * delta)] = val5;
  }

  __syncthreads();
 if (warp < 9) {
  val2 += sfacA[tid] * spartc[31 * order + (1 * delta + 0)];
  val6 += sfacA[tid] * spartc[31 * order + (5 * delta + 0)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 3 * delta)] = val3;
  }

  __syncthreads();
 if (warp < 9) {
  val4 += sfacA[tid] * spartc[31 * order + (3 * delta + 0)];
 }

  const int idx = tid - (block_size - order);
  if (idx >= 0) {
    fullcarry[chunk_id * order + idx] = val6;
    __threadfence();
    if (idx == 0) {
      status[chunk_id] = 2;
    }
  }

  if (chunk_id > 0) {
    __syncthreads();
    if (warp == 0) {
      const int cidm1 = chunk_id - 1;
      int flag = 1;
      do {
        if ((cidm1 - lane) >= 0) {
          flag = status[cidm1 - lane];
        }
      } while ((__any(flag == 0)) || (__all(flag != 2)));
      int mask = __ballot(flag == 2);
      const int pos = __ffs(mask) - 1;
      T X0 = fullcarry[cidm1 - pos];
      if (lane == 0) {
        sfullc[0] = X0;
      }
    }

    __syncthreads();
    T X0 = sfullc[0];
    val0 += sfacA[tid] * X0;
  }

  if (chunk_id == gridDim.x - 1) {
    if (offs + (0 * block_size) < items) output[offs + (0 * block_size)] = val0;
    if (offs + (1 * block_size) < items) output[offs + (1 * block_size)] = val1;
    if (offs + (2 * block_size) < items) output[offs + (2 * block_size)] = val2;
    if (offs + (3 * block_size) < items) output[offs + (3 * block_size)] = val3;
    if (offs + (4 * block_size) < items) output[offs + (4 * block_size)] = val4;
    if (offs + (5 * block_size) < items) output[offs + (5 * block_size)] = val5;
    if (offs + (6 * block_size) < items) output[offs + (6 * block_size)] = val6;
  } else {
    output[offs + (0 * block_size)] = val0;
    output[offs + (1 * block_size)] = val1;
    output[offs + (2 * block_size)] = val2;
    output[offs + (3 * block_size)] = val3;
    output[offs + (4 * block_size)] = val4;
    output[offs + (5 * block_size)] = val5;
    output[offs + (6 * block_size)] = val6;
  }
}

static __global__ __launch_bounds__(block_size, 2)
void Recurrence8(const int items, const T* const __restrict__ input, T* const __restrict__ output, volatile int* const __restrict__ status, volatile T* const __restrict__ partcarry, volatile T* const __restrict__ fullcarry)
{
  const int valsperthread = 8;
  const int chunk_size = valsperthread * block_size;
  __shared__ T spartc[chunk_size / warp_size * order];
  __shared__ T sfullc[order];
  __shared__ int cid;

  const int tid = threadIdx.x;
  const int warp = tid / warp_size;
  const int lane = tid % warp_size;

  __shared__ T sfacA[block_size];
  if (tid < 264) sfacA[tid] = facA[tid];
  else sfacA[tid] = 0;

  if (tid == 0) {
    cid = atomicInc(&counter, gridDim.x - 1);
  }

  __syncthreads();
  const int chunk_id = cid;
  const int offs = tid + chunk_id * chunk_size;

  T val0, val1, val2, val3, val4, val5, val6, val7;
  if (chunk_id == gridDim.x - 1) {
    val0 = 0;
    if (offs + (0 * block_size) < items) val0 = input[offs + (0 * block_size)];
    val1 = 0;
    if (offs + (1 * block_size) < items) val1 = input[offs + (1 * block_size)];
    val2 = 0;
    if (offs + (2 * block_size) < items) val2 = input[offs + (2 * block_size)];
    val3 = 0;
    if (offs + (3 * block_size) < items) val3 = input[offs + (3 * block_size)];
    val4 = 0;
    if (offs + (4 * block_size) < items) val4 = input[offs + (4 * block_size)];
    val5 = 0;
    if (offs + (5 * block_size) < items) val5 = input[offs + (5 * block_size)];
    val6 = 0;
    if (offs + (6 * block_size) < items) val6 = input[offs + (6 * block_size)];
    val7 = 0;
    if (offs + (7 * block_size) < items) val7 = input[offs + (7 * block_size)];
  } else {
    val0 = input[offs + (0 * block_size)];
    val1 = input[offs + (1 * block_size)];
    val2 = input[offs + (2 * block_size)];
    val3 = input[offs + (3 * block_size)];
    val4 = input[offs + (4 * block_size)];
    val5 = input[offs + (5 * block_size)];
    val6 = input[offs + (6 * block_size)];
    val7 = input[offs + (7 * block_size)];
  }

  val0 *= 2.809900e-01f;
  val1 *= 2.809900e-01f;
  val2 *= 2.809900e-01f;
  val3 *= 2.809900e-01f;
  val4 *= 2.809900e-01f;
  val5 *= 2.809900e-01f;
  val6 *= 2.809900e-01f;
  val7 *= 2.809900e-01f;

  const T sfA = sfacA[lane];

  int cond;
  T help, spc;

  help = 7.190100e-01f;
  cond = ((lane & 1) != 0);
  spc = help * __shfl(val0, 0, 2);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 2);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 2);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 0, 2);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 0, 2);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 0, 2);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 0, 2);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 0, 2);
  if (cond) val7 += spc;

  help = __shfl(sfA, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 1, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 1, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 1, 4);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 1, 4);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 1, 4);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 1, 4);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 1, 4);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 1, 4);
  if (cond) val7 += spc;

  help = __shfl(sfA, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 3, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 3, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 3, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 3, 8);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 3, 8);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 3, 8);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 3, 8);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 3, 8);
  if (cond) val7 += spc;

  help = __shfl(sfA, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 7, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 7, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 7, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 7, 16);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 7, 16);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 7, 16);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 7, 16);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 7, 16);
  if (cond) val7 += spc;

  help = __shfl(sfA, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 15, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 15, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 15, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 15, 32);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 15, 32);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 15, 32);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 15, 32);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 15, 32);
  if (cond) val7 += spc;

  const int delta = block_size / warp_size * order;
  const int clane = lane - (warp_size - order);
  const int clwo = clane + warp * order;

  if (((warp & 1) == 0) && (clane >= 0)) {
    spartc[clwo + 0 * delta] = val0;
    spartc[clwo + 1 * delta] = val1;
    spartc[clwo + 2 * delta] = val2;
    spartc[clwo + 3 * delta] = val3;
    spartc[clwo + 4 * delta] = val4;
    spartc[clwo + 5 * delta] = val5;
    spartc[clwo + 6 * delta] = val6;
    spartc[clwo + 7 * delta] = val7;
  }

  __syncthreads();
  if ((warp & 1) != 0) {
    const int cwarp = ((warp & ~1) | 0) * order;
    const T helpA = sfacA[tid % (warp_size * 1)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    val7 += helpA * spartc[cwarp + (7 * delta + 0)];
    if (((warp & 3) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
      spartc[clwo + 5 * delta] = val5;
      spartc[clwo + 6 * delta] = val6;
      spartc[clwo + 7 * delta] = val7;
    }
  }

  __syncthreads();
  if ((warp & 2) != 0) {
    const int cwarp = ((warp & ~3) | 1) * order;
    const T helpA = sfacA[tid % (warp_size * 2)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    val7 += helpA * spartc[cwarp + (7 * delta + 0)];
    if (((warp & 7) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
      spartc[clwo + 5 * delta] = val5;
      spartc[clwo + 6 * delta] = val6;
      spartc[clwo + 7 * delta] = val7;
    }
  }

  __syncthreads();
  if ((warp & 4) != 0) {
    const int cwarp = ((warp & ~7) | 3) * order;
    const T helpA = sfacA[tid % (warp_size * 4)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    val7 += helpA * spartc[cwarp + (7 * delta + 0)];
    if (((warp & 15) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
      spartc[clwo + 5 * delta] = val5;
      spartc[clwo + 6 * delta] = val6;
      spartc[clwo + 7 * delta] = val7;
    }
  }

  __syncthreads();
  if ((warp & 8) != 0) {
    const int cwarp = ((warp & ~15) | 7) * order;
    const T helpA = sfacA[tid % (warp_size * 8)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    val7 += helpA * spartc[cwarp + (7 * delta + 0)];
    if ((warp == 15) && (clane >= 0)) {
      spartc[clane + (15 * order + 0 * delta)] = val0;
      spartc[clane + (15 * order + 1 * delta)] = val1;
      spartc[clane + (15 * order + 2 * delta)] = val2;
      spartc[clane + (15 * order + 3 * delta)] = val3;
      spartc[clane + (15 * order + 4 * delta)] = val4;
      spartc[clane + (15 * order + 5 * delta)] = val5;
      spartc[clane + (15 * order + 6 * delta)] = val6;
      spartc[clane + (15 * order + 7 * delta)] = val7;
    }
  }

  __syncthreads();
  if ((warp & 16) != 0) {
   if ((warp & 15) < 9) {
    const int cwarp = 15 * order;
    const T helpA = sfacA[tid % (warp_size * 16)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    val7 += helpA * spartc[cwarp + (7 * delta + 0)];
   }
    if ((warp == 31) && (clane >= 0)) {
      spartc[clane + (31 * order + 0 * delta)] = val0;
      spartc[clane + (31 * order + 2 * delta)] = val2;
      spartc[clane + (31 * order + 4 * delta)] = val4;
      spartc[clane + (31 * order + 6 * delta)] = val6;
    }
  }

  __syncthreads();
 if (warp < 9) {
  val1 += sfacA[tid] * spartc[31 * order + (0 * delta + 0)];
  val3 += sfacA[tid] * spartc[31 * order + (2 * delta + 0)];
  val5 += sfacA[tid] * spartc[31 * order + (4 * delta + 0)];
  val7 += sfacA[tid] * spartc[31 * order + (6 * delta + 0)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 1 * delta)] = val1;
    spartc[clane + (31 * order + 5 * delta)] = val5;
  }

  __syncthreads();
 if (warp < 9) {
  val2 += sfacA[tid] * spartc[31 * order + (1 * delta + 0)];
  val6 += sfacA[tid] * spartc[31 * order + (5 * delta + 0)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 3 * delta)] = val3;
  }

  __syncthreads();
 if (warp < 9) {
  val4 += sfacA[tid] * spartc[31 * order + (3 * delta + 0)];
 }

  const int idx = tid - (block_size - order);
  if (idx >= 0) {
    fullcarry[chunk_id * order + idx] = val7;
    __threadfence();
    if (idx == 0) {
      status[chunk_id] = 2;
    }
  }

  if (chunk_id > 0) {
    __syncthreads();
    if (warp == 0) {
      const int cidm1 = chunk_id - 1;
      int flag = 1;
      do {
        if ((cidm1 - lane) >= 0) {
          flag = status[cidm1 - lane];
        }
      } while ((__any(flag == 0)) || (__all(flag != 2)));
      int mask = __ballot(flag == 2);
      const int pos = __ffs(mask) - 1;
      T X0 = fullcarry[cidm1 - pos];
      if (lane == 0) {
        sfullc[0] = X0;
      }
    }

    __syncthreads();
    T X0 = sfullc[0];
    val0 += sfacA[tid] * X0;
  }

  if (chunk_id == gridDim.x - 1) {
    if (offs + (0 * block_size) < items) output[offs + (0 * block_size)] = val0;
    if (offs + (1 * block_size) < items) output[offs + (1 * block_size)] = val1;
    if (offs + (2 * block_size) < items) output[offs + (2 * block_size)] = val2;
    if (offs + (3 * block_size) < items) output[offs + (3 * block_size)] = val3;
    if (offs + (4 * block_size) < items) output[offs + (4 * block_size)] = val4;
    if (offs + (5 * block_size) < items) output[offs + (5 * block_size)] = val5;
    if (offs + (6 * block_size) < items) output[offs + (6 * block_size)] = val6;
    if (offs + (7 * block_size) < items) output[offs + (7 * block_size)] = val7;
  } else {
    output[offs + (0 * block_size)] = val0;
    output[offs + (1 * block_size)] = val1;
    output[offs + (2 * block_size)] = val2;
    output[offs + (3 * block_size)] = val3;
    output[offs + (4 * block_size)] = val4;
    output[offs + (5 * block_size)] = val5;
    output[offs + (6 * block_size)] = val6;
    output[offs + (7 * block_size)] = val7;
  }
}

static __global__ __launch_bounds__(block_size, 2)
void Recurrence9(const int items, const T* const __restrict__ input, T* const __restrict__ output, volatile int* const __restrict__ status, volatile T* const __restrict__ partcarry, volatile T* const __restrict__ fullcarry)
{
  const int valsperthread = 9;
  const int chunk_size = valsperthread * block_size;
  __shared__ T spartc[chunk_size / warp_size * order];
  __shared__ T sfullc[order];
  __shared__ int cid;

  const int tid = threadIdx.x;
  const int warp = tid / warp_size;
  const int lane = tid % warp_size;

  __shared__ T sfacA[block_size];
  if (tid < 264) sfacA[tid] = facA[tid];
  else sfacA[tid] = 0;

  if (tid == 0) {
    cid = atomicInc(&counter, gridDim.x - 1);
  }

  __syncthreads();
  const int chunk_id = cid;
  const int offs = tid + chunk_id * chunk_size;

  T val0, val1, val2, val3, val4, val5, val6, val7, val8;
  if (chunk_id == gridDim.x - 1) {
    val0 = 0;
    if (offs + (0 * block_size) < items) val0 = input[offs + (0 * block_size)];
    val1 = 0;
    if (offs + (1 * block_size) < items) val1 = input[offs + (1 * block_size)];
    val2 = 0;
    if (offs + (2 * block_size) < items) val2 = input[offs + (2 * block_size)];
    val3 = 0;
    if (offs + (3 * block_size) < items) val3 = input[offs + (3 * block_size)];
    val4 = 0;
    if (offs + (4 * block_size) < items) val4 = input[offs + (4 * block_size)];
    val5 = 0;
    if (offs + (5 * block_size) < items) val5 = input[offs + (5 * block_size)];
    val6 = 0;
    if (offs + (6 * block_size) < items) val6 = input[offs + (6 * block_size)];
    val7 = 0;
    if (offs + (7 * block_size) < items) val7 = input[offs + (7 * block_size)];
    val8 = 0;
    if (offs + (8 * block_size) < items) val8 = input[offs + (8 * block_size)];
  } else {
    val0 = input[offs + (0 * block_size)];
    val1 = input[offs + (1 * block_size)];
    val2 = input[offs + (2 * block_size)];
    val3 = input[offs + (3 * block_size)];
    val4 = input[offs + (4 * block_size)];
    val5 = input[offs + (5 * block_size)];
    val6 = input[offs + (6 * block_size)];
    val7 = input[offs + (7 * block_size)];
    val8 = input[offs + (8 * block_size)];
  }

  val0 *= 2.809900e-01f;
  val1 *= 2.809900e-01f;
  val2 *= 2.809900e-01f;
  val3 *= 2.809900e-01f;
  val4 *= 2.809900e-01f;
  val5 *= 2.809900e-01f;
  val6 *= 2.809900e-01f;
  val7 *= 2.809900e-01f;
  val8 *= 2.809900e-01f;

  const T sfA = sfacA[lane];

  int cond;
  T help, spc;

  help = 7.190100e-01f;
  cond = ((lane & 1) != 0);
  spc = help * __shfl(val0, 0, 2);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 2);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 2);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 0, 2);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 0, 2);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 0, 2);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 0, 2);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 0, 2);
  if (cond) val7 += spc;
  spc = help * __shfl(val8, 0, 2);
  if (cond) val8 += spc;

  help = __shfl(sfA, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 1, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 1, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 1, 4);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 1, 4);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 1, 4);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 1, 4);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 1, 4);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 1, 4);
  if (cond) val7 += spc;
  spc = help * __shfl(val8, 1, 4);
  if (cond) val8 += spc;

  help = __shfl(sfA, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 3, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 3, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 3, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 3, 8);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 3, 8);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 3, 8);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 3, 8);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 3, 8);
  if (cond) val7 += spc;
  spc = help * __shfl(val8, 3, 8);
  if (cond) val8 += spc;

  help = __shfl(sfA, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 7, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 7, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 7, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 7, 16);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 7, 16);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 7, 16);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 7, 16);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 7, 16);
  if (cond) val7 += spc;
  spc = help * __shfl(val8, 7, 16);
  if (cond) val8 += spc;

  help = __shfl(sfA, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 15, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 15, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 15, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 15, 32);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 15, 32);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 15, 32);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 15, 32);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 15, 32);
  if (cond) val7 += spc;
  spc = help * __shfl(val8, 15, 32);
  if (cond) val8 += spc;

  const int delta = block_size / warp_size * order;
  const int clane = lane - (warp_size - order);
  const int clwo = clane + warp * order;

  if (((warp & 1) == 0) && (clane >= 0)) {
    spartc[clwo + 0 * delta] = val0;
    spartc[clwo + 1 * delta] = val1;
    spartc[clwo + 2 * delta] = val2;
    spartc[clwo + 3 * delta] = val3;
    spartc[clwo + 4 * delta] = val4;
    spartc[clwo + 5 * delta] = val5;
    spartc[clwo + 6 * delta] = val6;
    spartc[clwo + 7 * delta] = val7;
    spartc[clwo + 8 * delta] = val8;
  }

  __syncthreads();
  if ((warp & 1) != 0) {
    const int cwarp = ((warp & ~1) | 0) * order;
    const T helpA = sfacA[tid % (warp_size * 1)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    val7 += helpA * spartc[cwarp + (7 * delta + 0)];
    val8 += helpA * spartc[cwarp + (8 * delta + 0)];
    if (((warp & 3) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
      spartc[clwo + 5 * delta] = val5;
      spartc[clwo + 6 * delta] = val6;
      spartc[clwo + 7 * delta] = val7;
      spartc[clwo + 8 * delta] = val8;
    }
  }

  __syncthreads();
  if ((warp & 2) != 0) {
    const int cwarp = ((warp & ~3) | 1) * order;
    const T helpA = sfacA[tid % (warp_size * 2)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    val7 += helpA * spartc[cwarp + (7 * delta + 0)];
    val8 += helpA * spartc[cwarp + (8 * delta + 0)];
    if (((warp & 7) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
      spartc[clwo + 5 * delta] = val5;
      spartc[clwo + 6 * delta] = val6;
      spartc[clwo + 7 * delta] = val7;
      spartc[clwo + 8 * delta] = val8;
    }
  }

  __syncthreads();
  if ((warp & 4) != 0) {
    const int cwarp = ((warp & ~7) | 3) * order;
    const T helpA = sfacA[tid % (warp_size * 4)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    val7 += helpA * spartc[cwarp + (7 * delta + 0)];
    val8 += helpA * spartc[cwarp + (8 * delta + 0)];
    if (((warp & 15) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
      spartc[clwo + 5 * delta] = val5;
      spartc[clwo + 6 * delta] = val6;
      spartc[clwo + 7 * delta] = val7;
      spartc[clwo + 8 * delta] = val8;
    }
  }

  __syncthreads();
  if ((warp & 8) != 0) {
    const int cwarp = ((warp & ~15) | 7) * order;
    const T helpA = sfacA[tid % (warp_size * 8)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    val7 += helpA * spartc[cwarp + (7 * delta + 0)];
    val8 += helpA * spartc[cwarp + (8 * delta + 0)];
    if ((warp == 15) && (clane >= 0)) {
      spartc[clane + (15 * order + 0 * delta)] = val0;
      spartc[clane + (15 * order + 1 * delta)] = val1;
      spartc[clane + (15 * order + 2 * delta)] = val2;
      spartc[clane + (15 * order + 3 * delta)] = val3;
      spartc[clane + (15 * order + 4 * delta)] = val4;
      spartc[clane + (15 * order + 5 * delta)] = val5;
      spartc[clane + (15 * order + 6 * delta)] = val6;
      spartc[clane + (15 * order + 7 * delta)] = val7;
      spartc[clane + (15 * order + 8 * delta)] = val8;
    }
  }

  __syncthreads();
  if ((warp & 16) != 0) {
   if ((warp & 15) < 9) {
    const int cwarp = 15 * order;
    const T helpA = sfacA[tid % (warp_size * 16)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    val7 += helpA * spartc[cwarp + (7 * delta + 0)];
    val8 += helpA * spartc[cwarp + (8 * delta + 0)];
   }
    if ((warp == 31) && (clane >= 0)) {
      spartc[clane + (31 * order + 0 * delta)] = val0;
      spartc[clane + (31 * order + 2 * delta)] = val2;
      spartc[clane + (31 * order + 4 * delta)] = val4;
      spartc[clane + (31 * order + 6 * delta)] = val6;
      spartc[clane + (31 * order + 8 * delta)] = val8;
    }
  }

  __syncthreads();
 if (warp < 9) {
  val1 += sfacA[tid] * spartc[31 * order + (0 * delta + 0)];
  val3 += sfacA[tid] * spartc[31 * order + (2 * delta + 0)];
  val5 += sfacA[tid] * spartc[31 * order + (4 * delta + 0)];
  val7 += sfacA[tid] * spartc[31 * order + (6 * delta + 0)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 1 * delta)] = val1;
    spartc[clane + (31 * order + 5 * delta)] = val5;
  }

  __syncthreads();
 if (warp < 9) {
  val2 += sfacA[tid] * spartc[31 * order + (1 * delta + 0)];
  val6 += sfacA[tid] * spartc[31 * order + (5 * delta + 0)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 3 * delta)] = val3;
  }

  __syncthreads();
 if (warp < 9) {
  val4 += sfacA[tid] * spartc[31 * order + (3 * delta + 0)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 7 * delta)] = val7;
  }

  __syncthreads();
 if (warp < 9) {
  val8 += sfacA[tid] * spartc[31 * order + (7 * delta + 0)];
 }

  const int idx = tid - (block_size - order);
  if (idx >= 0) {
    fullcarry[chunk_id * order + idx] = val8;
    __threadfence();
    if (idx == 0) {
      status[chunk_id] = 2;
    }
  }

  if (chunk_id > 0) {
    __syncthreads();
    if (warp == 0) {
      const int cidm1 = chunk_id - 1;
      int flag = 1;
      do {
        if ((cidm1 - lane) >= 0) {
          flag = status[cidm1 - lane];
        }
      } while ((__any(flag == 0)) || (__all(flag != 2)));
      int mask = __ballot(flag == 2);
      const int pos = __ffs(mask) - 1;
      T X0 = fullcarry[cidm1 - pos];
      if (lane == 0) {
        sfullc[0] = X0;
      }
    }

    __syncthreads();
    T X0 = sfullc[0];
    val0 += sfacA[tid] * X0;
  }

  if (chunk_id == gridDim.x - 1) {
    if (offs + (0 * block_size) < items) output[offs + (0 * block_size)] = val0;
    if (offs + (1 * block_size) < items) output[offs + (1 * block_size)] = val1;
    if (offs + (2 * block_size) < items) output[offs + (2 * block_size)] = val2;
    if (offs + (3 * block_size) < items) output[offs + (3 * block_size)] = val3;
    if (offs + (4 * block_size) < items) output[offs + (4 * block_size)] = val4;
    if (offs + (5 * block_size) < items) output[offs + (5 * block_size)] = val5;
    if (offs + (6 * block_size) < items) output[offs + (6 * block_size)] = val6;
    if (offs + (7 * block_size) < items) output[offs + (7 * block_size)] = val7;
    if (offs + (8 * block_size) < items) output[offs + (8 * block_size)] = val8;
  } else {
    output[offs + (0 * block_size)] = val0;
    output[offs + (1 * block_size)] = val1;
    output[offs + (2 * block_size)] = val2;
    output[offs + (3 * block_size)] = val3;
    output[offs + (4 * block_size)] = val4;
    output[offs + (5 * block_size)] = val5;
    output[offs + (6 * block_size)] = val6;
    output[offs + (7 * block_size)] = val7;
    output[offs + (8 * block_size)] = val8;
  }
}

struct GPUTimer
{
  cudaEvent_t beg, end;
  GPUTimer() {cudaEventCreate(&beg);  cudaEventCreate(&end);}
  ~GPUTimer() {cudaEventDestroy(beg);  cudaEventDestroy(end);}
  void start() {cudaEventRecord(beg, 0);}
  double stop() {cudaEventRecord(end, 0);  cudaEventSynchronize(end);  float ms;  cudaEventElapsedTime(&ms, beg, end);  return 0.001 * ms;}
};

template< class T1, class T2 >
void check_cpu_reference(const T1 *ref,
                         const T2 *res,
                         const long int& ne,
                         T1& me, T1& mre) {
    mre = me = (T1)0;
    for (long int i = 0; i < ne; i++)
    {
        T1 a = (T1)(res[i]) - ref[i];
        if( a < (T1)0 ) a = -a;
        if( ref[i] != (T1)0 )
        {
            T1 r = (ref[i] < (T1)0) ? -ref[i] : ref[i];
            T1 b = a / r;
            mre = b > mre ? b : mre;
        }
        me = a > me ? a : me;
    }
}


int main(int argc, char *argv[])
{/*
  printf("Parallel Linear Recurrence Computation\n");
  printf("Copyright (c) 2018 Texas State University\n");
 */
  if (argc != 3) {
    fprintf(stderr, "USAGE: %s problem_size repeats\n", argv[0]);
    return -1;
  }

  const long int n = atol(argv[1]);
  const long int iterations = atol(argv[2]);
  if (n < 1) {fprintf(stderr, "ERROR: problem_size must be at least 1\n");  return -1;};

  int *d_status;
  T *h_in, *h_out, *h_sol, *d_in, *d_out, *d_partcarry, *d_fullcarry;

  const size_t size = n * sizeof(T);
  h_in = (T *)malloc(size);  assert(h_in != NULL);
  h_out = (T *)malloc(size);  assert(h_out != NULL);
  h_sol = (T *)malloc(size);  assert(h_sol != NULL);

  for (long int i = 0; i < n; i++) {
    h_in[i] = (i & 32) / 16 - 1;
    h_sol[i] = 0;
  }
  for (long int i = 0; i < n; i++) {
    if ((i - 0) >= 0) {
      h_sol[i] += 2.809900e-01f * h_in[i - 0];
    }
  }
  for (long int i = 1; i < n; i++) {
    if ((i - 1) >= 0) {
      h_sol[i] += 7.190100e-01f * h_sol[i - 1];
    }
  }

  cudaSetDevice(device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);

  const int SMs = deviceProp.multiProcessorCount;
  int valsperthread = 1;
  while ((valsperthread < 9) && (block_size * 2 * SMs * valsperthread < n)) {
    valsperthread++;
  }
  const int chunk_size = valsperthread * block_size;
//  const long int iterations = 5;

  assert(cudaSuccess == cudaMalloc(&d_in, size));
  assert(cudaSuccess == cudaMalloc(&d_out, size));
  assert(cudaSuccess == cudaMalloc(&d_status, (n + chunk_size - 1) / chunk_size * sizeof(int)));
  assert(cudaSuccess == cudaMalloc(&d_partcarry, (n + chunk_size - 1) / chunk_size * order * sizeof(T)));
  assert(cudaSuccess == cudaMalloc(&d_fullcarry, (n + chunk_size - 1) / chunk_size * order * sizeof(T)));
  assert(cudaSuccess == cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));
  assert(cudaSuccess == cudaMemcpy(d_out, d_in, size, cudaMemcpyDeviceToDevice));

  cudaMemset(d_status, 0, (n + chunk_size - 1) / chunk_size * sizeof(int));
  switch (valsperthread) {
    case 1:  Recurrence1<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
    case 2:  Recurrence2<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
    case 3:  Recurrence3<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
    case 4:  Recurrence4<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
    case 5:  Recurrence5<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
    case 6:  Recurrence6<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
    case 7:  Recurrence7<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
    case 8:  Recurrence8<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
    case 9:  Recurrence9<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
  }
  GPUTimer timer;
  timer.start();
  for (long int i = 0; i < iterations; i++) {
    cudaMemset(d_status, 0, (n + chunk_size - 1) / chunk_size * sizeof(int));
    switch (valsperthread) {
      case 1:  Recurrence1<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
      case 2:  Recurrence2<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
      case 3:  Recurrence3<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
      case 4:  Recurrence4<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
      case 5:  Recurrence5<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
      case 6:  Recurrence6<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
      case 7:  Recurrence7<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
      case 8:  Recurrence8<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
      case 9:  Recurrence9<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
    }
  }
  double runtime = timer.stop() / iterations;
  double throughput = 0.000000001 * n / runtime;
  assert(cudaSuccess == cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));
/*
  for (long int i = 0; i < n; i++) {
    T s = h_sol[i];
    T o = h_out[i];
    if (fabsf(o - s) > 0.001) {
      printf("result not correct at index %d: %e != %e\n", i, h_sol[i], h_out[i]);
      return -1;
    }
  }

  printf("size = %ld\tthroughput = %7.4f gigaitems/s\truntime = %7.4f s\tPassed!\n", n, throughput, runtime);

  printf("first elements of result are:\n");
  for (int i = 0; (i < 8) && (i < n); i++) {
    printf(" %f", h_out[i]);
  }
  printf("\n");
*/
  double mebissec = n / (runtime*1024*1024); // Mis/s
  T max_abs_err, max_rel_err;
  check_cpu_reference(h_sol, h_out, n, max_abs_err, max_rel_err);
  printf("%7.7f %e %e\n", mebissec, max_abs_err, max_rel_err);
         
  free(h_in);  free(h_out);  free(h_sol);
  cudaFree(d_in);  cudaFree(d_out);  cudaFree(d_status);  cudaFree(d_partcarry);  cudaFree(d_fullcarry);

  return 0;
}
