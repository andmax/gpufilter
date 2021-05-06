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


non-recursive coefficients: (0.143609)
recursive coefficients: (1.413320, -0.556927)

*/

#include <cstdio>
#include <cassert>
#include <cuda.h>

typedef float T;
static const int device = 0;
static const int order = 2;
static const int warp_size = 32;
static const int block_size = 1024;

static __device__ const T facA[300] = {-5.569270e-01f, -7.871161e-01f, -8.022791e-01f, -6.955109e-01f, -5.361685e-01f, -3.704288e-01f, -2.249277e-01f, -1.115930e-01f, -3.244826e-02f, 1.628936e-02f, 4.109339e-02f, 4.900613e-02f, 4.637531e-02f, 3.825032e-02f, 2.823227e-02f, 1.859860e-02f, 1.056245e-02f, 4.570063e-03f, 5.764468e-04f, -1.730488e-03f, -2.766772e-03f, -2.946578e-03f, -2.623568e-03f, -2.066912e-03f, -1.460072e-03f, -9.124296e-04f, -4.764014e-04f, -1.651509e-04f, 3.190975e-05f, 1.370757e-04f, 1.759604e-04f, 1.723472e-04f, 1.455846e-04f, 1.097728e-04f, 7.406413e-05f, 4.354086e-05f, 2.028884e-05f, 4.425547e-06f, -5.044692e-06f, -9.594471e-06f, -1.075053e-05f, -9.850521e-06f, -7.934676e-06f, -5.728214e-06f, -3.676764e-06f, -2.006246e-06f, -7.877786e-07f, 3.949594e-09f, 4.443173e-07f, 6.257628e-07f, 6.369507e-07f, 5.517110e-07f, 4.250091e-07f, 2.934110e-07f, 1.779846e-07f, 8.814068e-08f, 2.544655e-08f, -1.312382e-08f, -3.272002e-08f, -3.893485e-08f, -3.680474e-08f, -3.033299e-08f, -2.237267e-08f, -1.472648e-08f, -8.353289e-09f, -3.604292e-09f, -4.418452e-10f, 1.382859e-09f, 2.200498e-09f, 2.339856e-09f, 2.081448e-09f, 1.638623e-09f, 1.156684e-09f, 7.221714e-10f, 3.764705e-10f, 1.298765e-10f, -2.610956e-11f, -1.092329e-10f, -1.398399e-10f, -1.368038e-10f, -1.154669e-10f, -8.700197e-11f, -5.865497e-11f, -3.444449e-11f, -1.601455e-11f, -3.450613e-12f, 4.042114e-12f, 7.634540e-12f, 8.538885e-12f, 7.816294e-12f, 6.291389e-12f, 4.538639e-12f, 2.910705e-12f, 1.586066e-12f, 6.205688e-13f, -6.260834e-15f, -3.544601e-13f, -4.974787e-13f, -5.056882e-13f, -4.376398e-13f, -3.368937e-13f, -2.324051e-13f, -1.408376e-13f, -6.961586e-14f, -1.995324e-14f, 1.057065e-14f, 2.605221e-14f, 3.093302e-14f, 2.920908e-14f, 2.405433e-14f, 1.772914e-14f, 1.166044e-14f, 6.606096e-15f, 2.842512e-15f, 3.382656e-16f, -1.104994e-15f, -1.750099e-15f, -1.858050e-15f, -1.651341e-15f, -1.299075e-15f, -9.163323e-16f, -5.715807e-16f, -2.974962e-16f, -1.021285e-16f, 2.134338e-17f, 8.704316e-17f, 1.111331e-16f, 1.085900e-16f, 9.157935e-17f, 6.895423e-17f, 4.645137e-17f, 2.724817e-17f, 1.264036e-17f, 2.689634e-18f, -3.238447e-18f, -6.074891e-18f, -6.782186e-18f, -6.202127e-18f, -4.988408e-18f, -3.596083e-18f, -2.304237e-18f, -1.253868e-18f, -4.888254e-19f, 7.446590e-21f, 2.827645e-19f, 3.954895e-19f, 4.014740e-19f, 3.471525e-19f, 2.670458e-19f, 1.840826e-19f, 1.114425e-19f, 5.498341e-20f, 1.564378e-20f, -8.512076e-21f, -2.074273e-20f, -2.457551e-20f, -2.318087e-20f, -1.907522e-20f, -1.404934e-20f, -9.232702e-21f, -5.224306e-21f, -2.241674e-21f, -2.586462e-22f, 8.828993e-22f, 1.391866e-21f, 1.475442e-21f, 1.310104e-21f, 1.029882e-21f, 7.259208e-22f, 4.523891e-22f, 2.350857e-22f, 8.030357e-23f, -1.743095e-23f, -6.935873e-23f, -8.831832e-23f, -8.619429e-23f, -7.263325e-23f, -5.465009e-23f, -3.678664e-23f, -2.155518e-23f, -9.976893e-24f, -2.095878e-24f, 2.594255e-24f, 4.833763e-24f, 5.386843e-24f, 4.921280e-24f, 3.955264e-24f, 2.849260e-24f, 1.824122e-24f, 9.912387e-25f, 3.850344e-25f, -7.870761e-27f, -2.255600e-25f, -3.144050e-25f, -3.187344e-25f, -2.753731e-25f, -2.116784e-25f, -1.458066e-25f, -8.818200e-26f, -4.342573e-26f, -1.226351e-26f, 6.852704e-27f, 1.651494e-26f, 1.952444e-26f, 1.839666e-26f, 1.512668e-26f, 1.113324e-26f, 7.310376e-27f, 4.131496e-27f, 1.767780e-27f, 1.974972e-28f, -7.053979e-28f, -1.106944e-27f, -1.171611e-27f, -1.039375e-27f, -8.164666e-28f, -5.750728e-28f, -3.580495e-28f, -1.857649e-28f, -6.313783e-29f, 1.422354e-29f, 5.526558e-29f, 7.018647e-29f, 6.841704e-29f, 5.760642e-29f, 4.331301e-29f, 2.913257e-29f, 1.705146e-29f, 7.874449e-30f, 1.632697e-30f, -2.077970e-30f, -3.846129e-30f, -4.278534e-30f, -3.904925e-30f, -3.136076e-30f, -2.257521e-30f, -1.444034e-30f, -7.836079e-31f, -3.032669e-31f, 7.799217e-33f, 1.799203e-31f, 2.499414e-31f, 2.530446e-31f, 2.184339e-31f, 1.677896e-31f, 1.154887e-31f, 6.977586e-32f, 3.429686e-32f, 9.612367e-33f, -5.515497e-33f, -1.314855e-32f, -1.551138e-32f, -1.459975e-32f, -1.199542e-32f, -8.822366e-33f, -5.788254e-33f, -3.267240e-33f, -1.394021e-33f, -1.505832e-34f, 5.635457e-34f, 8.803342e-34f, 9.303400e-34f, 8.245861e-34f, 6.472725e-34f, 4.555688e-34f, 2.833809e-34f, 1.467894e-34f, 4.963785e-35f, -1.159680e-35f, -4.403466e-35f, -5.577648e-35f, -5.430593e-35f, -4.568822e-35f, -3.432763e-35f, -2.307092e-35f, -1.348861e-35f, -6.214901e-36f, -1.271472e-36f, 1.664249e-36f, 3.060234e-36f, 3.398224e-36f, 3.098451e-36f, 2.486539e-36f, 1.788665e-36f, 1.143135e-36f, 6.194592e-37f, 2.388514e-37f, 0.000000e+00f, -1.435098e-37f, -1.986928e-37f, -2.008920e-37f, -1.732673e-37f, -1.329999e-37f, -9.147423e-38f, -5.521109e-38f, -2.708647e-38f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 1.232305e-38f};

static __device__ const T facB[302] = {1.413320e+00f, 1.440546e+00f, 1.248837e+00f, 9.627268e-01f, 6.651301e-01f, 4.038730e-01f, 2.003729e-01f, 5.826317e-02f, -2.924857e-02f, -7.378592e-02f, -8.799380e-02f, -8.327001e-02f, -6.868105e-02f, -5.069298e-02f, -3.339506e-02f, -1.896562e-02f, -8.205878e-03f, -1.035064e-03f, 3.107198e-03f, 4.967920e-03f, 5.290777e-03f, 4.710793e-03f, 3.711280e-03f, 2.621658e-03f, 1.638330e-03f, 8.554121e-04f, 2.965407e-04f, -5.729517e-05f, -2.461280e-04f, -3.159483e-04f, -3.094607e-04f, -2.614069e-04f, -1.971045e-04f, -1.329872e-04f, -7.818059e-05f, -3.643004e-05f, -7.946423e-06f, 9.058038e-06f, 1.722748e-05f, 1.930328e-05f, 1.768726e-05f, 1.424724e-05f, 1.028539e-05f, 6.601879e-06f, 3.602355e-06f, 1.414515e-06f, -7.086328e-09f, -7.977969e-07f, -1.123596e-06f, -1.143686e-06f, -9.906330e-07f, -7.631319e-07f, -5.268392e-07f, -3.195836e-07f, -1.582629e-07f, -4.569137e-08f, 2.356437e-08f, 5.875075e-08f, 6.990997e-08f, 6.608528e-08f, 5.446489e-08f, 4.017163e-08f, 2.644241e-08f, 1.499891e-08f, 6.471772e-09f, 7.933840e-10f, -2.482999e-09f, -3.951129e-09f, -4.201360e-09f, -3.737376e-09f, -2.942257e-09f, -2.076905e-09f, -1.296709e-09f, -6.759798e-10f, -2.332037e-10f, 4.688003e-11f, 1.961339e-10f, 2.510912e-10f, 2.456400e-10f, 2.073284e-10f, 1.562178e-10f, 1.053189e-10f, 6.184744e-11f, 2.875526e-11f, 6.195877e-12f, -7.257825e-12f, -1.370828e-11f, -1.533211e-11f, -1.403466e-11f, -1.129660e-11f, -8.149431e-12f, -5.226370e-12f, -2.847895e-12f, -1.114280e-12f, 1.123537e-14f, 6.364519e-13f, 8.932529e-13f, 9.079948e-13f, 7.858105e-13f, 6.049148e-13f, 4.172990e-13f, 2.528837e-13f, 1.250004e-13f, 3.582783e-14f, -1.897992e-14f, -4.677819e-14f, -5.554212e-14f, -5.244675e-14f, -4.319113e-14f, -3.183388e-14f, -2.093715e-14f, -1.186174e-14f, -5.103968e-15f, -6.074179e-16f, 1.984062e-15f, 3.142402e-15f, 3.336241e-15f, 2.965088e-15f, 2.332575e-15f, 1.645337e-15f, 1.026314e-15f, 5.341768e-16f, 1.833809e-16f, -3.832154e-17f, -1.562904e-16f, -1.995460e-16f, -1.949800e-16f, -1.644366e-16f, -1.238119e-16f, -8.340665e-17f, -4.892609e-17f, -2.269679e-17f, -4.829567e-18f, 5.814733e-18f, 1.090779e-17f, 1.217782e-17f, 1.113631e-17f, 8.957013e-18f, 6.457012e-18f, 4.137422e-18f, 2.251417e-18f, 8.777300e-19f, -1.336155e-20f, -5.077158e-19f, -7.101234e-19f, -7.208709e-19f, -6.233344e-19f, -4.794984e-19f, -3.305329e-19f, -2.001031e-19f, -9.872703e-20f, -2.809005e-20f, 1.528353e-20f, 3.724462e-20f, 4.412675e-20f, 4.162269e-20f, 3.425080e-20f, 2.522653e-20f, 1.657797e-20f, 9.380639e-21f, 4.025123e-21f, 4.644559e-22f, -1.585275e-21f, -2.499169e-21f, -2.649243e-21f, -2.352373e-21f, -1.849221e-21f, -1.303441e-21f, -8.122974e-22f, -4.221149e-22f, -1.441929e-22f, 3.129643e-23f, 1.245368e-22f, 1.585805e-22f, 1.547671e-22f, 1.304177e-22f, 9.812792e-23f, 6.605300e-23f, 3.870394e-23f, 1.791435e-23f, 3.763434e-24f, -4.658047e-24f, -8.679269e-24f, -9.672392e-24f, -8.836464e-24f, -7.101935e-24f, -5.116040e-24f, -3.275343e-24f, -1.779846e-24f, -6.913652e-25f, 1.412426e-26f, 4.050021e-25f, 5.645313e-25f, 5.723068e-25f, 4.944498e-25f, 3.800827e-25f, 2.618060e-25f, 1.583373e-25f, 7.797446e-26f, 2.202053e-26f, -1.230403e-26f, -2.965336e-26f, -3.505723e-26f, -3.303233e-26f, -2.716093e-26f, -1.999049e-26f, -1.312630e-26f, -7.418418e-27f, -3.174208e-27f, -3.546535e-28f, 1.266563e-27f, 1.987575e-27f, 2.103696e-27f, 1.866262e-27f, 1.466019e-27f, 1.032583e-27f, 6.429039e-28f, 3.335558e-28f, 1.133704e-28f, -2.553753e-29f, -9.923175e-29f, -1.260237e-28f, -1.228469e-28f, -1.034360e-28f, -7.777140e-29f, -5.230958e-29f, -3.061717e-29f, -1.413924e-29f, -2.931744e-30f, 3.731034e-30f, 6.905912e-30f, 7.682349e-30f, 7.011528e-30f, 5.631024e-30f, 4.053530e-30f, 2.592864e-30f, 1.407027e-30f, 5.445428e-31f, -1.399812e-32f, -3.230544e-31f, -4.487833e-31f, -4.543566e-31f, -3.922118e-31f, -3.012772e-31f, -2.073678e-31f, -1.252875e-31f, -6.158268e-32f, -1.726001e-32f, 9.903146e-33f, 2.360888e-32f, 2.785157e-32f, 2.621476e-32f, 2.153855e-32f, 1.584116e-32f, 1.039322e-32f, 5.866577e-33f, 2.503085e-33f, 2.704048e-34f, -1.011867e-33f, -1.580688e-33f, -1.670481e-33f, -1.480597e-33f, -1.162221e-33f, -8.180058e-34f, -5.088315e-34f, -2.635722e-34f, -8.912982e-35f, 2.082153e-35f, 7.906629e-35f, 1.001499e-34f, 9.750968e-35f, 8.203620e-35f, 6.163762e-35f, 4.142550e-35f, 2.421983e-35f, 1.115939e-35f, 2.283108e-36f, -2.988202e-36f, -5.494810e-36f, -6.101715e-36f, -5.563467e-36f, -4.464749e-36f, -3.211674e-36f, -2.052583e-36f, -1.112289e-36f, -4.288806e-37f, 1.331812e-38f, 2.576779e-37f, 3.567642e-37f, 3.607141e-37f, 3.111128e-37f, 2.388105e-37f, 1.642485e-37f, 9.913571e-38f, 4.863603e-38f, 1.352691e-38f, 0.000000e+00f, -1.879606e-38f, -2.212677e-38f, -2.080417e-38f, -1.707995e-38f, -1.255303e-38f};

// shared memory size is 10508 bytes

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
  __shared__ T sfacB[block_size];
  if (tid < 300) sfacA[tid] = facA[tid];
  else sfacA[tid] = 0;
  if (tid < 302) sfacB[tid] = facB[tid];
  else sfacB[tid] = 0;

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

  val0 *= 1.436090e-01f;

  const T sfA = sfacA[lane];
  const T sfB = sfacB[lane];

  int cond;
  T help, spc;

  help = 1.413320e+00f;
  cond = ((lane & 1) != 0);
  spc = help * __shfl(val0, 0, 2);
  if (cond) val0 += spc;

  help = __shfl(sfA, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 0, 4);
  if (cond) val0 += spc;

  help = __shfl(sfB, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 1, 4);
  if (cond) val0 += spc;

  help = __shfl(sfA, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 2, 8);
  if (cond) val0 += spc;

  help = __shfl(sfB, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 3, 8);
  if (cond) val0 += spc;

  help = __shfl(sfA, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 6, 16);
  if (cond) val0 += spc;

  help = __shfl(sfB, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 7, 16);
  if (cond) val0 += spc;

  help = __shfl(sfA, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 14, 32);
  if (cond) val0 += spc;

  help = __shfl(sfB, lane % 16);
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
    const T helpB = sfacB[tid % (warp_size * 1)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    if (((warp & 3) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
    }
  }

  __syncthreads();
  if ((warp & 2) != 0) {
    const int cwarp = ((warp & ~3) | 1) * order;
    const T helpA = sfacA[tid % (warp_size * 2)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 2)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    if (((warp & 7) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
    }
  }

  __syncthreads();
  if ((warp & 4) != 0) {
    const int cwarp = ((warp & ~7) | 3) * order;
    const T helpA = sfacA[tid % (warp_size * 4)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 4)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    if (((warp & 15) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
    }
  }

  __syncthreads();
  if ((warp & 8) != 0) {
    const int cwarp = ((warp & ~15) | 7) * order;
    const T helpA = sfacA[tid % (warp_size * 8)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 8)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    if ((warp == 15) && (clane >= 0)) {
      spartc[clane + (15 * order + 0 * delta)] = val0;
    }
  }

  __syncthreads();
  if ((warp & 16) != 0) {
   if ((warp & 15) < 10) {
    const int cwarp = 15 * order;
    const T helpA = sfacA[tid % (warp_size * 16)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 16)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
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
      T fc;
      if (lane < order) {
        fc = fullcarry[(cidm1 - pos) * order + lane];
      }
      T X0 = __shfl(fc, 0);
      T X1 = __shfl(fc, 1);
      if (lane == 0) {
        sfullc[0] = X0;
        sfullc[1] = X1;
      }
    }

    __syncthreads();
    T X0 = sfullc[0];
    val0 += sfacA[tid] * X0;
    T X1 = sfullc[1];
    val0 += sfacB[tid] * X1;
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
  __shared__ T sfacB[block_size];
  if (tid < 300) sfacA[tid] = facA[tid];
  else sfacA[tid] = 0;
  if (tid < 302) sfacB[tid] = facB[tid];
  else sfacB[tid] = 0;

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

  val0 *= 1.436090e-01f;
  val1 *= 1.436090e-01f;

  const T sfA = sfacA[lane];
  const T sfB = sfacB[lane];

  int cond;
  T help, spc;

  help = 1.413320e+00f;
  cond = ((lane & 1) != 0);
  spc = help * __shfl(val0, 0, 2);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 2);
  if (cond) val1 += spc;

  help = __shfl(sfA, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 0, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 4);
  if (cond) val1 += spc;

  help = __shfl(sfB, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 1, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 1, 4);
  if (cond) val1 += spc;

  help = __shfl(sfA, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 2, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 2, 8);
  if (cond) val1 += spc;

  help = __shfl(sfB, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 3, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 3, 8);
  if (cond) val1 += spc;

  help = __shfl(sfA, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 6, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 6, 16);
  if (cond) val1 += spc;

  help = __shfl(sfB, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 7, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 7, 16);
  if (cond) val1 += spc;

  help = __shfl(sfA, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 14, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 14, 32);
  if (cond) val1 += spc;

  help = __shfl(sfB, lane % 16);
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
    const T helpB = sfacB[tid % (warp_size * 1)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
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
    const T helpB = sfacB[tid % (warp_size * 2)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
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
    const T helpB = sfacB[tid % (warp_size * 4)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
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
    const T helpB = sfacB[tid % (warp_size * 8)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    if ((warp == 15) && (clane >= 0)) {
      spartc[clane + (15 * order + 0 * delta)] = val0;
      spartc[clane + (15 * order + 1 * delta)] = val1;
    }
  }

  __syncthreads();
  if ((warp & 16) != 0) {
   if ((warp & 15) < 10) {
    const int cwarp = 15 * order;
    const T helpA = sfacA[tid % (warp_size * 16)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 16)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
   }
    if ((warp == 31) && (clane >= 0)) {
      spartc[clane + (31 * order + 0 * delta)] = val0;
    }
  }

  __syncthreads();
 if (warp < 10) {
  val1 += sfacA[tid] * spartc[31 * order + (0 * delta + 0)];
  val1 += sfacB[tid] * spartc[31 * order + (0 * delta + 1)];
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
      T fc;
      if (lane < order) {
        fc = fullcarry[(cidm1 - pos) * order + lane];
      }
      T X0 = __shfl(fc, 0);
      T X1 = __shfl(fc, 1);
      if (lane == 0) {
        sfullc[0] = X0;
        sfullc[1] = X1;
      }
    }

    __syncthreads();
    T X0 = sfullc[0];
    val0 += sfacA[tid] * X0;
    T X1 = sfullc[1];
    val0 += sfacB[tid] * X1;
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
  __shared__ T sfacB[block_size];
  if (tid < 300) sfacA[tid] = facA[tid];
  else sfacA[tid] = 0;
  if (tid < 302) sfacB[tid] = facB[tid];
  else sfacB[tid] = 0;

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

  val0 *= 1.436090e-01f;
  val1 *= 1.436090e-01f;
  val2 *= 1.436090e-01f;

  const T sfA = sfacA[lane];
  const T sfB = sfacB[lane];

  int cond;
  T help, spc;

  help = 1.413320e+00f;
  cond = ((lane & 1) != 0);
  spc = help * __shfl(val0, 0, 2);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 2);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 2);
  if (cond) val2 += spc;

  help = __shfl(sfA, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 0, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 4);
  if (cond) val2 += spc;

  help = __shfl(sfB, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 1, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 1, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 1, 4);
  if (cond) val2 += spc;

  help = __shfl(sfA, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 2, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 2, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 2, 8);
  if (cond) val2 += spc;

  help = __shfl(sfB, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 3, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 3, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 3, 8);
  if (cond) val2 += spc;

  help = __shfl(sfA, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 6, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 6, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 6, 16);
  if (cond) val2 += spc;

  help = __shfl(sfB, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 7, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 7, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 7, 16);
  if (cond) val2 += spc;

  help = __shfl(sfA, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 14, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 14, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 14, 32);
  if (cond) val2 += spc;

  help = __shfl(sfB, lane % 16);
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
    const T helpB = sfacB[tid % (warp_size * 1)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
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
    const T helpB = sfacB[tid % (warp_size * 2)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
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
    const T helpB = sfacB[tid % (warp_size * 4)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
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
    const T helpB = sfacB[tid % (warp_size * 8)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    if ((warp == 15) && (clane >= 0)) {
      spartc[clane + (15 * order + 0 * delta)] = val0;
      spartc[clane + (15 * order + 1 * delta)] = val1;
      spartc[clane + (15 * order + 2 * delta)] = val2;
    }
  }

  __syncthreads();
  if ((warp & 16) != 0) {
   if ((warp & 15) < 10) {
    const int cwarp = 15 * order;
    const T helpA = sfacA[tid % (warp_size * 16)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 16)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
   }
    if ((warp == 31) && (clane >= 0)) {
      spartc[clane + (31 * order + 0 * delta)] = val0;
      spartc[clane + (31 * order + 2 * delta)] = val2;
    }
  }

  __syncthreads();
 if (warp < 10) {
  val1 += sfacA[tid] * spartc[31 * order + (0 * delta + 0)];
  val1 += sfacB[tid] * spartc[31 * order + (0 * delta + 1)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 1 * delta)] = val1;
  }

  __syncthreads();
 if (warp < 10) {
  val2 += sfacA[tid] * spartc[31 * order + (1 * delta + 0)];
  val2 += sfacB[tid] * spartc[31 * order + (1 * delta + 1)];
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
      T fc;
      if (lane < order) {
        fc = fullcarry[(cidm1 - pos) * order + lane];
      }
      T X0 = __shfl(fc, 0);
      T X1 = __shfl(fc, 1);
      if (lane == 0) {
        sfullc[0] = X0;
        sfullc[1] = X1;
      }
    }

    __syncthreads();
    T X0 = sfullc[0];
    val0 += sfacA[tid] * X0;
    T X1 = sfullc[1];
    val0 += sfacB[tid] * X1;
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
  __shared__ T sfacB[block_size];
  if (tid < 300) sfacA[tid] = facA[tid];
  else sfacA[tid] = 0;
  if (tid < 302) sfacB[tid] = facB[tid];
  else sfacB[tid] = 0;

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

  val0 *= 1.436090e-01f;
  val1 *= 1.436090e-01f;
  val2 *= 1.436090e-01f;
  val3 *= 1.436090e-01f;

  const T sfA = sfacA[lane];
  const T sfB = sfacB[lane];

  int cond;
  T help, spc;

  help = 1.413320e+00f;
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
  spc = help * __shfl(val0, 0, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 4);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 0, 4);
  if (cond) val3 += spc;

  help = __shfl(sfB, lane % 2);
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
  spc = help * __shfl(val0, 2, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 2, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 2, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 2, 8);
  if (cond) val3 += spc;

  help = __shfl(sfB, lane % 4);
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
  spc = help * __shfl(val0, 6, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 6, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 6, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 6, 16);
  if (cond) val3 += spc;

  help = __shfl(sfB, lane % 8);
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
  spc = help * __shfl(val0, 14, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 14, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 14, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 14, 32);
  if (cond) val3 += spc;

  help = __shfl(sfB, lane % 16);
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
    const T helpB = sfacB[tid % (warp_size * 1)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
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
    const T helpB = sfacB[tid % (warp_size * 2)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
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
    const T helpB = sfacB[tid % (warp_size * 4)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
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
    const T helpB = sfacB[tid % (warp_size * 8)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    if ((warp == 15) && (clane >= 0)) {
      spartc[clane + (15 * order + 0 * delta)] = val0;
      spartc[clane + (15 * order + 1 * delta)] = val1;
      spartc[clane + (15 * order + 2 * delta)] = val2;
      spartc[clane + (15 * order + 3 * delta)] = val3;
    }
  }

  __syncthreads();
  if ((warp & 16) != 0) {
   if ((warp & 15) < 10) {
    const int cwarp = 15 * order;
    const T helpA = sfacA[tid % (warp_size * 16)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 16)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
   }
    if ((warp == 31) && (clane >= 0)) {
      spartc[clane + (31 * order + 0 * delta)] = val0;
      spartc[clane + (31 * order + 2 * delta)] = val2;
    }
  }

  __syncthreads();
 if (warp < 10) {
  val1 += sfacA[tid] * spartc[31 * order + (0 * delta + 0)];
  val3 += sfacA[tid] * spartc[31 * order + (2 * delta + 0)];
  val1 += sfacB[tid] * spartc[31 * order + (0 * delta + 1)];
  val3 += sfacB[tid] * spartc[31 * order + (2 * delta + 1)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 1 * delta)] = val1;
  }

  __syncthreads();
 if (warp < 10) {
  val2 += sfacA[tid] * spartc[31 * order + (1 * delta + 0)];
  val2 += sfacB[tid] * spartc[31 * order + (1 * delta + 1)];
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
      T fc;
      if (lane < order) {
        fc = fullcarry[(cidm1 - pos) * order + lane];
      }
      T X0 = __shfl(fc, 0);
      T X1 = __shfl(fc, 1);
      if (lane == 0) {
        sfullc[0] = X0;
        sfullc[1] = X1;
      }
    }

    __syncthreads();
    T X0 = sfullc[0];
    val0 += sfacA[tid] * X0;
    T X1 = sfullc[1];
    val0 += sfacB[tid] * X1;
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
  __shared__ T sfacB[block_size];
  if (tid < 300) sfacA[tid] = facA[tid];
  else sfacA[tid] = 0;
  if (tid < 302) sfacB[tid] = facB[tid];
  else sfacB[tid] = 0;

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

  val0 *= 1.436090e-01f;
  val1 *= 1.436090e-01f;
  val2 *= 1.436090e-01f;
  val3 *= 1.436090e-01f;
  val4 *= 1.436090e-01f;

  const T sfA = sfacA[lane];
  const T sfB = sfacB[lane];

  int cond;
  T help, spc;

  help = 1.413320e+00f;
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
  spc = help * __shfl(val0, 0, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 4);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 0, 4);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 0, 4);
  if (cond) val4 += spc;

  help = __shfl(sfB, lane % 2);
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
  spc = help * __shfl(val0, 2, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 2, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 2, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 2, 8);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 2, 8);
  if (cond) val4 += spc;

  help = __shfl(sfB, lane % 4);
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
  spc = help * __shfl(val0, 6, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 6, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 6, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 6, 16);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 6, 16);
  if (cond) val4 += spc;

  help = __shfl(sfB, lane % 8);
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
  spc = help * __shfl(val0, 14, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 14, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 14, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 14, 32);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 14, 32);
  if (cond) val4 += spc;

  help = __shfl(sfB, lane % 16);
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
    const T helpB = sfacB[tid % (warp_size * 1)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
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
    const T helpB = sfacB[tid % (warp_size * 2)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
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
    const T helpB = sfacB[tid % (warp_size * 4)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
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
    const T helpB = sfacB[tid % (warp_size * 8)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
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
   if ((warp & 15) < 10) {
    const int cwarp = 15 * order;
    const T helpA = sfacA[tid % (warp_size * 16)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 16)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
   }
    if ((warp == 31) && (clane >= 0)) {
      spartc[clane + (31 * order + 0 * delta)] = val0;
      spartc[clane + (31 * order + 2 * delta)] = val2;
      spartc[clane + (31 * order + 4 * delta)] = val4;
    }
  }

  __syncthreads();
 if (warp < 10) {
  val1 += sfacA[tid] * spartc[31 * order + (0 * delta + 0)];
  val3 += sfacA[tid] * spartc[31 * order + (2 * delta + 0)];
  val1 += sfacB[tid] * spartc[31 * order + (0 * delta + 1)];
  val3 += sfacB[tid] * spartc[31 * order + (2 * delta + 1)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 1 * delta)] = val1;
  }

  __syncthreads();
 if (warp < 10) {
  val2 += sfacA[tid] * spartc[31 * order + (1 * delta + 0)];
  val2 += sfacB[tid] * spartc[31 * order + (1 * delta + 1)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 3 * delta)] = val3;
  }

  __syncthreads();
 if (warp < 10) {
  val4 += sfacA[tid] * spartc[31 * order + (3 * delta + 0)];
  val4 += sfacB[tid] * spartc[31 * order + (3 * delta + 1)];
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
      T fc;
      if (lane < order) {
        fc = fullcarry[(cidm1 - pos) * order + lane];
      }
      T X0 = __shfl(fc, 0);
      T X1 = __shfl(fc, 1);
      if (lane == 0) {
        sfullc[0] = X0;
        sfullc[1] = X1;
      }
    }

    __syncthreads();
    T X0 = sfullc[0];
    val0 += sfacA[tid] * X0;
    T X1 = sfullc[1];
    val0 += sfacB[tid] * X1;
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
  __shared__ T sfacB[block_size];
  if (tid < 300) sfacA[tid] = facA[tid];
  else sfacA[tid] = 0;
  if (tid < 302) sfacB[tid] = facB[tid];
  else sfacB[tid] = 0;

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

  val0 *= 1.436090e-01f;
  val1 *= 1.436090e-01f;
  val2 *= 1.436090e-01f;
  val3 *= 1.436090e-01f;
  val4 *= 1.436090e-01f;
  val5 *= 1.436090e-01f;

  const T sfA = sfacA[lane];
  const T sfB = sfacB[lane];

  int cond;
  T help, spc;

  help = 1.413320e+00f;
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
  spc = help * __shfl(val0, 0, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 4);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 0, 4);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 0, 4);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 0, 4);
  if (cond) val5 += spc;

  help = __shfl(sfB, lane % 2);
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
  spc = help * __shfl(val0, 2, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 2, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 2, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 2, 8);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 2, 8);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 2, 8);
  if (cond) val5 += spc;

  help = __shfl(sfB, lane % 4);
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
  spc = help * __shfl(val0, 6, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 6, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 6, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 6, 16);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 6, 16);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 6, 16);
  if (cond) val5 += spc;

  help = __shfl(sfB, lane % 8);
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
  spc = help * __shfl(val0, 14, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 14, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 14, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 14, 32);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 14, 32);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 14, 32);
  if (cond) val5 += spc;

  help = __shfl(sfB, lane % 16);
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
    const T helpB = sfacB[tid % (warp_size * 1)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
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
    const T helpB = sfacB[tid % (warp_size * 2)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
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
    const T helpB = sfacB[tid % (warp_size * 4)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
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
    const T helpB = sfacB[tid % (warp_size * 8)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
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
   if ((warp & 15) < 10) {
    const int cwarp = 15 * order;
    const T helpA = sfacA[tid % (warp_size * 16)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 16)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
   }
    if ((warp == 31) && (clane >= 0)) {
      spartc[clane + (31 * order + 0 * delta)] = val0;
      spartc[clane + (31 * order + 2 * delta)] = val2;
      spartc[clane + (31 * order + 4 * delta)] = val4;
    }
  }

  __syncthreads();
 if (warp < 10) {
  val1 += sfacA[tid] * spartc[31 * order + (0 * delta + 0)];
  val3 += sfacA[tid] * spartc[31 * order + (2 * delta + 0)];
  val5 += sfacA[tid] * spartc[31 * order + (4 * delta + 0)];
  val1 += sfacB[tid] * spartc[31 * order + (0 * delta + 1)];
  val3 += sfacB[tid] * spartc[31 * order + (2 * delta + 1)];
  val5 += sfacB[tid] * spartc[31 * order + (4 * delta + 1)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 1 * delta)] = val1;
    spartc[clane + (31 * order + 5 * delta)] = val5;
  }

  __syncthreads();
 if (warp < 10) {
  val2 += sfacA[tid] * spartc[31 * order + (1 * delta + 0)];
  val2 += sfacB[tid] * spartc[31 * order + (1 * delta + 1)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 3 * delta)] = val3;
  }

  __syncthreads();
 if (warp < 10) {
  val4 += sfacA[tid] * spartc[31 * order + (3 * delta + 0)];
  val4 += sfacB[tid] * spartc[31 * order + (3 * delta + 1)];
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
      T fc;
      if (lane < order) {
        fc = fullcarry[(cidm1 - pos) * order + lane];
      }
      T X0 = __shfl(fc, 0);
      T X1 = __shfl(fc, 1);
      if (lane == 0) {
        sfullc[0] = X0;
        sfullc[1] = X1;
      }
    }

    __syncthreads();
    T X0 = sfullc[0];
    val0 += sfacA[tid] * X0;
    T X1 = sfullc[1];
    val0 += sfacB[tid] * X1;
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
  __shared__ T sfacB[block_size];
  if (tid < 300) sfacA[tid] = facA[tid];
  else sfacA[tid] = 0;
  if (tid < 302) sfacB[tid] = facB[tid];
  else sfacB[tid] = 0;

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

  val0 *= 1.436090e-01f;
  val1 *= 1.436090e-01f;
  val2 *= 1.436090e-01f;
  val3 *= 1.436090e-01f;
  val4 *= 1.436090e-01f;
  val5 *= 1.436090e-01f;
  val6 *= 1.436090e-01f;

  const T sfA = sfacA[lane];
  const T sfB = sfacB[lane];

  int cond;
  T help, spc;

  help = 1.413320e+00f;
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
  spc = help * __shfl(val0, 0, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 4);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 0, 4);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 0, 4);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 0, 4);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 0, 4);
  if (cond) val6 += spc;

  help = __shfl(sfB, lane % 2);
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
  spc = help * __shfl(val0, 2, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 2, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 2, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 2, 8);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 2, 8);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 2, 8);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 2, 8);
  if (cond) val6 += spc;

  help = __shfl(sfB, lane % 4);
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
  spc = help * __shfl(val0, 6, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 6, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 6, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 6, 16);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 6, 16);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 6, 16);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 6, 16);
  if (cond) val6 += spc;

  help = __shfl(sfB, lane % 8);
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
  spc = help * __shfl(val0, 14, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 14, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 14, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 14, 32);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 14, 32);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 14, 32);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 14, 32);
  if (cond) val6 += spc;

  help = __shfl(sfB, lane % 16);
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
    const T helpB = sfacB[tid % (warp_size * 1)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
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
    const T helpB = sfacB[tid % (warp_size * 2)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
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
    const T helpB = sfacB[tid % (warp_size * 4)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
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
    const T helpB = sfacB[tid % (warp_size * 8)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
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
   if ((warp & 15) < 10) {
    const int cwarp = 15 * order;
    const T helpA = sfacA[tid % (warp_size * 16)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 16)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
   }
    if ((warp == 31) && (clane >= 0)) {
      spartc[clane + (31 * order + 0 * delta)] = val0;
      spartc[clane + (31 * order + 2 * delta)] = val2;
      spartc[clane + (31 * order + 4 * delta)] = val4;
      spartc[clane + (31 * order + 6 * delta)] = val6;
    }
  }

  __syncthreads();
 if (warp < 10) {
  val1 += sfacA[tid] * spartc[31 * order + (0 * delta + 0)];
  val3 += sfacA[tid] * spartc[31 * order + (2 * delta + 0)];
  val5 += sfacA[tid] * spartc[31 * order + (4 * delta + 0)];
  val1 += sfacB[tid] * spartc[31 * order + (0 * delta + 1)];
  val3 += sfacB[tid] * spartc[31 * order + (2 * delta + 1)];
  val5 += sfacB[tid] * spartc[31 * order + (4 * delta + 1)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 1 * delta)] = val1;
    spartc[clane + (31 * order + 5 * delta)] = val5;
  }

  __syncthreads();
 if (warp < 10) {
  val2 += sfacA[tid] * spartc[31 * order + (1 * delta + 0)];
  val6 += sfacA[tid] * spartc[31 * order + (5 * delta + 0)];
  val2 += sfacB[tid] * spartc[31 * order + (1 * delta + 1)];
  val6 += sfacB[tid] * spartc[31 * order + (5 * delta + 1)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 3 * delta)] = val3;
  }

  __syncthreads();
 if (warp < 10) {
  val4 += sfacA[tid] * spartc[31 * order + (3 * delta + 0)];
  val4 += sfacB[tid] * spartc[31 * order + (3 * delta + 1)];
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
      T fc;
      if (lane < order) {
        fc = fullcarry[(cidm1 - pos) * order + lane];
      }
      T X0 = __shfl(fc, 0);
      T X1 = __shfl(fc, 1);
      if (lane == 0) {
        sfullc[0] = X0;
        sfullc[1] = X1;
      }
    }

    __syncthreads();
    T X0 = sfullc[0];
    val0 += sfacA[tid] * X0;
    T X1 = sfullc[1];
    val0 += sfacB[tid] * X1;
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
  __shared__ T sfacB[block_size];
  if (tid < 300) sfacA[tid] = facA[tid];
  else sfacA[tid] = 0;
  if (tid < 302) sfacB[tid] = facB[tid];
  else sfacB[tid] = 0;

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

  val0 *= 1.436090e-01f;
  val1 *= 1.436090e-01f;
  val2 *= 1.436090e-01f;
  val3 *= 1.436090e-01f;
  val4 *= 1.436090e-01f;
  val5 *= 1.436090e-01f;
  val6 *= 1.436090e-01f;
  val7 *= 1.436090e-01f;

  const T sfA = sfacA[lane];
  const T sfB = sfacB[lane];

  int cond;
  T help, spc;

  help = 1.413320e+00f;
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
  spc = help * __shfl(val0, 0, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 4);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 0, 4);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 0, 4);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 0, 4);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 0, 4);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 0, 4);
  if (cond) val7 += spc;

  help = __shfl(sfB, lane % 2);
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
  spc = help * __shfl(val0, 2, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 2, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 2, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 2, 8);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 2, 8);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 2, 8);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 2, 8);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 2, 8);
  if (cond) val7 += spc;

  help = __shfl(sfB, lane % 4);
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
  spc = help * __shfl(val0, 6, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 6, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 6, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 6, 16);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 6, 16);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 6, 16);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 6, 16);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 6, 16);
  if (cond) val7 += spc;

  help = __shfl(sfB, lane % 8);
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
  spc = help * __shfl(val0, 14, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 14, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 14, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 14, 32);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 14, 32);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 14, 32);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 14, 32);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 14, 32);
  if (cond) val7 += spc;

  help = __shfl(sfB, lane % 16);
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
    const T helpB = sfacB[tid % (warp_size * 1)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
    val7 += helpB * spartc[cwarp + (7 * delta + 1)];
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
    const T helpB = sfacB[tid % (warp_size * 2)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
    val7 += helpB * spartc[cwarp + (7 * delta + 1)];
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
    const T helpB = sfacB[tid % (warp_size * 4)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
    val7 += helpB * spartc[cwarp + (7 * delta + 1)];
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
    const T helpB = sfacB[tid % (warp_size * 8)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
    val7 += helpB * spartc[cwarp + (7 * delta + 1)];
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
   if ((warp & 15) < 10) {
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
    const T helpB = sfacB[tid % (warp_size * 16)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
    val7 += helpB * spartc[cwarp + (7 * delta + 1)];
   }
    if ((warp == 31) && (clane >= 0)) {
      spartc[clane + (31 * order + 0 * delta)] = val0;
      spartc[clane + (31 * order + 2 * delta)] = val2;
      spartc[clane + (31 * order + 4 * delta)] = val4;
      spartc[clane + (31 * order + 6 * delta)] = val6;
    }
  }

  __syncthreads();
 if (warp < 10) {
  val1 += sfacA[tid] * spartc[31 * order + (0 * delta + 0)];
  val3 += sfacA[tid] * spartc[31 * order + (2 * delta + 0)];
  val5 += sfacA[tid] * spartc[31 * order + (4 * delta + 0)];
  val7 += sfacA[tid] * spartc[31 * order + (6 * delta + 0)];
  val1 += sfacB[tid] * spartc[31 * order + (0 * delta + 1)];
  val3 += sfacB[tid] * spartc[31 * order + (2 * delta + 1)];
  val5 += sfacB[tid] * spartc[31 * order + (4 * delta + 1)];
  val7 += sfacB[tid] * spartc[31 * order + (6 * delta + 1)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 1 * delta)] = val1;
    spartc[clane + (31 * order + 5 * delta)] = val5;
  }

  __syncthreads();
 if (warp < 10) {
  val2 += sfacA[tid] * spartc[31 * order + (1 * delta + 0)];
  val6 += sfacA[tid] * spartc[31 * order + (5 * delta + 0)];
  val2 += sfacB[tid] * spartc[31 * order + (1 * delta + 1)];
  val6 += sfacB[tid] * spartc[31 * order + (5 * delta + 1)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 3 * delta)] = val3;
  }

  __syncthreads();
 if (warp < 10) {
  val4 += sfacA[tid] * spartc[31 * order + (3 * delta + 0)];
  val4 += sfacB[tid] * spartc[31 * order + (3 * delta + 1)];
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
      T fc;
      if (lane < order) {
        fc = fullcarry[(cidm1 - pos) * order + lane];
      }
      T X0 = __shfl(fc, 0);
      T X1 = __shfl(fc, 1);
      if (lane == 0) {
        sfullc[0] = X0;
        sfullc[1] = X1;
      }
    }

    __syncthreads();
    T X0 = sfullc[0];
    val0 += sfacA[tid] * X0;
    T X1 = sfullc[1];
    val0 += sfacB[tid] * X1;
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
  __shared__ T sfacB[block_size];
  if (tid < 300) sfacA[tid] = facA[tid];
  else sfacA[tid] = 0;
  if (tid < 302) sfacB[tid] = facB[tid];
  else sfacB[tid] = 0;

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

  val0 *= 1.436090e-01f;
  val1 *= 1.436090e-01f;
  val2 *= 1.436090e-01f;
  val3 *= 1.436090e-01f;
  val4 *= 1.436090e-01f;
  val5 *= 1.436090e-01f;
  val6 *= 1.436090e-01f;
  val7 *= 1.436090e-01f;
  val8 *= 1.436090e-01f;

  const T sfA = sfacA[lane];
  const T sfB = sfacB[lane];

  int cond;
  T help, spc;

  help = 1.413320e+00f;
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
  spc = help * __shfl(val0, 0, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 4);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 0, 4);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 0, 4);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 0, 4);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 0, 4);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 0, 4);
  if (cond) val7 += spc;
  spc = help * __shfl(val8, 0, 4);
  if (cond) val8 += spc;

  help = __shfl(sfB, lane % 2);
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
  spc = help * __shfl(val0, 2, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 2, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 2, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 2, 8);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 2, 8);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 2, 8);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 2, 8);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 2, 8);
  if (cond) val7 += spc;
  spc = help * __shfl(val8, 2, 8);
  if (cond) val8 += spc;

  help = __shfl(sfB, lane % 4);
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
  spc = help * __shfl(val0, 6, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 6, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 6, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 6, 16);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 6, 16);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 6, 16);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 6, 16);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 6, 16);
  if (cond) val7 += spc;
  spc = help * __shfl(val8, 6, 16);
  if (cond) val8 += spc;

  help = __shfl(sfB, lane % 8);
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
  spc = help * __shfl(val0, 14, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 14, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 14, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 14, 32);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 14, 32);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 14, 32);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 14, 32);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 14, 32);
  if (cond) val7 += spc;
  spc = help * __shfl(val8, 14, 32);
  if (cond) val8 += spc;

  help = __shfl(sfB, lane % 16);
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
    const T helpB = sfacB[tid % (warp_size * 1)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
    val7 += helpB * spartc[cwarp + (7 * delta + 1)];
    val8 += helpB * spartc[cwarp + (8 * delta + 1)];
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
    const T helpB = sfacB[tid % (warp_size * 2)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
    val7 += helpB * spartc[cwarp + (7 * delta + 1)];
    val8 += helpB * spartc[cwarp + (8 * delta + 1)];
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
    const T helpB = sfacB[tid % (warp_size * 4)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
    val7 += helpB * spartc[cwarp + (7 * delta + 1)];
    val8 += helpB * spartc[cwarp + (8 * delta + 1)];
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
    const T helpB = sfacB[tid % (warp_size * 8)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
    val7 += helpB * spartc[cwarp + (7 * delta + 1)];
    val8 += helpB * spartc[cwarp + (8 * delta + 1)];
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
   if ((warp & 15) < 10) {
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
    const T helpB = sfacB[tid % (warp_size * 16)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
    val7 += helpB * spartc[cwarp + (7 * delta + 1)];
    val8 += helpB * spartc[cwarp + (8 * delta + 1)];
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
 if (warp < 10) {
  val1 += sfacA[tid] * spartc[31 * order + (0 * delta + 0)];
  val3 += sfacA[tid] * spartc[31 * order + (2 * delta + 0)];
  val5 += sfacA[tid] * spartc[31 * order + (4 * delta + 0)];
  val7 += sfacA[tid] * spartc[31 * order + (6 * delta + 0)];
  val1 += sfacB[tid] * spartc[31 * order + (0 * delta + 1)];
  val3 += sfacB[tid] * spartc[31 * order + (2 * delta + 1)];
  val5 += sfacB[tid] * spartc[31 * order + (4 * delta + 1)];
  val7 += sfacB[tid] * spartc[31 * order + (6 * delta + 1)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 1 * delta)] = val1;
    spartc[clane + (31 * order + 5 * delta)] = val5;
  }

  __syncthreads();
 if (warp < 10) {
  val2 += sfacA[tid] * spartc[31 * order + (1 * delta + 0)];
  val6 += sfacA[tid] * spartc[31 * order + (5 * delta + 0)];
  val2 += sfacB[tid] * spartc[31 * order + (1 * delta + 1)];
  val6 += sfacB[tid] * spartc[31 * order + (5 * delta + 1)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 3 * delta)] = val3;
  }

  __syncthreads();
 if (warp < 10) {
  val4 += sfacA[tid] * spartc[31 * order + (3 * delta + 0)];
  val4 += sfacB[tid] * spartc[31 * order + (3 * delta + 1)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 7 * delta)] = val7;
  }

  __syncthreads();
 if (warp < 10) {
  val8 += sfacA[tid] * spartc[31 * order + (7 * delta + 0)];
  val8 += sfacB[tid] * spartc[31 * order + (7 * delta + 1)];
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
      T fc;
      if (lane < order) {
        fc = fullcarry[(cidm1 - pos) * order + lane];
      }
      T X0 = __shfl(fc, 0);
      T X1 = __shfl(fc, 1);
      if (lane == 0) {
        sfullc[0] = X0;
        sfullc[1] = X1;
      }
    }

    __syncthreads();
    T X0 = sfullc[0];
    val0 += sfacA[tid] * X0;
    T X1 = sfullc[1];
    val0 += sfacB[tid] * X1;
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

int main(int argc, char *argv[])
{
  printf("Parallel Linear Recurrence Computation\n");
  printf("Copyright (c) 2018 Texas State University\n");

  if (argc != 2) {
    fprintf(stderr, "USAGE: %s problem_size\n", argv[0]);
    return -1;
  }

  const int n = atoi(argv[1]);
  if (n < 1) {fprintf(stderr, "ERROR: problem_size must be at least 1\n");  return -1;};

  int *d_status;
  T *h_in, *h_out, *h_sol, *d_in, *d_out, *d_partcarry, *d_fullcarry;

  const size_t size = n * sizeof(T);
  h_in = (T *)malloc(size);  assert(h_in != NULL);
  h_out = (T *)malloc(size);  assert(h_out != NULL);
  h_sol = (T *)malloc(size);  assert(h_sol != NULL);

  for (int i = 0; i < n; i++) {
    h_in[i] = (i & 32) / 16 - 1;
    h_sol[i] = 0;
  }
  for (int i = 0; i < n; i++) {
    if ((i - 0) >= 0) {
      h_sol[i] += 1.436090e-01f * h_in[i - 0];
    }
  }
  for (int i = 1; i < n; i++) {
    if ((i - 1) >= 0) {
      h_sol[i] += 1.413320e+00f * h_sol[i - 1];
    }
    if ((i - 2) >= 0) {
      h_sol[i] += -5.569270e-01f * h_sol[i - 2];
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
  const int iterations = 5;

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
  for (long i = 0; i < iterations; i++) {
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

  for (int i = 0; i < n; i++) {
    T s = h_sol[i];
    T o = h_out[i];
    if (fabsf(o - s) > 0.001) {
      printf("result not correct at index %d: %e != %e\n", i, h_sol[i], h_out[i]);
      return -1;
    }
  }
  printf("size = %d\tthroughput = %7.4f gigaitems/s\truntime = %7.4f s\tPassed!\n", n, throughput, runtime);

  printf("first elements of result are:\n");
  for (int i = 0; (i < 8) && (i < n); i++) {
    printf(" %f", h_out[i]);
  }
  printf("\n");

  free(h_in);  free(h_out);  free(h_sol);
  cudaFree(d_in);  cudaFree(d_out);  cudaFree(d_status);  cudaFree(d_partcarry);  cudaFree(d_fullcarry);

  return 0;
}
