/*
PLR - Parallelized Linear Recurrences [int]
Copyright (c) 2018 Texas State University. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted for academic, research, experimental, or personal use provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
* Neither the name of Texas State University nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

For all other uses, please contact the Office for Commercialization and Industry Relations at Texas State University http://www.txstate.edu/ocir/.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Sepideh Maleki and Martin Burtscher


non-recursive coefficients: (1)
recursive coefficients: (1)

*/

#include <cstdlib>

#include <iostream>
#include <fstream>

#include <cstdio>
#include <cassert>
#include <cuda.h>

typedef long int T;
static const int device = 0;
static const int order = 1;
static const int warp_size = 32;
static const int block_size = 1024;

// shared memory size is 1416 bytes

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

    const T sfA = 1;

    int cond;
    T help, spc;

    help = sfA;
    cond = ((lane & 1) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 0, 2);
    if (cond) val0 += spc;

    help = sfA;
    cond = ((lane & 2) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 1, 4);
    if (cond) val0 += spc;

    help = sfA;
    cond = ((lane & 4) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 3, 8);
    if (cond) val0 += spc;

    help = sfA;
    cond = ((lane & 8) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 7, 16);
    if (cond) val0 += spc;

    help = sfA;
    cond = ((lane & 16) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 15, 32);
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
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
        }
        if (((warp & 3) != 0) && (clane >= 0)) {
            spartc[clwo + 0 * delta] = val0;
        }
    }

    __syncthreads();
    if ((warp & 2) != 0) {
        const int cwarp = ((warp & ~3) | 1) * order;
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
        }
        if (((warp & 7) != 0) && (clane >= 0)) {
            spartc[clwo + 0 * delta] = val0;
        }
    }

    __syncthreads();
    if ((warp & 4) != 0) {
        const int cwarp = ((warp & ~7) | 3) * order;
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
        }
        if (((warp & 15) != 0) && (clane >= 0)) {
            spartc[clwo + 0 * delta] = val0;
        }
    }

    __syncthreads();
    if ((warp & 8) != 0) {
        const int cwarp = ((warp & ~15) | 7) * order;
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
        }
        if ((warp == 15) && (clane >= 0)) {
            spartc[clane + (15 * order + 0 * delta)] = val0;
        }
    }

    __syncthreads();
    if ((warp & 16) != 0) {
        const int cwarp = 15 * order;
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
        }
        if ((warp == 31) && (clane >= 0)) {
            spartc[clane + (31 * order + 0 * delta)] = val0;
        }
    }

    const int idx = tid - (block_size - order);
    if (idx >= 0) {
        if (chunk_id == 0) {
            fullcarry[idx] = val0;
        } else {
            partcarry[chunk_id * order + idx] = val0;
        }
        __threadfence();
        if (idx == 0) {
            if (chunk_id == 0) {
                status[0] = 2;
            } else {
                status[chunk_id] = 1;
            }
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
            for (int i = ((chunk_id - pos) * order + lane); i < (chunk_id * order); i += warp_size) {
                spartc[i - (chunk_id - pos) * order] = partcarry[i];
            }
            if (lane < pos) {
                const T c0 = spartc[(pos - 1 - lane) * order + 0];
                for (int i = pos - 1; i >= 0; i--) {
                    const T h0 = __shfl(c0, i) + X0 * 1;
                    X0 = h0;
                }
            }
            if (lane == 0) {
                sfullc[0] = X0;
            }
        }

        __syncthreads();
        T X0 = sfullc[0];
        val0 += 1 * X0;

        if (idx >= 0) {
            fullcarry[chunk_id * order + idx] = val0;
            __threadfence();
            status[chunk_id] = 2;
        }
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

    const T sfA = 1;

    int cond;
    T help, spc;

    help = sfA;
    cond = ((lane & 1) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 0, 2);
    if (cond) val0 += spc;
    spc = __shfl(val1, 0, 2);
    if (cond) val1 += spc;

    help = sfA;
    cond = ((lane & 2) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 1, 4);
    if (cond) val0 += spc;
    spc = __shfl(val1, 1, 4);
    if (cond) val1 += spc;

    help = sfA;
    cond = ((lane & 4) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 3, 8);
    if (cond) val0 += spc;
    spc = __shfl(val1, 3, 8);
    if (cond) val1 += spc;

    help = sfA;
    cond = ((lane & 8) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 7, 16);
    if (cond) val0 += spc;
    spc = __shfl(val1, 7, 16);
    if (cond) val1 += spc;

    help = sfA;
    cond = ((lane & 16) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 15, 32);
    if (cond) val0 += spc;
    spc = __shfl(val1, 15, 32);
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
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
        }
        if (((warp & 3) != 0) && (clane >= 0)) {
            spartc[clwo + 0 * delta] = val0;
            spartc[clwo + 1 * delta] = val1;
        }
    }

    __syncthreads();
    if ((warp & 2) != 0) {
        const int cwarp = ((warp & ~3) | 1) * order;
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
        }
        if (((warp & 7) != 0) && (clane >= 0)) {
            spartc[clwo + 0 * delta] = val0;
            spartc[clwo + 1 * delta] = val1;
        }
    }

    __syncthreads();
    if ((warp & 4) != 0) {
        const int cwarp = ((warp & ~7) | 3) * order;
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
        }
        if (((warp & 15) != 0) && (clane >= 0)) {
            spartc[clwo + 0 * delta] = val0;
            spartc[clwo + 1 * delta] = val1;
        }
    }

    __syncthreads();
    if ((warp & 8) != 0) {
        const int cwarp = ((warp & ~15) | 7) * order;
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
        }
        if ((warp == 15) && (clane >= 0)) {
            spartc[clane + (15 * order + 0 * delta)] = val0;
            spartc[clane + (15 * order + 1 * delta)] = val1;
        }
    }

    __syncthreads();
    if ((warp & 16) != 0) {
        const int cwarp = 15 * order;
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
        }
        if ((warp == 31) && (clane >= 0)) {
            spartc[clane + (31 * order + 0 * delta)] = val0;
        }
    }

    __syncthreads();
    val1 += 1 * spartc[31 * order + (0 * delta + 0)];

    const int idx = tid - (block_size - order);
    if (idx >= 0) {
        if (chunk_id == 0) {
            fullcarry[idx] = val1;
        } else {
            partcarry[chunk_id * order + idx] = val1;
        }
        __threadfence();
        if (idx == 0) {
            if (chunk_id == 0) {
                status[0] = 2;
            } else {
                status[chunk_id] = 1;
            }
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
            for (int i = ((chunk_id - pos) * order + lane); i < (chunk_id * order); i += warp_size) {
                spartc[i - (chunk_id - pos) * order] = partcarry[i];
            }
            if (lane < pos) {
                const T c0 = spartc[(pos - 1 - lane) * order + 0];
                for (int i = pos - 1; i >= 0; i--) {
                    const T h0 = __shfl(c0, i) + X0 * 1;
                    X0 = h0;
                }
            }
            if (lane == 0) {
                sfullc[0] = X0;
            }
        }

        __syncthreads();
        T X0 = sfullc[0];
        val0 += 1 * X0;
        val1 += 1 * X0;

        if (idx >= 0) {
            fullcarry[chunk_id * order + idx] = val1;
            __threadfence();
            status[chunk_id] = 2;
        }
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

    const T sfA = 1;

    int cond;
    T help, spc;

    help = sfA;
    cond = ((lane & 1) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 0, 2);
    if (cond) val0 += spc;
    spc = __shfl(val1, 0, 2);
    if (cond) val1 += spc;
    spc = __shfl(val2, 0, 2);
    if (cond) val2 += spc;

    help = sfA;
    cond = ((lane & 2) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 1, 4);
    if (cond) val0 += spc;
    spc = __shfl(val1, 1, 4);
    if (cond) val1 += spc;
    spc = __shfl(val2, 1, 4);
    if (cond) val2 += spc;

    help = sfA;
    cond = ((lane & 4) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 3, 8);
    if (cond) val0 += spc;
    spc = __shfl(val1, 3, 8);
    if (cond) val1 += spc;
    spc = __shfl(val2, 3, 8);
    if (cond) val2 += spc;

    help = sfA;
    cond = ((lane & 8) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 7, 16);
    if (cond) val0 += spc;
    spc = __shfl(val1, 7, 16);
    if (cond) val1 += spc;
    spc = __shfl(val2, 7, 16);
    if (cond) val2 += spc;

    help = sfA;
    cond = ((lane & 16) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 15, 32);
    if (cond) val0 += spc;
    spc = __shfl(val1, 15, 32);
    if (cond) val1 += spc;
    spc = __shfl(val2, 15, 32);
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
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
        }
        if (((warp & 3) != 0) && (clane >= 0)) {
            spartc[clwo + 0 * delta] = val0;
            spartc[clwo + 1 * delta] = val1;
            spartc[clwo + 2 * delta] = val2;
        }
    }

    __syncthreads();
    if ((warp & 2) != 0) {
        const int cwarp = ((warp & ~3) | 1) * order;
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
        }
        if (((warp & 7) != 0) && (clane >= 0)) {
            spartc[clwo + 0 * delta] = val0;
            spartc[clwo + 1 * delta] = val1;
            spartc[clwo + 2 * delta] = val2;
        }
    }

    __syncthreads();
    if ((warp & 4) != 0) {
        const int cwarp = ((warp & ~7) | 3) * order;
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
        }
        if (((warp & 15) != 0) && (clane >= 0)) {
            spartc[clwo + 0 * delta] = val0;
            spartc[clwo + 1 * delta] = val1;
            spartc[clwo + 2 * delta] = val2;
        }
    }

    __syncthreads();
    if ((warp & 8) != 0) {
        const int cwarp = ((warp & ~15) | 7) * order;
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
        }
        if ((warp == 15) && (clane >= 0)) {
            spartc[clane + (15 * order + 0 * delta)] = val0;
            spartc[clane + (15 * order + 1 * delta)] = val1;
            spartc[clane + (15 * order + 2 * delta)] = val2;
        }
    }

    __syncthreads();
    if ((warp & 16) != 0) {
        const int cwarp = 15 * order;
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
        }
        if ((warp == 31) && (clane >= 0)) {
            spartc[clane + (31 * order + 0 * delta)] = val0;
            spartc[clane + (31 * order + 2 * delta)] = val2;
        }
    }

    __syncthreads();
    val1 += 1 * spartc[31 * order + (0 * delta + 0)];
    if ((warp == 31) && (clane >= 0)) {
        spartc[clane + (31 * order + 1 * delta)] = val1;
    }

    __syncthreads();
    val2 += 1 * spartc[31 * order + (1 * delta + 0)];

    const int idx = tid - (block_size - order);
    if (idx >= 0) {
        if (chunk_id == 0) {
            fullcarry[idx] = val2;
        } else {
            partcarry[chunk_id * order + idx] = val2;
        }
        __threadfence();
        if (idx == 0) {
            if (chunk_id == 0) {
                status[0] = 2;
            } else {
                status[chunk_id] = 1;
            }
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
            for (int i = ((chunk_id - pos) * order + lane); i < (chunk_id * order); i += warp_size) {
                spartc[i - (chunk_id - pos) * order] = partcarry[i];
            }
            if (lane < pos) {
                const T c0 = spartc[(pos - 1 - lane) * order + 0];
                for (int i = pos - 1; i >= 0; i--) {
                    const T h0 = __shfl(c0, i) + X0 * 1;
                    X0 = h0;
                }
            }
            if (lane == 0) {
                sfullc[0] = X0;
            }
        }

        __syncthreads();
        T X0 = sfullc[0];
        val0 += 1 * X0;
        val1 += 1 * X0;
        val2 += 1 * X0;

        if (idx >= 0) {
            fullcarry[chunk_id * order + idx] = val2;
            __threadfence();
            status[chunk_id] = 2;
        }
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

    const T sfA = 1;

    int cond;
    T help, spc;

    help = sfA;
    cond = ((lane & 1) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 0, 2);
    if (cond) val0 += spc;
    spc = __shfl(val1, 0, 2);
    if (cond) val1 += spc;
    spc = __shfl(val2, 0, 2);
    if (cond) val2 += spc;
    spc = __shfl(val3, 0, 2);
    if (cond) val3 += spc;

    help = sfA;
    cond = ((lane & 2) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 1, 4);
    if (cond) val0 += spc;
    spc = __shfl(val1, 1, 4);
    if (cond) val1 += spc;
    spc = __shfl(val2, 1, 4);
    if (cond) val2 += spc;
    spc = __shfl(val3, 1, 4);
    if (cond) val3 += spc;

    help = sfA;
    cond = ((lane & 4) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 3, 8);
    if (cond) val0 += spc;
    spc = __shfl(val1, 3, 8);
    if (cond) val1 += spc;
    spc = __shfl(val2, 3, 8);
    if (cond) val2 += spc;
    spc = __shfl(val3, 3, 8);
    if (cond) val3 += spc;

    help = sfA;
    cond = ((lane & 8) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 7, 16);
    if (cond) val0 += spc;
    spc = __shfl(val1, 7, 16);
    if (cond) val1 += spc;
    spc = __shfl(val2, 7, 16);
    if (cond) val2 += spc;
    spc = __shfl(val3, 7, 16);
    if (cond) val3 += spc;

    help = sfA;
    cond = ((lane & 16) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 15, 32);
    if (cond) val0 += spc;
    spc = __shfl(val1, 15, 32);
    if (cond) val1 += spc;
    spc = __shfl(val2, 15, 32);
    if (cond) val2 += spc;
    spc = __shfl(val3, 15, 32);
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
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
        }
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
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
        }
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
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
        }
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
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
        }
        if ((warp == 15) && (clane >= 0)) {
            spartc[clane + (15 * order + 0 * delta)] = val0;
            spartc[clane + (15 * order + 1 * delta)] = val1;
            spartc[clane + (15 * order + 2 * delta)] = val2;
            spartc[clane + (15 * order + 3 * delta)] = val3;
        }
    }

    __syncthreads();
    if ((warp & 16) != 0) {
        const int cwarp = 15 * order;
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
        }
        if ((warp == 31) && (clane >= 0)) {
            spartc[clane + (31 * order + 0 * delta)] = val0;
            spartc[clane + (31 * order + 2 * delta)] = val2;
        }
    }

    __syncthreads();
    val1 += 1 * spartc[31 * order + (0 * delta + 0)];
    val3 += 1 * spartc[31 * order + (2 * delta + 0)];
    if ((warp == 31) && (clane >= 0)) {
        spartc[clane + (31 * order + 1 * delta)] = val1;
    }

    __syncthreads();
    val2 += 1 * spartc[31 * order + (1 * delta + 0)];
    val3 += 1 * spartc[31 * order + (1 * delta + 0)];

    const int idx = tid - (block_size - order);
    if (idx >= 0) {
        if (chunk_id == 0) {
            fullcarry[idx] = val3;
        } else {
            partcarry[chunk_id * order + idx] = val3;
        }
        __threadfence();
        if (idx == 0) {
            if (chunk_id == 0) {
                status[0] = 2;
            } else {
                status[chunk_id] = 1;
            }
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
            for (int i = ((chunk_id - pos) * order + lane); i < (chunk_id * order); i += warp_size) {
                spartc[i - (chunk_id - pos) * order] = partcarry[i];
            }
            if (lane < pos) {
                const T c0 = spartc[(pos - 1 - lane) * order + 0];
                for (int i = pos - 1; i >= 0; i--) {
                    const T h0 = __shfl(c0, i) + X0 * 1;
                    X0 = h0;
                }
            }
            if (lane == 0) {
                sfullc[0] = X0;
            }
        }

        __syncthreads();
        T X0 = sfullc[0];
        val0 += 1 * X0;
        val1 += 1 * X0;
        val2 += 1 * X0;
        val3 += 1 * X0;

        if (idx >= 0) {
            fullcarry[chunk_id * order + idx] = val3;
            __threadfence();
            status[chunk_id] = 2;
        }
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

    const T sfA = 1;

    int cond;
    T help, spc;

    help = sfA;
    cond = ((lane & 1) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 0, 2);
    if (cond) val0 += spc;
    spc = __shfl(val1, 0, 2);
    if (cond) val1 += spc;
    spc = __shfl(val2, 0, 2);
    if (cond) val2 += spc;
    spc = __shfl(val3, 0, 2);
    if (cond) val3 += spc;
    spc = __shfl(val4, 0, 2);
    if (cond) val4 += spc;

    help = sfA;
    cond = ((lane & 2) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 1, 4);
    if (cond) val0 += spc;
    spc = __shfl(val1, 1, 4);
    if (cond) val1 += spc;
    spc = __shfl(val2, 1, 4);
    if (cond) val2 += spc;
    spc = __shfl(val3, 1, 4);
    if (cond) val3 += spc;
    spc = __shfl(val4, 1, 4);
    if (cond) val4 += spc;

    help = sfA;
    cond = ((lane & 4) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 3, 8);
    if (cond) val0 += spc;
    spc = __shfl(val1, 3, 8);
    if (cond) val1 += spc;
    spc = __shfl(val2, 3, 8);
    if (cond) val2 += spc;
    spc = __shfl(val3, 3, 8);
    if (cond) val3 += spc;
    spc = __shfl(val4, 3, 8);
    if (cond) val4 += spc;

    help = sfA;
    cond = ((lane & 8) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 7, 16);
    if (cond) val0 += spc;
    spc = __shfl(val1, 7, 16);
    if (cond) val1 += spc;
    spc = __shfl(val2, 7, 16);
    if (cond) val2 += spc;
    spc = __shfl(val3, 7, 16);
    if (cond) val3 += spc;
    spc = __shfl(val4, 7, 16);
    if (cond) val4 += spc;

    help = sfA;
    cond = ((lane & 16) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 15, 32);
    if (cond) val0 += spc;
    spc = __shfl(val1, 15, 32);
    if (cond) val1 += spc;
    spc = __shfl(val2, 15, 32);
    if (cond) val2 += spc;
    spc = __shfl(val3, 15, 32);
    if (cond) val3 += spc;
    spc = __shfl(val4, 15, 32);
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
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
        }
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
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
        }
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
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
        }
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
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
        }
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
        const int cwarp = 15 * order;
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
        }
        if ((warp == 31) && (clane >= 0)) {
            spartc[clane + (31 * order + 0 * delta)] = val0;
            spartc[clane + (31 * order + 2 * delta)] = val2;
            spartc[clane + (31 * order + 4 * delta)] = val4;
        }
    }

    __syncthreads();
    val1 += 1 * spartc[31 * order + (0 * delta + 0)];
    val3 += 1 * spartc[31 * order + (2 * delta + 0)];
    if ((warp == 31) && (clane >= 0)) {
        spartc[clane + (31 * order + 1 * delta)] = val1;
    }

    __syncthreads();
    val2 += 1 * spartc[31 * order + (1 * delta + 0)];
    val3 += 1 * spartc[31 * order + (1 * delta + 0)];
    if ((warp == 31) && (clane >= 0)) {
        spartc[clane + (31 * order + 3 * delta)] = val3;
    }

    __syncthreads();
    val4 += 1 * spartc[31 * order + (3 * delta + 0)];

    const int idx = tid - (block_size - order);
    if (idx >= 0) {
        if (chunk_id == 0) {
            fullcarry[idx] = val4;
        } else {
            partcarry[chunk_id * order + idx] = val4;
        }
        __threadfence();
        if (idx == 0) {
            if (chunk_id == 0) {
                status[0] = 2;
            } else {
                status[chunk_id] = 1;
            }
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
            for (int i = ((chunk_id - pos) * order + lane); i < (chunk_id * order); i += warp_size) {
                spartc[i - (chunk_id - pos) * order] = partcarry[i];
            }
            if (lane < pos) {
                const T c0 = spartc[(pos - 1 - lane) * order + 0];
                for (int i = pos - 1; i >= 0; i--) {
                    const T h0 = __shfl(c0, i) + X0 * 1;
                    X0 = h0;
                }
            }
            if (lane == 0) {
                sfullc[0] = X0;
            }
        }

        __syncthreads();
        T X0 = sfullc[0];
        val0 += 1 * X0;
        val1 += 1 * X0;
        val2 += 1 * X0;
        val3 += 1 * X0;
        val4 += 1 * X0;

        if (idx >= 0) {
            fullcarry[chunk_id * order + idx] = val4;
            __threadfence();
            status[chunk_id] = 2;
        }
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

    const T sfA = 1;

    int cond;
    T help, spc;

    help = sfA;
    cond = ((lane & 1) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 0, 2);
    if (cond) val0 += spc;
    spc = __shfl(val1, 0, 2);
    if (cond) val1 += spc;
    spc = __shfl(val2, 0, 2);
    if (cond) val2 += spc;
    spc = __shfl(val3, 0, 2);
    if (cond) val3 += spc;
    spc = __shfl(val4, 0, 2);
    if (cond) val4 += spc;
    spc = __shfl(val5, 0, 2);
    if (cond) val5 += spc;

    help = sfA;
    cond = ((lane & 2) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 1, 4);
    if (cond) val0 += spc;
    spc = __shfl(val1, 1, 4);
    if (cond) val1 += spc;
    spc = __shfl(val2, 1, 4);
    if (cond) val2 += spc;
    spc = __shfl(val3, 1, 4);
    if (cond) val3 += spc;
    spc = __shfl(val4, 1, 4);
    if (cond) val4 += spc;
    spc = __shfl(val5, 1, 4);
    if (cond) val5 += spc;

    help = sfA;
    cond = ((lane & 4) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 3, 8);
    if (cond) val0 += spc;
    spc = __shfl(val1, 3, 8);
    if (cond) val1 += spc;
    spc = __shfl(val2, 3, 8);
    if (cond) val2 += spc;
    spc = __shfl(val3, 3, 8);
    if (cond) val3 += spc;
    spc = __shfl(val4, 3, 8);
    if (cond) val4 += spc;
    spc = __shfl(val5, 3, 8);
    if (cond) val5 += spc;

    help = sfA;
    cond = ((lane & 8) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 7, 16);
    if (cond) val0 += spc;
    spc = __shfl(val1, 7, 16);
    if (cond) val1 += spc;
    spc = __shfl(val2, 7, 16);
    if (cond) val2 += spc;
    spc = __shfl(val3, 7, 16);
    if (cond) val3 += spc;
    spc = __shfl(val4, 7, 16);
    if (cond) val4 += spc;
    spc = __shfl(val5, 7, 16);
    if (cond) val5 += spc;

    help = sfA;
    cond = ((lane & 16) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 15, 32);
    if (cond) val0 += spc;
    spc = __shfl(val1, 15, 32);
    if (cond) val1 += spc;
    spc = __shfl(val2, 15, 32);
    if (cond) val2 += spc;
    spc = __shfl(val3, 15, 32);
    if (cond) val3 += spc;
    spc = __shfl(val4, 15, 32);
    if (cond) val4 += spc;
    spc = __shfl(val5, 15, 32);
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
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
        }
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
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
        }
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
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
        }
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
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
        }
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
        const int cwarp = 15 * order;
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
        }
        if ((warp == 31) && (clane >= 0)) {
            spartc[clane + (31 * order + 0 * delta)] = val0;
            spartc[clane + (31 * order + 2 * delta)] = val2;
            spartc[clane + (31 * order + 4 * delta)] = val4;
        }
    }

    __syncthreads();
    val1 += 1 * spartc[31 * order + (0 * delta + 0)];
    val3 += 1 * spartc[31 * order + (2 * delta + 0)];
    val5 += 1 * spartc[31 * order + (4 * delta + 0)];
    if ((warp == 31) && (clane >= 0)) {
        spartc[clane + (31 * order + 1 * delta)] = val1;
        spartc[clane + (31 * order + 5 * delta)] = val5;
    }

    __syncthreads();
    val2 += 1 * spartc[31 * order + (1 * delta + 0)];
    val3 += 1 * spartc[31 * order + (1 * delta + 0)];
    if ((warp == 31) && (clane >= 0)) {
        spartc[clane + (31 * order + 3 * delta)] = val3;
    }

    __syncthreads();
    val4 += 1 * spartc[31 * order + (3 * delta + 0)];
    val5 += 1 * spartc[31 * order + (3 * delta + 0)];

    const int idx = tid - (block_size - order);
    if (idx >= 0) {
        if (chunk_id == 0) {
            fullcarry[idx] = val5;
        } else {
            partcarry[chunk_id * order + idx] = val5;
        }
        __threadfence();
        if (idx == 0) {
            if (chunk_id == 0) {
                status[0] = 2;
            } else {
                status[chunk_id] = 1;
            }
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
            for (int i = ((chunk_id - pos) * order + lane); i < (chunk_id * order); i += warp_size) {
                spartc[i - (chunk_id - pos) * order] = partcarry[i];
            }
            if (lane < pos) {
                const T c0 = spartc[(pos - 1 - lane) * order + 0];
                for (int i = pos - 1; i >= 0; i--) {
                    const T h0 = __shfl(c0, i) + X0 * 1;
                    X0 = h0;
                }
            }
            if (lane == 0) {
                sfullc[0] = X0;
            }
        }

        __syncthreads();
        T X0 = sfullc[0];
        val0 += 1 * X0;
        val1 += 1 * X0;
        val2 += 1 * X0;
        val3 += 1 * X0;
        val4 += 1 * X0;
        val5 += 1 * X0;

        if (idx >= 0) {
            fullcarry[chunk_id * order + idx] = val5;
            __threadfence();
            status[chunk_id] = 2;
        }
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

    const T sfA = 1;

    int cond;
    T help, spc;

    help = sfA;
    cond = ((lane & 1) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 0, 2);
    if (cond) val0 += spc;
    spc = __shfl(val1, 0, 2);
    if (cond) val1 += spc;
    spc = __shfl(val2, 0, 2);
    if (cond) val2 += spc;
    spc = __shfl(val3, 0, 2);
    if (cond) val3 += spc;
    spc = __shfl(val4, 0, 2);
    if (cond) val4 += spc;
    spc = __shfl(val5, 0, 2);
    if (cond) val5 += spc;
    spc = __shfl(val6, 0, 2);
    if (cond) val6 += spc;

    help = sfA;
    cond = ((lane & 2) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 1, 4);
    if (cond) val0 += spc;
    spc = __shfl(val1, 1, 4);
    if (cond) val1 += spc;
    spc = __shfl(val2, 1, 4);
    if (cond) val2 += spc;
    spc = __shfl(val3, 1, 4);
    if (cond) val3 += spc;
    spc = __shfl(val4, 1, 4);
    if (cond) val4 += spc;
    spc = __shfl(val5, 1, 4);
    if (cond) val5 += spc;
    spc = __shfl(val6, 1, 4);
    if (cond) val6 += spc;

    help = sfA;
    cond = ((lane & 4) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 3, 8);
    if (cond) val0 += spc;
    spc = __shfl(val1, 3, 8);
    if (cond) val1 += spc;
    spc = __shfl(val2, 3, 8);
    if (cond) val2 += spc;
    spc = __shfl(val3, 3, 8);
    if (cond) val3 += spc;
    spc = __shfl(val4, 3, 8);
    if (cond) val4 += spc;
    spc = __shfl(val5, 3, 8);
    if (cond) val5 += spc;
    spc = __shfl(val6, 3, 8);
    if (cond) val6 += spc;

    help = sfA;
    cond = ((lane & 8) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 7, 16);
    if (cond) val0 += spc;
    spc = __shfl(val1, 7, 16);
    if (cond) val1 += spc;
    spc = __shfl(val2, 7, 16);
    if (cond) val2 += spc;
    spc = __shfl(val3, 7, 16);
    if (cond) val3 += spc;
    spc = __shfl(val4, 7, 16);
    if (cond) val4 += spc;
    spc = __shfl(val5, 7, 16);
    if (cond) val5 += spc;
    spc = __shfl(val6, 7, 16);
    if (cond) val6 += spc;

    help = sfA;
    cond = ((lane & 16) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 15, 32);
    if (cond) val0 += spc;
    spc = __shfl(val1, 15, 32);
    if (cond) val1 += spc;
    spc = __shfl(val2, 15, 32);
    if (cond) val2 += spc;
    spc = __shfl(val3, 15, 32);
    if (cond) val3 += spc;
    spc = __shfl(val4, 15, 32);
    if (cond) val4 += spc;
    spc = __shfl(val5, 15, 32);
    if (cond) val5 += spc;
    spc = __shfl(val6, 15, 32);
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
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
            val6 += spartc[cwarp + (6 * delta + 0)];
        }
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
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
            val6 += spartc[cwarp + (6 * delta + 0)];
        }
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
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
            val6 += spartc[cwarp + (6 * delta + 0)];
        }
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
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
            val6 += spartc[cwarp + (6 * delta + 0)];
        }
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
        const int cwarp = 15 * order;
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
            val6 += spartc[cwarp + (6 * delta + 0)];
        }
        if ((warp == 31) && (clane >= 0)) {
            spartc[clane + (31 * order + 0 * delta)] = val0;
            spartc[clane + (31 * order + 2 * delta)] = val2;
            spartc[clane + (31 * order + 4 * delta)] = val4;
            spartc[clane + (31 * order + 6 * delta)] = val6;
        }
    }

    __syncthreads();
    val1 += 1 * spartc[31 * order + (0 * delta + 0)];
    val3 += 1 * spartc[31 * order + (2 * delta + 0)];
    val5 += 1 * spartc[31 * order + (4 * delta + 0)];
    if ((warp == 31) && (clane >= 0)) {
        spartc[clane + (31 * order + 1 * delta)] = val1;
        spartc[clane + (31 * order + 5 * delta)] = val5;
    }

    __syncthreads();
    val2 += 1 * spartc[31 * order + (1 * delta + 0)];
    val3 += 1 * spartc[31 * order + (1 * delta + 0)];
    val6 += 1 * spartc[31 * order + (5 * delta + 0)];
    if ((warp == 31) && (clane >= 0)) {
        spartc[clane + (31 * order + 3 * delta)] = val3;
    }

    __syncthreads();
    val4 += 1 * spartc[31 * order + (3 * delta + 0)];
    val5 += 1 * spartc[31 * order + (3 * delta + 0)];
    val6 += 1 * spartc[31 * order + (3 * delta + 0)];

    const int idx = tid - (block_size - order);
    if (idx >= 0) {
        if (chunk_id == 0) {
            fullcarry[idx] = val6;
        } else {
            partcarry[chunk_id * order + idx] = val6;
        }
        __threadfence();
        if (idx == 0) {
            if (chunk_id == 0) {
                status[0] = 2;
            } else {
                status[chunk_id] = 1;
            }
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
            for (int i = ((chunk_id - pos) * order + lane); i < (chunk_id * order); i += warp_size) {
                spartc[i - (chunk_id - pos) * order] = partcarry[i];
            }
            if (lane < pos) {
                const T c0 = spartc[(pos - 1 - lane) * order + 0];
                for (int i = pos - 1; i >= 0; i--) {
                    const T h0 = __shfl(c0, i) + X0 * 1;
                    X0 = h0;
                }
            }
            if (lane == 0) {
                sfullc[0] = X0;
            }
        }

        __syncthreads();
        T X0 = sfullc[0];
        val0 += 1 * X0;
        val1 += 1 * X0;
        val2 += 1 * X0;
        val3 += 1 * X0;
        val4 += 1 * X0;
        val5 += 1 * X0;
        val6 += 1 * X0;

        if (idx >= 0) {
            fullcarry[chunk_id * order + idx] = val6;
            __threadfence();
            status[chunk_id] = 2;
        }
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

    const T sfA = 1;

    int cond;
    T help, spc;

    help = sfA;
    cond = ((lane & 1) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 0, 2);
    if (cond) val0 += spc;
    spc = __shfl(val1, 0, 2);
    if (cond) val1 += spc;
    spc = __shfl(val2, 0, 2);
    if (cond) val2 += spc;
    spc = __shfl(val3, 0, 2);
    if (cond) val3 += spc;
    spc = __shfl(val4, 0, 2);
    if (cond) val4 += spc;
    spc = __shfl(val5, 0, 2);
    if (cond) val5 += spc;
    spc = __shfl(val6, 0, 2);
    if (cond) val6 += spc;
    spc = __shfl(val7, 0, 2);
    if (cond) val7 += spc;

    help = sfA;
    cond = ((lane & 2) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 1, 4);
    if (cond) val0 += spc;
    spc = __shfl(val1, 1, 4);
    if (cond) val1 += spc;
    spc = __shfl(val2, 1, 4);
    if (cond) val2 += spc;
    spc = __shfl(val3, 1, 4);
    if (cond) val3 += spc;
    spc = __shfl(val4, 1, 4);
    if (cond) val4 += spc;
    spc = __shfl(val5, 1, 4);
    if (cond) val5 += spc;
    spc = __shfl(val6, 1, 4);
    if (cond) val6 += spc;
    spc = __shfl(val7, 1, 4);
    if (cond) val7 += spc;

    help = sfA;
    cond = ((lane & 4) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 3, 8);
    if (cond) val0 += spc;
    spc = __shfl(val1, 3, 8);
    if (cond) val1 += spc;
    spc = __shfl(val2, 3, 8);
    if (cond) val2 += spc;
    spc = __shfl(val3, 3, 8);
    if (cond) val3 += spc;
    spc = __shfl(val4, 3, 8);
    if (cond) val4 += spc;
    spc = __shfl(val5, 3, 8);
    if (cond) val5 += spc;
    spc = __shfl(val6, 3, 8);
    if (cond) val6 += spc;
    spc = __shfl(val7, 3, 8);
    if (cond) val7 += spc;

    help = sfA;
    cond = ((lane & 8) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 7, 16);
    if (cond) val0 += spc;
    spc = __shfl(val1, 7, 16);
    if (cond) val1 += spc;
    spc = __shfl(val2, 7, 16);
    if (cond) val2 += spc;
    spc = __shfl(val3, 7, 16);
    if (cond) val3 += spc;
    spc = __shfl(val4, 7, 16);
    if (cond) val4 += spc;
    spc = __shfl(val5, 7, 16);
    if (cond) val5 += spc;
    spc = __shfl(val6, 7, 16);
    if (cond) val6 += spc;
    spc = __shfl(val7, 7, 16);
    if (cond) val7 += spc;

    help = sfA;
    cond = ((lane & 16) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 15, 32);
    if (cond) val0 += spc;
    spc = __shfl(val1, 15, 32);
    if (cond) val1 += spc;
    spc = __shfl(val2, 15, 32);
    if (cond) val2 += spc;
    spc = __shfl(val3, 15, 32);
    if (cond) val3 += spc;
    spc = __shfl(val4, 15, 32);
    if (cond) val4 += spc;
    spc = __shfl(val5, 15, 32);
    if (cond) val5 += spc;
    spc = __shfl(val6, 15, 32);
    if (cond) val6 += spc;
    spc = __shfl(val7, 15, 32);
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
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
            val6 += spartc[cwarp + (6 * delta + 0)];
            val7 += spartc[cwarp + (7 * delta + 0)];
        }
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
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
            val6 += spartc[cwarp + (6 * delta + 0)];
            val7 += spartc[cwarp + (7 * delta + 0)];
        }
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
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
            val6 += spartc[cwarp + (6 * delta + 0)];
            val7 += spartc[cwarp + (7 * delta + 0)];
        }
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
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
            val6 += spartc[cwarp + (6 * delta + 0)];
            val7 += spartc[cwarp + (7 * delta + 0)];
        }
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
        const int cwarp = 15 * order;
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
            val6 += spartc[cwarp + (6 * delta + 0)];
            val7 += spartc[cwarp + (7 * delta + 0)];
        }
        if ((warp == 31) && (clane >= 0)) {
            spartc[clane + (31 * order + 0 * delta)] = val0;
            spartc[clane + (31 * order + 2 * delta)] = val2;
            spartc[clane + (31 * order + 4 * delta)] = val4;
            spartc[clane + (31 * order + 6 * delta)] = val6;
        }
    }

    __syncthreads();
    val1 += 1 * spartc[31 * order + (0 * delta + 0)];
    val3 += 1 * spartc[31 * order + (2 * delta + 0)];
    val5 += 1 * spartc[31 * order + (4 * delta + 0)];
    val7 += 1 * spartc[31 * order + (6 * delta + 0)];
    if ((warp == 31) && (clane >= 0)) {
        spartc[clane + (31 * order + 1 * delta)] = val1;
        spartc[clane + (31 * order + 5 * delta)] = val5;
    }

    __syncthreads();
    val2 += 1 * spartc[31 * order + (1 * delta + 0)];
    val3 += 1 * spartc[31 * order + (1 * delta + 0)];
    val6 += 1 * spartc[31 * order + (5 * delta + 0)];
    val7 += 1 * spartc[31 * order + (5 * delta + 0)];
    if ((warp == 31) && (clane >= 0)) {
        spartc[clane + (31 * order + 3 * delta)] = val3;
    }

    __syncthreads();
    val4 += 1 * spartc[31 * order + (3 * delta + 0)];
    val5 += 1 * spartc[31 * order + (3 * delta + 0)];
    val6 += 1 * spartc[31 * order + (3 * delta + 0)];
    val7 += 1 * spartc[31 * order + (3 * delta + 0)];

    const int idx = tid - (block_size - order);
    if (idx >= 0) {
        if (chunk_id == 0) {
            fullcarry[idx] = val7;
        } else {
            partcarry[chunk_id * order + idx] = val7;
        }
        __threadfence();
        if (idx == 0) {
            if (chunk_id == 0) {
                status[0] = 2;
            } else {
                status[chunk_id] = 1;
            }
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
            for (int i = ((chunk_id - pos) * order + lane); i < (chunk_id * order); i += warp_size) {
                spartc[i - (chunk_id - pos) * order] = partcarry[i];
            }
            if (lane < pos) {
                const T c0 = spartc[(pos - 1 - lane) * order + 0];
                for (int i = pos - 1; i >= 0; i--) {
                    const T h0 = __shfl(c0, i) + X0 * 1;
                    X0 = h0;
                }
            }
            if (lane == 0) {
                sfullc[0] = X0;
            }
        }

        __syncthreads();
        T X0 = sfullc[0];
        val0 += 1 * X0;
        val1 += 1 * X0;
        val2 += 1 * X0;
        val3 += 1 * X0;
        val4 += 1 * X0;
        val5 += 1 * X0;
        val6 += 1 * X0;
        val7 += 1 * X0;

        if (idx >= 0) {
            fullcarry[chunk_id * order + idx] = val7;
            __threadfence();
            status[chunk_id] = 2;
        }
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

    const T sfA = 1;

    int cond;
    T help, spc;

    help = sfA;
    cond = ((lane & 1) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 0, 2);
    if (cond) val0 += spc;
    spc = __shfl(val1, 0, 2);
    if (cond) val1 += spc;
    spc = __shfl(val2, 0, 2);
    if (cond) val2 += spc;
    spc = __shfl(val3, 0, 2);
    if (cond) val3 += spc;
    spc = __shfl(val4, 0, 2);
    if (cond) val4 += spc;
    spc = __shfl(val5, 0, 2);
    if (cond) val5 += spc;
    spc = __shfl(val6, 0, 2);
    if (cond) val6 += spc;
    spc = __shfl(val7, 0, 2);
    if (cond) val7 += spc;
    spc = __shfl(val8, 0, 2);
    if (cond) val8 += spc;

    help = sfA;
    cond = ((lane & 2) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 1, 4);
    if (cond) val0 += spc;
    spc = __shfl(val1, 1, 4);
    if (cond) val1 += spc;
    spc = __shfl(val2, 1, 4);
    if (cond) val2 += spc;
    spc = __shfl(val3, 1, 4);
    if (cond) val3 += spc;
    spc = __shfl(val4, 1, 4);
    if (cond) val4 += spc;
    spc = __shfl(val5, 1, 4);
    if (cond) val5 += spc;
    spc = __shfl(val6, 1, 4);
    if (cond) val6 += spc;
    spc = __shfl(val7, 1, 4);
    if (cond) val7 += spc;
    spc = __shfl(val8, 1, 4);
    if (cond) val8 += spc;

    help = sfA;
    cond = ((lane & 4) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 3, 8);
    if (cond) val0 += spc;
    spc = __shfl(val1, 3, 8);
    if (cond) val1 += spc;
    spc = __shfl(val2, 3, 8);
    if (cond) val2 += spc;
    spc = __shfl(val3, 3, 8);
    if (cond) val3 += spc;
    spc = __shfl(val4, 3, 8);
    if (cond) val4 += spc;
    spc = __shfl(val5, 3, 8);
    if (cond) val5 += spc;
    spc = __shfl(val6, 3, 8);
    if (cond) val6 += spc;
    spc = __shfl(val7, 3, 8);
    if (cond) val7 += spc;
    spc = __shfl(val8, 3, 8);
    if (cond) val8 += spc;

    help = sfA;
    cond = ((lane & 8) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 7, 16);
    if (cond) val0 += spc;
    spc = __shfl(val1, 7, 16);
    if (cond) val1 += spc;
    spc = __shfl(val2, 7, 16);
    if (cond) val2 += spc;
    spc = __shfl(val3, 7, 16);
    if (cond) val3 += spc;
    spc = __shfl(val4, 7, 16);
    if (cond) val4 += spc;
    spc = __shfl(val5, 7, 16);
    if (cond) val5 += spc;
    spc = __shfl(val6, 7, 16);
    if (cond) val6 += spc;
    spc = __shfl(val7, 7, 16);
    if (cond) val7 += spc;
    spc = __shfl(val8, 7, 16);
    if (cond) val8 += spc;

    help = sfA;
    cond = ((lane & 16) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 15, 32);
    if (cond) val0 += spc;
    spc = __shfl(val1, 15, 32);
    if (cond) val1 += spc;
    spc = __shfl(val2, 15, 32);
    if (cond) val2 += spc;
    spc = __shfl(val3, 15, 32);
    if (cond) val3 += spc;
    spc = __shfl(val4, 15, 32);
    if (cond) val4 += spc;
    spc = __shfl(val5, 15, 32);
    if (cond) val5 += spc;
    spc = __shfl(val6, 15, 32);
    if (cond) val6 += spc;
    spc = __shfl(val7, 15, 32);
    if (cond) val7 += spc;
    spc = __shfl(val8, 15, 32);
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
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
            val6 += spartc[cwarp + (6 * delta + 0)];
            val7 += spartc[cwarp + (7 * delta + 0)];
            val8 += spartc[cwarp + (8 * delta + 0)];
        }
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
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
            val6 += spartc[cwarp + (6 * delta + 0)];
            val7 += spartc[cwarp + (7 * delta + 0)];
            val8 += spartc[cwarp + (8 * delta + 0)];
        }
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
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
            val6 += spartc[cwarp + (6 * delta + 0)];
            val7 += spartc[cwarp + (7 * delta + 0)];
            val8 += spartc[cwarp + (8 * delta + 0)];
        }
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
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
            val6 += spartc[cwarp + (6 * delta + 0)];
            val7 += spartc[cwarp + (7 * delta + 0)];
            val8 += spartc[cwarp + (8 * delta + 0)];
        }
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
        const int cwarp = 15 * order;
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
            val6 += spartc[cwarp + (6 * delta + 0)];
            val7 += spartc[cwarp + (7 * delta + 0)];
            val8 += spartc[cwarp + (8 * delta + 0)];
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
    val1 += 1 * spartc[31 * order + (0 * delta + 0)];
    val3 += 1 * spartc[31 * order + (2 * delta + 0)];
    val5 += 1 * spartc[31 * order + (4 * delta + 0)];
    val7 += 1 * spartc[31 * order + (6 * delta + 0)];
    if ((warp == 31) && (clane >= 0)) {
        spartc[clane + (31 * order + 1 * delta)] = val1;
        spartc[clane + (31 * order + 5 * delta)] = val5;
    }

    __syncthreads();
    val2 += 1 * spartc[31 * order + (1 * delta + 0)];
    val3 += 1 * spartc[31 * order + (1 * delta + 0)];
    val6 += 1 * spartc[31 * order + (5 * delta + 0)];
    val7 += 1 * spartc[31 * order + (5 * delta + 0)];
    if ((warp == 31) && (clane >= 0)) {
        spartc[clane + (31 * order + 3 * delta)] = val3;
    }

    __syncthreads();
    val4 += 1 * spartc[31 * order + (3 * delta + 0)];
    val5 += 1 * spartc[31 * order + (3 * delta + 0)];
    val6 += 1 * spartc[31 * order + (3 * delta + 0)];
    val7 += 1 * spartc[31 * order + (3 * delta + 0)];
    if ((warp == 31) && (clane >= 0)) {
        spartc[clane + (31 * order + 7 * delta)] = val7;
    }

    __syncthreads();
    val8 += 1 * spartc[31 * order + (7 * delta + 0)];

    const int idx = tid - (block_size - order);
    if (idx >= 0) {
        if (chunk_id == 0) {
            fullcarry[idx] = val8;
        } else {
            partcarry[chunk_id * order + idx] = val8;
        }
        __threadfence();
        if (idx == 0) {
            if (chunk_id == 0) {
                status[0] = 2;
            } else {
                status[chunk_id] = 1;
            }
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
            for (int i = ((chunk_id - pos) * order + lane); i < (chunk_id * order); i += warp_size) {
                spartc[i - (chunk_id - pos) * order] = partcarry[i];
            }
            if (lane < pos) {
                const T c0 = spartc[(pos - 1 - lane) * order + 0];
                for (int i = pos - 1; i >= 0; i--) {
                    const T h0 = __shfl(c0, i) + X0 * 1;
                    X0 = h0;
                }
            }
            if (lane == 0) {
                sfullc[0] = X0;
            }
        }

        __syncthreads();
        T X0 = sfullc[0];
        val0 += 1 * X0;
        val1 += 1 * X0;
        val2 += 1 * X0;
        val3 += 1 * X0;
        val4 += 1 * X0;
        val5 += 1 * X0;
        val6 += 1 * X0;
        val7 += 1 * X0;
        val8 += 1 * X0;

        if (idx >= 0) {
            fullcarry[chunk_id * order + idx] = val8;
            __threadfence();
            status[chunk_id] = 2;
        }
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

static __global__ __launch_bounds__(block_size, 2)
    void Recurrence10(const int items, const T* const __restrict__ input, T* const __restrict__ output, volatile int* const __restrict__ status, volatile T* const __restrict__ partcarry, volatile T* const __restrict__ fullcarry)
{
    const int valsperthread = 10;
    const int chunk_size = valsperthread * block_size;
    __shared__ T spartc[chunk_size / warp_size * order];
    __shared__ T sfullc[order];
    __shared__ int cid;

    const int tid = threadIdx.x;
    const int warp = tid / warp_size;
    const int lane = tid % warp_size;

    if (tid == 0) {
        cid = atomicInc(&counter, gridDim.x - 1);
    }

    __syncthreads();
    const int chunk_id = cid;
    const int offs = tid + chunk_id * chunk_size;

    T val0, val1, val2, val3, val4, val5, val6, val7, val8, val9;
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
        val9 = 0;
        if (offs + (9 * block_size) < items) val9 = input[offs + (9 * block_size)];
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
        val9 = input[offs + (9 * block_size)];
    }

    const T sfA = 1;

    int cond;
    T help, spc;

    help = sfA;
    cond = ((lane & 1) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 0, 2);
    if (cond) val0 += spc;
    spc = __shfl(val1, 0, 2);
    if (cond) val1 += spc;
    spc = __shfl(val2, 0, 2);
    if (cond) val2 += spc;
    spc = __shfl(val3, 0, 2);
    if (cond) val3 += spc;
    spc = __shfl(val4, 0, 2);
    if (cond) val4 += spc;
    spc = __shfl(val5, 0, 2);
    if (cond) val5 += spc;
    spc = __shfl(val6, 0, 2);
    if (cond) val6 += spc;
    spc = __shfl(val7, 0, 2);
    if (cond) val7 += spc;
    spc = __shfl(val8, 0, 2);
    if (cond) val8 += spc;
    spc = __shfl(val9, 0, 2);
    if (cond) val9 += spc;

    help = sfA;
    cond = ((lane & 2) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 1, 4);
    if (cond) val0 += spc;
    spc = __shfl(val1, 1, 4);
    if (cond) val1 += spc;
    spc = __shfl(val2, 1, 4);
    if (cond) val2 += spc;
    spc = __shfl(val3, 1, 4);
    if (cond) val3 += spc;
    spc = __shfl(val4, 1, 4);
    if (cond) val4 += spc;
    spc = __shfl(val5, 1, 4);
    if (cond) val5 += spc;
    spc = __shfl(val6, 1, 4);
    if (cond) val6 += spc;
    spc = __shfl(val7, 1, 4);
    if (cond) val7 += spc;
    spc = __shfl(val8, 1, 4);
    if (cond) val8 += spc;
    spc = __shfl(val9, 1, 4);
    if (cond) val9 += spc;

    help = sfA;
    cond = ((lane & 4) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 3, 8);
    if (cond) val0 += spc;
    spc = __shfl(val1, 3, 8);
    if (cond) val1 += spc;
    spc = __shfl(val2, 3, 8);
    if (cond) val2 += spc;
    spc = __shfl(val3, 3, 8);
    if (cond) val3 += spc;
    spc = __shfl(val4, 3, 8);
    if (cond) val4 += spc;
    spc = __shfl(val5, 3, 8);
    if (cond) val5 += spc;
    spc = __shfl(val6, 3, 8);
    if (cond) val6 += spc;
    spc = __shfl(val7, 3, 8);
    if (cond) val7 += spc;
    spc = __shfl(val8, 3, 8);
    if (cond) val8 += spc;
    spc = __shfl(val9, 3, 8);
    if (cond) val9 += spc;

    help = sfA;
    cond = ((lane & 8) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 7, 16);
    if (cond) val0 += spc;
    spc = __shfl(val1, 7, 16);
    if (cond) val1 += spc;
    spc = __shfl(val2, 7, 16);
    if (cond) val2 += spc;
    spc = __shfl(val3, 7, 16);
    if (cond) val3 += spc;
    spc = __shfl(val4, 7, 16);
    if (cond) val4 += spc;
    spc = __shfl(val5, 7, 16);
    if (cond) val5 += spc;
    spc = __shfl(val6, 7, 16);
    if (cond) val6 += spc;
    spc = __shfl(val7, 7, 16);
    if (cond) val7 += spc;
    spc = __shfl(val8, 7, 16);
    if (cond) val8 += spc;
    spc = __shfl(val9, 7, 16);
    if (cond) val9 += spc;

    help = sfA;
    cond = ((lane & 16) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 15, 32);
    if (cond) val0 += spc;
    spc = __shfl(val1, 15, 32);
    if (cond) val1 += spc;
    spc = __shfl(val2, 15, 32);
    if (cond) val2 += spc;
    spc = __shfl(val3, 15, 32);
    if (cond) val3 += spc;
    spc = __shfl(val4, 15, 32);
    if (cond) val4 += spc;
    spc = __shfl(val5, 15, 32);
    if (cond) val5 += spc;
    spc = __shfl(val6, 15, 32);
    if (cond) val6 += spc;
    spc = __shfl(val7, 15, 32);
    if (cond) val7 += spc;
    spc = __shfl(val8, 15, 32);
    if (cond) val8 += spc;
    spc = __shfl(val9, 15, 32);
    if (cond) val9 += spc;

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
        spartc[clwo + 9 * delta] = val9;
    }

    __syncthreads();
    if ((warp & 1) != 0) {
        const int cwarp = ((warp & ~1) | 0) * order;
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
            val6 += spartc[cwarp + (6 * delta + 0)];
            val7 += spartc[cwarp + (7 * delta + 0)];
            val8 += spartc[cwarp + (8 * delta + 0)];
            val9 += spartc[cwarp + (9 * delta + 0)];
        }
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
            spartc[clwo + 9 * delta] = val9;
        }
    }

    __syncthreads();
    if ((warp & 2) != 0) {
        const int cwarp = ((warp & ~3) | 1) * order;
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
            val6 += spartc[cwarp + (6 * delta + 0)];
            val7 += spartc[cwarp + (7 * delta + 0)];
            val8 += spartc[cwarp + (8 * delta + 0)];
            val9 += spartc[cwarp + (9 * delta + 0)];
        }
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
            spartc[clwo + 9 * delta] = val9;
        }
    }

    __syncthreads();
    if ((warp & 4) != 0) {
        const int cwarp = ((warp & ~7) | 3) * order;
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
            val6 += spartc[cwarp + (6 * delta + 0)];
            val7 += spartc[cwarp + (7 * delta + 0)];
            val8 += spartc[cwarp + (8 * delta + 0)];
            val9 += spartc[cwarp + (9 * delta + 0)];
        }
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
            spartc[clwo + 9 * delta] = val9;
        }
    }

    __syncthreads();
    if ((warp & 8) != 0) {
        const int cwarp = ((warp & ~15) | 7) * order;
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
            val6 += spartc[cwarp + (6 * delta + 0)];
            val7 += spartc[cwarp + (7 * delta + 0)];
            val8 += spartc[cwarp + (8 * delta + 0)];
            val9 += spartc[cwarp + (9 * delta + 0)];
        }
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
            spartc[clane + (15 * order + 9 * delta)] = val9;
        }
    }

    __syncthreads();
    if ((warp & 16) != 0) {
        const int cwarp = 15 * order;
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
            val6 += spartc[cwarp + (6 * delta + 0)];
            val7 += spartc[cwarp + (7 * delta + 0)];
            val8 += spartc[cwarp + (8 * delta + 0)];
            val9 += spartc[cwarp + (9 * delta + 0)];
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
    val1 += 1 * spartc[31 * order + (0 * delta + 0)];
    val3 += 1 * spartc[31 * order + (2 * delta + 0)];
    val5 += 1 * spartc[31 * order + (4 * delta + 0)];
    val7 += 1 * spartc[31 * order + (6 * delta + 0)];
    val9 += 1 * spartc[31 * order + (8 * delta + 0)];
    if ((warp == 31) && (clane >= 0)) {
        spartc[clane + (31 * order + 1 * delta)] = val1;
        spartc[clane + (31 * order + 5 * delta)] = val5;
        spartc[clane + (31 * order + 9 * delta)] = val9;
    }

    __syncthreads();
    val2 += 1 * spartc[31 * order + (1 * delta + 0)];
    val3 += 1 * spartc[31 * order + (1 * delta + 0)];
    val6 += 1 * spartc[31 * order + (5 * delta + 0)];
    val7 += 1 * spartc[31 * order + (5 * delta + 0)];
    if ((warp == 31) && (clane >= 0)) {
        spartc[clane + (31 * order + 3 * delta)] = val3;
    }

    __syncthreads();
    val4 += 1 * spartc[31 * order + (3 * delta + 0)];
    val5 += 1 * spartc[31 * order + (3 * delta + 0)];
    val6 += 1 * spartc[31 * order + (3 * delta + 0)];
    val7 += 1 * spartc[31 * order + (3 * delta + 0)];
    if ((warp == 31) && (clane >= 0)) {
        spartc[clane + (31 * order + 7 * delta)] = val7;
    }

    __syncthreads();
    val8 += 1 * spartc[31 * order + (7 * delta + 0)];
    val9 += 1 * spartc[31 * order + (7 * delta + 0)];

    const int idx = tid - (block_size - order);
    if (idx >= 0) {
        if (chunk_id == 0) {
            fullcarry[idx] = val9;
        } else {
            partcarry[chunk_id * order + idx] = val9;
        }
        __threadfence();
        if (idx == 0) {
            if (chunk_id == 0) {
                status[0] = 2;
            } else {
                status[chunk_id] = 1;
            }
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
            for (int i = ((chunk_id - pos) * order + lane); i < (chunk_id * order); i += warp_size) {
                spartc[i - (chunk_id - pos) * order] = partcarry[i];
            }
            if (lane < pos) {
                const T c0 = spartc[(pos - 1 - lane) * order + 0];
                for (int i = pos - 1; i >= 0; i--) {
                    const T h0 = __shfl(c0, i) + X0 * 1;
                    X0 = h0;
                }
            }
            if (lane == 0) {
                sfullc[0] = X0;
            }
        }

        __syncthreads();
        T X0 = sfullc[0];
        val0 += 1 * X0;
        val1 += 1 * X0;
        val2 += 1 * X0;
        val3 += 1 * X0;
        val4 += 1 * X0;
        val5 += 1 * X0;
        val6 += 1 * X0;
        val7 += 1 * X0;
        val8 += 1 * X0;
        val9 += 1 * X0;

        if (idx >= 0) {
            fullcarry[chunk_id * order + idx] = val9;
            __threadfence();
            status[chunk_id] = 2;
        }
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
        if (offs + (9 * block_size) < items) output[offs + (9 * block_size)] = val9;
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
        output[offs + (9 * block_size)] = val9;
    }
}

static __global__ __launch_bounds__(block_size, 2)
    void Recurrence11(const int items, const T* const __restrict__ input, T* const __restrict__ output, volatile int* const __restrict__ status, volatile T* const __restrict__ partcarry, volatile T* const __restrict__ fullcarry)
{
    const int valsperthread = 11;
    const int chunk_size = valsperthread * block_size;
    __shared__ T spartc[chunk_size / warp_size * order];
    __shared__ T sfullc[order];
    __shared__ int cid;

    const int tid = threadIdx.x;
    const int warp = tid / warp_size;
    const int lane = tid % warp_size;

    if (tid == 0) {
        cid = atomicInc(&counter, gridDim.x - 1);
    }

    __syncthreads();
    const int chunk_id = cid;
    const int offs = tid + chunk_id * chunk_size;

    T val0, val1, val2, val3, val4, val5, val6, val7, val8, val9, val10;
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
        val9 = 0;
        if (offs + (9 * block_size) < items) val9 = input[offs + (9 * block_size)];
        val10 = 0;
        if (offs + (10 * block_size) < items) val10 = input[offs + (10 * block_size)];
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
        val9 = input[offs + (9 * block_size)];
        val10 = input[offs + (10 * block_size)];
    }

    const T sfA = 1;

    int cond;
    T help, spc;

    help = sfA;
    cond = ((lane & 1) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 0, 2);
    if (cond) val0 += spc;
    spc = __shfl(val1, 0, 2);
    if (cond) val1 += spc;
    spc = __shfl(val2, 0, 2);
    if (cond) val2 += spc;
    spc = __shfl(val3, 0, 2);
    if (cond) val3 += spc;
    spc = __shfl(val4, 0, 2);
    if (cond) val4 += spc;
    spc = __shfl(val5, 0, 2);
    if (cond) val5 += spc;
    spc = __shfl(val6, 0, 2);
    if (cond) val6 += spc;
    spc = __shfl(val7, 0, 2);
    if (cond) val7 += spc;
    spc = __shfl(val8, 0, 2);
    if (cond) val8 += spc;
    spc = __shfl(val9, 0, 2);
    if (cond) val9 += spc;
    spc = __shfl(val10, 0, 2);
    if (cond) val10 += spc;

    help = sfA;
    cond = ((lane & 2) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 1, 4);
    if (cond) val0 += spc;
    spc = __shfl(val1, 1, 4);
    if (cond) val1 += spc;
    spc = __shfl(val2, 1, 4);
    if (cond) val2 += spc;
    spc = __shfl(val3, 1, 4);
    if (cond) val3 += spc;
    spc = __shfl(val4, 1, 4);
    if (cond) val4 += spc;
    spc = __shfl(val5, 1, 4);
    if (cond) val5 += spc;
    spc = __shfl(val6, 1, 4);
    if (cond) val6 += spc;
    spc = __shfl(val7, 1, 4);
    if (cond) val7 += spc;
    spc = __shfl(val8, 1, 4);
    if (cond) val8 += spc;
    spc = __shfl(val9, 1, 4);
    if (cond) val9 += spc;
    spc = __shfl(val10, 1, 4);
    if (cond) val10 += spc;

    help = sfA;
    cond = ((lane & 4) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 3, 8);
    if (cond) val0 += spc;
    spc = __shfl(val1, 3, 8);
    if (cond) val1 += spc;
    spc = __shfl(val2, 3, 8);
    if (cond) val2 += spc;
    spc = __shfl(val3, 3, 8);
    if (cond) val3 += spc;
    spc = __shfl(val4, 3, 8);
    if (cond) val4 += spc;
    spc = __shfl(val5, 3, 8);
    if (cond) val5 += spc;
    spc = __shfl(val6, 3, 8);
    if (cond) val6 += spc;
    spc = __shfl(val7, 3, 8);
    if (cond) val7 += spc;
    spc = __shfl(val8, 3, 8);
    if (cond) val8 += spc;
    spc = __shfl(val9, 3, 8);
    if (cond) val9 += spc;
    spc = __shfl(val10, 3, 8);
    if (cond) val10 += spc;

    help = sfA;
    cond = ((lane & 8) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 7, 16);
    if (cond) val0 += spc;
    spc = __shfl(val1, 7, 16);
    if (cond) val1 += spc;
    spc = __shfl(val2, 7, 16);
    if (cond) val2 += spc;
    spc = __shfl(val3, 7, 16);
    if (cond) val3 += spc;
    spc = __shfl(val4, 7, 16);
    if (cond) val4 += spc;
    spc = __shfl(val5, 7, 16);
    if (cond) val5 += spc;
    spc = __shfl(val6, 7, 16);
    if (cond) val6 += spc;
    spc = __shfl(val7, 7, 16);
    if (cond) val7 += spc;
    spc = __shfl(val8, 7, 16);
    if (cond) val8 += spc;
    spc = __shfl(val9, 7, 16);
    if (cond) val9 += spc;
    spc = __shfl(val10, 7, 16);
    if (cond) val10 += spc;

    help = sfA;
    cond = ((lane & 16) != 0);
    if (!help) cond = 0;
    spc = __shfl(val0, 15, 32);
    if (cond) val0 += spc;
    spc = __shfl(val1, 15, 32);
    if (cond) val1 += spc;
    spc = __shfl(val2, 15, 32);
    if (cond) val2 += spc;
    spc = __shfl(val3, 15, 32);
    if (cond) val3 += spc;
    spc = __shfl(val4, 15, 32);
    if (cond) val4 += spc;
    spc = __shfl(val5, 15, 32);
    if (cond) val5 += spc;
    spc = __shfl(val6, 15, 32);
    if (cond) val6 += spc;
    spc = __shfl(val7, 15, 32);
    if (cond) val7 += spc;
    spc = __shfl(val8, 15, 32);
    if (cond) val8 += spc;
    spc = __shfl(val9, 15, 32);
    if (cond) val9 += spc;
    spc = __shfl(val10, 15, 32);
    if (cond) val10 += spc;

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
        spartc[clwo + 9 * delta] = val9;
        spartc[clwo + 10 * delta] = val10;
    }

    __syncthreads();
    if ((warp & 1) != 0) {
        const int cwarp = ((warp & ~1) | 0) * order;
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
            val6 += spartc[cwarp + (6 * delta + 0)];
            val7 += spartc[cwarp + (7 * delta + 0)];
            val8 += spartc[cwarp + (8 * delta + 0)];
            val9 += spartc[cwarp + (9 * delta + 0)];
            val10 += spartc[cwarp + (10 * delta + 0)];
        }
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
            spartc[clwo + 9 * delta] = val9;
            spartc[clwo + 10 * delta] = val10;
        }
    }

    __syncthreads();
    if ((warp & 2) != 0) {
        const int cwarp = ((warp & ~3) | 1) * order;
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
            val6 += spartc[cwarp + (6 * delta + 0)];
            val7 += spartc[cwarp + (7 * delta + 0)];
            val8 += spartc[cwarp + (8 * delta + 0)];
            val9 += spartc[cwarp + (9 * delta + 0)];
            val10 += spartc[cwarp + (10 * delta + 0)];
        }
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
            spartc[clwo + 9 * delta] = val9;
            spartc[clwo + 10 * delta] = val10;
        }
    }

    __syncthreads();
    if ((warp & 4) != 0) {
        const int cwarp = ((warp & ~7) | 3) * order;
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
            val6 += spartc[cwarp + (6 * delta + 0)];
            val7 += spartc[cwarp + (7 * delta + 0)];
            val8 += spartc[cwarp + (8 * delta + 0)];
            val9 += spartc[cwarp + (9 * delta + 0)];
            val10 += spartc[cwarp + (10 * delta + 0)];
        }
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
            spartc[clwo + 9 * delta] = val9;
            spartc[clwo + 10 * delta] = val10;
        }
    }

    __syncthreads();
    if ((warp & 8) != 0) {
        const int cwarp = ((warp & ~15) | 7) * order;
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
            val6 += spartc[cwarp + (6 * delta + 0)];
            val7 += spartc[cwarp + (7 * delta + 0)];
            val8 += spartc[cwarp + (8 * delta + 0)];
            val9 += spartc[cwarp + (9 * delta + 0)];
            val10 += spartc[cwarp + (10 * delta + 0)];
        }
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
            spartc[clane + (15 * order + 9 * delta)] = val9;
            spartc[clane + (15 * order + 10 * delta)] = val10;
        }
    }

    __syncthreads();
    if ((warp & 16) != 0) {
        const int cwarp = 15 * order;
        const T helpA = 1;
        if (helpA) {
            val0 += spartc[cwarp + (0 * delta + 0)];
            val1 += spartc[cwarp + (1 * delta + 0)];
            val2 += spartc[cwarp + (2 * delta + 0)];
            val3 += spartc[cwarp + (3 * delta + 0)];
            val4 += spartc[cwarp + (4 * delta + 0)];
            val5 += spartc[cwarp + (5 * delta + 0)];
            val6 += spartc[cwarp + (6 * delta + 0)];
            val7 += spartc[cwarp + (7 * delta + 0)];
            val8 += spartc[cwarp + (8 * delta + 0)];
            val9 += spartc[cwarp + (9 * delta + 0)];
            val10 += spartc[cwarp + (10 * delta + 0)];
        }
        if ((warp == 31) && (clane >= 0)) {
            spartc[clane + (31 * order + 0 * delta)] = val0;
            spartc[clane + (31 * order + 2 * delta)] = val2;
            spartc[clane + (31 * order + 4 * delta)] = val4;
            spartc[clane + (31 * order + 6 * delta)] = val6;
            spartc[clane + (31 * order + 8 * delta)] = val8;
            spartc[clane + (31 * order + 10 * delta)] = val10;
        }
    }

    __syncthreads();
    val1 += 1 * spartc[31 * order + (0 * delta + 0)];
    val3 += 1 * spartc[31 * order + (2 * delta + 0)];
    val5 += 1 * spartc[31 * order + (4 * delta + 0)];
    val7 += 1 * spartc[31 * order + (6 * delta + 0)];
    val9 += 1 * spartc[31 * order + (8 * delta + 0)];
    if ((warp == 31) && (clane >= 0)) {
        spartc[clane + (31 * order + 1 * delta)] = val1;
        spartc[clane + (31 * order + 5 * delta)] = val5;
        spartc[clane + (31 * order + 9 * delta)] = val9;
    }

    __syncthreads();
    val2 += 1 * spartc[31 * order + (1 * delta + 0)];
    val3 += 1 * spartc[31 * order + (1 * delta + 0)];
    val6 += 1 * spartc[31 * order + (5 * delta + 0)];
    val7 += 1 * spartc[31 * order + (5 * delta + 0)];
    val10 += 1 * spartc[31 * order + (9 * delta + 0)];
    if ((warp == 31) && (clane >= 0)) {
        spartc[clane + (31 * order + 3 * delta)] = val3;
    }

    __syncthreads();
    val4 += 1 * spartc[31 * order + (3 * delta + 0)];
    val5 += 1 * spartc[31 * order + (3 * delta + 0)];
    val6 += 1 * spartc[31 * order + (3 * delta + 0)];
    val7 += 1 * spartc[31 * order + (3 * delta + 0)];
    if ((warp == 31) && (clane >= 0)) {
        spartc[clane + (31 * order + 7 * delta)] = val7;
    }

    __syncthreads();
    val8 += 1 * spartc[31 * order + (7 * delta + 0)];
    val9 += 1 * spartc[31 * order + (7 * delta + 0)];
    val10 += 1 * spartc[31 * order + (7 * delta + 0)];

    const int idx = tid - (block_size - order);
    if (idx >= 0) {
        if (chunk_id == 0) {
            fullcarry[idx] = val10;
        } else {
            partcarry[chunk_id * order + idx] = val10;
        }
        __threadfence();
        if (idx == 0) {
            if (chunk_id == 0) {
                status[0] = 2;
            } else {
                status[chunk_id] = 1;
            }
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
            for (int i = ((chunk_id - pos) * order + lane); i < (chunk_id * order); i += warp_size) {
                spartc[i - (chunk_id - pos) * order] = partcarry[i];
            }
            if (lane < pos) {
                const T c0 = spartc[(pos - 1 - lane) * order + 0];
                for (int i = pos - 1; i >= 0; i--) {
                    const T h0 = __shfl(c0, i) + X0 * 1;
                    X0 = h0;
                }
            }
            if (lane == 0) {
                sfullc[0] = X0;
            }
        }

        __syncthreads();
        T X0 = sfullc[0];
        val0 += 1 * X0;
        val1 += 1 * X0;
        val2 += 1 * X0;
        val3 += 1 * X0;
        val4 += 1 * X0;
        val5 += 1 * X0;
        val6 += 1 * X0;
        val7 += 1 * X0;
        val8 += 1 * X0;
        val9 += 1 * X0;
        val10 += 1 * X0;

        if (idx >= 0) {
            fullcarry[chunk_id * order + idx] = val10;
            __threadfence();
            status[chunk_id] = 2;
        }
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
        if (offs + (9 * block_size) < items) output[offs + (9 * block_size)] = val9;
        if (offs + (10 * block_size) < items) output[offs + (10 * block_size)] = val10;
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
        output[offs + (9 * block_size)] = val9;
        output[offs + (10 * block_size)] = val10;
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

int main(int argc, char *argv[])
{/*
    printf("Parallel Linear Recurrence Computation\n");
    printf("Copyright (c) 2018 Texas State University\n");
 */
    if (argc != 4) {
        fprintf(stderr, "USAGE: %s problem_size repeats array_bin_fn\n", argv[0]);
        return -1;
    }

    const long int n = atol(argv[1]);
    const long int iterations = atol(argv[2]);
    char array_bin_fn[200] = "../bin/random_array_double.bin";
    sscanf(argv[3], "%s", array_bin_fn);

    if (n < 1) {fprintf(stderr, "ERROR: problem_size must be at least 1\n");  return -1;};

    int *d_status;
    T *h_in, *h_out, *h_sol, *d_in, *d_out, *d_partcarry, *d_fullcarry;

    const size_t size = n * sizeof(T);
    h_in = (T *)malloc(size);  assert(h_in != NULL);
    h_out = (T *)malloc(size);  assert(h_out != NULL);
    h_sol = (T *)malloc(size);  assert(h_sol != NULL);

    std::ifstream in_file(array_bin_fn, std::ios::binary);
    in_file.read(reinterpret_cast<char*>(h_in),
                 sizeof(T)*n);
    in_file.close();

    for (long int i = 0; i < n; i++) {
//        h_in[i] = (i & 32) / 16 - 1;
        h_sol[i] = 0;
    }
    for (long int i = 0; i < n; i++) {
        if ((i - 0) >= 0) {
            h_sol[i] += 1 * h_in[i - 0];
        }
    }
    for (long int i = 1; i < n; i++) {
        if ((i - 1) >= 0) {
            h_sol[i] += 1 * h_sol[i - 1];
        }
    }

    cudaSetDevice(device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    const int SMs = deviceProp.multiProcessorCount;
    int valsperthread = 1;
    while ((valsperthread < 11) && (block_size * 2 * SMs * valsperthread < n)) {
        valsperthread++;
    }
    const int chunk_size = valsperthread * block_size;
//    const int iterations = 5;

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
    case 10:  Recurrence10<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
    case 11:  Recurrence11<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
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
        case 10:  Recurrence10<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
        case 11:  Recurrence11<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
        }
    }
    double runtime = timer.stop() / iterations;
    double throughput = 0.000000001 * n / runtime;
    assert(cudaSuccess == cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));
/*
    for (long int i = 0; i < n; i++) {
        if (h_sol[i] != h_out[i]) {
            printf("result not correct at index %d: %d != %d\n", i, h_sol[i], h_out[i]);
            return -1;
        }
    }

    printf("size = %d\tthroughput = %7.4f gigaitems/s\truntime = %7.4f s\tPassed!\n", n, throughput, runtime);

    printf("first elements of result are:\n");
    for (int i = 0; (i < 8) && (i < n); i++) {
        printf(" %d", h_out[i]);
    }
    printf("\n");
*/
    double mebissec = n / (runtime*1024*1024); // Mis/s
    double max_abs_err, max_rel_err;
    check_cpu_reference(h_sol, h_out, n, max_abs_err, max_rel_err);
    printf("%7.7f %e %e\n", mebissec, max_abs_err, max_rel_err);

    free(h_in);  free(h_out);  free(h_sol);
    cudaFree(d_in);  cudaFree(d_out);  cudaFree(d_status);  cudaFree(d_partcarry);  cudaFree(d_fullcarry);

    return 0;
}

