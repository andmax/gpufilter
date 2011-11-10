/**
 *  @file gpufilter.cu
 *  @brief CUDA device code for GPU-Efficient Recursive Filtering
 *  @author Rodolfo Lima
 *  @date September, 2011
 */

//== INCLUDES =================================================================

#include <cmath>
#include <cstdio>
#include <cfloat>
#include <cassert>
#include <iostream>
#include <algorithm>

#include <timer.h>
#include <symbol.h>
#include <dvector.h>

#include <gpufilter.h>
#include <gpudefs.cuh>
#include <gpuconsts.cuh>

#include <gpufilter.cuh>

//== NAMESPACES ===============================================================

namespace gpufilter {

//== IMPLEMENTATION ===========================================================

//-- Device -------------------------------------------------------------------

template <class T> 
__device__
inline void swap(T& a, T& b) {
    T c = a;
    a = b;
    b = c;
}

__device__ inline
float2 multgpu(const float M[2][2], const float2& v)
{
    return make_float2(M[0][0]*v.x + M[0][1]*v.y, M[1][0]*v.x + M[1][1]*v.y);
}

__device__ inline
float2 addgpu(const float2& u, const float2& v)
{
    return make_float2(u.x+v.x, u.y+v.y);
}

__device__ inline
float2 addgpu(const float2& u, const float2& v, const float2& w)
{
    return make_float2(u.x+v.x+w.x, u.y+v.y+w.y);
}

//-- Algorithm 4_2 ------------------------------------------------------------

__global__ __launch_bounds__(WS*SOW, MBO)
void algorithm4_stage1( const float *g_inout,
                        float2 *g_transp_ybar,
                        float2 *g_transp_zhat )
{
    int m = blockIdx.y, n = blockIdx.x;
    int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float block[WS][WS+1];

    g_inout += (m*WS + ty)*c_width + n*WS + tx;

#pragma unroll
    for(int i=0; i<WS-2; i+=SOW)
    {
        block[ty+i][tx] = g_inout[i*c_width];
    }

    if(ty < 2)
    {
        block[ty+WS-2][tx] = g_inout[(WS-2)*c_width];
    }

    __syncthreads();

    if(ty == 0) // one warp computing
    {
        float *bdata = block[tx];
        int outidx = n*c_height + m*WS + tx; // transposed!

        float2 accum;

        if(n < c_n_size-1)
        {
            accum.x = bdata[0];
            accum.y = bdata[1] -= c_Minf*accum.x;

            for(int i=2; i<WS; ++i)
            {
                accum.x = bdata[i] -= c_Minf*accum.y + c_Linf2*accum.x;
                swap(accum.x, accum.y);
            }
        }
        else
        {
            accum.x = bdata[0];
            accum.y = bdata[1] -= c_Minf*accum.x;

            for(int i=2; i<WS-1; ++i)
            {
                accum.x = bdata[i] -= c_Minf*accum.y + c_Linf2*accum.x;
                swap(accum.x, accum.y);
            }
            accum.x = bdata[WS-1] -= c_Minf*accum.y + c_Linf2*accum.x;
            swap(accum.x, accum.y);
        }

        g_transp_ybar[outidx] = accum;

        if(n < c_n_size-1)
        {
            accum.y = bdata[WS-1] * c_Linf2;
            accum.x = (bdata[WS-2] - accum.y*c_Ninf)*c_Linf2;

#pragma unroll
            for(int i=WS-3; i>=0; --i)
            {
                accum.y = (bdata[i] - accum.x*c_Ninf - accum.y)*c_Linf2;
                swap(accum.x, accum.y);
            }
        }
        else // last block
        {
            int i = WS-1;

            accum.y = bdata[i--] * c_Llast2;
            accum.x = (bdata[i--] - accum.y*c_Ninf)*c_Linf2;

            for(; i>=0; --i)
            {
                accum.y = (bdata[i] - accum.x*c_Ninf - accum.y)*c_Linf2;
                swap(accum.x, accum.y);
            }

        }

        g_transp_zhat[outidx] = accum;
    }
}

__global__ __launch_bounds__(MTS, MBO)
void algorithm4_stage2_3_or_5_6( float2 *g_transp_ybar,
                                 float2 *g_transp_zhat )
{
    int m = blockIdx.x;
    int tx = threadIdx.x;

    int row = m*blockDim.x + tx;

    if(row >= c_height) 
        return;

    g_transp_ybar += row;
    g_transp_zhat += row;

    float2 accum = g_transp_ybar[0];

    for(int j=1; j<c_n_size; ++j)
    {
        g_transp_ybar += c_height;

        *g_transp_ybar = accum = addgpu(*g_transp_ybar,
                                        multgpu(c_Af, accum));
    }

    g_transp_zhat += (c_n_size-1)*c_height;
    g_transp_ybar -= c_height;

    *g_transp_zhat = accum = addgpu(*g_transp_zhat,
                                    multgpu(c_Arf, *g_transp_ybar));

    for(int j=c_n_size-2; j>=1; --j)
    {
        g_transp_ybar -= c_height;
        g_transp_zhat -= c_height;

        *g_transp_zhat = accum = addgpu(*g_transp_zhat,
                                        multgpu(c_Ar, accum),
                                        multgpu(c_Arf, *g_transp_ybar));
    }
}

__global__ __launch_bounds__(WS*SOW, ONB)
void algorithm4_stage4( float *g_inout,
                        const float2 *g_transp_y,
                        const float2 *g_transp_z,
                        float2 *g_ubar,
                        float2 *g_vhat )
{
    int m = blockIdx.y*2, n = blockIdx.x;
    int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float block[WS*2][WS+1];

    g_inout += (m*WS + ty)*c_width + n*WS + tx;

#pragma unroll
    for(int i=0; i<WS-2; i+=SOW)
    {
        block[ty+i][tx] = g_inout[i*c_width];
        block[ty+i+WS][tx] = g_inout[(i+WS)*c_width];
    }

    if(ty < 2)
    {
        block[ty+WS-2][tx] = g_inout[(WS-2)*c_width];
        block[ty+WS-2+WS][tx] = g_inout[(WS-2+WS)*c_width];
    }

    __syncthreads();

    if(ty < 2)
    {
        m += ty;

        int outidx = n*c_height + m*WS + tx; // transposed!

        float *bdata = block[tx+ty*WS];

        if(n < c_n_size-1)
        {
            float2 accum;
            
            if(n == 0)
            {
                accum.x = bdata[0];
                accum.y = bdata[1] -= c_Minf*accum.x;
            }
            else
                accum = g_transp_y[outidx-c_height];

#pragma unroll
            for(int i=0; i<WS; ++i)
            {
                accum.x = bdata[i] -= c_Minf*accum.y + c_Linf2*accum.x;
                swap(accum.x, accum.y);
            }
        }
        else
        {
            float2 accum = g_transp_y[outidx-c_height];

            for(int i=0; i<WS-1; ++i)
            {
                accum.x = bdata[i] -= c_Minf*accum.y + c_Linf2*accum.x;
                swap(accum.x, accum.y);
            }
            bdata[WS-1] -= c_Minf*accum.y + c_Linf2*accum.x;

        }

        if(n < c_n_size-1)
        {
            float2 accum = g_transp_z[outidx+c_height];

#pragma unroll
            for(int i=WS-1; i>=0; --i)
            {
                accum.y = bdata[i] = (bdata[i] - accum.x*c_Ninf - accum.y)*c_Linf2;
                swap(accum.x,accum.y);
            }
        }
        // last block
        else
        {
            float2 accum;
            int i = WS-1;

            accum.y = bdata[i--] *= c_Linf2;
            accum.x = bdata[i] = (bdata[i] - accum.y*c_Ninf)*c_Linf2;
            --i;

            for(; i>=0; --i)
            {
                accum.y = bdata[i] = (bdata[i] - accum.x*c_Ninf - accum.y)*c_Linf2;
                swap(accum.x,accum.y);
            }
        }

        m -= ty;
    }

    __syncthreads();

#pragma unroll
    for(int i=0; i<WS-2; i+=SOW)
    {
        g_inout[i*c_width] = block[ty+i][tx];
        g_inout[(i+WS)*c_width] = block[ty+i+WS][tx];
    }

    if(ty < 2)
    {
        g_inout[(WS-2)*c_width] = block[ty+WS-2][tx];
        g_inout[(WS-2+WS)*c_width] = block[ty+WS-2+WS][tx];
    }

    if(ty < 2)
    {
        m += ty;

        float (*bdata)[WS+1] = (float (*)[WS+1]) &block[ty*WS][tx];

        int outidx = m*c_width + n*WS + tx; 
        float2 accum;

        // first block
        if(m < c_m_size-1)
        {
            accum.x = bdata[0][0];
            accum.y = bdata[1][0] -= c_Minf*accum.x;

#pragma unroll
            for(int i=2; i<WS; ++i)
            {
                accum.x = bdata[i][0] -= c_Minf*accum.y + c_Linf2*accum.x;
                swap(accum.x, accum.y);
            }
        }
        else
        {
            accum.x = bdata[0][0];
            accum.y = bdata[1][0] -= c_Minf*accum.x;

            for(int i=2; i<WS-1; ++i)
            {
                accum.x = bdata[i][0] -= c_Minf*accum.y + c_Linf2*accum.x;
                swap(accum.x, accum.y);
            }
            accum.x = bdata[WS-1][0] -= c_Minf*accum.y + c_Linf2*accum.x;
            swap(accum.x, accum.y);
        }

        g_ubar[outidx] = accum;

        if(m < c_m_size-1)
        {
            accum.y = bdata[WS-1][0] * c_Linf2;
            accum.x = (bdata[WS-2][0] - accum.y*c_Ninf)*c_Linf2;

#pragma unroll
            for(int i=WS-3; i>=0; --i)
            {
                accum.y = (bdata[i][0] - accum.x*c_Ninf - accum.y)*c_Linf2;
                swap(accum.x,accum.y);
            }
        }
        else
        {
            int i = WS-1;

            accum.y = bdata[i--][0] * c_Linf2;
            accum.x = (bdata[i--][0] - accum.y*c_Ninf)*c_Linf2;
            for(; i>=0; --i)
            {
                accum.y = (bdata[i][0] - accum.x*c_Ninf - accum.y)*c_Linf2;
                swap(accum.x,accum.y);
            }
        }

        g_vhat[outidx] = accum;
    }
}

__global__ __launch_bounds__(WS*SOW, DNB)
void algorithm4_stage7( float *g_inout,
                        const float2 *g_u,
                        const float2 *g_v )
{
    int m = blockIdx.y, n = blockIdx.x;
    int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float block[WS][WS+1];

    g_inout += (m*WS + ty)*c_width + n*WS + tx;

#pragma unroll
    for(int i=0; i<WS-2; i+=SOW)
    {
        block[tx][ty+i] = g_inout[i*c_width];
    }

    if(ty < 2)
    {
        block[tx][ty+WS-2] = g_inout[(WS-2)*c_width];
    }

    __syncthreads();

    if(ty == 0)
    {
        int outidx = m*c_width + n*WS + tx; 

        float *bdata = block[tx];

        // first block
        if(m < c_m_size-1)
        {
            float2 accum;

            if(m == 0)
            {
                accum.x = bdata[0];
                accum.y = bdata[1] -= c_Minf*accum.x;
            }
            else
                accum = g_u[outidx-c_width];

#pragma unroll
            for(int i=0; i<WS; ++i)
            {
                accum.x = bdata[i] -= c_Minf*accum.y + c_Linf2*accum.x;
                swap(accum.x, accum.y);
            }
        }
        else
        {
            float2 accum  = g_u[outidx-c_width];

            for(int i=0; i<WS-1; ++i)
            {
                accum.x = bdata[i] -= c_Minf*accum.y + c_Linf2*accum.x;
                swap(accum.x, accum.y);
            }
            bdata[WS-1] -= c_Minf*accum.y + c_Linf2*accum.x;
        }

        if(m < c_m_size-1)
        {
            float2 accum = g_v[outidx+c_width];

#pragma unroll
            for(int i=WS-1; i>=0; --i)
            {
                bdata[i] = accum.y = (bdata[i] - accum.x*c_Ninf - accum.y)*c_Linf2;
                swap(accum.x,accum.y);
            }
        }
        // last block
        else
        {
            int i = WS-1;

            float2 accum;
            accum.y = bdata[i--] *= c_Linf2;
            accum.x = bdata[i] = (bdata[i] - accum.y*c_Ninf)*c_Linf2;
            --i;
            for(; i>=0; --i)
            {
                bdata[i] = accum.y = (bdata[i] - accum.x*c_Ninf - accum.y)*c_Linf2;
                swap(accum.x,accum.y);
            }
        }
    }

    __syncthreads();

#pragma unroll
    for(int i=0; i<WS-2; i+=SOW)
    {
        g_inout[i*c_width] = block[tx][ty+i] * c_iR2;
    }

    if(ty < 2)
    {
        g_inout[(WS-2)*c_width] = block[tx][ty+WS-2] * c_iR2;
    }
}

//-- Algorithm 5_1 ------------------------------------------------------------

__global__ __launch_bounds__(WS*DW, ONB)
void algorithm5_stage1( const float *g_in,
                        float *g_transp_ybar,
                        float *g_transp_zhat,
                        float *g_ucheck,
                        float *g_vtilde )
{
    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.y*2, n = blockIdx.x;

    // each cuda block will work on two WSxWS input data blocks, so allocate
    // enough shared memory for these.
    __shared__ float block[WS*2][WS+1];

    // make g_in point to the data we'll work with
    g_in += (m*WS + ty)*c_width + n*WS + tx;

    // load data into shared memory
#pragma unroll
    for(int i=0; i<WS; i+=DW)
    {
        // load data for the first warp
        block[ty+i][tx] = g_in[i*c_width];

        // load data for the second warp
        block[ty+i+WS][tx] = g_in[(i+WS)*c_width];
    }

    __syncthreads();

    // use 2 warps for calculations, one for each scheduler of sm_20
    if(ty < 2)
    {
        // adjust 'm' for the second warp
        m += ty;

        {
            // ty*WS makes this warp's bdata point to the right data
            float *bdata = block[tx+ty*WS];

            // We use a transposed matrix for ybar and zhat to have
            // coalesced memory operations. This is the index for these
            // transposed buffers.
            int outidx = n*c_height + m*WS + tx;

            float prev;

            // calculate ybar --------------------------------

            prev = bdata[0];

#pragma unroll
            for(int j=1; j<WS; ++j)
                prev = bdata[j] -= prev*c_Linf1;

            g_transp_ybar[outidx] = prev;

            // calculate zhat --------------------------------

            prev = bdata[WS-1] *= c_Linf1;

            for(int j=WS-2; j>=0; --j)
                bdata[j] = prev = (bdata[j] - prev)*c_Linf1;

            g_transp_zhat[outidx] = prev;
        }

        {
            float (*bdata)[WS+1] = (float (*)[WS+1]) &block[ty*WS][tx];
            int outidx = m*c_width + n*WS+tx;
            float prev;

            // calculate ucheck ------------------------------

            prev = bdata[0][0];

#pragma unroll
            for(int i=1; i<WS; ++i)
                prev = bdata[i][0] -= prev*c_Linf1;

            g_ucheck[outidx] = prev;

            // calculate vtilde ------------------------------

            prev = bdata[WS-1][0] *= c_Linf1;

            for(int i=WS-2; i>=0; --i)
                bdata[i][0] = prev = (bdata[i][0] - prev)*c_Linf1;

            g_vtilde[outidx] = prev;
        }
    }
}

__global__ __launch_bounds__(WS*DW, DNB)
void algorithm5_stage2_3( float *g_transp_ybar,
                          float *g_transp_zhat )
{
    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.y*2;

    __shared__ float block[DW][WS*2];

    // ybar -> y processing --------------------------------------

    g_transp_ybar += m*WS+tx + ty*c_height;

    // first column-block

    block[ty][tx] = g_transp_ybar[0];
    block[ty][tx+WS] = g_transp_ybar[WS];

    __syncthreads();

    float py; // P(Y)
    if(ty < 2)
    {
        int i = tx+ty*WS;
        py = block[0][i];

#pragma unroll
        for(int j=1; j<blockDim.y; ++j)
            block[j][i] = py = block[j][i] + c_Svm*py;

    }

    __syncthreads();


    if(ty > 0)
    {
        g_transp_ybar[0] = block[ty][tx];
        g_transp_ybar[WS] = block[ty][tx+WS];
    }

    g_transp_ybar += c_height*blockDim.y;

    // middle column-blocks

    int n = blockDim.y;
    if(blockDim.y == DW)
    {
        for(; n<c_n_size-DW; 
            n+=DW, g_transp_ybar+=c_height*DW)
        {
            block[ty][tx] = g_transp_ybar[0];
            block[ty][tx+WS] = g_transp_ybar[WS];

            __syncthreads();

            if(ty < 2)
            {
                int i = tx+ty*WS;
#pragma unroll
                for(int j=0; j<DW; ++j)
                    block[j][i] = py = block[j][i] + c_Svm*py;
            }

            __syncthreads();

            g_transp_ybar[0] = block[ty][tx];
            g_transp_ybar[WS] = block[ty][tx+WS];
        }
    }

    // remaining column-blocks

    if(c_n_size > n)
    {
        int remaining = c_n_size-n;
        if(ty < remaining)
        {
            block[ty][tx] = g_transp_ybar[0];
            block[ty][tx+WS] = g_transp_ybar[WS];
        }

        __syncthreads();

        if(ty < 2)
        {
            int i = tx+ty*WS;
#pragma unroll
            for(int j=0; j<remaining; ++j)
                block[j][i] = py = block[j][i] + c_Svm*py;

        }

        __syncthreads();

        if(ty < remaining)
        {
            g_transp_ybar[0] = block[ty][tx];
            g_transp_ybar[WS] = block[ty][tx+WS];
        }

        g_transp_ybar += remaining*c_height;
    }

    // zhat -> z processing --------------------------------------

    const float *g_y = g_transp_ybar - (2*ty+2)*c_height;

    __shared__ float block_y[DW][WS*2];

    float ez; // E(Z)

    // last column-block

    g_transp_zhat += m*WS+tx + (c_n_size-(ty+1))*c_height;

    block[ty][tx] = g_transp_zhat[0];
    block[ty][tx+WS] = g_transp_zhat[WS];

    block_y[ty][tx] = g_y[0];
    block_y[ty][tx+WS] = g_y[WS];

    __syncthreads();

    if(ty < 2)
    {
        int i = tx+ty*WS;
        block[0][i] = ez = block[0][i] - block_y[0][i]*c_Alpha;

#pragma unroll
        for(int j=1; j<blockDim.y; ++j)
        {
            float ezhat = block[j][i] - block_y[j][i]*c_Alpha;
            block[j][i] = ez = ezhat + c_Stm*ez;
        }
    }

    __syncthreads();

    g_transp_zhat[0] = block[ty][tx];
    g_transp_zhat[WS] = block[ty][tx+WS];

    g_transp_zhat -= c_height*blockDim.y;
    g_y -= c_height*blockDim.y;

    // middle column-blocks
    n = c_n_size-blockDim.y-1;
    if(blockDim.y == DW)
    {
        for(; n>=DW; n-=DW, g_transp_zhat-=DW*c_height,
                g_y-=DW*c_height)
        {
            block[ty][tx] = g_transp_zhat[0];
            block[ty][tx+WS] = g_transp_zhat[WS];

            block_y[ty][tx] = g_y[0];
            block_y[ty][tx+WS] = g_y[WS];

            __syncthreads();

            if(ty < 2)
            {
                int i = tx+ty*WS;
#pragma unroll
                for(int j=0; j<DW; ++j)
                {
                    float ezhat = block[j][i] - block_y[j][i]*c_Alpha;
                    block[j][i] = ez = ezhat + c_Stm*ez;
                }
            }

            __syncthreads();

            g_transp_zhat[0] = block[ty][tx];
            g_transp_zhat[WS] = block[ty][tx+WS];
        }
    }

    // remaining column-blocks

    if(n > 0)
    {
        int remaining = n;
        if(ty < remaining)
        {
            block[ty][tx] = g_transp_zhat[0];
            block[ty][tx+WS] = g_transp_zhat[WS];

            block_y[ty][tx] = g_y[0];
            block_y[ty][tx+WS] = g_y[WS];
        }

        __syncthreads();

        if(ty < 2)
        {
            int i = tx+ty*WS;
#pragma unroll
            for(int j=0; j<remaining; ++j)
            {
                float ezhat = block[j][i] - block_y[j][i]*c_Alpha;
                block[j][i] = ez = ezhat + c_Stm*ez;
            }
        }

        __syncthreads();

        if(ty < remaining)
        {
            g_transp_zhat[0] = block[ty][tx];
            g_transp_zhat[WS] = block[ty][tx+WS];
        }
    }
}

__global__ __launch_bounds__(WS*OW, DNB)
void algorithm5_stage4_5_step1( float *g_ucheck,
                                float *g_vtilde,
                                const float *g_transp_y,
                                const float *g_transp_z )
{
    int tx = threadIdx.x, ty = threadIdx.y, n = blockIdx.x*OW + ty, m = blockIdx.y;

    if(n >= c_n_size)
        return;

    __shared__ float shared_yimb1[OW][WS], shared_zim1b[OW][WS];

    float delta_x, delta_y;

    delta_x = c_Delta_x_tail[tx];
    delta_y = c_Delta_y[tx];

    g_transp_y += (n-1)*c_height + m*WS; 
    g_transp_z += (n+1)*c_height + m*WS;

    float *yimb1 = shared_yimb1[ty], *zim1b = shared_zim1b[ty];

    if(n > 0)
        yimb1[tx] = g_transp_y[tx];
    
    if(n < c_n_size-1)
        zim1b[tx] = g_transp_z[tx];

    int e;

    if(m == c_m_size-1)
    {
        e = c_height%WS;
        if(e == 0)
            e = WS;
    }
    else
        e = WS;

    g_vtilde += m*c_width + n*WS+tx;
    g_ucheck += m*c_width + n*WS+tx;

    // update vtilde to vcheck
    float sum_vtilde = *g_vtilde, sum_ucheck = *g_ucheck, inner_sum_vtilde = 0;
    float sign = 1;

    if(n == 0)
    {
        if(m == c_m_size-1)
        {
#pragma unroll
            for(int i=0; i<e; ++i)
            {
                float delta = zim1b[i]*delta_x;
                if(i>0)
                    inner_sum_vtilde = inner_sum_vtilde*c_Linf1;
                inner_sum_vtilde += delta*sign;
                sum_vtilde += inner_sum_vtilde*c_ProdLinf[i];
                sum_ucheck += delta*c_SignRevProdLinf[i];

                sign *= -1;
            }
        }
        else
        {
#pragma unroll
            for(int i=0; i<WS; ++i)
            {
                float delta = zim1b[i]*delta_x;
                if(i>0)
                    inner_sum_vtilde = inner_sum_vtilde*c_Linf1;
                inner_sum_vtilde += delta*sign;
                sum_vtilde += inner_sum_vtilde*c_ProdLinf[i];
                sum_ucheck += delta*c_SignRevProdLinf[i];

                sign *= -1;
            }
        }
    }
    else if(n == c_n_size-1)
    {
        if(m == c_m_size-1)
        {
#pragma unroll
            for(int i=0; i<e; ++i)
            {
                float delta = yimb1[i]*delta_y;
                if(i>0)
                    inner_sum_vtilde = inner_sum_vtilde*c_Linf1;
                inner_sum_vtilde += delta*sign;
                sum_vtilde += inner_sum_vtilde*c_ProdLinf[i];
                sum_ucheck += delta*c_SignRevProdLinf[i];

                sign *= -1;
            }
        }
        else
        {
#pragma unroll
            for(int i=0; i<WS; ++i)
            {
                float delta = yimb1[i]*delta_y;
                if(i>0)
                    inner_sum_vtilde = inner_sum_vtilde*c_Linf1;
                inner_sum_vtilde += delta*sign;
                sum_vtilde += inner_sum_vtilde*c_ProdLinf[i];
                sum_ucheck += delta*c_SignRevProdLinf[i];

                sign *= -1;
            }
        }
    }
    else
    {
        if(m == c_m_size-1)
        {
#pragma unroll
            for(int i=0; i<e; ++i)
            {
                float delta = zim1b[i]*delta_x + yimb1[i]*delta_y;
                if(i>0)
                    inner_sum_vtilde = inner_sum_vtilde*c_Linf1;
                inner_sum_vtilde += delta*sign;
                sum_vtilde += inner_sum_vtilde*c_ProdLinf[i];
                sum_ucheck += delta*c_SignRevProdLinf[i];

                sign *= -1;
            }
        }
        else
        {
            float inner_sum_vtilde = zim1b[0]*delta_x + yimb1[0]*delta_y;
            sum_vtilde += inner_sum_vtilde*c_Linf1;
            float sign = -1;
#pragma unroll
            for(int i=1; i<WS; ++i)
            {
                float delta = zim1b[i]*delta_x + yimb1[i]*delta_y;
                inner_sum_vtilde = inner_sum_vtilde*c_Linf1 + delta*sign;
                sum_vtilde += inner_sum_vtilde*c_ProdLinf[i];
                sum_ucheck += delta*c_SignRevProdLinf[i];

                sign *= -1;
            }
        }
    }

    *g_vtilde = sum_vtilde;
    *g_ucheck = sum_ucheck;
}

__global__ __launch_bounds__(WS*DW, DNB)
void algorithm5_stage4_5_step2( float *g_ubar,
                                float *g_vcheck )
{
    int tx = threadIdx.x, ty = threadIdx.y, n = blockIdx.x*2;

    __shared__ float block[DW][WS*2];

    // ubar -> u processing --------------------------------------

    g_ubar += n*WS+tx + ty*c_width;

    float u;

    // first row-block

    block[ty][tx] = g_ubar[0];
    block[ty][tx+WS] = g_ubar[WS];

    __syncthreads();

    if(ty < 2)
    {
        int j = tx+ty*WS;
        u = block[0][j];
#pragma unroll
        for(int i=1; i<blockDim.y; ++i)
            block[i][j] = u = block[i][j] + c_Svm*u;
    }

    __syncthreads();

    if(ty > 0)
    {
        g_ubar[0] = block[ty][tx];
        g_ubar[WS] = block[ty][tx+WS];
    }

    g_ubar += c_width*blockDim.y;

    // middle row-blocks

    int m = blockDim.y;
    if(blockDim.y == DW)
    {
        for(; m<c_m_size-DW; 
            m+=DW, g_ubar+=c_width*DW)
        {
            block[ty][tx] = g_ubar[0];
            block[ty][tx+WS] = g_ubar[WS];

            __syncthreads();

            if(ty < 2)
            {
                int j = tx+ty*WS;
#pragma unroll
                for(int i=0; i<DW; ++i)
                    block[i][j] = u = block[i][j] + c_Svm*u;
            }

            __syncthreads();

            g_ubar[0] = block[ty][tx];
            g_ubar[WS] = block[ty][tx+WS];
        }
    }

    // remaining row-blocks

    if(c_m_size > m)
    {
        int remaining = c_m_size-m;
        if(ty < remaining)
        {
            block[ty][tx] = g_ubar[0];
            block[ty][tx+WS] = g_ubar[WS];
        }

        __syncthreads();

        if(ty < 2)
        {
            int j = tx+ty*WS;
#pragma unroll
            for(int i=0; i<remaining; ++i)
                block[i][j] = u = block[i][j] + c_Svm*u;
        }

        __syncthreads();

        if(ty < remaining)
        {
            g_ubar[0] = block[ty][tx];
            g_ubar[WS] = block[ty][tx+WS];
        }

        g_ubar += remaining*c_width;
    }

    // vcheck -> v processing --------------------------------------

    const float *g_u = g_ubar - (2*ty+2)*c_width;

    __shared__ float block_y[DW][WS*2];

    float v;

    // last row-block

    g_vcheck += n*WS+tx + (c_m_size-(ty+1))*c_width;

    block[ty][tx] = g_vcheck[0];
    block[ty][tx+WS] = g_vcheck[WS];

    block_y[ty][tx] = g_u[0];
    block_y[ty][tx+WS] = g_u[WS];

    __syncthreads();

    if(ty < 2)
    {
        int j = tx+ty*WS;
        block[0][j] = v = block[0][j] - block_y[0][j]*c_Alpha;

#pragma unroll
        for(int i=1; i<blockDim.y; ++i)
        {
            float vimb = block[i][j] - block_y[i][j]*c_Alpha;
            block[i][j] = v = vimb + c_Stm*v;
        }
    }

    __syncthreads();

    g_vcheck[0] = block[ty][tx];
    g_vcheck[WS] = block[ty][tx+WS];

    g_vcheck -= c_width*blockDim.y;
    g_u -= c_width*blockDim.y;

    // middle row-blocks

    m = c_m_size-blockDim.y-1;
    if(blockDim.y == DW)
    {
        for(; m>=DW; m-=DW, g_vcheck-=DW*c_width,
                g_u-=DW*c_width)
        {
            block[ty][tx] = g_vcheck[0];
            block[ty][tx+WS] = g_vcheck[WS];

            block_y[ty][tx] = g_u[0];
            block_y[ty][tx+WS] = g_u[WS];

            __syncthreads();

            if(ty < 2)
            {
                int j = tx+ty*WS;
#pragma unroll
                for(int i=0; i<DW; ++i)
                {
                    float vimb = block[i][j] - block_y[i][j]*c_Alpha;
                    block[i][j] = v = vimb + c_Stm*v;
                }
            }

            __syncthreads();

            g_vcheck[0] = block[ty][tx];
            g_vcheck[WS] = block[ty][tx+WS];
        }
    }

    // remaining row-blocks

    if(m > 0)
    {
        int remaining = m;
        if(ty < remaining)
        {
            block[ty][tx] = g_vcheck[0];
            block[ty][tx+WS] = g_vcheck[WS];

            block_y[ty][tx] = g_u[0];
            block_y[ty][tx+WS] = g_u[WS];
        }

        __syncthreads();

        if(ty < 2)
        {
            int j = tx+ty*WS;
#pragma unroll
            for(int i=0; i<remaining; ++i)
            {
                float vimb = block[i][j] - block_y[i][j]*c_Alpha;
                block[i][j] = v = vimb + c_Stm*v;
            }
        }

        __syncthreads();

        if(ty < remaining)
        {
            g_vcheck[0] = block[ty][tx];
            g_vcheck[WS] = block[ty][tx+WS];
        }
    }
}

__global__ __launch_bounds__(WS*DW, ONB)
void algorithm5_stage6( float *g_inout,
                        const float *g_transp_y,
                        const float *g_transp_z,
                        const float *g_u,
                        const float *g_v )
{
    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.y*2, n = blockIdx.x;

    __shared__ float block[WS*2][WS+1];

    const float *in_data = g_inout + (m*WS + ty)*c_width + n*WS + tx;

#pragma unroll
    for(int i=0; i<WS; i+=DW)
    {
        block[ty+i][tx] = in_data[i*c_width];
        block[ty+i+WS][tx] = in_data[(i+WS)*c_width];
    }

    __syncthreads();

    if(ty < 2)
    {

        m += ty;

        {
            float *bdata = block[tx+ty*WS];
            float prev;

            // calculate y ---------------------

            prev = bdata[0];

            if(n > 0)
                bdata[0] = prev -= g_transp_y[(n-1)*c_height + m*WS+tx]*c_Linf1;
#pragma unroll
            for(int j=1; j<WS; ++j)
                prev = bdata[j] -= prev*c_Linf1;

            // calculate z ---------------------

            if(n < c_n_size-1)
                bdata[WS-1] = prev = (bdata[WS-1] - g_transp_z[(n+1)*c_height+m*WS+tx])*c_Linf1;
            else
                prev = bdata[WS-1] *= c_Linf1;

            for(int j=WS-2; j>=0; --j)
                bdata[j] = prev = (bdata[j] - prev)*c_Linf1;
        }

        {
            float (*bdata)[WS+1] = (float (*)[WS+1]) &block[ty*WS][tx];
            float prev;

            // calculate u ---------------------

            prev = bdata[0][0];

            if(m > 0)
                bdata[0][0] = prev -= g_u[(m-1)*c_width + n*WS+tx]*c_Linf1;

#pragma unroll
            for(int i=1; i<WS; ++i)
                prev = bdata[i][0] -= prev*c_Linf1;

            // calculate v ---------------------

            float *out_data = g_inout + (m*WS+WS-1)*c_width + n*WS + tx;

            prev = bdata[WS-1][0];
            if(m == c_m_size-1)
                prev *= c_Linf1;
            else
                prev = (prev - g_v[(m+1)*c_width + n*WS+tx])*c_Linf1;

            prev *= c_iR1;

            *out_data = prev;

            for(int i=WS-2; i>=0; --i)
            {
                out_data -= c_width;
                *out_data = prev = (bdata[i][0]*c_iR1 - prev)*c_Linf1;
            }
        }
    }
}

//-- Fusion -------------------------------------------------------------------

__global__ __launch_bounds__(WS*DW, ONB)
void algorithm5_stage6_fusion_algorithm4_stage1( float *g_inout,
                                                 const float *g_transp_y,
                                                 const float *g_transp_z,
                                                 const float *g_u,
                                                 const float *g_v,
                                                 float2 *g_transp_ybar,
                                                 float2 *g_transp_zhat )
{
    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.y*2, n = blockIdx.x;

    __shared__ float block[WS*2][WS+1];

    float *in_data = g_inout + (m*WS + ty)*c_width + n*WS + tx;

#pragma unroll
    for(int i=0; i<WS; i+=DW)
    {
        block[ty+i][tx] = in_data[i*c_width];
        block[ty+i+WS][tx] = in_data[(i+WS)*c_width];
    }

    __syncthreads();

    if(ty < 2)
    {
        // 1st ORDER ENDING ----------------------------
        m += ty;
        {
            float *bdata = block[tx+ty*WS];
            float prev;

            prev = bdata[0];

            if(n > 0)
                bdata[0] = prev -= g_transp_y[(n-1)*c_height + m*WS+tx]*c_Linf1;
#pragma unroll
            for(int j=1; j<WS; ++j)
                prev = bdata[j] -= prev*c_Linf1;

            if(n < c_n_size-1)
                bdata[WS-1] = prev = (bdata[WS-1] - g_transp_z[(n+1)*c_height+m*WS+tx])*c_Linf1;
            else
                prev = bdata[WS-1] *= c_Linf1;

            for(int j=WS-2; j>=0; --j)
                bdata[j] = prev = (bdata[j] - prev)*c_Linf1;
        }

        {
            float (*bdata)[WS+1] = (float (*)[WS+1]) &block[ty*WS][tx];
            float prev;

            prev = bdata[0][0];

            if(m > 0)
                bdata[0][0] = prev -= g_u[(m-1)*c_width + n*WS+tx]*c_Linf1;

#pragma unroll
            for(int i=1; i<WS; ++i)
                prev = bdata[i][0] -= prev*c_Linf1;

            float *out_data = g_inout + (m*WS+WS-1)*c_width + n*WS + tx;

            prev = bdata[WS-1][0];
            if(m == c_m_size-1)
                prev *= c_Linf1;
            else
                prev = (prev - g_v[(m+1)*c_width + n*WS+tx])*c_Linf1;

            bdata[WS-1][0] = prev *= c_iR1;

            *out_data = prev;

            for(int i=WS-2; i>=0; --i)
            {
                out_data -= c_width;
                *out_data = bdata[i][0] = prev = (bdata[i][0]*c_iR1 - prev)*c_Linf1;
            }
        }
    }

    // 2nd ORDER BEGINNING ----------------------------
    if(ty < 2)
    {
        float *bdata = block[tx+ty*WS];
        int outidx = n*c_height + m*WS + tx; // transposed!

        float2 accum;

        if(n < c_n_size-1)
        {
            accum.x = bdata[0];
            accum.y = bdata[1] -= c_Minf*accum.x;

            for(int i=2; i<WS; ++i)
            {
                accum.x = bdata[i] -= c_Minf*accum.y + c_Linf2*accum.x;
                swap(accum.x, accum.y);
            }
        }
        else
        {
            accum.x = bdata[0];
            accum.y = bdata[1] -= c_Minf*accum.x;

            for(int i=2; i<WS-1; ++i)
            {
                accum.x = bdata[i] -= c_Minf*accum.y + c_Linf2*accum.x;
                swap(accum.x, accum.y);
            }
            accum.x = bdata[WS-1] -= c_Minf*accum.y + c_Linf2*accum.x;
            swap(accum.x, accum.y);
        }

        g_transp_ybar[outidx] = accum;

        if(n < c_n_size-1)
        {
            accum.y = bdata[WS-1] * c_Linf2;
            accum.x = (bdata[WS-2] - accum.y*c_Ninf)*c_Linf2;

#pragma unroll
            for(int i=WS-3; i>=0; --i)
            {
                accum.y = (bdata[i] - accum.x*c_Ninf - accum.y)*c_Linf2;
                swap(accum.x, accum.y);
            }

        }
        else // last block
        {
            int i = WS-1;

            accum.y = bdata[i--] * c_Llast2;
            accum.x = (bdata[i--] - accum.y*c_Ninf)*c_Linf2;

            for(; i>=0; --i)
            {
                accum.y = (bdata[i] - accum.x*c_Ninf - accum.y)*c_Linf2;
                swap(accum.x, accum.y);
            }

        }

        g_transp_zhat[outidx] = accum;
    }
}

//-- Host ---------------------------------------------------------------------

__host__
void algorithm4( float *inout,
                 const int& h,
                 const int& w,
                 const float& b0,
                 const float& a1,
                 const float& a2 )
{
    dvector<float> d_img(inout, h*w);

    constants_coefficients2( b0, a1, a2 );

    const int m_size = (h+WS-1)/WS, n_size = (w+WS-1)/WS;

    copy_to_symbol("c_height", h);
    copy_to_symbol("c_width", w);
    copy_to_symbol("c_m_size", m_size);
    copy_to_symbol("c_n_size", n_size);

    dvector<float2> d_transp_ybar(m_size*w),
        d_transp_zhat(m_size*w),
        d_ubar(n_size*h),
        d_vhat(n_size*h);

    dvector<float2> d_transp_y, d_transp_z, d_u, d_v;

    algorithm4_stage1<<< dim3(n_size, m_size), dim3(WS, SOW) >>>(
        d_img, d_transp_ybar, d_transp_zhat );

    algorithm4_stage2_3_or_5_6<<< dim3(m_size, 1), dim3(MTS, 1) >>>(
        d_transp_ybar, d_transp_zhat );

    swap( d_transp_ybar, d_transp_y );
    swap( d_transp_zhat, d_transp_z );

    algorithm4_stage4<<< dim3(n_size, (m_size+2-1)/2), dim3(WS, SOW) >>>(
        d_img, d_transp_y, d_transp_z, d_ubar, d_vhat );

    algorithm4_stage2_3_or_5_6<<< dim3(n_size, 1), dim3(MTS, 1) >>>(
        d_ubar, d_vhat );

    swap( d_ubar, d_u );
    swap( d_vhat, d_v );

    algorithm4_stage7<<< dim3(n_size, m_size), dim3(WS, SOW) >>>(
        d_img, d_u, d_v );

    d_img.copy_to(inout, h*w);
}

__host__
void algorithm5( float *inout,
                 const int& h,
                 const int& w,
                 const float& b0,
                 const float& a1 )
{
    dvector<float> d_img(inout, h*w);

    constants_coefficients1( b0, a1 );

    const int m_size = (h+WS-1)/WS, n_size = (w+WS-1)/WS;

    copy_to_symbol("c_height", h);
    copy_to_symbol("c_width", w);
    copy_to_symbol("c_m_size", m_size);
    copy_to_symbol("c_n_size", n_size);

    dvector<float> d_transp_ybar(n_size*h),
        d_transp_zhat(n_size*h),
        d_ucheck(m_size*w),
        d_vtilde(m_size*w);
                   
    dvector<float> d_transp_y, d_transp_z, d_ubar, d_vcheck, d_u, d_v;

    algorithm5_stage1<<< dim3(n_size, (m_size+2-1)/2), dim3(WS, DW) >>>(
        d_img, d_transp_ybar, d_transp_zhat, d_ucheck, d_vtilde );

    algorithm5_stage2_3<<< dim3(1, (m_size+2-1)/2), dim3(WS, std::min(n_size, DW)) >>>(
        d_transp_ybar, d_transp_zhat );

    swap(d_transp_ybar, d_transp_y);
    swap(d_transp_zhat, d_transp_z);

    algorithm5_stage4_5_step1<<< dim3((n_size+OW-1)/OW, m_size), dim3(WS, OW) >>>(
        d_ucheck, d_vtilde, d_transp_y, d_transp_z );

    swap(d_ucheck, d_ubar);
    swap(d_vtilde, d_vcheck);

    algorithm5_stage4_5_step2<<< dim3((n_size+2-1)/2), dim3(WS, std::min(m_size, DW)) >>>(
        d_ubar, d_vcheck );

    swap(d_ubar, d_u);
    swap(d_vcheck, d_v);

    algorithm5_stage6<<< dim3(n_size, (m_size+2-1)/2), dim3(WS, DW) >>>(
        d_img, d_transp_y, d_transp_z, d_u, d_v );

    d_img.copy_to(inout, h*w);
}

__host__
void gaussian_gpu( float *inout,
                   const int& h,
                   const int& w,
                   const float& s )
{
    dvector<float> d_img(inout, h*w);

    float b10, a11, b20, a21, a22;

    weights1( s, b10, a11 );
        
    constants_coefficients1( b10, a11 );

    weights2( s, b20, a21, a22 );

    constants_coefficients2( b20, a21, a22 );

    const int m_size = (h+WS-1)/WS, n_size = (w+WS-1)/WS;

    copy_to_symbol("c_height", h);
    copy_to_symbol("c_width", w);
    copy_to_symbol("c_m_size", m_size);
    copy_to_symbol("c_n_size", n_size);

    // for order 1
    dvector<float> d_transp_ybar1(n_size*h),
        d_transp_zhat1(n_size*h),
        d_ucheck1(m_size*w),
        d_vtilde1(m_size*w);

    dvector<float> d_transp_y1, d_transp_z1, d_ubar1, d_vcheck1, d_u1, d_v1;

    // for order 2
    dvector<float2> d_transp_ybar2(m_size*w),
        d_transp_zhat2(m_size*w),
        d_ubar2(n_size*h),
        d_vhat2(n_size*h);

    dvector<float2> d_transp_y2, d_transp_z2, d_u2, d_v2;
   
    algorithm5_stage1<<< dim3(n_size, (m_size+2-1)/2), dim3(WS, DW) >>>(
        d_img, d_transp_ybar1, d_transp_zhat1, d_ucheck1, d_vtilde1 );

    algorithm5_stage2_3<<< dim3(1, (m_size+2-1)/2), dim3(WS, std::min(n_size, DW)) >>>(
        d_transp_ybar1, d_transp_zhat1 );
	    
    swap(d_transp_ybar1, d_transp_y1);
    swap(d_transp_zhat1, d_transp_z1);

    algorithm5_stage4_5_step1<<< dim3((n_size+OW-1)/OW, m_size), dim3(WS, OW) >>>(
        d_ucheck1, d_vtilde1, d_transp_y1, d_transp_z1 );

    swap(d_ucheck1, d_ubar1);
    swap(d_vtilde1, d_vcheck1);

    algorithm5_stage4_5_step2<<< dim3((n_size+2-1)/2), dim3(WS, std::min(m_size, DW)) >>>(
        d_ubar1, d_vcheck1 );

    swap(d_ubar1, d_u1);
    swap(d_vcheck1, d_v1);

    algorithm5_stage6_fusion_algorithm4_stage1<<< dim3(n_size, (m_size+2-1)/2), dim3(WS, DW) >>>(
        d_img, d_transp_y1, d_transp_z1, d_u1, d_v1, d_transp_ybar2, d_transp_zhat2 );

    // algorithm5_stage6<<< dim3(n_size, (m_size+2-1)/2), dim3(WS, DW) >>>(
    //     d_img, d_transp_y1, d_transp_z1, d_u1, d_v1 );

    // algorithm4_stage1<<< dim3(n_size, m_size), dim3(WS, SOW) >>>(
    //     d_img, d_transp_ybar2, d_transp_zhat2 );

    algorithm4_stage2_3_or_5_6<<< dim3(m_size, 1), dim3(MTS, 1) >>>(
        d_transp_ybar2, d_transp_zhat2 );

    swap( d_transp_ybar2, d_transp_y2 );
    swap( d_transp_zhat2, d_transp_z2 );

    algorithm4_stage4<<< dim3(n_size, (m_size+2-1)/2), dim3(WS, SOW) >>>(
        d_img, d_transp_y2, d_transp_z2, d_ubar2, d_vhat2 );

    algorithm4_stage2_3_or_5_6<<< dim3(n_size, 1), dim3(MTS, 1) >>>(
        d_ubar2, d_vhat2 );

    swap( d_ubar2, d_u2 );
    swap( d_vhat2, d_v2 );

    algorithm4_stage7<<< dim3(n_size, m_size), dim3(WS, SOW) >>>(
        d_img, d_u2, d_v2 );

    d_img.copy_to(inout, h*w);

}

//=============================================================================
} // namespace gpufilter
//=============================================================================
// vi: ai ts=4 sw=4
