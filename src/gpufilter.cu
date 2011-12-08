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
void alg4_stage1( const float *g_in,
                  float2 *g_transp_ybar,
                  float2 *g_transp_zhat )
{
    int m = blockIdx.y, n = blockIdx.x;
    int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float block[WS][WS+1];

    g_in += (m*WS + ty)*c_width + n*WS + tx;

#pragma unroll
    for(int i=0; i<WS-2; i+=SOW)
    {
        block[ty+i][tx] = g_in[i*c_width];
    }

    if(ty < 2)
    {
        block[ty+WS-2][tx] = g_in[(WS-2)*c_width];
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
void alg4_stage2_3_or_5_6( float2 *g_transp_ybar,
                           float2 *g_transp_zhat )
{
    int m = blockIdx.x;
    int tx = threadIdx.x;

    int row = m*blockDim.x + tx;

    if(row >= c_height) return;

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
void alg4_stage4( float *g_inout,
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
void alg4_stage7( float *g_inout,
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

__global__ __launch_bounds__(WS*SOW, MBO)
void alg5_stage1( const float *g_in,
                  float *g_transp_pybar,
                  float *g_transp_ezhat,
                  float *g_ptucheck,
                  float *g_etvtilde )
{
    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n = blockIdx.y;
    __shared__ float block[WS][WS+1];

    // make g_in point to the data we'll work with
    g_in += (n*WS+ty)*c_width + m*WS+tx;

    float (*bdata)[WS+1] = (float (*)[WS+1]) &block[ty][tx];

    // load data into shared memory
    int i;
#pragma unroll
    for(i=0; i<WS-(WS%SOW); i+=SOW)
    {
        **bdata = *g_in;
        bdata += SOW;
        g_in += SOW*c_width;
    }

    if(ty < WS%SOW)
    {
        **bdata = *g_in;
    }

    // We use a transposed matrix for pybar and ezhat to have
    // coalesced memory accesses. This is the movement for these
    // transposed buffers.
    g_transp_pybar += m*c_height + n*WS + tx; 
    g_transp_ezhat += m*c_height + n*WS + tx;
    g_ptucheck += n*c_width + m*WS + tx;
    g_etvtilde += n*c_width + m*WS + tx;

    __syncthreads();

    float prev;

    if(ty == 0)
    {
        // scan columns
        {
            float *bdata = block[tx];

            // calculate pybar, scan left -> right

            prev = *bdata++;

#pragma unroll
            for(int j=1; j<WS; ++j, ++bdata)
                prev = *bdata -= prev*c_a1;

            *g_transp_pybar = prev*c_b0;
            
            // calculate ezhat, scan right -> left

            prev = *--bdata;
            --bdata;

#pragma unroll
            for(int j=WS-2; j>=0; --j, --bdata)
                prev = *bdata -= prev*c_a1;

            *g_transp_ezhat = prev*c_b0*c_b0;
        }

        // scan rows
        {
            float (*bdata)[WS+1] = (float (*)[WS+1]) &block[0][tx];

            // calculate ptucheck, scan top -> down

            prev = **bdata++;

#pragma unroll
            for(int i=1; i<WS; ++i, ++bdata)
                prev = **bdata -= prev*c_a1;

            *g_ptucheck = prev*c_b0*c_b0*c_b0;

            // calculate etvtilde, scan bottom -> up

            prev = **--bdata;
            --bdata;

            for(int i=WS-2; i>=0; --i, --bdata)
                prev = **bdata - prev*c_a1;

            *g_etvtilde = prev*c_b0*c_b0*c_b0*c_b0;
        }
    }
}

__global__ __launch_bounds__(WS*DW, DNB)
void alg5_stage2_3( float *g_transp_pybar,
                    float *g_transp_ezhat )
{
    int tx = threadIdx.x, ty = threadIdx.y, n = blockIdx.y;

    __shared__ float transp_block[DW][WS];
    float *bdata = &transp_block[ty][tx];

    // P(ybar) -> P(y) processing --------------------------------------

    float *transp_pybar = g_transp_pybar + ty*c_height + n*WS+tx;

    // first column-transp_block

    // read P(ybar)
    *bdata = *transp_pybar;

    float py; // P(Y)

    __syncthreads();

    if(ty == 0)
    {
        float (*bdata)[WS] = (float (*)[WS]) &transp_block[0][tx];

        // (24): P_m(y) = P_m(ybar) + A^b_F * P_{m-1}(y)
        py = **bdata++;

#pragma unroll
        for(int m=1; m<blockDim.y; ++m, ++bdata)
            **bdata = py = **bdata + c_AbF*py;
    }

    __syncthreads();

    // write P(y)
    if(ty > 0) // first one doesn't need fixing
        *transp_pybar = *bdata;

    transp_pybar += c_height*blockDim.y;

    // middle column-blocks

    int m = blockDim.y;
    if(m == DW)
    {
        for(; m<c_m_size-(c_m_size%DW); m+=DW)
        {
            *bdata = *transp_pybar;

            __syncthreads();

            if(ty == 0)
            {
                float (*bdata)[WS] = (float (*)[WS]) &transp_block[0][tx];
#pragma unroll
                for(int dm=0; dm<DW; ++dm, ++bdata)
                    **bdata = py = **bdata + c_AbF*py;
            }

            __syncthreads();

            *transp_pybar = *bdata;
            transp_pybar += c_height*DW;
        }
    }

    // remaining column-transp_blocks

    if(m < c_m_size)
    {
        int remaining = c_m_size - m;

        if(remaining > 0)
            *bdata = *transp_pybar;

        __syncthreads();

        if(ty == 0)
        {
            float (*bdata)[WS] = (float (*)[WS]) &transp_block[0][tx];
#pragma unroll
            for(int dm=0; dm<remaining; ++dm, ++bdata)
                **bdata = py = **bdata + c_AbF*py;

        }

        __syncthreads();

        if(remaining > 0)
            *transp_pybar = *bdata;
    }

    // E(zhat) -> E(z) processing --------------------------------------

    int idx = (c_m_size-1-ty)*c_height + n*WS+tx;

    const float *transp_pm1y = g_transp_pybar + idx - c_height;

    // last column-transp_block

    float *transp_ezhat = g_transp_ezhat + idx;

    // all pybars must be updated!
    __syncthreads();

    float ez;

    {
        *bdata = *transp_ezhat;

        if(m-ty > 0)
            *bdata += *transp_pm1y*c_HARB_AFP;

        __syncthreads();

        if(ty == 0)
        {
            float (*bdata)[WS] = (float (*)[WS]) &transp_block[0][tx];
            ez = **bdata++;

#pragma unroll
            for(int dm=1; dm<blockDim.y; ++dm, ++bdata)
                **bdata = ez = **bdata + c_AbR*ez;
        }

        __syncthreads();

        *transp_ezhat = *bdata;

        transp_ezhat -= c_height*blockDim.y;
        transp_pm1y -= c_height*blockDim.y;
    }

    // middle column-transp_blocks
    m = c_m_size-1 - blockDim.y;
    if(blockDim.y == DW)
    {
        for(; m>=c_m_size%DW; m-=DW)
        {
            *bdata = *transp_ezhat;

            if(m-ty > 0)
                *bdata += *transp_pm1y*c_HARB_AFP;

            __syncthreads();

            if(ty == 0)
            {
                float (*bdata)[WS] = (float (*)[WS]) &transp_block[0][tx];
#pragma unroll
                for(int dm=0; dm<DW; ++dm, ++bdata)
                    **bdata = ez = **bdata + c_AbR*ez;
            }

            __syncthreads();

            *transp_ezhat = *bdata;

            transp_ezhat -= DW*c_height;
            transp_pm1y -= DW*c_height;
        }
    }

    // remaining column-blocks

    if(m >= 0)
    {
        int remaining = m+1;

        if(m-ty >= 0)
        {
            *bdata = *transp_ezhat;
        
            if(m-ty > 0)
                *bdata += *transp_pm1y*c_HARB_AFP;
        }

        __syncthreads();

        if(ty == 0)
        {
            float (*bdata)[WS] = (float (*)[WS]) &transp_block[0][tx];
            // (24): P_m(y) = P_m(ybar) + A^b_F * P_{m-1}(y)
#pragma unroll
            for(int dm=0; dm<remaining; ++dm, ++bdata)
                **bdata = ez = **bdata + c_AbR*ez;
        }

        __syncthreads();

        if(m-ty >= 0)
            *transp_ezhat = *bdata;
    }
}

__global__ __launch_bounds__(WS*CFW, ONB)
void alg5_stage4_5( float *g_ptucheck,
                    float *g_etvtilde,
                    const float *g_transp_py,
                    const float *g_transp_ez )
{
    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x;

    __shared__ float block[CFW][WS];
    float *bdata = &block[ty][tx];

    // P(ucheck) -> P(u) processing --------------------------------------

	volatile __shared__ float block_RD_raw[CFW][16+32+1];
	volatile float (*block_RD)[16+32+1] 
        = (float (*)[16+32+1]) &block_RD_raw[0][16];
    if(ty < CFW)
        block_RD_raw[ty][tx] = 0;

#define CALC_DOT(RES, V1, V2) \
    block_RD[ty][tx] = V1*V2; \
    block_RD[ty][tx] += block_RD[ty][tx-1]; \
    block_RD[ty][tx] += block_RD[ty][tx-2]; \
    block_RD[ty][tx] += block_RD[ty][tx-4]; \
    block_RD[ty][tx] += block_RD[ty][tx-8]; \
    block_RD[ty][tx] += block_RD[ty][tx-16]; \
    float RES = block_RD[ty][31];

    float *ptucheck = g_ptucheck + m*WS+tx + ty*c_width;

    // first row-block

    int idx = m*c_height + ty*WS+tx;

    const float *transp_pm1ybar = g_transp_py + idx - c_height,
                *transp_em1zhat = g_transp_ez + idx + c_height;

    float ptu;

    {
        // read P(ucheck)
        *bdata = *ptucheck;

        if(m > 0)
        {
            CALC_DOT(dot, *transp_pm1ybar, c_TAFB[tx]);
            *bdata += dot*c_ARB_AFP_T[tx];
        }

        if(m < c_m_size-1)
        {
            CALC_DOT(dot, *transp_em1zhat, c_TAFB[tx]);
            *bdata += dot*c_ARE_T[tx];
        }

        transp_pm1ybar += WS*blockDim.y;
        transp_em1zhat += WS*blockDim.y;

        __syncthreads();

        if(ty == 0)
        {
            float (*bdata2)[WS] = (float (*)[WS]) &block[0][tx];

            ptu = **bdata2++;

#pragma unroll
            for(int n=1; n<blockDim.y; ++n, ++bdata2)
                **bdata2 = ptu = **bdata2 + c_AbF*ptu;
        }

        __syncthreads();

        // write P(u)
        *ptucheck = *bdata;

        ptucheck += blockDim.y*c_width;
    }

    // middle row-blocks

    int n = blockDim.y;
    if(n == CFW)
    {
        int nmax = c_n_size-(c_n_size%CFW);
        for(; n<nmax; n+=CFW)
        {
            *bdata = *ptucheck;

            if(m > 0)
            {
                CALC_DOT(dot, *transp_pm1ybar, c_TAFB[tx]);
                *bdata += dot*c_ARB_AFP_T[tx];
            }

            if(m < c_m_size-1)
            {
                CALC_DOT(dot, *transp_em1zhat, c_TAFB[tx]);
                *bdata += dot*c_ARE_T[tx];
            }

            transp_pm1ybar += WS*CFW;
            transp_em1zhat += WS*CFW;

            __syncthreads();

            if(ty == 0)
            {
                float (*bdata2)[WS] = (float (*)[WS]) &block[0][tx];

#pragma unroll
                for(int dn=0; dn<CFW; ++dn, ++bdata2)
                    **bdata2 = ptu = **bdata2 + c_AbF*ptu;
            }

            __syncthreads();

            *ptucheck = *bdata;

            ptucheck += CFW*c_width;

        }
    }

    // remaining row-blocks

    if(n < c_n_size)
    {

        if(n+ty < c_n_size)
        {
            *bdata = *ptucheck;

            if(m > 0)
            {
                CALC_DOT(dot, *transp_pm1ybar, c_TAFB[tx]);
                *bdata += dot*c_ARB_AFP_T[tx];
            }

            if(m < c_m_size-1)
            {
                CALC_DOT(dot, *transp_em1zhat, c_TAFB[tx]);
                *bdata += dot*c_ARE_T[tx];
            }
        }

        int remaining = c_n_size-n;
        __syncthreads();

        if(ty == 0)
        {
            float (*bdata2)[WS] = (float (*)[WS]) &block[0][tx];
#pragma unroll
            for(int dn=0; dn<remaining; ++dn, ++bdata2)
                **bdata2 = ptu = **bdata2 + c_AbF*ptu;
        }

        __syncthreads();

        if(n+ty < c_n_size)
            *ptucheck = *bdata;
    }

    // E(utilde) -> E(u) processing --------------------------------------

    // last row-block

    idx = (c_n_size-1-ty)*c_width + m*WS+tx;
    int transp_idx = m*c_height + (c_n_size-1-ty)*WS+tx;

    float *etvtilde = g_etvtilde + idx;

    transp_pm1ybar = g_transp_py + transp_idx-c_height;
    transp_em1zhat = g_transp_ez + transp_idx+c_height;

    const float *ptmn1u = g_ptucheck + idx - c_width;

    // all ptuchecks must be updated!
    __syncthreads();

    float etv;

    n = c_n_size-1;

    {
        block[ty][tx] = *etvtilde;


        if(m > 0)
        {
            CALC_DOT(dot, *transp_pm1ybar, c_HARB_AFB[tx]);
            *bdata += dot*c_ARB_AFP_T[tx];
        }

        if(m < c_m_size-1)
        {
            CALC_DOT(dot, *transp_em1zhat, c_HARB_AFB[tx]);
            *bdata += dot*c_ARE_T[tx];
        }

        if(n-ty > 0)
            *bdata += *ptmn1u*c_HARB_AFP;

        transp_pm1ybar -= WS*blockDim.y;
        transp_em1zhat -= WS*blockDim.y;
        ptmn1u -= c_width*blockDim.y;

        __syncthreads();

        if(ty == 0)
        {
            float (*bdata2)[WS] = (float (*)[WS]) &block[0][tx];

            etv = **bdata2++;

#pragma unroll
            for(int dn=1; dn<blockDim.y; ++dn, ++bdata2)
                **bdata2 = etv = **bdata2 + c_AbR*etv;
        }

        __syncthreads();

        *etvtilde = *bdata;

        etvtilde -= c_width*blockDim.y;

        n -= blockDim.y;
    }

    // middle row-blocks
    if(blockDim.y == CFW)
    {
        int nmin = c_n_size%CFW;
        for(; n>=nmin; n-=CFW)
        {

            *bdata = *etvtilde;


            if(m > 0)
            {
                CALC_DOT(dot, *transp_pm1ybar, c_HARB_AFB[tx]);
                *bdata += dot*c_ARB_AFP_T[tx];
            }

            if(m < c_m_size-1)
            {
                CALC_DOT(dot, *transp_em1zhat, c_HARB_AFB[tx]);
                *bdata += dot*c_ARE_T[tx];
            }

            if(n-ty > 0)
                *bdata += *ptmn1u*c_HARB_AFP;

            transp_pm1ybar -= WS*CFW;
            transp_em1zhat -= WS*CFW;
            ptmn1u -= CFW*c_width;

            __syncthreads();

            if(ty == 0)
            {
                float (*bdata2)[WS] = (float (*)[WS]) &block[0][tx];
#pragma unroll
                for(int dn=0; dn<CFW; ++dn, ++bdata2)
                    **bdata2 = etv = **bdata2 + c_AbR*etv;
            }

            __syncthreads();

            *etvtilde = *bdata;

            etvtilde -= CFW*c_width;
        }
    }

    // remaining row-blocks

    if(n >= 0)
    {

        if(n-ty >= 0)
        {
            *bdata = *etvtilde;
            if(n-ty > 0)
                *bdata += *ptmn1u*c_HARB_AFP;

            if(m > 0)
            {
                CALC_DOT(dot, *transp_pm1ybar, c_HARB_AFB[tx]);
                *bdata += dot*c_ARB_AFP_T[tx];
            }

            if(m < c_m_size-1)
            {
                CALC_DOT(dot, *transp_em1zhat, c_HARB_AFB[tx]);
                *bdata += dot*c_ARE_T[tx];
            }
        }

        int remaining = n+1;
        __syncthreads();

        if(ty == 0)
        {
            float (*bdata2)[WS] = (float (*)[WS]) &block[0][tx];
#pragma unroll
            for(int dn=0; dn<remaining; ++dn, ++bdata2)
                **bdata2 = etv = **bdata2 + c_AbR*etv;
        }

        __syncthreads();

        if(n-ty >= 0)
            *etvtilde = *bdata;
    }
#undef CALC_DOT
}

__global__ __launch_bounds__(WS*SOW, MBO)
void alg5_stage6( float *g_inout,
                  const float *g_transp_py,
                  const float *g_transp_ez,
                  const float *g_ptu,
                  const float *g_etv )
{
    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n = blockIdx.y;

    __shared__ float block[WS][WS+1];

    const float *in = g_inout + (n*WS+ty)*c_width + m*WS+tx;

    float (*curb)[WS+1] = (float (*)[WS+1]) &block[ty][tx];

    // load data into shared memory
    int i;
#pragma unroll
    for(i=0; i<WS-(WS%SOW); i+=SOW)
    {
        **curb = *in;
        in += SOW*c_width;
        curb += SOW;
    }

    if(ty < WS%SOW)
    {
        **curb = *in;
    }

    __shared__ float py[WS], ez[WS], ptu[WS], etv[WS];

    switch(ty)
    {
    case 0:
        if(m > 0)
            py[tx] = g_transp_py[(n*WS + tx) + (m-1)*c_height] / c_b0;
        else
            py[tx] = 0;
        break;
    case 1:
        if(m < c_m_size-1)
            ez[tx] = g_transp_ez[(n*WS + tx) + (m+1)*c_height];
        else
            ez[tx] = 0;
        break;
    case 2:
        if(n > 0)
            ptu[tx] = g_ptu[(m*WS + tx) + (n-1)*c_width] / c_b0;
        else
            ptu[tx] = 0;
        break;
    case 3:
        if(n < c_n_size-1)
            etv[tx] = g_etv[(m*WS + tx) + (n+1)*c_width];
        else
            etv[tx] = 0;
        break;
    }

    __syncthreads();

    if(ty == 0)
    {
        float b0_2 = c_b0*c_b0;
        {
            float *bdata = block[tx];

            // calculate y ---------------------

            float prev = py[tx];

#pragma unroll
            for(int j=0; j<WS; ++j, ++bdata)
                *bdata = prev = *bdata - prev*c_a1;

            // calculate z ---------------------

            prev = ez[tx];

            --bdata;
            for(int j=WS-1; j>=0; --j, --bdata)
                *bdata = prev = *bdata*b0_2 - prev*c_a1;
        }

        {
            float (*bdata)[WS+1] = (float (*)[WS+1]) &block[0][tx];

            // calculate u ---------------------

            float prev = ptu[tx];

#pragma unroll
            for(int i=0; i<WS; ++i, ++bdata)
                **bdata = prev = **bdata - prev*c_a1;

            // calculate v ---------------------
            float *out = g_inout + ((n+1)*WS-1)*c_width + m*WS+tx;

            prev = etv[tx];

            --bdata;
            for(int i=WS-1; i>=0; --i) 
            {
                *out = prev = **bdata-- *b0_2 - prev*c_a1;
                out -= c_width;
            }
        }
    }
}

//-- Host ---------------------------------------------------------------------

__host__
void alg4( float *inout,
           const int& h,
           const int& w,
           const float& b0,
           const float& a1,
           const float& a2 )
{
    up_constants_coefficients2( b0, a1, a2 );

    dim3 cg_img;
    up_constants_sizes( cg_img, h, w );

    dvector<float> d_img(inout, h*w);

    dvector<float2> d_transp_ybar(cg_img.y*w),
        d_transp_zhat(cg_img.y*w),
        d_ubar(cg_img.x*h),
        d_vhat(cg_img.x*h);

    dvector<float2> d_transp_y, d_transp_z, d_u, d_v;

    alg4_stage1<<< cg_img, dim3(WS, SOW) >>>(
        d_img, d_transp_ybar, d_transp_zhat );

    alg4_stage2_3_or_5_6<<< dim3((h+MTS-1)/MTS, 1), dim3(MTS, 1) >>>(
        d_transp_ybar, d_transp_zhat );

    swap( d_transp_ybar, d_transp_y );
    swap( d_transp_zhat, d_transp_z );

    alg4_stage4<<< dim3(cg_img.x, (cg_img.y+2-1)/2), dim3(WS, SOW) >>>(
        d_img, d_transp_y, d_transp_z, d_ubar, d_vhat );

    alg4_stage2_3_or_5_6<<< dim3((w+MTS-1)/MTS, 1), dim3(MTS, 1) >>>(
        d_ubar, d_vhat );

    swap( d_ubar, d_u );
    swap( d_vhat, d_v );

    alg4_stage7<<< cg_img, dim3(WS, SOW) >>>(
        d_img, d_u, d_v );

    d_img.copy_to(inout, h*w);
}

__host__
void alg5( float *inout,
           const int& h,
           const int& w,
           const float& b0,
           const float& a1 )
{
    up_constants_coefficients1( b0, a1 );

    dim3 cg_img;
    up_constants_sizes( cg_img, h, w );

    dvector<float> d_img(inout, h*w);

    dvector<float> d_transp_pybar(cg_img.x*h),
        d_transp_ezhat(cg_img.x*h),
        d_ptucheck(cg_img.y*w),
        d_etvtilde(cg_img.y*w);
                   
    dvector<float> d_transp_py, d_transp_ez, d_ptu, d_etv;

    alg5_stage1<<< cg_img, dim3(WS, SOW) >>>(
        d_img, d_transp_pybar, d_transp_ezhat, d_ptucheck, d_etvtilde );

    alg5_stage2_3<<< dim3(1, cg_img.y), dim3(WS, std::min<int>(cg_img.x, DW)) >>>(
        d_transp_pybar, d_transp_ezhat );

    swap(d_transp_pybar, d_transp_py);
    swap(d_transp_ezhat, d_transp_ez);

    alg5_stage4_5<<< dim3(cg_img.x, 1), dim3(WS, std::min<int>(cg_img.y, CFW)) >>>(
        d_ptucheck, d_etvtilde, d_transp_py, d_transp_ez );

    swap(d_ptucheck, d_ptu);
    swap(d_etvtilde, d_etv);

    alg5_stage6<<< cg_img, dim3(WS, SOW) >>>(
        d_img, d_transp_py, d_transp_ez, d_ptu, d_etv );

    d_img.copy_to(inout, h*w);
}

__host__
void gaussian_gpu( float **inout,
                   const int& h,
                   const int& w,
                   const int& d,
                   const float& s )
{
    float b10, a11, b20, a21, a22;
    weights1( s, b10, a11 );
    weights2( s, b20, a21, a22 );
    for (int c = 0; c < d; c++) {
        alg5( inout[c], h, w, b10, a11 );
        alg4( inout[c], h, w, b20, a21, a22 );
    }
}

__host__
void gaussian_gpu( float *inout,
                   const int& h,
                   const int& w,
                   const float& s )
{
    float b10, a11, b20, a21, a22;
    weights1( s, b10, a11 );
    weights2( s, b20, a21, a22 );
    alg5( inout, h, w, b10, a11 );
    alg4( inout, h, w, b20, a21, a22 );
}

__host__
void bspline3i_gpu( float **inout,
                    const int& h,
                    const int& w,
                    const int& d )
{
    const float alpha = 2.f - sqrt(3.f);
    for (int c = 0; c < d; c++) {
        alg5( inout[c], h, w, 1.f+alpha, alpha );
    }
}

__host__
void bspline3i_gpu( float *inout,
                    const int& h,
                    const int& w )
{
    const float alpha = 2.f - sqrt(3.f);
    alg5( inout, h, w, 1.f+alpha, alpha );
}

//=============================================================================
} // namespace gpufilter
//=============================================================================
// vi: ai ts=4 sw=4
