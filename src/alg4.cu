/**
 *  @file gpufilter.cu
 *  @brief CUDA device code for GPU-Efficient Recursive Filtering Algorithm 4
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

#include <dvector.h>
#include <gpufilter.h>
#include <gpudefs.cuh>
#include <gpuconsts.cuh>

#include <alg4.cuh>

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

//=============================================================================
} // namespace gpufilter
//=============================================================================
// vi: ai ts=4 sw=4
