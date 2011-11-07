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
#include <complex>
#include <cassert>
#include <iostream>
#include <algorithm>

#include <symbol.h>
#include <dvector.h>

#include <gpufilter.h>
#include <gpudefs.cuh>

#include <gpufilter.cuh>

//== NAMESPACES ===============================================================

namespace gpufilter {

//== IMPLEMENTATION ===========================================================

__global__
__launch_bounds__(WS*DW, ONB)
void algorithm5_stage1( const float *g_in,
                        float *g_transp_ybar,
                        float *g_transp_zhat,
                        float *g_ucheck,
                        float *g_vtilde ) {

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

__global__
__launch_bounds__(WS*DW, DNB)
void algorithm5_stage2_3( float *g_transp_ybar,
                          float *g_transp_zhat ) {

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

__global__
__launch_bounds__(WS*OW, DNB)
void algorithm5_stage4_5_step1( float *g_ucheck,
                                float *g_vtilde,
                                const float *g_y,
                                const float *g_z ) {

    int tx = threadIdx.x, ty = threadIdx.y, n = blockIdx.x*OW + ty, m = blockIdx.y;

    if(n >= c_n_size)
        return;

    __shared__ float shared_yimb1[OW][WS], shared_zim1b[OW][WS];

    float delta_x, delta_y;

    delta_x = c_Delta_x_tail[tx];
    delta_y = c_Delta_y[tx];

    g_y += (n-1)*c_height + m*WS; 
    g_z += (n+1)*c_height + m*WS;

    float *yimb1 = shared_yimb1[ty], *zim1b = shared_zim1b[ty];

    if(n > 0)
        yimb1[tx] = g_y[tx];
    
    if(n < c_n_size-1)
        zim1b[tx] = g_z[tx];

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

__global__
__launch_bounds__(WS*DW, DNB)
void algorithm5_stage4_5_step2( float *g_ubar,
                                float *g_vcheck ) {

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

__global__
__launch_bounds__(WS*DW, ONB)
void algorithm5_stage6( float *g_in,
                        const float *g_y,
                        const float *g_z,
                        const float *g_u,
                        const float *g_v ) {

    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.y*2, n = blockIdx.x;

    __shared__ float block[WS*2][WS+1];

    const float *in_data = g_in + (m*WS + ty)*c_width + n*WS + tx;

#pragma unroll
    for(int i=0; i<WS; i+=8)
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
                bdata[0] = prev -= g_y[(n-1)*c_height + m*WS+tx]*c_Linf1;
#pragma unroll
            for(int j=1; j<WS; ++j)
                prev = bdata[j] -= prev*c_Linf1;

            // calculate z ---------------------

            if(n < c_n_size-1)
                bdata[WS-1] = prev = (bdata[WS-1] - g_z[(n+1)*c_height+m*WS+tx])*c_Linf1;
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

            float *out_data = g_in + (m*WS+WS-1)*c_width + n*WS + tx;

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

__host__
void algorithm5( float *inout,
                 const int& h,
                 const int& w,
                 const float& b0,
                 const float& a1 ) {

    dvector<float> d_img( inout, h*w );

    const float Linf = a1, iR = b0*b0*b0*b0/Linf/Linf;

    std::vector<float> signrevprodLinf(WS);

    signrevprodLinf[WS-1] = 1;

    for(int i=WS-2; i>=0; --i)
        signrevprodLinf[i] = -signrevprodLinf[i+1]*Linf;

    copy_to_symbol("c_SignRevProdLinf", signrevprodLinf);

    std::vector<float> prodLinf(WS);

    prodLinf[0] = Linf;
    for(int i=1; i<WS; ++i)
        prodLinf[i] = prodLinf[i-1]*Linf;

    copy_to_symbol("c_ProdLinf", prodLinf);

    copy_to_symbol("c_iR1", iR);
    copy_to_symbol("c_Linf1", Linf);

    const float Linf2 = Linf*Linf,
                alpha = Linf2*(1-pow(Linf2,WS))/(1-Linf2),
                stm = (WS & 1 ? -1 : 1)*pow(Linf, WS);

    copy_to_symbol("c_Stm", stm);
    copy_to_symbol("c_Svm", stm);
    copy_to_symbol("c_Alpha", alpha);

    std::vector<float> delta_x_tail(WS), delta_y(WS);

    delta_x_tail[WS-1] = -Linf;
    for(int j=WS-2; j>=0; --j)
        delta_x_tail[j] = -delta_x_tail[j+1]*Linf;

    float sign = WS & 1 ? -1 : 1;
    for(int j=WS-1; j>=0; --j)
    {
        delta_y[j] = sign*pow(Linf,2+j)*(1-pow(Linf,2*(WS+1-j)))/(1-Linf*Linf);
        sign *= -1;
    }

    copy_to_symbol("c_Delta_x_tail", delta_x_tail);
    copy_to_symbol("c_Delta_y", delta_y);

    const int m_size = (h+WS-1)/WS, n_size = (w+WS-1)/WS;

    copy_to_symbol("c_height", h);
    copy_to_symbol("c_width", w);
    copy_to_symbol("c_m_size", m_size);
    copy_to_symbol("c_n_size", n_size);

    dvector<float> d_transp_ybar(n_size*h),
        d_transp_zhat(n_size*h),
        d_ucheck(m_size*w),
        d_vtilde(m_size*w);
                   
    dvector<float> d_y, d_z, d_ubar, d_u, d_vcheck, d_v;

    algorithm5_stage1<<< dim3(n_size, (m_size+2-1)/2), dim3(WS, DW) >>>( d_img, d_transp_ybar, d_transp_zhat, d_ucheck, d_vtilde );

    algorithm5_stage2_3<<< dim3(1, (m_size+2-1)/2), dim3(WS, std::min(n_size, DW)) >>>( d_transp_ybar, d_transp_zhat );

    swap(d_transp_ybar, d_y);
    swap(d_transp_zhat, d_z);

    algorithm5_stage4_5_step1<<< dim3((n_size+OW-1)/OW, m_size), dim3(WS, OW) >>>( d_ucheck, d_vtilde, d_y, d_z );

    swap(d_ucheck, d_ubar);
    swap(d_vtilde, d_vcheck);

    algorithm5_stage4_5_step2<<< dim3((n_size+2-1)/2), dim3(WS, std::min(m_size, DW)) >>>( d_ubar, d_vcheck );

    swap(d_ubar, d_u);
    swap(d_vcheck, d_v);

    algorithm5_stage6<<< dim3(n_size, (m_size+2-1)/2), dim3(WS, DW) >>>( d_img, d_y, d_z, d_u, d_v );

    d_img.copy_to( inout, h*w );

}

//=============================================================================
} // namespace gpufilter
//=============================================================================
// vi: ai ts=4 sw=4
