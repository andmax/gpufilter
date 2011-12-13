/**
 *  @file alg5.cu
 *  @brief CUDA device code for GPU-Efficient Recursive Filtering Algorithm 5
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
#include <extension.h>

#include <gpudefs.h>
#include <gpufilter.h>
#include <gpuconsts.cuh>

#include <alg5.cuh>

//== NAMESPACES ===============================================================

namespace gpufilter {

//== IMPLEMENTATION ===========================================================

//-- Algorithm 5_1 Stage 1 ----------------------------------------------------

__global__ __launch_bounds__(WS*SOW, MBO)
void alg5_stage1( float *g_transp_pybar,
                  float *g_transp_ezhat,
                  float *g_ptucheck,
                  float *g_etvtilde,
                  int extb )
{
    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n = blockIdx.y,
        mwstx = m*WS+tx,
        nwstx = n*WS+tx,
        mewstx = (m-extb)*WS+tx,
        newsty = (n-extb)*WS+ty;
    __shared__ float s_block[WS][WS+1];

    float tu = ( mewstx + 0.5f ) * c_tex_width;
    float tv = ( newsty + 0.5f ) * c_tex_height;

    float (*brow)[WS+1] = (float (*)[WS+1]) &s_block[ty][tx];

    // load data into shared memory
#pragma unroll
    for(int i=0; i<WS-(WS%SOW); i+=SOW)
    {
        **brow = tex2D( t_in, tu, tv );
        brow += SOW;
        tv += SOW * c_tex_height;
    }

    if(ty < WS%SOW)
    {
        **brow = tex2D( t_in, tu, tv );
    }

    // We use a transposed matrix for pybar and ezhat to have
    // coalesced memory accesses. This is the movement for these
    // transposed buffers.
    g_transp_pybar += m*c_height + nwstx; 
    g_transp_ezhat += m*c_height + nwstx;
    g_ptucheck += n*c_width + mwstx;
    g_etvtilde += n*c_width + mwstx;

    __syncthreads();

    if(ty == 0)
    {
        // scan columns
        {
            float *bcol = s_block[tx];

            // calculate pybar, scan left -> right

            float prev = *bcol++;

#pragma unroll
            for(int j=1; j<WS; ++j, ++bcol)
                prev = *bcol -= prev*c_a1;

            *g_transp_pybar = prev*c_b0;
            
            // calculate ezhat, scan right -> left

            prev = *--bcol;
            --bcol;

#pragma unroll
            for(int j=WS-2; j>=0; --j, --bcol)
                prev = *bcol -= prev*c_a1;

            *g_transp_ezhat = prev*c_b0*c_b0;
        }

        // scan rows
        {
            brow = (float (*)[WS+1]) &s_block[0][tx];

            // calculate ptucheck, scan top -> down

            float prev = **brow++;

#pragma unroll
            for(int i=1; i<WS; ++i, ++brow)
                prev = **brow -= prev*c_a1;

            *g_ptucheck = prev*c_b0*c_b0*c_b0;

            // calculate etvtilde, scan bottom -> up

            prev = **--brow;
            --brow;

            for(int i=WS-2; i>=0; --i, --brow)
                prev = **brow - prev*c_a1;

            *g_etvtilde = prev*c_b0*c_b0*c_b0*c_b0;
        }
    }
}

//-- Algorithm 5_1 Stage 2 and 3 ----------------------------------------------

__global__ __launch_bounds__(WS*DW, DNB)
void alg5_stage2_3( float *g_transp_pybar,
                    float *g_transp_ezhat )
{
    int tx = threadIdx.x, ty = threadIdx.y, n = blockIdx.y;

    __shared__ float s_transp_block[DW][WS];
    float *bdata = &s_transp_block[ty][tx];

    // P(ybar) -> P(y) processing --------------------------------------

    float *transp_pybar = g_transp_pybar + ty*c_height + n*WS+tx;

    // first column-transp_block

    // read P(ybar)
    *bdata = *transp_pybar;

    float py; // P(Y)

    __syncthreads();

    if(ty == 0)
    {
        float (*bdata)[WS] = (float (*)[WS]) &s_transp_block[0][tx];

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
                float (*bdata)[WS] = (float (*)[WS]) &s_transp_block[0][tx];
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
            float (*bdata)[WS] = (float (*)[WS]) &s_transp_block[0][tx];
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
            float (*bdata)[WS] = (float (*)[WS]) &s_transp_block[0][tx];
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
                float (*bdata)[WS] = (float (*)[WS]) &s_transp_block[0][tx];
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
            float (*bdata)[WS] = (float (*)[WS]) &s_transp_block[0][tx];
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

//-- Algorithm 5_1 Stage 4 and 5 ----------------------------------------------

__global__ __launch_bounds__(WS*DW, ONB)
void alg5_stage4_5( float *g_ptucheck,
                    float *g_etvtilde,
                    const float *g_transp_py,
                    const float *g_transp_ez )
{
    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x;

    __shared__ float s_block[DW][WS];
    float *bdata = &s_block[ty][tx];

    // P(ucheck) -> P(u) processing --------------------------------------

	volatile __shared__ float s_block_RD_raw[DW][16+32+1];
	volatile float (*block_RD)[16+32+1] 
        = (float (*)[16+32+1]) &s_block_RD_raw[0][16];
    s_block_RD_raw[ty][tx] = 0;

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
            float (*bdata2)[WS] = (float (*)[WS]) &s_block[0][tx];

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
    if(n == DW)
    {
        int nmax = c_n_size-(c_n_size%DW);
        for(; n<nmax; n+=DW)
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

            transp_pm1ybar += WS*DW;
            transp_em1zhat += WS*DW;

            __syncthreads();

            if(ty == 0)
            {
                float (*bdata2)[WS] = (float (*)[WS]) &s_block[0][tx];

#pragma unroll
                for(int dn=0; dn<DW; ++dn, ++bdata2)
                    **bdata2 = ptu = **bdata2 + c_AbF*ptu;
            }

            __syncthreads();

            *ptucheck = *bdata;

            ptucheck += DW*c_width;

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
            float (*bdata2)[WS] = (float (*)[WS]) &s_block[0][tx];
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
        s_block[ty][tx] = *etvtilde;


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
            float (*bdata2)[WS] = (float (*)[WS]) &s_block[0][tx];

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
    if(blockDim.y == DW)
    {
        int nmin = c_n_size%DW;
        for(; n>=nmin; n-=DW)
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

            transp_pm1ybar -= WS*DW;
            transp_em1zhat -= WS*DW;
            ptmn1u -= DW*c_width;

            __syncthreads();

            if(ty == 0)
            {
                float (*bdata2)[WS] = (float (*)[WS]) &s_block[0][tx];
#pragma unroll
                for(int dn=0; dn<DW; ++dn, ++bdata2)
                    **bdata2 = etv = **bdata2 + c_AbR*etv;
            }

            __syncthreads();

            *etvtilde = *bdata;

            etvtilde -= DW*c_width;
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
            float (*bdata2)[WS] = (float (*)[WS]) &s_block[0][tx];
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

//-- Algorithm 5_1 Stage 6 ----------------------------------------------------

__global__ __launch_bounds__(WS*SOW, MBO)
void alg5_stage6( float *g_out,
                  const float *g_transp_py,
                  const float *g_transp_ez,
                  const float *g_ptu,
                  const float *g_etv,
                  int extb )
{
    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n = blockIdx.y,
        mwstx = m*WS+tx,
        nwstx = n*WS+tx,
        mewstx = (m-extb)*WS+tx,
        newsty = (n-extb)*WS+ty,
        newsip = ((n-extb+1)*WS-1)*c_img_pitch;
    __shared__ float s_block[WS][WS+1];

    float tu = ( mewstx + 0.5f ) * c_tex_width;
    float tv = ( newsty + 0.5f ) * c_tex_height;

    float (*brow)[WS+1] = (float (*)[WS+1]) &s_block[ty][tx];

    if( (extb > 0) && (m < extb || n < extb || m > (c_m_size-extb-1) || n > (c_n_size-extb-1)) ) return;
    
    // load data into shared memory
    int i;
#pragma unroll
    for(i=0; i<WS-(WS%SOW); i+=SOW)
    {
        **brow = tex2D( t_in, tu, tv );
        brow += SOW;
        tv += SOW * c_tex_height;
    }

    if(ty < WS%SOW)
    {
        **brow = tex2D( t_in, tu, tv );
    }

    g_out += newsip + mewstx;

    // It is strangely better to make all warps read the carries to
    // registers than to separate carries in four block boundaries
    // reading them with four warps to shared memory
    float py=0.f, ez=0.f, ptu=0.f, etv=0.f;

    if(m > 0)
        py = g_transp_py[nwstx + (m-1)*c_height] * c_1_b0;

    if(m < c_m_size-1)
        ez = g_transp_ez[nwstx + (m+1)*c_height];

    if(n > 0)
        ptu = g_ptu[mwstx + (n-1)*c_width] * c_1_b0;

    if(n < c_n_size-1)
        etv = g_etv[mwstx + (n+1)*c_width];

    __syncthreads();

    if(ty == 0)
    {
        float b0_2 = c_b0*c_b0;
        {
            float *bcol = s_block[tx];

            // calculate y ---------------------

            float prev = py;

#pragma unroll
            for(int j=0; j<WS; ++j, ++bcol)
                *bcol = prev = *bcol - prev*c_a1;

            // calculate z ---------------------

            prev = ez;

            --bcol;
            for(int j=WS-1; j>=0; --j, --bcol)
                *bcol = prev = *bcol*b0_2 - prev*c_a1;
        }

        {
            brow = (float (*)[WS+1]) &s_block[0][tx];

            // calculate u ---------------------

            float prev = ptu;

#pragma unroll
            for(int i=0; i<WS; ++i, ++brow)
                **brow = prev = **brow - prev*c_a1;

            // calculate v ---------------------

            prev = etv;

            --brow;
            for(int i=WS-1; i>=0; --i, --brow)
            {
                *g_out = prev = **brow *b0_2 - prev*c_a1;
                g_out -= c_img_pitch;
            }
        }
    }
}

//-- Host ---------------------------------------------------------------------

__host__
void prepare_alg5( dvector<float>& d_out,
                   cudaArray *& a_in,
                   dvector<float>& d_transp_pybar,
                   dvector<float>& d_transp_ezhat,
                   dvector<float>& d_ptucheck,
                   dvector<float>& d_etvtilde,
                   dim3& cg_img,
                   const float *h_in,
                   const int& h,
                   const int& w,
                   const float& b0,
                   const float& a1,
                   const initcond& ic,
                   const int& extb )
{

    up_constants_coefficients1( b0, a1 );

    up_constants_texture( h, w );

    // cuda channel descriptor for texture
    cudaChannelFormatDesc ccd = cudaCreateChannelDesc<float>();
    cudaMallocArray( &a_in, &ccd, w, h );
    cudaMemcpyToArray( a_in, 0, 0, h_in, h*w*sizeof(float), cudaMemcpyHostToDevice );

    d_out.resize( h * w );

    // 2 times the number of extension blocks, one before and one
    // after the image, on both dimensions
    int ext_h = h + 2*extb*WS;
    int ext_w = w + 2*extb*WS;

    up_constants_sizes( cg_img, ext_h, ext_w, w );

    d_transp_pybar.resize( cg_img.x * ext_h );
    d_transp_ezhat.resize( cg_img.x * ext_h );
    d_ptucheck.resize( cg_img.y * ext_w );
    d_etvtilde.resize( cg_img.y * ext_w );

    t_in.normalized = true;
    t_in.filterMode = cudaFilterModePoint;

    switch( ic ) {
    case zero:
        t_in.addressMode[0] = cudaAddressModeBorder; // defaults to zero-border
        t_in.addressMode[1] = cudaAddressModeBorder;
        break;
    case clamp:
        t_in.addressMode[0] = cudaAddressModeClamp;
        t_in.addressMode[1] = cudaAddressModeClamp;
        break;
    case repeat:
        t_in.addressMode[0] = cudaAddressModeWrap; // implements repeat
        t_in.addressMode[1] = cudaAddressModeWrap;
        break;
    case mirror:
        t_in.addressMode[0] = cudaAddressModeMirror;
        t_in.addressMode[1] = cudaAddressModeMirror;
        break;
    }

}

__host__
void alg5( float *h_inout,
           const int& h,
           const int& w,
           const float& b0,
           const float& a1,
           const initcond& ic,
           const int& extb )
{

    dim3 cg_img;
    dvector<float> d_out, d_transp_pybar, d_transp_ezhat, d_ptucheck, d_etvtilde;
    cudaArray *a_in;

    prepare_alg5( d_out, a_in, d_transp_pybar, d_transp_ezhat, d_ptucheck, d_etvtilde, cg_img, h_inout, h, w, b0, a1, ic, extb );

    alg5( d_out, a_in, d_transp_pybar, d_transp_ezhat, d_ptucheck, d_etvtilde, cg_img, extb );

    d_out.copy_to( h_inout, h * w );

    cudaFreeArray( a_in );

}

__host__
void alg5( dvector<float>& d_out,
           const cudaArray *a_in,
           dvector<float>& d_transp_pybar,
           dvector<float>& d_transp_ezhat,
           dvector<float>& d_ptucheck,
           dvector<float>& d_etvtilde,
           const dim3& cg_img,
           const int& extb )
{

    dvector<float> d_transp_py, d_transp_ez, d_ptu, d_etv;

    cudaBindTextureToArray( t_in, a_in );

    alg5_stage1<<< cg_img, dim3(WS, SOW) >>>( d_transp_pybar, d_transp_ezhat, d_ptucheck, d_etvtilde, extb );

    alg5_stage2_3<<< dim3(1, cg_img.y), dim3(WS, DW) >>>( d_transp_pybar, d_transp_ezhat );

    swap(d_transp_pybar, d_transp_py);
    swap(d_transp_ezhat, d_transp_ez);

    alg5_stage4_5<<< dim3(cg_img.x, 1), dim3(WS, DW) >>>( d_ptucheck, d_etvtilde, d_transp_py, d_transp_ez );

    swap(d_ptucheck, d_ptu);
    swap(d_etvtilde, d_etv);

    alg5_stage6<<< cg_img, dim3(WS, SOW) >>>( d_out, d_transp_py, d_transp_ez, d_ptu, d_etv, extb );

    swap(d_ptu, d_ptucheck);
    swap(d_etv, d_etvtilde);
    swap(d_transp_py, d_transp_pybar);
    swap(d_transp_ez, d_transp_ezhat);

    cudaUnbindTexture( t_in );

}

//=============================================================================
} // namespace gpufilter
//=============================================================================
// vi: ai ts=4 sw=4
