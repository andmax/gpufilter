/**
 *  @file sat.cu
 *  @brief CUDA device code for GPU-Efficient Summed-Area Tables
 *  @author Andre Maximo
 *  @date September, 2011
 */

//== INCLUDES =================================================================

#include <dvector.h>

#include <gpufilter.h>
#include <gpudefs.cuh>
#include <gpuconsts.cuh>

#include <sat.cuh>

//== NAMESPACES ===============================================================

namespace gpufilter {

//== IMPLEMENTATION ===========================================================

__global__ __launch_bounds__( WS * SOW, MBO )
void algorithmSAT_stage1( const float *g_in,
                          float *g_ybar,
                          float *g_vhat ) {

	const int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x, by = blockIdx.y, col = bx*WS+tx, row0 = by*WS;

	__shared__ float block[ WS ][ WS+1 ];

	g_in += (row0+ty)*c_width+col;
	g_ybar += bx*c_height+row0+tx;
	g_vhat += by*c_width+col;

#pragma unroll
    for (int i = 0; i < WS/SOW; ++i) {
        block[ty+i*SOW][tx] = *g_in;
        g_in += SOW * c_width;
    }
    if( ty < (WS-((WS/SOW)*SOW)) ) {
        block[ty+((WS/SOW)*SOW)][tx] = *g_in;
    }

	__syncthreads();

	if( ty == 0 ) {

        // if( c_width-bx*WS < WS ) {
        //     for (int i = 1; i < c_width-bx*WS; ++i)
        //         block[tx][i] += block[tx][i-1];
        // } else
#pragma unroll
            for (int i = 1; i < WS; ++i)
                block[tx][i] += block[tx][i-1];

		*g_ybar = block[tx][WS-1];

        // if( c_height-by*WS < WS ) {
        //     for (int i = 1; i < c_height-by*WS; ++i)
        //         block[i][tx] += block[i-1][tx];
        // } else
#pragma unroll
            for (int i = 1; i < WS; ++i)
                block[i][tx] += block[i-1][tx];

		*g_vhat = block[WS-1][tx];

	}

}

__global__ __launch_bounds__( WS * MW, MBO )
void algorithmSAT_stage2( float *g_ybar,
                          float *g_ysum ) {

	const int tx = threadIdx.x, ty = threadIdx.y, by = blockIdx.y, row0 = by*MW+ty, row = row0*WS+tx;

	if( row >= c_height ) return;

	g_ybar += row;
	float y = *g_ybar;
	int ln = HWS+tx;

	if( tx == WS-1 )
		g_ysum += row0;

	volatile __shared__ float block[ MW ][ HWS+WS+1 ];

	if( tx < HWS ) block[ty][tx] = 0.f;
	else block[ty][ln] = 0.f;

	for (int n = 1; n < c_n_size; ++n) {

		block[ty][ln] = y;

		block[ty][ln] += block[ty][ln-1];
		block[ty][ln] += block[ty][ln-2];
		block[ty][ln] += block[ty][ln-4];
		block[ty][ln] += block[ty][ln-8];
		block[ty][ln] += block[ty][ln-16];

		if( tx == WS-1 ) {
			*g_ysum = block[ty][ln];
			g_ysum += c_m_size;
		}

		g_ybar += c_height;
		y = *g_ybar += y;

	}

}

__global__ __launch_bounds__( WS * MW, MBO )
void algorithmSAT_stage3( const float *g_ysum,
                          float *g_vhat ) {

	const int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x, col0 = bx*MW+ty, col = col0*WS+tx;

	if( col >= c_width ) return;

	g_vhat += col;
	float y = 0.f, v = 0.f;

	if( col0 > 0 )
		g_ysum += (col0-1)*c_m_size;

	for (int m = 0; m < c_m_size; ++m) {

		if( col0 > 0 ) {
			y = *g_ysum;
			g_ysum += 1;
		}

		v = *g_vhat += v + y;
		g_vhat += c_width;

	}

}

__global__ __launch_bounds__( WS * SOW, MBO )
void algorithmSAT_stage4( float *g_inout,
                          const float *g_y,
                          const float *g_v ) {

	const int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x, by = blockIdx.y, col = bx*WS+tx, row0 = by*WS;

	__shared__ float block[ WS+1 ][ WS+1 ];

	g_inout += (row0+ty)*c_width+col;
	if( bx > 0 ) g_y += (bx-1)*c_height+row0+tx;
	if( by > 0 ) g_v += (by-1)*c_width+col;

#pragma unroll
    for (int i = 0; i < WS/SOW; ++i) {
        block[ty+1+i*SOW][tx+1] = *g_inout;
        g_inout += SOW * c_width;
    }
    if( ty < (WS-((WS/SOW)*SOW)) ) {
        block[ty+1+((WS/SOW)*SOW)][tx+1] = *g_inout;
    } else if( ty == (WS-((WS/SOW)*SOW)) ) {
		if( bx > 0 ) block[tx+1][0] = *g_y;
		else block[tx+1][0] = 0.f;
    } else if( ty == 1+(WS-((WS/SOW)*SOW)) ) {
		if( by > 0 ) block[0][tx+1] = *g_v;
		else block[0][tx+1] = 0.f;
    }

	__syncthreads();

	if( ty == 0 ) {

        // if( c_width-bx*WS < WS ) {
        //     for (int i = 1; i < c_width-bx*WS+1; ++i)
        //         block[tx+1][i] += block[tx+1][i-1];
        // } else
#pragma unroll
            for (int i = 1; i < WS+1; ++i)
                block[tx+1][i] += block[tx+1][i-1];

        // if( c_height-by*WS < WS ) {
        //     for (int i = 1; i < c_height-by*WS+1; ++i)
        //         block[i][tx+1] += block[i-1][tx+1];
        // } else
#pragma unroll
            for (int i = 1; i < WS+1; ++i)
                block[i][tx+1] += block[i-1][tx+1];

	}

	__syncthreads();

	g_inout -= 30*c_width;

#pragma unroll
    for (int i = 0; i < WS/SOW; ++i) {
        *g_inout = block[ty+1+i*SOW][tx+1];
        g_inout += SOW * c_width;
    }
    if( ty < (WS-((WS/SOW)*SOW)) ) {
        *g_inout = block[ty+1+((WS/SOW)*SOW)][tx+1];
    }

}

__host__
void algorithmSAT( dvector<float>& d_img,
                   dvector<float>& d_ybar,
                   dvector<float>& d_vhat,
                   dvector<float>& d_ysum,
                   const dim3& cg_img,
                   const dim3& cg_ybar,
                   const dim3& cg_vhat ) {

    algorithmSAT_stage1<<< cg_img, dim3(WS, SOW) >>>( d_img, d_ybar, d_vhat );

    algorithmSAT_stage2<<< cg_ybar, dim3(WS, MW) >>>( d_ybar, d_ysum );

    algorithmSAT_stage3<<< cg_vhat, dim3(WS, MW) >>>( d_ysum, d_vhat );

    algorithmSAT_stage4<<< cg_img, dim3(WS, SOW) >>>( d_img, d_ybar, d_vhat );

}

__host__
void algorithmSAT( float *inout,
                   const int& h,
                   const int& w ) {

    int h_out = h, w_out = w;

    if( h % 32 > 0 ) h_out += (32 - (h % 32));
    if( w % 32 > 0 ) w_out += (32 - (w % 32));

    float *co_inout = inout; // coalesced inout

    if( w_out > w or h_out > h ) {
        
        co_inout = new float[ h_out * w_out ];

        for (int i = 0; i < h; ++i)
            for (int j = 0; j < w; ++j)
                co_inout[i*w_out+j] = inout[i*w+j];

    }

    dim3 cg_img; // computational grid of input image
    up_constants_sizes( cg_img, h_out, w_out );

    dvector<float> d_out( co_inout, h_out * w_out );

    dvector<float> d_ybar( cg_img.x * h_out ), d_vhat( cg_img.x * h_out ), d_ysum( cg_img.x * cg_img.y );

	const int nWm = (w_out+MTS-1)/MTS, nHm = (h_out+MTS-1)/MTS;
    dim3 cg_ybar(1, nHm), cg_vhat(nWm, 1);

    algorithmSAT( d_out, d_ybar, d_vhat, d_ysum, cg_img, cg_ybar, cg_vhat );

    d_out.copy_to( co_inout, h_out * w_out );

    if( co_inout != inout ) {

        for (int i = 0; i < h; ++i)
            for (int j = 0; j < w; ++j)
                inout[i*w+j] = co_inout[i*w_out+j];

        delete [] co_inout;

    }

}

//=============================================================================
} // namespace gpufilter
//=============================================================================
