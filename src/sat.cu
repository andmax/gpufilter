/**
 *  @file sat.cu
 *  @brief CUDA device code for GPU-Efficient Summed-Area Tables
 *  @author Andre Maximo
 *  @date September, 2011
 */

//== INCLUDES =================================================================

#include <symbol.h>
#include <dvector.h>

#include <gpufilter.h>
#include <gpudefs.cuh>

#include <sat.cuh>

//== NAMESPACES ===============================================================

namespace gpufilter {

//== IMPLEMENTATION ===========================================================

__global__ __launch_bounds__( WS * SOW, MBO )
void algorithmSAT_stage1( const float *g_in,
                          float *g_xbar,
                          float *g_yhat ) {

	const int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x, by = blockIdx.y, col = bx*WS+tx, row0 = by*WS;

	__shared__ float block[ WS ][ WS+1 ];

	g_in += (row0+ty)*c_width+col;
	g_xbar += bx*c_height+row0+tx;
	g_yhat += by*c_width+col;

	block[ty][tx] = *g_in; g_in += 5*c_width;
	block[ty+5][tx] = *g_in; g_in += 5*c_width;
	block[ty+10][tx] = *g_in; g_in += 5*c_width;
	block[ty+15][tx] = *g_in; g_in += 5*c_width;
	block[ty+20][tx] = *g_in; g_in += 5*c_width;
	block[ty+25][tx] = *g_in; g_in += 5*c_width;
	if( ty == 0 || ty == 1 ) block[ty+30][tx] = *g_in;

	__syncthreads();

	if( ty == 0 ) {

#pragma unroll
		for (int i = 1; i < WS; ++i)
			block[tx][i] += block[tx][i-1];
		*g_xbar = block[tx][31];

#pragma unroll
		for (int i = 1; i < WS; ++i)
			block[i][tx] += block[i-1][tx];
		*g_yhat = block[31][tx];

	}

}

__global__ __launch_bounds__( WS * MW, MBO )
void algorithmSAT_stage2( float *g_xbar,
                          float *g_xsum ) {

	const int tx = threadIdx.x, ty = threadIdx.y, by = blockIdx.y, row0 = by*MW+ty, row = row0*WS+tx;

	if( row >= c_height ) return;

	g_xbar += row;
	float x = *g_xbar;
	int ln = HWS+tx;

	if( tx == 31 )
		g_xsum += row0;

	volatile __shared__ float block[ MW ][ HWS+WS+1 ];

	if( tx < HWS ) block[ty][tx] = 0.f;
	else block[ty][ln] = 0.f;

	for (int n = 1; n < c_n_size; ++n) {

		block[ty][ln] = x;

		block[ty][ln] += block[ty][ln-1];
		block[ty][ln] += block[ty][ln-2];
		block[ty][ln] += block[ty][ln-4];
		block[ty][ln] += block[ty][ln-8];
		block[ty][ln] += block[ty][ln-16];

		if( tx == 31 ) {
			*g_xsum = block[ty][ln];
			g_xsum += c_m_size;
		}

		g_xbar += c_height;
		x = *g_xbar += x;

	}

}

__global__ __launch_bounds__( WS * MW, MBO )
void algorithmSAT_stage3( const float *g_xsum,
                          float *g_yhat ) {

	const int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x, col0 = bx*MW+ty, col = col0*WS+tx;

	if( col >= c_width ) return;

	g_yhat += col;
	float y = 0.f, x = 0.f;

	if( col0 > 0 )
		g_xsum += (col0-1)*c_m_size;

	for (int m = 0; m < c_m_size; ++m) {

		if( col0 > 0 ) {
			x = *g_xsum;
			g_xsum += 1;
		}

		y = *g_yhat += y + x;
		g_yhat += c_width;

	}

}

__global__ __launch_bounds__( WS * SOW, MBO )
void algorithmSAT_stage4( float *g_inout,
                          const float *g_xbar,
                          const float *g_ybar ) {

	const int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x, by = blockIdx.y, col = bx*WS+tx, row0 = by*WS;

	__shared__ float block[ WS+1 ][ WS+1 ];

	g_inout += (row0+ty)*c_width+col;
	if( bx > 0 ) g_xbar += (bx-1)*c_height+row0+tx;
	if( by > 0 ) g_ybar += (by-1)*c_width+col;

	block[ty+1][tx+1] = *g_inout; g_inout += 5*c_width;
	block[ty+6][tx+1] = *g_inout; g_inout += 5*c_width;
	block[ty+11][tx+1] = *g_inout; g_inout += 5*c_width;
	block[ty+16][tx+1] = *g_inout; g_inout += 5*c_width;
	block[ty+21][tx+1] = *g_inout; g_inout += 5*c_width;
	block[ty+26][tx+1] = *g_inout; g_inout += 5*c_width;

	if( ty == 0 || ty == 1 ) {

        block[ty+31][tx+1] = *g_inout;

    } else if( ty == 2 ) {

		if( bx > 0 ) block[tx+1][0] = *g_xbar;
		else block[tx+1][0] = 0.f;

	} else if( ty == 3 ) {

		if( by > 0 ) block[0][tx+1] = *g_ybar;
		else block[0][tx+1] = 0.f;

	}

	g_inout -= 30*c_width;

	__syncthreads();

	if( ty == 0 ) {

#pragma unroll
		for (int i = 1; i < 33; ++i)
			block[tx+1][i] += block[tx+1][i-1];

#pragma unroll
		for (int i = 1; i < 33; ++i)
			block[i][tx+1] += block[i-1][tx+1];

	}

	__syncthreads();

	*g_inout = block[ty+1][tx+1]; g_inout += 5*c_width;
	*g_inout = block[ty+6][tx+1]; g_inout += 5*c_width;
	*g_inout = block[ty+11][tx+1]; g_inout += 5*c_width;
	*g_inout = block[ty+16][tx+1]; g_inout += 5*c_width;
	*g_inout = block[ty+21][tx+1]; g_inout += 5*c_width;
	*g_inout = block[ty+26][tx+1]; g_inout += 5*c_width;

	if( ty == 0 || ty == 1 ) *g_inout = block[ty+31][tx+1];

}

__host__
void algorithmSAT( float *inout,
                   const int& h,
                   const int& w ) {

    dvector<float> d_img( inout, w*h );

	const int nWm = (w+MTS-1)/MTS, nHm = (h+MTS-1)/MTS;

    const int m_size = (h+WS-1)/WS, n_size = (w+WS-1)/WS;

    copy_to_symbol("c_height", h);
	copy_to_symbol("c_width", w);

	copy_to_symbol("c_n_size", n_size);
    copy_to_symbol("c_m_size", m_size);

    dvector<float> d_xbar(n_size*h), d_ybar(n_size*h), d_xsum(n_size*m_size);

    algorithmSAT_stage1<<< dim3(n_size, m_size), dim3(WS, SOW) >>>( d_img, d_xbar, d_ybar );

    algorithmSAT_stage2<<< dim3(1, nHm), dim3(WS, MW) >>>( d_xbar, d_xsum );

    algorithmSAT_stage3<<< dim3(nWm, 1), dim3(WS, MW) >>>( d_xsum, d_ybar );

    algorithmSAT_stage4<<< dim3(n_size, m_size), dim3(WS, SOW) >>>( d_img, d_xbar, d_ybar );

    d_img.copy_to( inout, h*w );

}

//=============================================================================
} // namespace gpufilter
//=============================================================================
