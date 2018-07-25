/**
 *    Gaussian Blur Computation
 *
 *  Maximo, Andre -- Sep, 2011
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <cfloat>

#include <cufft.h>

#include <util/util.h>
#include <util/symbol.h>
//#include <util/chronos.h>
#include <util/dvector.h>
#include <util/timer.h>
#include <util/recfilter.h>

//#include "pgm.h"

#include "gauss_dir_cpu.h"

using namespace gpufilter;

#define WS 32 ///< Warp Size
#define NW 8 ///< Number of warps per block (with all warps computing)
#define NB 8 ///< Number of blocks per SM
#define WS1 32 ///< for gauss1
#define NW1 8 ///< for gauss1
#define NB1 8 ///< for gauss1
#define WS2 16 ///< for gauss2
#define NW2 8 ///< for gauss2
#define NB2 16 ///< for gauss2
#define WS3 32 ///< for gauss3
#define NW3 16 ///< for gauss3
#define NB3 3 ///< for gauss3
#define MNB 2 ///< Maximum number of blocks per SM
#define MNT 1024 ///< Maximum number of threads per block (with MNB)

unsigned RUN_TIMES=1024;
int res_out;
#define DEFAULT_RES_OUT 1 // res_out: 0 verbose; 1 perf; 2 error

#define KERNEL_RADIUS 33
//#define KERNEL_RADIUS 47

//#define COMPUTE_IN_CPU // compute in cpu debug flag: cpu can be very slow..
//#define RUN_GAUSS_0
//#define RUN_GAUSS_1
//#define RUN_GAUSS_2
//#define RUN_GAUSS_3
//#define RUN_GAUSS_FFT
//#define RUN_GAUSS_REC

// get cactus with Diego Nehab
//#define USE_CACTUS // use cactus debug flag: read/write file names below
#define CACTUS_IN "cactus-1024.pgm"
#define CACTUS_GPU "cactus-1024-direct-gpu.pgm"
#define CACTUS_CPU "cactus-1024-direct-cpu.pgm"
#define CACTUS_FFT_GPU "cactus-1024-fft-gpu.pgm"
#define CACTUS_RECURSIVE_GPU "cactus-1024-recursive-gpu.pgm"

/// Naming conventions are: c_ constant; t_ texture; g_ global memory;
/// s_ shared memory; d_ device pointer; a_ cuda-array; p_ template
/// parameter; f_ surface.

__constant__ int c_width, c_height;
__constant__ Vector< float, KERNEL_RADIUS * 2 + 1 > c_kernel;
__constant__ int c_fft_width;

texture< float, 2, cudaReadModeElementType > t_in_img;

surface< void, 2 > f_in_img;
surface< void, 2 > f_out_img;

/// Memory types are: TMEM texture memory; SMEM shared memory; GMEM
/// global memory

///------------------------- AUXILIARY FUNCTIONS ----------------------------
__device__ inline // auxiliary reflect function in device
int d_reflect( const int& i, const int& n ) {
    if( i < 0 ) return (-i-1);
    else if( i >= n ) return (2*n-i-1);
    else return i;
}

__device__ inline // auxiliary clamp function in device
int d_clamp( const int& i, const int& n ) {
    if( i < 0 ) return 0;
    else if( i >= n ) return n-1;
    else return i;
}

///----------------- GAUSS-0 - TWO-PASS CONVOLUTION IN GMEM -----------------
template< int p_radius >
__global__ __launch_bounds__( WS*NW, NB )
void gauss0_rows( float *g_out, const float *g_in ) {

    const int tx = threadIdx.x, ty = threadIdx.y,
        bx = blockIdx.x, by = blockIdx.y, col = bx*WS+tx, row = by*NW+ty;

    if( row >= c_height or col >= c_width ) return;

    g_in += row*c_width;
    g_out += row*c_width + col;

    float s = 0.f;

#pragma unroll
    for (int k = -p_radius; k <= p_radius; ++k) {
        s += g_in[ d_clamp(col+k, c_width) ] * c_kernel[ k + p_radius ];
    }

    *g_out = s;

}

template< int p_radius >
__global__ __launch_bounds__( WS*NW, NB )
void gauss0_cols( float *g_out, const float *g_in ) {

    const int tx = threadIdx.x, ty = threadIdx.y,
        bx = blockIdx.x, by = blockIdx.y, col = bx*WS+tx, row = by*NW+ty;

    if( row >= c_height or col >= c_width ) return;

    g_in += col;
    g_out += row*c_width + col;

    float s = 0.f;

#pragma unroll
    for (int k = -p_radius; k <= p_radius; ++k) {
        s += g_in[ d_clamp(row+k, c_height)*c_width ] * c_kernel[ k + p_radius ];
    }

    *g_out = s;

}

///----------------- GAUSS-1 - TWO-PASS CONVOLUTION IN TMEM -----------------
template< int p_radius >
__global__ __launch_bounds__( WS1*NW1, NB1 )
void gauss1_rows( void ) {

    const int tx = threadIdx.x, ty = threadIdx.y,
        bx = blockIdx.x, by = blockIdx.y, col = bx*WS1+tx, row = by*NW1+ty;

    if( row >= c_height or col >= c_width ) return;

    float tu = col + .5f, tv = row + .5f, s = 0.f;

#pragma unroll
    for (int k = -p_radius; k <= p_radius; ++k) {
        s += tex2D( t_in_img, tu+k, tv ) * c_kernel[ k + p_radius ];
    }

    surf2Dwrite( s, f_out_img, col*4, row, cudaBoundaryModeTrap ); // trap kills kernel if outside boundary

}

template< int p_radius >
__global__ __launch_bounds__( WS1*NW1, NB1 )
void gauss1_cols( float *g_out ) {
    const int tx = threadIdx.x, ty = threadIdx.y,
        bx = blockIdx.x, by = blockIdx.y, col = bx*WS1+tx, row = by*NW1+ty;
    if( row >= c_height or col >= c_width ) return;
    g_out += row*c_width + col;
    float tu = col + .5f, tv = row + .5f;
    float s = 0.f;

#pragma unroll
    for (int k = -p_radius; k <= p_radius; ++k) {
        s += tex2D( t_in_img, tu, tv+k ) * c_kernel[ k + p_radius ];
    }

    *g_out = s;

}

///----------------- GAUSS-2 - TWO-PASS CONVOLUTION IN SMEM -----------------
template< int p_radius >
__global__ __launch_bounds__( MNT, MNB )
void gauss2_rows( void ) {
    const int tx = threadIdx.x, bx = blockIdx.x, by = blockIdx.y;

    float tu = bx*MNT+tx + .5f, tv = by + .5f;
    float s = 0.f;
    volatile __shared__ float s_row[ MNT + p_radius*2 ];

    s_row[ p_radius + tx ] = tex2D( t_in_img, tu, tv );

    if( tx < p_radius ) s_row[ tx ] = tex2D( t_in_img, tu - p_radius, tv );
    else if( tx < 2*p_radius ) s_row[ MNT + tx ] = tex2D( t_in_img, tu - p_radius + MNT, tv );

    __syncthreads();
    if( bx*MNT+tx >= c_width ) return;

#pragma unroll
    for (int k = -p_radius; k <= p_radius; ++k) {
        s += s_row[ p_radius + tx + k ] * c_kernel[ k + p_radius ];
    }

    surf2Dwrite( s, f_out_img, (bx*MNT+tx)*4, by, cudaBoundaryModeTrap );

}

template< int p_radius >
__global__ __launch_bounds__( WS2*NW2, NB2 )
void gauss2_cols( float *g_out ) {
    const int tx = threadIdx.x, ty = threadIdx.y,
        bx = blockIdx.x, by = blockIdx.y;

    float tu = bx*WS2+tx + .5f, tv = by*NW2+ty + .5f;
    float s = 0.f;

    volatile __shared__ float s_cols[ WS2 ][ NW2 + p_radius*2 + 1 ];

    s_cols[ tx ][ p_radius + ty ] = tex2D( t_in_img, tu, tv );

    if( p_radius <= NW2/2 ) {
        if( ty < p_radius ) s_cols[ tx ][ ty ] = tex2D( t_in_img, tu, tv - p_radius );
        else if( ty < 2*p_radius ) s_cols[ tx ][ NW2 + ty ] = tex2D( t_in_img, tu, tv - p_radius + NW2 );
    } else if( p_radius <= NW2 ) {
        if( ty < p_radius ) {
            s_cols[ tx ][ ty ] = tex2D( t_in_img, tu, tv - p_radius );
            s_cols[ tx ][ p_radius + NW2 + ty ] = tex2D( t_in_img, tu, tv + NW2 );
        }
    } else {
        for (int i = 0; i < (p_radius+NW2-1)/NW2; ++i) {
            int wy = i*NW2+ty;
            if( wy < p_radius ) {
                s_cols[ tx ][ wy ] = tex2D( t_in_img, tu, tv - p_radius + i*NW2 );
                s_cols[ tx ][ p_radius + NW2 + wy ] = tex2D( t_in_img, tu, tv + NW2 + i*NW2 );
            }
        }
    }

    __syncthreads();
    if( bx*WS2+tx >= c_width or by*NW2+ty >= c_height ) return;

    g_out += (by*NW2+ty)*c_width + bx*WS2+tx;

#pragma unroll
    for (int k = -p_radius; k <= p_radius; ++k) {
        s += s_cols[tx][ p_radius + ty + k ] * c_kernel[ k + p_radius ];
    }

    *g_out = s;

}

///----------------- GAUSS-3 - ONE-PASS CONVOLUTION -----------------
template< int p_radius >
__device__
void load_convolve_rows( volatile float *s_in, const int& tx, const float& tu, const float& tv ) {

    // load middle data
    s_in[ p_radius + tx ] = tex2D( t_in_img, tu, tv );

    // load left and right data
    if( p_radius <= WS3/2 ) {
        if( tx < p_radius ) s_in[ tx ] = tex2D( t_in_img, tu - p_radius, tv );
        else if( tx < p_radius*2 ) s_in[ WS3 + tx ] = tex2D( t_in_img, tu - p_radius + WS3, tv );
    } else if( p_radius <= WS3 ) {
        if( tx < p_radius ) {
            s_in[ tx ] = tex2D( t_in_img, tu - p_radius, tv );
            s_in[ p_radius + WS3 + tx ] = tex2D( t_in_img, tu + WS3, tv );
        }
    } else {
        for (int i = 0; i < (p_radius+WS3-1)/WS3; ++i) {
            int wx = i*WS3+tx;
            if( wx < p_radius ) {
                s_in[ wx ] = tex2D( t_in_img, tu - p_radius + i*WS3, tv );
                s_in[ p_radius + WS3 + wx ] = tex2D( t_in_img, tu + WS3 + i*WS3, tv );
            }
        }
    }

    // convolve row
    float s = 0.f;
    for (int k = -p_radius; k <= p_radius; ++k) {

        s += s_in[ p_radius + tx + k ] * c_kernel[ k + p_radius ];

    }

    s_in[ p_radius + tx ] = s;

}

template< int p_radius >
__global__ __launch_bounds__( WS3*NW3, NB3 )
void gauss3( float *g_out ) {
    int tx = threadIdx.x, ty = threadIdx.y,
        bx = blockIdx.x, by = blockIdx.y, col = bx*WS3+tx, row = by*NW3+ty;
    if( row >= c_height or col >= c_width ) bx = -1;
    float tu = col + .5f, tv = row + .5f;
    volatile __shared__ float s_inblock[ NW3 + p_radius*2 ][ WS3 + p_radius*2 ];

    // load middle data
    load_convolve_rows< p_radius >( &s_inblock[ p_radius + ty ][0], tx, tu, tv );

    // load upper and lower data
    if( p_radius <= NW3/2 ) {
        if( ty < p_radius ) load_convolve_rows< p_radius >( &s_inblock[ ty ][0], tx, tu, tv - p_radius );
        else if( ty < p_radius*2 ) load_convolve_rows< p_radius >( &s_inblock[ NW3 + ty ][0], tx, tu, tv - p_radius + NW3 );
    } else if( p_radius <= NW3 ) {
        if( ty < p_radius ) {
            load_convolve_rows< p_radius >( &s_inblock[ ty ][0], tx, tu, tv - p_radius );
            load_convolve_rows< p_radius >( &s_inblock[ p_radius + NW3 + ty ][0], tx, tu, tv + NW3 );
        }
    } else {
        for (int i = 0; i < (p_radius+NW3-1)/NW3; ++i) {
            int wy = i*NW3+ty;
            if( wy < p_radius ) {
                load_convolve_rows< p_radius >( &s_inblock[ wy ][0], tx, tu, tv - p_radius + i*NW3 );
                load_convolve_rows< p_radius >( &s_inblock[ p_radius + NW3 + wy ][0], tx, tu, tv + NW3 + i*NW3 );
            }
        }
    }

    __syncthreads();
    if( bx == -1 ) return;
    g_out += row*c_width + col;

    // convolve cols
    float s = 0.f;
    for (int k = -p_radius; k <= p_radius; ++k) {

        s += s_inblock[ p_radius + ty + k ][ p_radius + tx ] * c_kernel[ k + p_radius ];

    }

    *g_out = s;

}

///----------------- GAUSS-FFT - CONVOLUTION THROUGH FFT -----------------
__global__ __launch_bounds__( WS*NW, NB )
void apply_gauss_hat_kernel( float2 *img, float cte ) {

    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int x = blockDim.x * blockIdx.x + threadIdx.x;

    if( x >= c_fft_width or y >= c_height ) return;

    float wx = float(x)/c_fft_width/2.f, wy;

    if( y < c_height/2 ) wy = float(y)/c_height;
    else wy = float(y-c_height)/c_height;

    float g = exp( -cte * (wx*wx + wy*wy)/2.f );

    float2 val = img[y*c_fft_width+x];

    val.x *= g;
    val.y *= g;

    img[y*c_fft_width+x] = val;

}

template< int radius >
void apply_gauss_hat( float2 *img_spectrum, const dim3& dim_grid, const dim3& dim_block ) {

    float sigma = (float)radius/2;
    float cte = sigma*2*M_PI;
    cte *= cte;

    apply_gauss_hat_kernel<<< dim_grid, dim_block >>>( img_spectrum, cte );

}




///---------------- MAIN ----------------
int main( int argc, char** argv ) {

    int width = 4096, height = 2048;

    if( argc > 2 && ( sscanf( argv[1], "%d", &width ) != 1 || sscanf( argv[2], "%d", &height ) != 1 || width < 1 || height < 1 ) ) { fprintf( stderr, "[error] Invalid argument: %s %s\n[usage] %s [width=%d] [height=%d] [output=0-no|1-perf|2-err]\n", argv[1], argv[2], argv[0], width, height ); return 1; }
    else if( argc == 2 || argc > 4 ) { fprintf( stderr, "[usage] %s [width=%d] [height=%d] [output=0-no|1-perf|2-err]\n", argv[0], width, height ); return 1; }

    res_out = ( argc == 4 ) ? res_out = argv[3][0]-48 : DEFAULT_RES_OUT;

    if( res_out != 0 && res_out != 1 && res_out != 2 ) { fprintf( stderr, "[error] Invalid output type %d\n\tValids are: 0 - no output; 1 - performance; 2 - error\n", res_out ); return 1; }

    int ne = width * height; // number of elements
    float dt = 0.f;

    std::vector< float > kernel;
    int radius = KERNEL_RADIUS;
    compute_gauss( kernel, radius );

    std::vector< float > h_in_img;

#ifdef USE_CACTUS
    int maxval;

    if( !res_out ) {
        printf( "[gauss] Reading %s ... ", CACTUS_IN ); fflush(stdin);
        if( load_pgm( h_in_img, width, height, maxval, CACTUS_IN ) ) printf( "done! (image: %d x %d)\n", width, height );
        else { fprintf( stderr, "!!! error!" ); return 1; }
        ne = width * height;
    } else {
        h_in_img.resize( ne );
        maxval = 255;
        srand( 1234 );
        for (int i = 0; i < ne; ++i)
            h_in_img[i] = rand() / (double)RAND_MAX;
    }

#else

    if( !res_out ) { printf( "[gauss] Generating random image (%dx%d) ... ", width, height ); fflush(stdin); }

    h_in_img.resize( ne );

    srand( 1234 );
    for (int i = 0; i < ne; ++i)
        h_in_img[i] = rand() / (double)RAND_MAX;

    if( !res_out ) printf( "done!\n" );

#endif

    if( !res_out ) { printf( "[gauss] Allocating memory in CPU ... " ); fflush(stdin); }

    std::vector< float > h_ref_img( ne ), h_out_img( ne ), h_fft_img( ne );

    if( !res_out ) printf( "done!\n");

#ifdef COMPUTE_IN_CPU

    if( !res_out ) { printf( "[gauss] Computing in CPU ... " ); fflush(stdin); }

    Chronos te; // time elapsed computation

    te.reset();
    gauss_cpu( &h_ref_img[0], &h_in_img[0], width, height, kernel, radius );
    dt = te.elapsed();

    if( !res_out ) printf( "done!\n[CPU] reference done in %gs @ %g MiP/s\n", dt, ne/(1024.*1024.*dt) );

#endif

#ifdef USE_CACTUS

    if( !res_out ) {
        printf( "[gauss] Writing %s ... ", CACTUS_CPU ); fflush(stdin);
        if( save_pgm( h_ref_img, width, height, maxval, CACTUS_CPU ) ) printf( "done!\n" );
        else { fprintf( stderr, "!!! error!" ); return 1; }
    }

#endif

    if( !res_out ) { printf( "[gauss] Allocating memory in GPU ... " ); fflush(stdin); }

    cudaChannelFormatDesc ccd = cudaCreateChannelDesc<float>(); // cuda channel descriptor for texture
    cudaArray *a_in_img; cudaMallocArray( &a_in_img, &ccd, width, height );

    cudaArray *a_fin_img; cudaMallocArray( &a_fin_img, &ccd, width, height, cudaArraySurfaceLoadStore ); // array for surface in
    cudaArray *a_out_img; cudaMallocArray( &a_out_img, &ccd, width, height, cudaArraySurfaceLoadStore ); // array for surface out

    float *d_in_img = 0; cudaMalloc( (void**)&d_in_img, sizeof(float)*ne );
    float *d_out_img = 0; cudaMalloc( (void**)&d_out_img, sizeof(float)*ne );
    int fft_width = width/2+1;
    Vector< float, KERNEL_RADIUS*2+1 > kernel_gpu;
    for (int i=0; i<KERNEL_RADIUS*2+1; ++i)
        kernel_gpu[i] = kernel[i];

    copy_to_symbol(c_width, width);
    copy_to_symbol(c_height, height);
    copy_to_symbol(c_kernel, kernel_gpu);
    copy_to_symbol(c_fft_width, fft_width);

    t_in_img.addressMode[0] = cudaAddressModeClamp;
    t_in_img.addressMode[1] = cudaAddressModeClamp;
    t_in_img.filterMode = cudaFilterModePoint;
    t_in_img.normalized = false;

    if( !a_in_img || !a_fin_img || !a_out_img || !d_in_img || !d_out_img ) { fprintf( stderr, "!!! error!\n" ); return 1; }

    if( !res_out ) { printf( "done!\n[gauss] Computing in GPU ...\n" ); }

    if( !res_out ) { printf( "[gauss] Info: r = %d ; b = %dx%d\n", radius, WS, NW ); fflush(stdin); }

    //if( res_out ) { printf( "%d %d", width, height ); fflush(stdin); }

    float me = 0.f, mre = 0.f; // maximum error and maximum relative error
    //int w_ws = (width+WS-1)/WS, h_nw = (height+NW-1)/NW, w_hws = (width+HWS-1)/HWS, w_mnt = (width+MNT-1)/MNT;

    cudaEvent_t start_device, stop_device;
    cudaEventCreate(&start_device); cudaEventCreate(&stop_device);

    //cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
#ifdef RUN_GAUSS_0
    { // gauss-0
        cudaMemcpy( d_in_img, &h_in_img[0], sizeof(float)*ne, cudaMemcpyHostToDevice );
        cudaEventRecord( start_device, 0 );

        for (int i = 0; i <= res_out*(RUN_TIMES-1); ++i) {

            gauss0_rows< KERNEL_RADIUS >
                <<< dim3((width+WS-1)/WS, (height+NW-1)/NW), dim3(WS, NW) >>>(
                    d_out_img, d_in_img );

            gauss0_cols< KERNEL_RADIUS >
                <<< dim3((width+WS-1)/WS, (height+NW-1)/NW), dim3(WS, NW) >>>(
                    d_in_img, d_out_img );

        }

        cudaEventRecord( stop_device, 0 );
        cudaEventSynchronize( stop_device );

        dt = 0.f; cudaEventElapsedTime(&dt, start_device, stop_device); dt /= 1000.f;
        if( res_out == 1 ) dt /= float(RUN_TIMES);

        cudaMemcpy( &h_out_img[0], d_in_img, sizeof(float)*ne, cudaMemcpyDeviceToHost );

#ifdef COMPUTE_IN_CPU
        check_cpu_reference(&h_ref_img[0], &h_out_img[0], ne, me, mre);
#endif
        if( res_out == 1 )
            printf( " %g", ne/(1024.*1024.*dt) );
        else if( res_out == 2 )
            printf( " %g %g", me, mre );
        else
            printf( "[GPU] gauss0 done with max_err = %g in %gs @ %g MiP/s\n", me, dt, ne/(1024.*1024.*dt) );
    }
#endif
#ifdef RUN_GAUSS_1
    { // gauss-1
        cudaMemcpyToArray( a_in_img, 0, 0, &h_in_img[0], sizeof(float)*ne, cudaMemcpyHostToDevice );
        cudaEventRecord( start_device, 0 );
        cudaBindSurfaceToArray( f_out_img, a_out_img );

        for (int i = 0; i <= res_out*(RUN_TIMES-1); ++i) {

            cudaBindTextureToArray( t_in_img, a_in_img );

            gauss1_rows< KERNEL_RADIUS >
                <<< dim3((width+WS1-1)/WS1, (height+NW1-1)/NW1), dim3(WS1, NW1) >>>();

            // fortunately output surface can be used as input texture afterwards
            cudaBindTextureToArray( t_in_img, a_out_img );

            gauss1_cols< KERNEL_RADIUS >
                <<< dim3((width+WS1-1)/WS1, (height+NW1-1)/NW1), dim3(WS1, NW1) >>>(
                    d_out_img );

        }

        cudaEventRecord( stop_device, 0 );
        cudaEventSynchronize( stop_device );
        cudaUnbindTexture( t_in_img );

        dt = 0.f; cudaEventElapsedTime(&dt, start_device, stop_device); dt /= 1000.f;
        if( res_out == 1 ) dt /= float(RUN_TIMES);

        cudaMemcpy( &h_out_img[0], d_out_img, sizeof(float)*ne, cudaMemcpyDeviceToHost );

#ifdef COMPUTE_IN_CPU
        check_cpu_reference(&h_ref_img[0], &h_out_img[0], ne, me, mre);
#endif

        if( res_out == 1 )
            printf( " %g", ne/(1024.*1024.*dt) );
        else if( res_out == 2 )
            printf( " %g %g", me, mre );
        else
            printf( "[GPU] gauss1 done with max_err = %g in %gs @ %g MiP/s\n", me, dt, ne/(1024.*1024.*dt) );
    }
#endif
    cudaDeviceSetCacheConfig( cudaFuncCachePreferShared );
#ifdef RUN_GAUSS_2
    { // gauss-2
        cudaMemcpyToArray( a_in_img, 0, 0, &h_in_img[0], sizeof(float)*ne, cudaMemcpyHostToDevice );
        cudaEventRecord( start_device, 0 );
        cudaBindSurfaceToArray( f_out_img, a_out_img );

        for (int i = 0; i <= res_out*(RUN_TIMES-1); ++i) {

            cudaBindTextureToArray( t_in_img, a_in_img );

            gauss2_rows< KERNEL_RADIUS >
                <<< dim3((width+MNT-1)/MNT, height), dim3(MNT, 1) >>>();

            cudaBindTextureToArray( t_in_img, a_out_img );

            gauss2_cols< KERNEL_RADIUS >
                <<< dim3((width+WS2-1)/WS2, (height+NW2-1)/NW2), dim3(WS2, NW2) >>>(
                    d_out_img );

        }

        cudaEventRecord( stop_device, 0 );
        cudaEventSynchronize( stop_device );
        cudaUnbindTexture( t_in_img );

        dt = 0.f; cudaEventElapsedTime(&dt, start_device, stop_device); dt /= 1000.f;
        if( res_out == 1 ) dt /= float(RUN_TIMES);

        cudaMemcpy( &h_out_img[0], d_out_img, sizeof(float)*ne, cudaMemcpyDeviceToHost );

#ifdef COMPUTE_IN_CPU
        check_cpu_reference(&h_ref_img[0], &h_out_img[0], ne, me, mre);
#endif

        if( res_out == 1 )
            printf( " %g", ne/(1024.*1024.*dt) );
        else if( res_out == 2 )
            printf( " %g %g", me, mre );
        else
            printf( "[GPU] gauss2 done with max_err = %g in %gs @ %g MiP/s\n", me, dt, ne/(1024.*1024.*dt) );

#ifdef USE_CACTUS
        if( !res_out ) {
            printf( "[gauss] Writing %s ... ", CACTUS_GPU ); fflush(stdin);
            if( save_pgm( h_out_img, width, height, maxval, CACTUS_GPU ) ) printf( "done!\n" );
            else { fprintf( stderr, "!!! error!" ); return 1; }
        }
#endif
    }
#endif
#ifdef RUN_GAUSS_3
    { // gauss-3
        cudaMemcpyToArray( a_in_img, 0, 0, &h_in_img[0], sizeof(float)*ne, cudaMemcpyHostToDevice );
        cudaEventRecord( start_device, 0 );
        cudaBindTextureToArray( t_in_img, a_in_img );

        for (int i = 0; i <= res_out*(RUN_TIMES-1); ++i) {

            gauss3< KERNEL_RADIUS >
                <<< dim3((width+WS3-1)/WS3, (height+NW3-1)/NW3), dim3(WS3, NW3) >>>(
                    d_out_img );
        }

        cudaEventRecord( stop_device, 0 );
        cudaEventSynchronize( stop_device );
        cudaUnbindTexture( t_in_img );

        dt = 0.f; cudaEventElapsedTime(&dt, start_device, stop_device); dt /= 1000.f;
        if( res_out == 1 ) dt /= float(RUN_TIMES);

        cudaMemcpy( &h_out_img[0], d_out_img, sizeof(float)*ne, cudaMemcpyDeviceToHost );

#ifdef COMPUTE_IN_CPU
        check_cpu_reference(&h_ref_img[0], &h_out_img[0], ne, me, mre);
#endif

        if( res_out == 1 )
            printf( " %g", ne/(1024.*1024.*dt) );
        else if( res_out == 2 )
            printf( " %g %g", me, mre );
        else
            printf( "[GPU] gauss3 done with max_err = %g in %gs @ %g MiP/s\n", me, dt, ne/(1024.*1024.*dt) );
    }
#endif
#ifdef RUN_GAUSS_FFT
    { // gauss-fft
        dvector< float > d_img( h_in_img );
        int fftW = width/2+1, fftH = height;
        dvector< float2 > d_img_hat( fftW * fftH );
        dim3 dim_block( WS, NW ), dim_grid( (fftW+1)/dim_block.x, (fftH+1)/dim_block.y );

        cufftHandle planFwd, planInv;
        cufftPlan2d(&planFwd, width, height, CUFFT_R2C);
        cufftPlan2d(&planInv, width, height, CUFFT_C2R);

        cudaEventRecord( start_device, 0 );

        for (int i = 0; i <= res_out*(RUN_TIMES-1); ++i) {

            cufftExecR2C(planFwd, d_img, (cufftComplex *)(float2 *)d_img_hat);

            apply_gauss_hat< KERNEL_RADIUS >(d_img_hat, dim_grid, dim_block );

            cufftExecC2R(planInv, (cufftComplex *)(float2 *)d_img_hat, d_img);

        }

        cudaEventRecord( stop_device, 0 );
        cudaEventSynchronize( stop_device );
        dt = 0.f; cudaEventElapsedTime(&dt, start_device, stop_device); dt /= 1000.f;
        if( res_out == 1 ) dt /= float(RUN_TIMES);

        cufftDestroy(planFwd);
        cufftDestroy(planInv);

        h_fft_img = to_cpu(d_img);

        for (int i=0; i<ne; ++i) h_fft_img[i] /= (float)ne;

        // gauss-fft calculates the exact convolution, so it might be different
        // from the cpu reference
#ifdef COMPUTE_IN_CPU
        check_cpu_reference(&h_ref_img[0], &h_out_img[0], ne, me, mre);
#endif

        if( res_out == 1 )
            printf( " %g", ne/(1024.*1024.*dt) );
        else if( res_out == 2 )
            printf( " %g %g", me, mre );
        else
            printf( "[GPU] gauss-fft done with max_err = %g in %gs @ %g MiP/s\n", me, dt, ne/(1024.*1024.*dt) );
#ifdef USE_CACTUS
        if( !res_out ) {
            printf( "[gauss] Writing %s ... ", CACTUS_FFT_GPU ); fflush(stdin);
            if( save_pgm( h_fft_img, width, height, maxval, CACTUS_FFT_GPU ) ) printf( "done!\n" );
            else { fprintf( stderr, "!!! error!" ); return 1; }
        }
#endif
    }
#endif
#ifdef RUN_GAUSS_REC
    { // gauss-recursive
        void rec_gauss(float *h_img, int width, int height, float sigma);

        std::vector<float> h_img = h_in_img;

        rec_gauss(&h_img[0], width, height, radius/2);

#ifdef USE_CACTUS
        if( !res_out ) {
            printf( "[gauss] Writing %s ... ", CACTUS_RECURSIVE_GPU ); fflush(stdin);
            if( save_pgm(h_img, width, height, maxval, CACTUS_RECURSIVE_GPU) ) printf( "done!\n" );
            else { fprintf( stderr, "!!! error!" ); return 1; }
        }
#endif
    }
#endif
    //if( res_out ) printf( "\n" );

    cudaEventDestroy(start_device); cudaEventDestroy(stop_device);

    cudaFreeArray( a_in_img );
    cudaFreeArray( a_fin_img );
    cudaFreeArray( a_out_img );

    cudaFree( d_in_img );
    cudaFree( d_out_img );

    return 0;

}
