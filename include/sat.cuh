/**
 *  @file sat.cuh
 *  @brief CUDA device code for GPU-Efficient Summed-Area Tables
 *  @author Andre Maximo
 *  @date September, 2011
 */

#ifndef SAT_CUH
#define SAT_CUH

//== NAMESPACES ===============================================================

namespace gpufilter {

//== EXTERNS ==================================================================

/**
 *  @ingroup gpu
 *  @{
 */

__global__
void algorithmSAT_stage1( const float *g_in,
                          float *g_xbar,
                          float *g_yhat );

__global__
void algorithmSAT_stage2( float *g_xbar,
                          float *g_xsum );

__global__
void algorithmSAT_stage3( const float *g_xsum,
                          float *g_yhat );

__global__
void algorithmSAT_stage4( float *g_inout,
                          const float *g_xbar,
                          const float *g_ybar );

/**
 *  @}
 */

//=============================================================================
} // namespace gpufilter
//=============================================================================
#endif // SAT_CUH
//=============================================================================
