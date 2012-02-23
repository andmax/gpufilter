/**
 *  GPU definitions (header)
 *  @author Andre Maximo
 *  @author Rodolfo Lima
 *  @date March, 2011
 */

#ifndef GPUDEFS_H
#define GPUDEFS_H

//== INCLUDES =================================================================

#include <cuda_runtime.h>

//== DEFINITIONS ===============================================================

#define WS 32 // Warp size (defines b x b block size where b = WS)
#define HWS 16 // Half Warp Size
#define DW 8 // Default number of warps (computational block height)
#define CHW 7 // Carry-heavy number of warps (computational block height for some kernels)
#define OW 6 // Optimized number of warps (computational block height for some kernels)
#define DNB 6 // Default number of blocks per SM (minimum blocks per SM launch bounds)
#define ONB 5 // Optimized number of blocks per SM (minimum blocks per SM for some kernels)
#define MTS 192 // Maximum number of threads per block with 8 blocks per SM
#define MBO 8 // Maximum number of blocks per SM using optimize or maximum warps
#define CHB 7 // Carry-heavy number of blocks per SM using default number of warps
#define MW 6 // Maximum number of warps per block with 8 blocks per SM (with all warps computing)
#define SOW 5 // Dual-scheduler optimized number of warps per block (with 8 blocks per SM and to use the dual scheduler with 1 computing warp)
#define MBH 3 // Maximum number of blocks per SM using half-warp size

//== NAMESPACES ===============================================================

namespace gpufilter {

//== PROTOTYPES ===============================================================

/**
 *  @ingroup api_gpu
 *  @brief Calculate image borders
 *
 *  Given the image size and an extension to consider calculates the
 *  four-directional border pixels (left, top, right and bottom)
 *  needed by algorithm 4 and 5.
 *
 *  @param[out] left Extension (in pixels) in the left border of the image
 *  @param[out] top Extension (in pixels) in the top border of the image
 *  @param[out] right Extension (in pixels) in the right border of the image
 *  @param[out] bottom Extension (in pixels) in the bottom border of the image
 *  @param[in] h Height of the image
 *  @param[in] w Width of the image
 *  @param[in] extb Extension (in blocks) to consider outside image (default 0)
 */
extern
void calc_borders( int& left,
                   int& top,
                   int& right,
                   int& bottom,
                   const int& h,
                   const int& w,
                   const int& extb );

/**
 *  @ingroup api_gpu
 *  @brief Upload device constants sizes
 *
 *  Given the dimensions of the 2D work image, upload to the device
 *  constant memory size-related values.  It returns the computational
 *  grid domain.
 *
 *  @param[out] g Computation grid dimension considering a b x b block size where b = 32
 *  @param[in] h Height of the work image
 *  @param[in] w Width of the work image
 */
extern
void up_constants_sizes( dim3& g,
                         const int& h,
                         const int& w );


/**
 *  @ingroup api_gpu
 *  @overload void up_constants_sizes( dim3& g, int& ext_h, int& ext_w, const int& h, const int& w, const int& extb )
 *  @brief Upload device constants sizes
 *
 *  Given the dimensions of the 2D work image, upload to the device
 *  constant memory size-related values.  The work image is the
 *  original image plus extension blocks to compute filtering
 *  out-of-bounds.  It returns the extended image size and
 *  computational grid domain.
 *
 *  @param[out] g Computation grid dimension considering a b x b block size where b = 32
 *  @param[out] ext_h Height of the extended image
 *  @param[out] ext_w Width of the extended image
 *  @param[in] h Height of the work image
 *  @param[in] w Width of the work image
 *  @param[in] extb Extension (in blocks) to consider outside image (default 0)
 */
extern
void up_constants_sizes( dim3& g,
                         int& ext_h,
                         int& ext_w,
                         const int& h,
                         const int& w,
                         const int& extb = 0 );

/**
 *  @ingroup api_gpu
 *  @brief Upload device constants first-order coefficients
 *
 *  Given the first-order coefficients of the recursive filter, upload
 *  to the device constant memory the coefficients-related values.
 *
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 */
extern
void up_constants_coefficients1( const float& b0,
                                 const float& a1 );

/**
 *  @ingroup api_gpu
 *  @brief Upload device constants second-order coefficients
 *
 *  Given the second-order coefficients of the recursive filter,
 *  upload to the device constant memory the coefficients-related
 *  values.
 *
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @param[in] a2 Feedback second-order coefficient
 */
extern
void up_constants_coefficients2( const float& b0,
                                 const float& a1,
                                 const float& a2 );

//=============================================================================
} // namespace gpufilter
//=============================================================================
#endif // GPUDEFS_H
//=============================================================================
