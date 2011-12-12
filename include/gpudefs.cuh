/**
 *  GPU definitions (header)
 *  @author Andre Maximo
 *  @author Rodolfo Lima
 *  @date March, 2011
 */

#ifndef GPUDEFS_CUH
#define GPUDEFS_CUH

//== DEFINITIONS ===============================================================

#define WS 32 // Warp size (defines b x b block size where b = WS)
#define HWS 16 // Half Warp Size
#define DW 8 // Default number of warps (computational block height)
#define OW 6 // Optimized number of warps (computational block height for some kernels)
#define DNB 6 // Default number of blocks per SM (minimum blocks per SM launch bounds)
#define ONB 5 // Optimized number of blocks per SM (minimum blocks per SM for some kernels)
#define MTS 192 // Maximum number of threads per block with 8 blocks per SM
#define MBO 8 // Maximum number of blocks per SM using optimize or maximum warps
#define MW 6 // Maximum number of warps per block with 8 blocks per SM (with all warps computing)
#define SOW 5 // Dual-scheduler optimized number of warps per block (with 8 blocks per SM and to use the dual scheduler with 1 computing warp)
#define MBH 3 // Maximum number of blocks per SM using half-warp size

//== NAMESPACES ===============================================================

namespace gpufilter {

//== PROTOTYPES ===============================================================

/**
 *  @ingroup api_gpu
 *  @brief Upload device constants sizes
 *
 *  Given the dimensions of the 2D input image, upload in device
 *  constant memory all values related to size.
 *
 *  @param[out] g Computation grid dimension considering a b x b block size where b = 32
 *  @param[in] h Height of the input image
 *  @param[in] w Width of the input image
 */
void up_constants_sizes( dim3& g,
                         const int& h,
                         const int& w );

/**
 *  @ingroup api_gpu
 *  @brief Upload device constants first-order coefficients
 *
 *  Given the first-order coefficients of the recursive filter, upload
 *  in device constant memory all values related to the coefficients.
 *
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 */
void up_constants_coefficients1( const float& b0,
                                 const float& a1 );

/**
 *  @ingroup api_gpu
 *  @brief Upload device constants second-order coefficients
 *
 *  Given the second-order coefficients of the recursive filter,
 *  upload in device constant memory all values related to the
 *  coefficients.
 *
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @param[in] a2 Feedback second-order coefficient
 */
void up_constants_coefficients2( const float& b0,
                                 const float& a1,
                                 const float& a2 );

//=============================================================================
} // namespace gpufilter
//=============================================================================
#endif // GPUDEFS_CUH
//=============================================================================
