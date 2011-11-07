/**
 *  GPU definitions
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

__constant__ int c_width, c_height, c_m_size, c_n_size;

__constant__ float c_Linf1, c_Svm, c_Stm, c_Alpha, c_iR1;

__constant__ float c_Delta_x_tail[WS], c_Delta_y[WS],
    c_SignRevProdLinf[WS], c_ProdLinf[WS];
    
__constant__ float c_Linf2, c_Llast2, c_iR2, c_Minf, c_Ninf;
__constant__ float c_Af[2][2], c_Ar[2][2], c_Arf[2][2];

//=============================================================================
#endif // GPUDEFS_CUH
//=============================================================================
