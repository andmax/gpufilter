/**
 *  GPU constants
 *  @author Andre Maximo
 *  @author Rodolfo Lima
 *  @date March, 2011
 */

#ifndef GPUCONSTS_CUH
#define GPUCONSTS_CUH

//== DEFINITIONS ===============================================================

// Naming conventions are: c_ constant; t_ texture; g_ global memory;
// s_ shared memory; d_ device pointer; a_ cuda-array; p_ template
// parameter in kernels; f_ surface; h_ host pointer.

__constant__ int c_width, c_height, c_m_size, c_n_size,
    c_last_m, c_last_n, c_border, c_carry_width, c_carry_height;

__constant__ float c_inv_width, c_inv_height;

__constant__ float c_b0, c_a1, c_inv_b0, c_AbF, c_AbR, c_HARB_AFP,
    c_TAFB[WS], c_HARB_AFB[WS], c_ARE[WS], c_ARB_AFP_T[WS];

__constant__ int c_transp_out_height;
__constant__ float c_a2;
__constant__ float c_AbF2[2][2], c_AbR2[2][2], c_AFP_HARB[2][2];

texture< float, cudaTextureType2D, cudaReadModeElementType > t_in;

//=============================================================================
#endif // GPUCONSTS_CUH
//=============================================================================
