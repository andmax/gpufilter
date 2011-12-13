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

__constant__
int c_width, c_height, c_img_pitch, c_m_size, c_n_size;

__constant__
float c_tex_width, c_tex_height;

__constant__
float c_b0, c_a1, c_1_b0, c_AbF, c_AbR, c_HARB_AFP, c_HARB_AFP_T,
    c_TAFB[WS], c_HARB_AFB[WS], c_ARE_T[WS], c_ARB_AFP_T[WS];

__constant__
float c_Linf2, c_Llast2, c_iR2, c_Minf, c_Ninf;

__constant__
float c_Af[2][2], c_Ar[2][2], c_Arf[2][2];

texture< float, cudaTextureType2D, cudaReadModeElementType > t_in;

//=============================================================================
#endif // GPUCONSTS_CUH
//=============================================================================
