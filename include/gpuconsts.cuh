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
// parameter in kernels; f_ surface.

__constant__
int c_width, c_height, c_m_size, c_n_size;

__constant__
float c_Linf1, c_Svm, c_Stm, c_Alpha, c_iR1;

__constant__
float c_Delta_x_tail[WS], c_Delta_y[WS], c_SignRevProdLinf[WS], c_ProdLinf[WS];

__constant__
float c_Linf2, c_Llast2, c_iR2, c_Minf, c_Ninf;

__constant__
float c_Af[2][2], c_Ar[2][2], c_Arf[2][2];

//=============================================================================
#endif // GPUCONSTS_CUH
//=============================================================================
