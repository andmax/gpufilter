/**
 *  GPU constants
 *  @author Andre Maximo
 *  @author Rodolfo Lima
 *  @date March, 2011
 */

#ifndef GPUCONSTS_CUH
#define GPUCONSTS_CUH

//== DEFINITIONS ===============================================================

__constant__ int c_width, c_height, c_m_size, c_n_size, c_last_m, c_last_n,
    c_border, c_carry_width, c_carry_height;

__constant__ float c_inv_width, c_inv_height, c_b0, c_a1, c_a2, c_inv_b0,
    c_AbF, c_AbR, c_HARB_AFP, c_TAFB[WS], c_HARB_AFB[WS], c_ARE[WS],
    c_ARB_AFP_T[WS], c_AbF2[2][2], c_AbR2[2][2], c_AFP_HARB[2][2];

//=============================================================================
#endif // GPUCONSTS_CUH
//=============================================================================
