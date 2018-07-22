/**
 *  @file gpud_aux.cuh
 *  @brief Auxiliary GPU definitions (Deriche) and implementations
 *  @author Andre Maximo
 *  @date Jun, 2014
 */

#ifndef GPUD_AUX_CUH
#define GPUD_AUX_CUH

//=== GLOBAL-SCOPE DEFINITIONS ==================================================

#define NBAC 5 // # of blocks to adjust carriers for cols (smem=2x bxb)

// basics
__constant__ Vector<float, 6> c_coefs;

// alg4d
__constant__ Matrix<float,R,R> c_TAFB_AC1P_T, c_HARB_AC2E_T;

// alg5d
__constant__ Matrix<float,R,WS> c_AFP_T;
__constant__ Matrix<float,R,WS> c_TAFB_AC1B, c_HARB_AC2B,
    c_AC1P_AFB_T, c_AC2E_ARB_T;

#endif // GPUD_AUX_CUH
