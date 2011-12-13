/**
 *  @file gpufilter.cu
 *  @brief CUDA device code for GPU-Efficient Recursive Filtering Algorithms
 *  @author Diego Nehab
 *  @author Andre Maximo
 *  @date September, 2011
 */

//== INCLUDES =================================================================

#include <cmath>
#include <cstdio>
#include <cfloat>
#include <cassert>
#include <iostream>
#include <algorithm>

#include "sat.cu"
#include "alg4.cu"
#include "alg5.cu"

//== NAMESPACES ===============================================================

namespace gpufilter {

//== IMPLEMENTATION ===========================================================

//-- Host ---------------------------------------------------------------------

__host__
void gaussian_gpu( float **inout,
                   const int& h,
                   const int& w,
                   const int& d,
                   const float& s,
                   const initcond& ic,
                   const int& extb ) {
    float b10, a11, b20, a21, a22;
    weights1( s, b10, a11 );
    weights2( s, b20, a21, a22 );
    for (int c = 0; c < d; c++) {
        alg5( inout[c], h, w, b10, a11 );
        alg4( inout[c], h, w, b20, a21, a22 );
    }
}

__host__
void gaussian_gpu( float *inout,
                   const int& h,
                   const int& w,
                   const float& s,
                   const initcond& ic,
                   const int& extb ) {
    float b10, a11, b20, a21, a22;
    weights1( s, b10, a11 );
    weights2( s, b20, a21, a22 );
    alg5( inout, h, w, b10, a11 );
    alg4( inout, h, w, b20, a21, a22 );
}

__host__
void bspline3i_gpu( float **inout,
                    const int& h,
                    const int& w,
                    const int& d,
                    const initcond& ic,
                    const int& extb ) {
    const float alpha = 2.f - sqrt(3.f);
    for (int c = 0; c < d; c++) {
        alg5( inout[c], h, w, 1.f+alpha, alpha, ic, extb );
    }
}

__host__
void bspline3i_gpu( float *inout,
                    const int& h,
                    const int& w,
                    const initcond& ic,
                    const int& extb ) {
    const float alpha = 2.f - sqrt(3.f);
    alg5( inout, h, w, 1.f+alpha, alpha, ic, extb );
}

//=============================================================================
} // namespace gpufilter
//=============================================================================
