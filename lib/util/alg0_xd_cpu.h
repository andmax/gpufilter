/**
 *  @file alg0_xd_cpu.h
 *  @brief Algorithm 0 in the CPU for 1D, 2D or 3D
 *  @author Andre Maximo
 *  @date Jun, 2019
 *  @copyright The MIT License
 */

#ifndef ALG0_XD_CPU_H
#define ALG0_XD_CPU_H

//== INCLUDES =================================================================

#include "recfilter.h"

// Disclaimer: this is optimized for usability, not speed

//== NAMESPACES ================================================================

namespace gpufilter {

//== IMPLEMENTATION ============================================================

template <int AXIS, bool FWD, int R, typename T1, typename T2>
void recursive_3d( T1 *inout,
                   const T2& w, const T2& h, const T2& d,
                   const Vector<T1,R+1> &weights ) {

    if (AXIS == 0) {
        for (T2 z = 0; z < d; ++z) {
            for (T2 y = 0; y < h; ++y) {
                Vector<T1,R> c = zeros<T1,R>();
                if (FWD) {
                    for (T2 x = 0; x < w; ++x) {
                        inout[z*h*w+y*w+x] = fwd(
                            c, inout[z*h*w+y*w+x]*weights[0], weights);
                    }
                } else {
                    for (T2 x = w-1; x >= 0; --x) {
                        inout[z*h*w+y*w+x] = rev(
                            inout[z*h*w+y*w+x]*weights[0], c, weights);
                    }
                }
            }
        }
    } else if (AXIS == 1) {
        for (T2 z = 0; z < d; ++z) {
            for (T2 x = 0; x < w; ++x) {
                Vector<T1,R> c = zeros<T1,R>();
                if (FWD) {
                    for (T2 y = 0; y < h; ++y) {
                        inout[z*h*w+y*w+x] = fwd(
                            c, inout[z*h*w+y*w+x]*weights[0], weights);
                    }
                } else {
                    for (T2 y = h-1; y >= 0; --y) {
                        inout[z*h*w+y*w+x] = rev(
                            inout[z*h*w+y*w+x]*weights[0], c, weights);
                    }
                }
            }
        }
    } else if (AXIS == 2) {
        for (T2 x = 0; x < w; ++x) {
            for (T2 y = 0; y < h; ++y) {
                Vector<T1,R> c = zeros<T1,R>();
                if (FWD) {
                    for (T2 z = 0; z < d; ++z) {
                        inout[z*h*w+y*w+x] = fwd(
                            c, inout[z*h*w+y*w+x]*weights[0], weights);
                    }
                } else {
                    for (T2 z = d-1; z >= 0; --z) {
                        inout[z*h*w+y*w+x] = rev(
                            inout[z*h*w+y*w+x]*weights[0], c, weights);
                    }
                }
            }
        }
    }

}

template <int AXIS, bool FWD, int R, typename T1, typename T2>
void recursive_2d( T1 *inout,
                   const T2& h, const T2& w,
                   const Vector<T1,R+1> &weights ) {
    
    recursive_3d<AXIS, FWD, R, T1, T2>(inout, w, h, 1, weights);

}

template <int AXIS, bool FWD, int R, typename T1, typename T2>
void recursive_1d( T1 *inout,
                   const T2& n,
                   const Vector<T1,R+1> &weights ) {
    
    recursive_3d<AXIS, FWD, R, T1, T2>(inout, n, 1, 1, weights);

}

//==============================================================================
} // namespace gpufilter
//==============================================================================
#endif // ALG0_XD_CPU_H
//==============================================================================
