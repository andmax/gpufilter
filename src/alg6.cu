/**
 *  @file alg6.cu
 *  @brief Algorithm 6 in the GPU
 *  @author Andre Maximo
 *  @date Dec, 2012
 *  @copyright The MIT License
 */

#ifndef ORDER
#define ORDER 1 // default filter order r=1
#endif
#define APPNAME "[alg6_" << ORDER << "]"

//== INCLUDES ==================================================================

#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <complex>
#include <iomanip>

#include <util/error.h>
#include <util/symbol.h>
#include <util/dvector.h>
#include <util/timer.h>
#include <util/recfilter.h>
#include <util/image.h>
#include <util/gaussian.h>

#include "cpudefs.h"
#include "gpudefs.h"
#include "alg0_cpu.h"
#include "alg6_gpu.cuh"
#include "alg6_clamp.cuh"
#include "alg6_repeat.cuh"
#include "alg6_reflect.cuh"

//== IMPLEMENTATION ============================================================

/**
 *  @ingroup api_gpu
 *  @brief Compute Algorithm 6 (order agnostic) with any boundary condition
 *
 *  This function computes R-order recursive filtering with given
 *  weights of an input 2D image using algorithm \f$6_r\f$ and any
 *  given boundary condition.  That is, it can compute using border
 *  input padding \f$6^b_r\f$, constant padding extension (or clamp to
 *  border) \f$6^c_r\f$, periodic extension (a.k.a. repeat)
 *  \f$6^p_r\f$ or even-periodic extension (a.k.a. mirror or reflect)
 *  \f$6^e_r\f$.
 *
 *  All flavors of the algorithm 6 are discussed in depth in our paper:
 *
 *  @verbatim
@inproceedings{NehabMaximo:2016,
  title = {{P}arallel {R}ecursive {F}iltering of {I}nfinite {I}nput {E}xtesions},
  author = {{N}ehab, {D}. and {M}aximo, {A}.},
  journal = {{ACM} {T}ransactions on {G}raphics ({P}roceedings of the {ACM} {SIGGRAPH} {A}sia 2016)},
  year = {2016},
  volume = {},
  number = {},
  doi = {},
  publisher = {ACM}
}   @endverbatim
 *
 *  @param[in,out] h_img The in(out)put 2D image to filter in host memory
 *  @param[in] width Image width
 *  @param[in] height Image height
 *  @param[in] runtimes Number of run times (1 for debug and 1000 for performance measurements)
 *  @param[in] w Filter weights (feedforward and feedback coefficients)
 *  @param[in] border Number of border blocks (32x32) outside image
 *  @param[in] btype Border type (either zero, clamp, repeat or reflect)
 *  @tparam R Filter order
 */
template<int R>
void alg6( float *h_img,
           const int& width, const int& height, const int& runtimes,
           const gpufilter::Vector<float, R+1>& w,
           const int& border,
           const gpufilter::BorderType& btype ) {

    if (border == 0) {
        if (btype == gpufilter::CLAMP_TO_ZERO)
            gpufilter::alg6_gpu<false, R>(h_img, width, height, runtimes, w);
        else if (btype == gpufilter::CLAMP_TO_EDGE)
            gpufilter::alg6_clamp<R>(h_img, width, height, runtimes, w);
        else if (btype == gpufilter::REPEAT)
            gpufilter::alg6_repeat<R>(h_img, width, height, runtimes, w);
        else if (btype == gpufilter::REFLECT)
            gpufilter::alg6_reflect<R>(h_img, width, height, runtimes, w);
    } else if (border > 0) {
        gpufilter::alg6_gpu<true, R>(h_img, width, height, runtimes, w,
                                     border, btype);
    }

}

// Main ------------------------------------------------------------------------

int main( int argc, char** argv ) {

    int width, height, runtimes, border, a0border;
    gpufilter::BorderType btype;
    std::vector<float> cpu_img, gpu_img;
    gpufilter::Vector<float, ORDER+1> w;
    float me, mre;

    initial_setup(width, height, runtimes, btype, border,
                  cpu_img, gpu_img, w, a0border, me, mre,
                  argc, argv);

    if (runtimes == 1) // running for debugging
        print_info(width, height, btype, border, a0border, w);

    gpufilter::alg0_cpu<ORDER>(&cpu_img[0], width, height, w, a0border, btype);

    alg6<ORDER>(&gpu_img[0], width, height, runtimes, w, border, btype);

    gpufilter::check_cpu_reference( &cpu_img[0], &gpu_img[0], width*height, me, mre );

    if (runtimes == 1) // running for debugging
        std::cout << APPNAME << " [max-error] [max-relative-error]:";

    std::cout << " " << std::scientific << me << " "
              << std::scientific << mre << "\n";

    return 0;

}
