/**
 *  @file alg5.cu
 *  @brief Algorithm 5 in the GPU
 *  @author Andre Maximo
 *  @date Sep, 2012
 *  @copyright The MIT License
 */

#ifndef ORDER
#define ORDER 1 // default filter order r=1
#endif
#define APPNAME "[alg5_" << ORDER << "]"

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

#include <util/symbol.h>
#include <util/dvector.h>
#include <util/timer.h>
#include <util/recfilter.h>
#include <util/image.h>
#include <util/gaussian.h>
#include <util/util.h>

#include "cpudefs.h"
#include "gpudefs.h"
#include "alg0_cpu.h"
#include "alg5_gpu.cuh"

//== IMPLEMENTATION ============================================================

/**
 *  @ingroup api_gpu
 *  @brief Compute Algorithm 5 (order agnostic) with any boundary condition
 *
 *  This function computes R-order recursive filtering with given
 *  weights of an input 2D image using algorithm \f$5_r\f$ and any
 *  given boundary condition.  All choices of boundary conditions
 *  imply border input padding, that is the algorithm 5 is restricted
 *  to the discussions and considerations in our 2011 paper:
 *
 *  @verbatim
@inproceedings{NehabEtAl:2011,
  title = {{GPU}-{E}fficient {R}ecursive {F}iltering and {S}ummed-{A}rea {T}ables},
  author = {{N}ehab, {D}. and {M}aximo, {A}. and {L}ima, {R}. {S}. and {H}oppe, {H}.},
  journal = {{ACM} {T}ransactions on {G}raphics ({P}roceedings of the {ACM} {SIGGRAPH} {A}sia 2011)},
  year = {2011},
  volume = {30},
  number = {6},
  doi = {},
  publisher = {ACM},
  address = {{N}ew {Y}ork, {NY}, {USA}}
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
void alg5( float *h_img,
           const int& width, const int& height, const int& runtimes,
           const gpufilter::Vector<float, R+1>& w,
           const int& border,
           const gpufilter::BorderType& btype ) {

    if (border == 0) {
        gpufilter::alg5_gpu<false, R>(h_img, width, height, runtimes, w);
    } else if (border > 0) {
        gpufilter::alg5_gpu<true, R>(h_img, width, height, runtimes, w,
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

    alg5<ORDER>(&gpu_img[0], width, height, runtimes, w, border, btype);

    gpufilter::check_cpu_reference( &cpu_img[0], &gpu_img[0], width*height, me, mre );

    if (runtimes == 1) // running for debugging
        std::cout << APPNAME << " [max-error] [max-relative-error]:";

    std::cout << " " << std::scientific << me << " "
              << std::scientific << mre << "\n";

    return 0;

}
