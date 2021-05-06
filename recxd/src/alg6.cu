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

#include <util/util.h>
#include <util/symbol.h>
#include <util/dvector.h>
#include <util/timer.h>
#include <util/recfilter.h>
#include <util/image.h>
#include <util/gaussian.h>

#include "cpudefs.h"
#include "gpudefs.h"
#include "alg0_cpu.h"
#include "alg6.cuh"

//== IMPLEMENTATION ============================================================

template<int R>
void alg6( float *h_img,
           const int& width, const int& height, const int& runtimes,
           const gpufilter::Vector<float, R+1>& w,
           const int& border,
           const gpufilter::BorderType& btype ) {

    gpufilter::alg6i_gpu<false, R>(h_img, width, height, runtimes, w);

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

    gpufilter::check_cpu_reference( &cpu_img[0], &gpu_img[0], width*height,
                                    me, mre );

    if (runtimes == 1) // running for debugging
        std::cout << APPNAME << " [max-error] [max-relative-error]:";

    std::cout << " " << std::scientific << me << " "
              << std::scientific << mre << "\n";

    return 0;

}
