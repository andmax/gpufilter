/**
 *  @file alg5varc.cu
 *  @brief Algorithm 5 in the GPU using variable coefficients
 *  @author Andre Maximo
 *  @date Jan, 2012
 *  @copyright The MIT License
 */

#define ORDER 1 // filter order is fixed
#define APPNAME "[alg5varc]"

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

#include "vardefs.h"
#include "gpudefs.h"
#include "alg0_cpu.h"
#include "alg5varc_gpu.cuh"

//== IMPLEMENTATION ============================================================

// Main ------------------------------------------------------------------------

int main( int argc, char** argv ) {

    assert(ORDER==1); // Currently supporting order 1 only!

    int width = 1024, height = 1024;
    int runtimes = 1; // # of run times (1 for debug; 1000 for performance)
    float me = 0.f, mre = 0.f; // maximum error and maximum relative error
    int border = 1;

    if ((argc > 1 && argc < 5) ||
        (argc==5 && (sscanf(argv[1], "%d", &width) != 1 ||
                     sscanf(argv[2], "%d", &height) != 1 ||
                     sscanf(argv[3], "%d", &runtimes) != 1 ||
                     sscanf(argv[4], "%d", &border) != 1))) {
        std::cerr << APPNAME << " Bad arguments!\n";
        std::cout << APPNAME << " Usage: " << argv[0]
                  << " [width height runtimes border]\n";
        return 1;
    }

    std::vector< float > cpu_img(width*height), gpu_img(width*height);

    srand( 1234 );
    for (int i = 0; i < width*height; ++i)
        gpu_img[i] = cpu_img[i] = rand() / (float)RAND_MAX;

    gpufilter::Vector<float, ORDER+1> w;
    w[0] = spline::w0;
    w[1] = spline::w1;

    if (runtimes == 1) { // running for debugging
        std::cout << APPNAME << " Size: " << width << " x " << height
                  << "  Order: 1  Run-times: 1\n";
        std::cout << APPNAME << " Boundary: reflect  Border: " << border << "\n";
        std::cout << APPNAME << " Weights: " << w << "\n";
        std::cout << APPNAME << " (1) Runs the reference in the CPU (ref)\n";
        std::cout << APPNAME << " (2) Runs the algorithm in the GPU (res)\n";
        std::cout << APPNAME << " (3) Checks computations (ref x res)\n";
    }

    gpufilter::nehab_hoppe_tr2011_recfilter(&cpu_img[0], width, height);

    gpufilter::alg5varc_gpu<ORDER>(&gpu_img[0], width, height, runtimes, border);

    if (runtimes == 1) // running for debugging
        std::cout << APPNAME << " [max-error] [max-relative-error]:";

    gpufilter::check_cpu_reference( &cpu_img[0], &gpu_img[0], width*height, me, mre );

    std::cout << " " << std::scientific << me << " "
              << std::scientific << mre << "\n";

    return 0;

}
