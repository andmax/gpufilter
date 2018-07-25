/**
 *  @file alg5f4.cu
 *  @brief Algorithm 5_1 fusioned with algorithm 4_2 in the GPU
 *  @author Andre Maximo
 *  @date Jan, 2014
 *  @copyright The MIT License
 */

#define ORDER 1 // not used, it is in fact two filter orders: 1 and 2
#define APPNAME "[alg5f4]"

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

#include "gpudefs.h"
#include "alg0_cpu.h"
#include "alg5f4_gpu.cuh"

//== IMPLEMENTATION ============================================================

// Main ------------------------------------------------------------------------

int main( int argc, char** argv ) {

    int width = 1024, height = 1024;
    int runtimes = 1; // # of run times (1 for debug; 1000 for performance)
    float me = 0.f, mre = 0.f; // maximum error and maximum relative error

    if ((argc > 1 && argc < 4) ||
        (argc >= 4 && (sscanf(argv[1], "%d", &width) != 1 ||
                       sscanf(argv[2], "%d", &height) != 1 ||
                       sscanf(argv[3], "%d", &runtimes) != 1))) {
        std::cerr << APPNAME << " Bad arguments!\n";
        std::cout << APPNAME << " Usage: " << argv[0]
                  << " [width height runtimes]\n";
        return 1;
    }

    std::vector< float > cpu_img(width*height), gpu_img(width*height);

    srand( 1234 );
    for (int i = 0; i < width*height; ++i)
        gpu_img[i] = cpu_img[i] = rand() / (float)RAND_MAX;

    float sigma = 4.f; // width / 6.f;

    gpufilter::Vector<float, R1+1> w1;
    gpufilter::Vector<float, R2+1> w2;
    gpufilter::weights(sigma, w1);
    gpufilter::weights(sigma, w2);

    if (runtimes == 1) { // running for debugging
        std::cout << APPNAME << " Size: " << width << " x " << height
                  << "  Orders: 1->2  Run-times: 1\n";
        std::cout << APPNAME << " Boundary: zero  Border: 0\n";
        std::cout << APPNAME << " Weights1: " << w1 << "\n";
        std::cout << APPNAME << " Weights2: " << w2 << "\n";
        std::cout << APPNAME << " (1) Runs the reference in the CPU (ref)\n";
        std::cout << APPNAME << " (2) Runs the algorithm in the GPU (res)\n";
        std::cout << APPNAME << " (3) Checks computations (ref x res)\n";
    }

    gpufilter::alg0_cpu<R1>(&cpu_img[0], width, height, w1);
    gpufilter::alg0_cpu<R2>(&cpu_img[0], width, height, w2);

    gpufilter::alg5f4_gpu<false>(&gpu_img[0], width, height, runtimes, w1, w2);

    gpufilter::check_cpu_reference( &cpu_img[0], &gpu_img[0], width*height, me, mre );

    if (runtimes == 1) // running for debugging
        std::cout << APPNAME << " [max-error] [max-relative-error]:";

    std::cout << " " << std::scientific << me << " "
              << std::scientific << mre << "\n";

    return 0;

}
