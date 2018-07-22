/**
 *  @file recAlg_cuda.cuh
 *  @brief Recursive Algorithms in the GPU
 *  @author Andre Maximo
 *  @date Jun, 2014
 */

#include "recursiveGaussian.h"

#include "alg0_cpu.h"
#include "alg5_cpu.h"
#include "gpu_aux.h"
#include "alg5_gpu.h"
#include "alg5f4_aux.h"
#include "alg5f4_gpu.h"
#include "alg5f4_gauss_gpu.h"

#include "gpud_aux.cuh"
#include "alg4pd_gpu.cuh"
#include "alg4d_gpu.cuh"
#include "alg5d_gpu.cuh"

/**
 * @brief Run alg5f4 - Recursive Gaussian filter van Vliet (our alg gray float)
 *
 * @param[in,out] iod Input/output data
 * @param[in] width Image width
 * @param[in] height Image height
 * @param[in] sigma Sigma of the Gaussian kernel
 */

extern "C"
void runAlg5f4(float *iod, int width, int height, float sigma) {

#if CLAMPTOEDGE
    const BorderType border_type = CLAMP_TO_EDGE;
    const int border = 1; // 1 block outside
#else
    const BorderType border_type = CLAMP_TO_ZERO;
    const int border = 0;
#endif

    Vector<float, 1+1> w1;
    Vector<float, 2+1> w2;
    Vector<float, 3+1> w3;
    weights(sigma, w1);
    weights(sigma, w2);
    weights(sigma, w3); // to use r=3 change R in recursiveGaussian.h

    // For lena image in gray 512x512
    // Deriche (original): 246 MiP/s
    // Deriche (row/col): 289 MiP/s
    // Alg5r3: 272 MiP/s
    // Alg5f5: 378 MiP/s
    // Alg5f4: 418 MiP/s

    //alg0_cpu<1>(iod, width, height, w1, border, border_type);
    //alg0_cpu<2>(iod, width, height, w2, border, border_type);
    //alg0_cpu<3>(iod, width, height, w3, border, border_type);
    //alg5_gpu<true, 3>(iod, width, height, w3, border, border_type);
    //alg5f5_gpu<true, 1, 2>(iod, width, height, w1, w2, border, border_type);
    alg5f4_gpu<true, 1, 2>(iod, width, height, w1, w2, border, border_type);

}

/**
 * @brief Run alg4pd - Recursive Gaussian filter Deriche (our alg gray float)
 *
 * @param[in,out] iod Input/output data
 * @param[in] width Image width
 * @param[in] height Image height
 * @param[in] sigma Sigma of the Gaussian kernel
 * @param[in] order Filter order
 */

// R in recursiveGaussian.h must be 2

extern "C"
void runAlg4pd(float *iod, int width, int height, float sigma, int order) {

    alg4pd_gpu(iod, width, height, sigma, order);

}

extern "C"
void runAlg4d(float *iod, int width, int height, float sigma, int order) {

    alg4d_gpu(iod, width, height, sigma, order);

}

extern "C"
void runAlg5d(float *iod, int width, int height, float sigma, int order) {

    alg5d_gpu(iod, width, height, sigma, order);

}
