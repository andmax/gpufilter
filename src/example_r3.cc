/**
 *  @file example_r3.cc
 *  @brief Third R (Recursive Filtering) example
 *  @author Andre Maximo
 *  @date November, 2011
 */

#include <ctime>
#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <iomanip>

#include <timer.h>
#include <dvector.h>

#include <cpuground.h>
#include <gpufilter.h>

#include <alg5.cuh>

#define REPEATS 100

// Check computation
void check_reference( const float *ref,
                      const float *res,
                      const int& ne,
                      float& me,
                      float& mre ) {
    mre = me = (float)0;
    for (int i = 0; i < ne; i++) {
        float a = (float)(res[i]) - ref[i];
        if( a < (float)0 ) a = -a;
        if( ref[i] != (float)0 ) {
            float r = (ref[i] < (float)0) ? -ref[i] : ref[i];
            float b = a / r;
            mre = b > mre ? b : mre;
        }
        me = a > me ? a : me;
    }
}

// Main
int main(int argc, char *argv[]) {

    int w_in = 4096, h_in = 4096;

    if( argc == 3 ) { sscanf(argv[1], "%d", &w_in); sscanf(argv[2], "%d", &h_in); }

    const float b0 = 1.267949192f, a1 = 0.267949192f;

    std::cout << "[r3] Generating random input image (" << w_in << "x" << h_in << ") ... " << std::flush;

    float *in_cpu = new float[h_in*w_in];
    float *in_gpu = new float[h_in*w_in];

    srand(time(0));

    for (int i = 0; i < h_in*w_in; ++i)
        in_gpu[i] = in_cpu[i] = rand() / (float)RAND_MAX;

    std::cout << "done!\n[r3] Recursive filter: y_i = b0 * x_i - a1 * y_{i-1}\n";
    std::cout << "[r3] Considering forward and reverse on rows and columns\n";
    std::cout << "[r3] Feedforward and feedback coefficients are: b0 = " << b0 << " ; a1 = " << a1 << "\n";
    std::cout << "[r3] CPU Computing first-order recursive filtering ... " << std::flush;

    std::cout << std::fixed << std::setprecision(2);

    {
        gpufilter::scoped_timer_stop sts( gpufilter::timers.cpu_add("CPU", h_in*w_in, "iP") );

        gpufilter::r( in_cpu, h_in, w_in, b0, a1 );
    }

    std::cout << "done!\n[r3] Configuring the GPU to run ... " << std::flush;

    dim3 cg_img;
    gpufilter::dvector<float> d_out, d_transp_pybar, d_transp_ezhat, d_ptucheck, d_etvtilde;
    cudaArray *a_in;

    gpufilter::prepare_alg5( d_out, a_in, d_transp_pybar, d_transp_ezhat, d_ptucheck, d_etvtilde, cg_img, in_gpu, h_in, w_in, b0, a1 );

    std::cout << "done!\n[r3] GPU Computing first-order recursive filtering using Algorithm 5 ... " << std::flush;

    {
        gpufilter::scoped_timer_stop sts( gpufilter::timers.gpu_add("GPU", h_in*w_in*REPEATS, "iP") );

        for (int i = 0; i < REPEATS; ++i)
            gpufilter::alg5( d_out, a_in, d_transp_pybar, d_transp_ezhat, d_ptucheck, d_etvtilde, cg_img );
    }

    std::cout << "done!\n";

    gpufilter::timers.flush();

    std::cout << "[r3] Copying result back from the GPU ... " << std::flush;

    d_out.copy_to( in_gpu, h_in * w_in );

    cudaFreeArray( a_in );

    std::cout << "done!\n[r3] Checking GPU result with CPU reference values\n";

    float me, mre;

    check_reference( in_cpu, in_gpu, h_in*w_in, me, mre );

    std::cout << std::scientific;

    std::cout << "[r3] Maximum error: " << me << "\n";

    delete [] in_cpu;
    delete [] in_gpu;

    return 0;

}
