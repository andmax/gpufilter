/**
 *  @file example_gauss.cc
 *  @brief Gaussian filter example
 *  @author Andre Maximo
 *  @date November, 2011
 */

#include <ctime>
#include <cstdlib>

#include <iostream>
#include <iomanip>

#include <timer.h>
#include <cpuground.h>
#include <gpufilter.h>

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

    const int w_in = 1024, h_in = 1024;
    const float sigma = 2.f;

    std::cout << "[gauss] Generating random input image (" << w_in << "x" << h_in << ") ... " << std::flush;

    float *in_cpu = new float[h_in*w_in];
    float *in_gpu = new float[h_in*w_in];

    srand(time(0));

    for (int i = 0; i < h_in*w_in; ++i)
        in_gpu[i] = in_cpu[i] = rand() / (float)RAND_MAX;

    std::cout << "done!\n[gauss] Applying Gaussian filter with sigma = " << sigma << "\n";
    std::cout << "[gauss] Considering zero-border as initial condition.\n";

    std::cout << "[gauss] Computing in the CPU ... " << std::flush;

    std::cout << std::fixed << std::setprecision(2);

    {
        gpufilter::scoped_timer_stop sts( gpufilter::timers.cpu_add("CPU") );

        float b10, a11;

        gpufilter::weights1(sigma, b10, a11);

        float b20, a21, a22;

        gpufilter::weights2(sigma, b20, a21, a22);

        gpufilter::r_0(in_cpu, h_in, w_in, b10, a11);
        gpufilter::r_0(in_cpu, h_in, w_in, b20, a21, a22);

        std::cout << "done!\n[gauss] CPU Timing: " << sts.elapsed()*1000 << " ms\n";
    }

    std::cout << "[gauss] Computing in the GPU ... " << std::flush;

    {
        gpufilter::scoped_timer_stop sts( gpufilter::timers.gpu_add("GPU") );

        gpufilter::gaussian_gpu( in_gpu, h_in, w_in, sigma );

        std::cout << "done!\n[gauss] GPU Timing: " << sts.elapsed()*1000 << " ms\n";
    }

    std::cout << "[gauss] GPU Timing includes memory transfers from and to the CPU\n";

    std::cout << "[gauss] Checking GPU result with CPU reference values\n";

    float me, mre;

    check_reference( in_cpu, in_gpu, h_in*w_in, me, mre );

    std::cout << std::scientific;

    std::cout << "[gauss] Maximum error: " << me << " ; Maximum relative error: " << mre << "\n";

    delete [] in_cpu;
    delete [] in_gpu;

    return 0;

}
