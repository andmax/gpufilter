/**
 *  @file example_bspline.cc
 *  @brief Bicubic B-Spline interpolation example
 *  @author Andre Maximo
 *  @date December, 2011
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

    std::cout << "[bspline] Generating random input image (" << w_in << "x" << h_in << ") ... " << std::flush;

    float *in_cpu = new float[h_in*w_in];
    float *in_gpu = new float[h_in*w_in];

    srand(time(0));

    for (int i = 0; i < h_in*w_in; ++i)
        in_gpu[i] = in_cpu[i] = rand() / (float)RAND_MAX;

    std::cout << "done!\n[bspline] Applying bspline3i filter\n";
    std::cout << "[bspline] Considering zero-border as initial condition.\n";

    std::cout << "[bspline] Computing in the CPU ... " << std::flush;

    std::cout << std::fixed << std::setprecision(2);

    {
        gpufilter::scoped_timer_stop sts( gpufilter::timers.cpu_add("CPU") );

        gpufilter::bspline3i_cpu( in_cpu, h_in, w_in, gpufilter::zero, 0 );

        std::cout << "done!\n[bspline] CPU Timing: " << sts.elapsed()*1000 << " ms\n";
    }

    std::cout << "[bspline] Computing in the GPU ... " << std::flush;

    {
        gpufilter::scoped_timer_stop sts( gpufilter::timers.gpu_add("GPU") );

        gpufilter::bspline3i_gpu( in_gpu, h_in, w_in );

        std::cout << "done!\n[bspline] GPU Timing: " << sts.elapsed()*1000 << " ms\n";
    }

    std::cout << "[bspline] GPU Timing includes memory transfers from and to the CPU\n";

    std::cout << "[bspline] Checking GPU result with CPU reference values\n";

    float me, mre;

    check_reference( in_cpu, in_gpu, h_in*w_in, me, mre );

    std::cout << std::scientific;

    std::cout << "[bspline] Maximum error: " << me << "\n";

    delete [] in_cpu;
    delete [] in_gpu;

    return 0;

}
