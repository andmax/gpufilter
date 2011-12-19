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
#include <gpudefs.h>

#define EXTEND_IMAGE

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

    const int in_w = 1024, in_h = 1024;
    const gpufilter::initcond ic = gpufilter::mirror;
    const int extb = 1, ext = WS*extb;

    std::cout << "[bspline] Generating random input image (" << in_w << "x" << in_h << ") ... " << std::flush;

    float *in_cpu = new float[in_h*in_w];
    float *in_gpu = new float[in_h*in_w];

    srand(time(0));

    for (int i = 0; i < in_h*in_w; ++i)
        in_gpu[i] = in_cpu[i] = rand() / (float)RAND_MAX;

    std::cout << "done!\n[bspline] Applying bspline3i filter\n";
    std::cout << "[bspline] Considering zero-border as initial condition.\n";

    std::cout << "[bspline] Computing in the CPU ... " << std::flush;

    std::cout << std::fixed << std::setprecision(2);

#ifdef EXTEND_IMAGE
    int in_h2, in_w2;
    in_h2 = in_h + 2*ext;
    in_w2 = in_w + 2*ext;
    float *in_cpu2 = new float[in_h2*in_w2];
    for (int i = -ext; i < in_h+ext; ++i) {
        gpufilter::_mirror m;
        int mi = m( i, in_h );
        for (int j = -ext; j < in_w+ext; ++j) {
            int mj = m( j, in_w );
            in_cpu2[(i+ext)*in_w2+(j+ext)] = in_cpu[mi*in_w+mj];
        }
    }
#endif

    {
        gpufilter::scoped_timer_stop sts( gpufilter::timers.cpu_add("CPU") );

#ifdef EXTEND_IMAGE
        gpufilter::bspline3i_cpu( in_cpu2, in_h2, in_w2, gpufilter::zero, 0 );
#else
        gpufilter::bspline3i_cpu( in_cpu, in_h, in_w, ic, ext );
#endif
        std::cout << "done!\n[bspline] CPU Timing: " << sts.elapsed()*1000 << " ms\n";
    }

    std::cout << "[bspline] Computing in the GPU ... " << std::flush;

    {
        gpufilter::scoped_timer_stop sts( gpufilter::timers.gpu_add("GPU") );

        gpufilter::bspline3i_gpu( in_gpu, in_h, in_w, ic, extb );

        std::cout << "done!\n[bspline] GPU Timing: " << sts.elapsed()*1000 << " ms\n";
    }

    std::cout << "[bspline] GPU Timing includes memory transfers from and to the CPU\n";

    std::cout << "[bspline] Checking GPU result with CPU reference values\n";

#ifdef EXTEND_IMAGE
    for (int i = 0; i < in_h; ++i)
        for (int j = 0; j < in_w; ++j)
            in_cpu[i*in_w+j] = in_cpu2[(i+ext)*in_w2+(j+ext)];
    delete [] in_cpu2;
#endif

    float me, mre;

    check_reference( in_cpu, in_gpu, in_h*in_w, me, mre );

    std::cout << std::scientific;

    std::cout << "[bspline] Maximum error: " << me << "\n";

    delete [] in_cpu;
    delete [] in_gpu;

    return 0;

}
