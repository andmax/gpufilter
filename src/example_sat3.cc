/**
 *  @file example_sat3.cc
 *  @brief Third SAT (Summed-Area Table) example
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
#include <gpudefs.cuh>

#include <sat.cuh>

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

    std::cout << "[sat3] Generating random input image (" << w_in << "x" << h_in << ") ... " << std::flush;

    float *in_cpu = new float[h_in*w_in];
    float *in_gpu = new float[h_in*w_in];

    srand(time(0));

    for (int i = 0; i < h_in*w_in; ++i)
        in_gpu[i] = in_cpu[i] = rand() % 256;

    std::cout << "done!\n[sat3] Computing summed-area table in the CPU ... " << std::flush;

    {
        gpufilter::scoped_timer_stop sts( gpufilter::timers.cpu_add("CPU", h_in*w_in, "iP") );

        gpufilter::sat_cpu( in_cpu, h_in, w_in );
    }

    std::cout << "done!\n[sat3] Configuring the GPU to run ... " << std::flush;

    dim3 cg_img, cg_ybar, cg_vhat;
    gpufilter::dvector<float> d_in_gpu, d_ybar, d_vhat, d_ysum;
    int h_out, w_out;

    gpufilter::prepareSAT( d_in_gpu, d_ybar, d_vhat, d_ysum, cg_img, cg_ybar, cg_vhat, h_out, w_out, in_gpu, h_in, w_in );

    gpufilter::dvector<float> d_out_gpu( h_out*w_out );

    std::cout << "done!\n[sat3] Computing summed-area table in the GPU ... " << std::flush;

    {
        gpufilter::scoped_timer_stop sts( gpufilter::timers.gpu_add("GPU", h_in*w_in*REPEATS, "iP") );

        for (int i = 0; i < REPEATS; ++i)
            gpufilter::algSAT( d_out_gpu, d_in_gpu, d_ybar, d_vhat, d_ysum, cg_img, cg_ybar, cg_vhat );
    }

    std::cout << "done!\n";

    gpufilter::timers.flush();

    std::cout << "[sat3] Copying result back from the GPU ... " << std::flush;

    d_out_gpu.copy_to( in_gpu, h_out, w_out, h_in, w_in );

    std::cout << "done!\n[sat3] Checking GPU result with CPU reference values\n";

    float me, mre;

    check_reference( in_cpu, in_gpu, h_in*w_in, me, mre );

    std::cout << "[sat3] Maximum relative error: " << mre << "\n";

    delete [] in_cpu;
    delete [] in_gpu;

    return 0;

}
