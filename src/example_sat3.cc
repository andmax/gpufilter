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

    int h_out = h_in, w_out = w_in;

    if( h_in % 32 > 0 ) h_out += (32 - (h_in % 32));
    if( w_in % 32 > 0 ) w_out += (32 - (w_in % 32));

    dim3 cg_img;
    gpufilter::up_constants_sizes( cg_img, h_out, w_out );

    gpufilter::dvector<float> d_in_gpu( in_gpu, h_in, w_in, h_out, w_out );

    gpufilter::dvector<float> d_ybar( cg_img.x * h_out ), d_vhat( cg_img.x * h_out ), d_ysum( cg_img.x * cg_img.y );

	const int nWm = (w_out+MTS-1)/MTS, nHm = (h_out+MTS-1)/MTS;
    dim3 cg_ybar(1, nHm), cg_vhat(nWm, 1);

    std::cout << "done!\n[sat3] Computing summed-area table in the GPU ... " << std::flush;

    {
        gpufilter::scoped_timer_stop sts( gpufilter::timers.gpu_add("GPU", h_in*w_in, "iP") );

        gpufilter::algorithmSAT( d_in_gpu, d_ybar, d_vhat, d_ysum, cg_img, cg_ybar, cg_vhat );
    }

    std::cout << "done![sat3] Timings:\n";

    gpufilter::timers.flush();

    gpufilter::dvector<float> d_in_gpu2( in_gpu, h_in, w_in, h_out, w_out );

    {
        gpufilter::scoped_timer_stop sts( gpufilter::timers.gpu_add("GPU", h_in*w_in, "iP") );

        for (int i = 0; i < 100; ++i)
            gpufilter::algorithmSAT( d_in_gpu2, d_ybar, d_vhat, d_ysum, cg_img, cg_ybar, cg_vhat );

        std::cout << "[sat3] GPU 100x Timing: " << sts.elapsed()*10 << " ms\n";
    }

    std::cout << "[sat3] Copying result back from the GPU ... " << std::flush;

    d_in_gpu.copy_to( in_gpu, h_out, w_out, h_in, w_in );

    std::cout << "done!\n[sat3] Checking GPU result with CPU reference values\n";

    float me, mre;

    check_reference( in_cpu, in_gpu, h_in*w_in, me, mre );

    std::cout << "[sat3] Big values in SAT may lead to big maximum error (look at relative error instead)\n";

    std::cout << "[sat3] Maximum error: " << me << " ; Maximum relative error: " << mre << "\n";

    delete [] in_cpu;
    delete [] in_gpu;

    return 0;

}
