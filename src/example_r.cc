
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
    const float b0 = 1.f, a1 = .5f;

    std::cout << "[r] Generating random input image (" << w_in << "x" << h_in << ") ... " << std::flush;

    float *in_cpu = new float[w_in*h_in];
    float *in_gpu = new float[w_in*h_in];

    srand(time(0));

    for (int i = 0; i < w_in*h_in; ++i)
        in_gpu[i] = in_cpu[i] = rand() / (float)RAND_MAX;

    std::cout << "done!\n[r] Recursive filter: y_i = b0 * x_i - a1 * y_{i-1}\n";
    std::cout << "[r] Considering forward and reverse on rows and columns\n";
    std::cout << "[r] Feedforward and feedback coefficients are: b0 = " << b0 << " ; a1 = " << a1 << "\n";
    std::cout << "[r] CPU Computing first-order recursive filtering with zero-border ... " << std::flush;

    std::cout << std::fixed << std::setprecision(2);

    {
        gpufilter::scoped_timer_stop sts( gpufilter::timers.cpu_add("CPU") );

        gpufilter::r_0( in_cpu, h_in, w_in, b0, a1 );

        std::cout << "done!\n[r] CPU Timing: " << sts.elapsed()*1000 << " ms\n";
    }

    std::cout << "[r] GPU Computing first-order recursive filtering with zero-border ... " << std::flush;

    {
        gpufilter::scoped_timer_stop sts( gpufilter::timers.gpu_add("GPU") );

        gpufilter::algorithm5_1( in_gpu, h_in, w_in, b0, a1 );

        std::cout << "done!\n[r] GPU Timing: " << sts.elapsed()*1000 << " ms\n";
    }

    std::cout << "[r] GPU Timing includes memory transfers from and to the CPU\n";

    std::cout << "[r] Checking GPU result with CPU reference values\n";

    float me, mre;

    check_reference( in_cpu, in_gpu, w_in*h_in, me, mre );

    std::cout << std::scientific;

    std::cout << "[r] Maximum error: " << me << " ; Maximum relative error: " << mre << "\n";

    delete [] in_cpu;
    delete [] in_gpu;

    return 0;

}
