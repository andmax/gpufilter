
#include <ctime>
#include <cstdlib>

#include <iostream>
#include <iomanip>

#include <cpuground.h>

#include <gpufilter.cuh>

// Check device computation
void check_cpu_reference( const float *ref,
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

// Print a matrix of values
void print_matrix( const float *img,
                   const int& h,
                   const int& w,
                   const int& fw = 8 ) {
    std::cout << std::setprecision(4) << std::fixed;
    for (int i = 0; i < h; ++i) {
        std::cout << std::setw(fw) << img[i*w];
        for (int j = 1; j < w; ++j)
            std::cout << " " << std::setw(fw) << img[i*w+j];
        std::cout << "\n";
    }
    std::cout << std::resetiosflags( std::ios_base::fixed );
}

// Main
int main(int argc, char *argv[]) {

    std::cout << "[r] Generating random input matrix ... " << std::flush;

    const int w_in = 32, h_in = 32;
    const float b0 = 1.f, a1 = .5f;

    float *in_cpu = new float[w_in*h_in];
    float *in_gpu = new float[w_in*h_in];

    srand(time(0));

    for (int i = 0; i < w_in*h_in; ++i)
        in_gpu[i] = in_cpu[i] = rand() / (float)RAND_MAX;

    std::cout << "done!\n[r] Input matrix " << w_in << " x " << h_in << " :\n";

    print_matrix( in_cpu, h_in, w_in );

    std::cout << "[r] Recursive filter: y_i = b0 * x_i - a1 * y_{i-1}\n";
    std::cout << "[r] Considering forward and reverse on rows and columns\n";
    std::cout << "[r] Feedforward and feedback coefficients are: b0 = " << b0 << " ; a1 = " << a1 << "\n";

    std::cout << "[r] CPU Computing first-order recursive filtering with zero-border ... " << std::flush;

    gpufilter::r_0( in_cpu, h_in, w_in, b0, a1 );

    std::cout << "done!\n[r] Output matrix " << w_in << " x " << h_in << " :\n";

    print_matrix( in_cpu, h_in, w_in );

    std::cout << "[r] GPU Computing first-order recursive filtering with zero-border ... " << std::flush;

    gpufilter::algorithm5_1( in_gpu, h_in, w_in, b0, a1 );

    std::cout << "done!\n[r] Output matrix " << w_in << " x " << h_in << " :\n";

    print_matrix( in_gpu, h_in, w_in );

    float me, mre;

    check_cpu_reference( in_cpu, in_gpu, w_in*h_in, me, mre );

    std::cout << "me = " << me << " mre = " << mre << "\n";

    delete [] in_cpu;
    delete [] in_gpu;

    return 0;

}
