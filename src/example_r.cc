
#include <ctime>
#include <cstdlib>

#include <iostream>
#include <iomanip>

#include <cpuground.h>

#include <gpufilter.cuh>

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

    const int w_in = 8, h_in = 8;
    const float b0 = 1.f, a1 = .5f;

    float *in = new float[w_in*h_in];

    srand(time(0));

    for (int i = 0; i < w_in*h_in; ++i)
        in[i] = rand() / (float)RAND_MAX;

    std::cout << "done!\n[r] Input matrix " << w_in << " x " << h_in << " :\n";

    print_matrix(in, h_in, w_in);

    std::cout << "[r] Recursive filter: y_i = b0 * x_i - a1 * y_{i-1}\n";
    std::cout << "[r] Considering forward and reverse on rows and columns\n";
    std::cout << "[r] Feedforward and feedback coefficients are: b0 = " << b0 << " ; a1 = " << a1 << "\n";

    std::cout << "[r] Computing first-order recursive filtering with zero-border ... " << std::flush;

    gpufilter::r_0( in, h_in, w_in, b0, a1 );

    std::cout << "done!\n[r] Output matrix " << w_in << " x " << h_in << " :\n";

    print_matrix(in, h_in, w_in);

    delete [] in;

    return 0;

}
