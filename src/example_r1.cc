/**
 *  @file example_r1.cc
 *  @brief First R (Recursive Filtering) example
 *  @author Andre Maximo
 *  @date November, 2011
 */

#include <ctime>
#include <cstdlib>

#include <iostream>
#include <iomanip>

#include <cpuground.h>

// Print a matrix of values
void print_matrix( const float *img,
                   const int& h,
                   const int& w,
                   const int& fw = 4 ) {
    for (int i = 0; i < h; ++i) {
        std::cout << std::setw(fw) << img[i*w];
        for (int j = 1; j < w; ++j)
            std::cout << " " << std::setw(fw) << img[i*w+j];
        std::cout << "\n";
    }
}

// Main
int main(int argc, char *argv[]) {

    const int w_in = 8, h_in = 8;
    const float b0 = 1.f, a1 = -1.f;

    std::cout << "[r1] Generating random input image (" << w_in << "x" << h_in << ") ... " << std::flush;

    float *in = new float[h_in*w_in];

    srand(time(0));

    for (int i = 0; i < h_in*w_in; ++i)
        in[i] = rand() % 8;

    std::cout << "done!\n";

    print_matrix(in, h_in, w_in, 2);

    std::cout << "[r1] Recursive filter: y_i = b0 * x_i - a1 * y_{i-1}\n";
    std::cout << "[r1] Considering causal filter (only forward) on each row\n";
    std::cout << "[r1] Feedforward and feedback coefficients are: b0 = " << b0 << " ; a1 = " << a1 << "\n";
    std::cout << "[r1] This is equivalent to an inclusive multi-scan with the plus operator\n";
    std::cout << "[r1] CPU Computing first-order recursive filtering with zero-border ... " << std::flush;

    gpufilter::rrfr_0( in, h_in, w_in, b0, a1, true );

    std::cout << "done!\n[r1] Output matrix " << w_in << " x " << h_in << " :\n";

    print_matrix(in, h_in, w_in, 4);

    delete [] in;

    return 0;

}
