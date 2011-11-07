/**
 *  @file example_sat1.cc
 *  @brief First SAT (Summed-Area Table) example
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

    std::cout << "[sat1] Generating random input matrix ... " << std::flush;

    const int w_in = 8, h_in = 8;

    float *in = new float[w_in*h_in];

    srand(time(0));

    for (int i = 0; i < w_in*h_in; ++i)
        in[i] = rand() % 8;

    std::cout << "done!\n[sat1] Input matrix " << w_in << " x " << h_in << " :\n";

    print_matrix(in, h_in, w_in, 2);

    std::cout << "[sat1] Computing summed-area table ... " << std::flush;

    gpufilter::sat( in, h_in, w_in );

    std::cout << "done!\n[sat1] Output matrix " << w_in << " x " << h_in << " :\n";

    print_matrix(in, h_in, w_in, 4);

    delete [] in;

    return 0;

}
