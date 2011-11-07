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

    const int w_in = 8, h_in = 8;

    std::cout << "[sat1] Generating random input image (" << w_in << "x" << h_in << ") ... " << std::flush;

    float *in = new float[h_in*w_in];

    srand(time(0));

    for (int i = 0; i < h_in*w_in; ++i)
        in[i] = rand() % 8;

    std::cout << "done!\n";

    print_matrix(in, h_in, w_in, 2);

    std::cout << "[sat1] Computing summed-area table in the CPU ... " << std::flush;

    gpufilter::sat_cpu( in, h_in, w_in );

    std::cout << "done!\n[sat1] Output matrix " << w_in << " x " << h_in << " :\n";

    print_matrix(in, h_in, w_in, 4);

    delete [] in;

    return 0;

}
