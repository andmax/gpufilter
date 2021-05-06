/**
 *  @file alg3_3d.cu
 *  @brief Recursive filtering algorithm 3 for 3D
 *  @author Andre Maximo
 *  @date Jun, 2019
 *  @copyright The MIT License
 */

#include <cstdlib>

#include <iostream>
#include <fstream>

#include <util/util.h>
#include <util/timer.h>
#include <util/symbol.h>
#include <util/linalg.h>
#include <util/dvector.h>
#include <util/gaussian.h>
#include <util/recfilter.h>
#include <util/alg0_xd_cpu.h>

#include "alg3_3d.cuh"


int main(int argc, char** argv) {

    long int width = 1 << 9, height = 1 << 9, depth = 1 << 5; // defaults
    long int num_repeats = 1; // defaults
    char array_bin_fn[200] = "../bin/random_array.bin";
    
    if ((argc != 1 && argc != 6)
        || (argc==6 && (sscanf(argv[1], "%ld", &width) != 1 ||
                        sscanf(argv[2], "%ld", &height) != 1 ||
                        sscanf(argv[3], "%ld", &depth) != 1 ||
                        sscanf(argv[4], "%ld", &num_repeats) != 1 ||
                        sscanf(argv[5], "%s", array_bin_fn) != 1))) {
        std::cerr << APPNAME << " Bad arguments!\n";
        std::cerr << APPNAME << " Usage: " << argv[0]
                  << " [width height depth num_repeats array_bin_fn] ->"
                  << " Output: Mis/s MAE MRE\n";
        std::cerr << APPNAME << " Where: width, height, depth are the shape "
                  << "of the 3D array to run this on (up to 1Gi total)\n";
        std::cerr << APPNAME << " Where: num_repeats = number of repetitions "
                  << "to measure the run timing performance\n";
        std::cerr << APPNAME << " Where: array_bin_fn = array of inputs in "
                  << "binary to read 1D input data from\n";
        std::cerr << APPNAME << " Where: Mis/s = Mebi samples per second; "
                  << "MAE = max. abs. error; MRE = max. rel. error\n";
        return EXIT_FAILURE;
    }

    if (num_repeats == 1) { // running for debugging
        std::cout << gpufilter::get_cuda_device_properties();
    }

    float gaussian_sigma = 4.0;
    gpufilter::Vector<float, ORDER+1> iir_weights;
    gpufilter::weights(gaussian_sigma, iir_weights);

    long int num_samples = width * height * depth;
    
    float *cpu_arr = new float[num_samples];
    float *gpu_arr = new float[num_samples];

    std::ifstream in_file(array_bin_fn, std::ios::binary);
    in_file.read(reinterpret_cast<char*>(cpu_arr),
                 sizeof(float)*num_samples);
    in_file.close();

    memcpy(gpu_arr, cpu_arr, sizeof(float) * num_samples);

    gpufilter::recursive_3d<0,true,ORDER>(
        cpu_arr, width, height, depth, iir_weights);
    gpufilter::recursive_3d<0,false,ORDER>(
        cpu_arr, width, height, depth, iir_weights);
    gpufilter::recursive_3d<1,true,ORDER>(
        cpu_arr, width, height, depth, iir_weights);
    gpufilter::recursive_3d<1,false,ORDER>(
        cpu_arr, width, height, depth, iir_weights);
    gpufilter::recursive_3d<2,true,ORDER>(
        cpu_arr, width, height, depth, iir_weights);
    gpufilter::recursive_3d<2,false,ORDER>(
        cpu_arr, width, height, depth, iir_weights);

    gpufilter::alg3_3d_gpu<ORDER>(
        gpu_arr, width, height, depth, num_repeats, iir_weights);

    float max_abs_err, max_rel_err;
    gpufilter::check_cpu_reference(cpu_arr, gpu_arr, num_samples,
                                   max_abs_err, max_rel_err);

    if (num_repeats == 1) // running for debugging
        std::cout << APPNAME << " [max-absolute-error] [max-relative-error]:";

    std::cout << " " << std::scientific << max_abs_err << " "
              << std::scientific << max_rel_err << "\n";

    if (num_repeats == 1) { // running for debugging
        std::cout << "width: " << width << " height: " << height
                  << " depth: " << depth << "\n";
    }

    if (cpu_arr) delete [] cpu_arr;
    if (gpu_arr) delete [] gpu_arr;

    return EXIT_SUCCESS;

}
