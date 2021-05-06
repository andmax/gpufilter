/**
 *  @file alg_rd.cu
 *  @brief Algorithm recursive doubling RD
 *  @author Andre Maximo
 *  @date Aug, 2019
 *  @copyright The MIT License
 */

#include <cstdlib>

#include <algorithm>
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

#include "alg_rd.cuh"


int main(int argc, char** argv) {

    long int num_samples = 1 << 23, num_repeats = 1; // defaults
    char array_bin_fn[200] = "../bin/random_array.bin";
    
    if ((argc != 1 && argc != 4)
        || (argc==4 && (sscanf(argv[1], "%ld", &num_samples) != 1 ||
                        sscanf(argv[2], "%ld", &num_repeats) != 1 ||
                        sscanf(argv[3], "%s", array_bin_fn) != 1))) {
        std::cerr << APPNAME << " Bad arguments!\n";
        std::cerr << APPNAME << " Usage: " << argv[0]
                  << " [num_samples num_repeats array_bin_fn] ->"
                  << " Output: Mis/s MAE MRE\n";
        std::cerr << APPNAME << " Where: num_samples = number of samples "
                  << "in the 1D array to run this on (up to 1Gi)\n";
        std::cerr << APPNAME << " Where: num_repeats = number of repetitions "
                  << "to measure the run timing performance\n";
        std::cerr << APPNAME << " Where: array_bin_fn = array of inputs in "
                  << "binary to read 1D input data from\n";
        std::cerr << APPNAME << " Where: Mis/s = Mebi samples per second; "
                  << "MAE = max. abs. error; MRE = max. rel. error\n";
        return EXIT_FAILURE;
    }

    gpufilter::Vector<SAMPLETYPE, ORDER+1> iir_weights;
    SAMPLETYPE gaussian_sigma = 4.0;
    gpufilter::weights(gaussian_sigma, iir_weights);

    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
    if (num_repeats == 1) { // running for debugging
        std::cout << gpufilter::get_cuda_device_properties();
        std::cout << "IIR weights: " << iir_weights << "\n";
    }

    SAMPLETYPE *cpu_arr = new SAMPLETYPE[num_samples];
    SAMPLETYPE *gpu_arr = new SAMPLETYPE[num_samples];

    std::ifstream in_file(array_bin_fn, std::ios::binary);
    in_file.read(reinterpret_cast<char*>(cpu_arr),
                 sizeof(SAMPLETYPE)*num_samples);
    in_file.close();

    memcpy(gpu_arr, cpu_arr, sizeof(SAMPLETYPE) * num_samples);

    if (num_repeats == 1) { // running for debugging
        std::cout << std::fixed << std::flush;
        gpufilter::print_array(cpu_arr, 32, "cpu bi=0 wi=0:");
    }

    for (long int i=0; i<num_samples/WARPSIZE; ++i) {
        cpu_arr[i*WARPSIZE+0] *= iir_weights[0];
        for (int j=1; j<WARPSIZE; ++j) {
            cpu_arr[i*WARPSIZE+j] = cpu_arr[i*WARPSIZE+j] * iir_weights[0]
                - cpu_arr[i*WARPSIZE+j-1] * iir_weights[1];
        }
    }

    gpufilter::alg_rd_gpu<SAMPLETYPE,ORDER>(
        gpu_arr, num_samples, num_repeats, iir_weights);

    SAMPLETYPE max_abs_err, max_rel_err;
    gpufilter::check_cpu_reference(cpu_arr, gpu_arr, num_samples,
                                   max_abs_err, max_rel_err);

    if (num_repeats == 1) // running for debugging
        std::cout << APPNAME << " [max-absolute-error] [max-relative-error]:";

    std::cout << " " << std::scientific << max_abs_err << " "
              << std::scientific << max_rel_err << "\n";

    if (num_repeats == 1) { // running for debugging
        std::cout << std::fixed << std::flush;
        gpufilter::print_array(gpu_arr, 32, "gpu bi=0 wi=0:");
        gpufilter::print_array(cpu_arr, 32, "cpu bi=0 wi=0:");
    }

    if (cpu_arr) delete [] cpu_arr;
    if (gpu_arr) delete [] gpu_arr;

    return EXIT_SUCCESS;

}
