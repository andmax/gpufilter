/**
 *  @file alg3_1d.cu
 *  @brief Recursive filtering algorithm 3 for 1D
 *  @author Andre Maximo
 *  @date Jan, 2018
 *  @copyright The MIT License
 */

#include <cstdlib>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <typeinfo>

#include <util/util.h>
#include <util/timer.h>
#include <util/symbol.h>
#include <util/linalg.h>
#include <util/dvector.h>
#include <util/gaussian.h>
#include <util/recfilter.h>
#include <util/alg0_xd_cpu.h>

#ifdef FWD_ONLY
#include "alg3_1d_fwd.cuh"
#else
#ifdef DO_STEP2
#include "alg3_1d_step2.cuh"
#else
#include "alg3_1d.cuh"
#endif
#endif


int main(int argc, char** argv) {

    long int num_samples = 1 << 23, num_repeats = 1; // defaults
#ifdef FWD_ONLY
    char array_bin_fn[200] = "../bin/random_array_double.bin";
#else
    char array_bin_fn[200] = "../bin/random_array.bin";
#endif
    
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

    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
    if (num_repeats == 1) { // running for debugging
        std::cout << gpufilter::get_cuda_device_properties();
        std::cout << "SAMPLE TYPE is: " << typeid(SAMPLETYPE).name() << "\n";
    }

    gpufilter::Vector<SAMPLETYPE, ORDER+1> iir_weights;
    double gaussian_sigma = 4.0;
    gpufilter::weights(gaussian_sigma, iir_weights);

#ifdef FWD_ONLY
    iir_weights[0] = 1.;
    iir_weights[1] = -1.;
#endif

    SAMPLETYPE *cpu_arr = new SAMPLETYPE[num_samples];
    SAMPLETYPE *gpu_arr = new SAMPLETYPE[num_samples];

    std::ifstream in_file(array_bin_fn, std::ios::binary);
    in_file.read(reinterpret_cast<char*>(cpu_arr),
                 sizeof(SAMPLETYPE)*num_samples);
    in_file.close();

    memcpy(gpu_arr, cpu_arr, sizeof(SAMPLETYPE) * num_samples);

#ifdef FWD_ONLY
    gpufilter::recursive_1d<0,true,ORDER>(cpu_arr, num_samples, iir_weights);
#else
    gpufilter::recursive_1d<0,true,ORDER>(cpu_arr, num_samples, iir_weights);
    gpufilter::recursive_1d<0,false,ORDER>(cpu_arr, num_samples, iir_weights);
#endif

    gpufilter::alg3_1d_gpu<SAMPLETYPE,ORDER>(
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
        /*
        gpufilter::print_array(gpu_arr, 32, "gpu bi=0 wi=0:");
        gpufilter::print_array(cpu_arr, 32, "cpu bi=0 wi=0:");
        gpufilter::print_array(gpu_arr+32, 32, "gpu bi=0 wi=1:");
        gpufilter::print_array(cpu_arr+32, 32, "cpu bi=0 wi=1:");
        gpufilter::print_array(gpu_arr+31*32, 32, "gpu bi=0 wi=31:");
        gpufilter::print_array(cpu_arr+31*32, 32, "cpu bi=0 wi=31:");
        gpufilter::print_array(gpu_arr+1024, 32, "gpu bi=1 wi=0:");
        gpufilter::print_array(cpu_arr+1024, 32, "cpu bi=1 wi=0:");
        gpufilter::print_array(gpu_arr+1024+32, 32, "gpu bi=1 wi=1:");
        gpufilter::print_array(cpu_arr+1024+32, 32, "cpu bi=1 wi=1:");
        gpufilter::print_array(gpu_arr+2048, 32, "gpu bi=2 wi=0:");
        gpufilter::print_array(cpu_arr+2048, 32, "cpu bi=2 wi=0:");
        */
        gpufilter::print_array(gpu_arr+32*1024, 32, "gpu bi=32 wi=0:");
        gpufilter::print_array(cpu_arr+32*1024, 32, "cpu bi=32 wi=0:");
        gpufilter::print_array(gpu_arr+33*1024, 32, "gpu bi=33 wi=0:");
        gpufilter::print_array(cpu_arr+33*1024, 32, "cpu bi=33 wi=0:");
        gpufilter::print_errors(cpu_arr, gpu_arr, num_samples, (long int)32);
#ifdef FWD_ONLY
        std::cout << "Running forward-only prefix sum\n";
#endif
    }

    if (cpu_arr) delete [] cpu_arr;
    if (gpu_arr) delete [] gpu_arr;

    return EXIT_SUCCESS;

}
