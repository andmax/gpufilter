/**
 *  @file thrustps.cu
 *  @brief Thrust prefix sum
 *  @author Andre Maximo
 *  @date Jul, 2019
 *  @copyright The MIT License
 */

#include <cstdlib>

#include <iostream>
#include <fstream>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>

#include <util/util.h>
#include <util/timer.h>
#include <util/gaussian.h>
#include <util/recfilter.h>
#include <util/alg0_xd_cpu.h>

#define ORDER 1 // it may ONLY be one
#define APPNAME "[thurstps]"

typedef long int T;


int main(int argc, char** argv) {

    long int num_samples = 1 << 15, num_repeats = 100; // defaults
    char array_bin_fn[200] = "../bin/random_array_double.bin";
    
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

    gpufilter::Vector<T, ORDER+1> iir_weights;
    iir_weights[0] = 1.;
    iir_weights[1] = -1.;

    T *h_in = new T[num_samples];
    T *h_out = new T[num_samples];

    T *d_in = NULL;
    T *d_out = NULL;

    std::ifstream in_file(array_bin_fn, std::ios::binary);
    in_file.read(reinterpret_cast<char*>(h_in),
                 sizeof(T)*num_samples);
    in_file.close();
    
    cudaMalloc(&d_in, sizeof(T) * num_samples);
    cudaMalloc(&d_out, sizeof(T) * num_samples);

    cudaMemcpy(d_in, h_in, sizeof(T) * num_samples,
               cudaMemcpyHostToDevice);

    thrust::device_ptr<T> dd_in(d_in);
    thrust::device_ptr<T> dd_out(d_out);

    thrust::inclusive_scan(dd_in, dd_in + num_samples, dd_out);

    gpufilter::base_timer &timer_total = gpufilter::timers.gpu_add(
        APPNAME, num_samples, "is");

    for (int r = 0; r < num_repeats; ++r) {

        thrust::inclusive_scan(dd_in, dd_in + num_samples, dd_out);

    }

    timer_total.stop();

    cudaMemcpy(h_out, d_out, sizeof(T) * num_samples,
               cudaMemcpyDeviceToHost);

    std::size_t proc_samples = timer_total.data_size()*num_repeats;
    double time_sec_inv_mebi = timer_total.elapsed()*1024*1024;
    std::cout << std::fixed << proc_samples/time_sec_inv_mebi << std::flush;

    gpufilter::recursive_1d<0,true,ORDER>(h_in, num_samples, iir_weights);

    T max_abs_err, max_rel_err;
    gpufilter::check_cpu_reference(h_in, h_out, num_samples,
                                   max_abs_err, max_rel_err);

    std::cout << " " << std::scientific << max_abs_err << " "
              << std::scientific << max_rel_err << "\n";

    if (h_in) delete [] h_in;
    if (h_out) delete [] h_out;
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);

    return EXIT_SUCCESS;

}
