/**
 *  @file memcpy.cu
 *  @brief Mem-copy upper-bound test
 *  @author Andre Maximo
 *  @date Jan, 2018
 *  @copyright The MIT License
 */

#include <cstdlib>

#include <iostream>
#include <fstream>

#include <util/util.h>
#include <util/timer.h>

#define APPNAME "[memcpy]"

#ifdef DOUBLE
typedef long int T;
#else
typedef float T;
#endif


int main(int argc, char** argv) {

    long int num_samples = 1 << 23, num_repeats = 100; // defaults
#ifdef DOUBLE
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

    gpufilter::base_timer &timer_total = gpufilter::timers.gpu_add(
        APPNAME, num_samples, "is");

    for (int i = 0; i < num_repeats; ++i) {
        cudaMemcpy(d_out, d_in, sizeof(T) * num_samples,
                   cudaMemcpyDeviceToDevice);
    }

    timer_total.stop();

    cudaMemcpy(h_out, d_out, sizeof(T) * num_samples,
               cudaMemcpyDeviceToHost);

    T max_abs_err, max_rel_err;
    gpufilter::check_cpu_reference(h_in, h_out, num_samples,
                                   max_abs_err, max_rel_err);

    std::size_t proc_samples = timer_total.data_size()*num_repeats;
    double time_sec_inv_mebi = timer_total.elapsed()*1024*1024;
    std::cout << std::fixed << proc_samples/time_sec_inv_mebi << std::flush;

    std::cout << " " << std::scientific << max_abs_err << " "
              << std::scientific << max_rel_err << "\n";

    if (h_in) delete [] h_in;
    if (h_out) delete [] h_out;
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);

    return EXIT_SUCCESS;

}
