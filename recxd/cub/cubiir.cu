/**
 *  @file cubiir.cu
 *  @brief CUB scan-based IIR filter
 *  @author Andre Maximo
 *  @date Jan, 2018
 *  @copyright The MIT License
 */

#include <cstdlib>

#include <iostream>
#include <fstream>

#include <cub/device/device_scan.cuh>
#include <test/test_util.h>

#include <util/util.h>
#include <util/gaussian.h>
#include <util/recfilter.h>
#include <util/alg0_xd_cpu.h>

#define ORDER 1 // it may ONLY be one
#define APPNAME "[cubiir]"

template <typename T>
struct RecOp
{
    const T w0, w1;
    
    RecOp(const T& _w0, const T& _w1) : w0(_w0), w1(_w1) { }
    
    __host__ __device__  __forceinline__
    T operator()(const T &a, const T &x) const {
        return w0 * x - w1 * a;
    }
};

int main(int argc, char** argv) {

    // This is based on simple1Gis.cu and it inherits the same problems
    // it seems the recursive filter as CUB inclusive scan does not work
    // it returns high MRE (max. rel. error) with incredible performance
    // even better than memcpy (45 x 42 Gis/s for 1Gis)

    long int num_samples = 1 << 23, num_repeats = 100; // defaults
    
    if ((argc != 1 && argc != 3)
        || (argc==3 && (sscanf(argv[1], "%ld", &num_samples) != 1 ||
                        sscanf(argv[2], "%ld", &num_repeats) != 1))) {
        std::cerr << APPNAME << " Bad arguments!\n";
        std::cerr << APPNAME << " Usage: " << argv[0]
                  << " [num_samples num_repeats] ->"
                  << " Output: Mis/s MAE MRE\n";
        std::cerr << APPNAME << " Where: num_samples = number of samples "
                  << "in the 1D array to run this on (up to 1Gi)\n";
        std::cerr << APPNAME << " Where: num_repeats = number of repetitions "
                  << "to measure the run timing performance\n";
        std::cerr << APPNAME << " Where: Mis/s = Mebi samples per second; "
                  << "MAE = max. abs. error; MRE = max. rel. error\n";
        return EXIT_FAILURE;
    }

    float gaussian_sigma = 4.f;
    gpufilter::Vector<float, ORDER+1> iir_weights;
    gpufilter::weights(gaussian_sigma, iir_weights);
    RecOp<float> rec_op(iir_weights[0], iir_weights[1]);

    float *h_in = new float[num_samples];
    float *h_out = new float[num_samples];

    float *d_in = NULL;
    float *d_out = NULL;
    
    cudaMalloc(&d_in, sizeof(float) * num_samples);
    cudaMalloc(&d_out, sizeof(float) * num_samples);

    std::ifstream in_file("../bin/random_array.bin", std::ios::binary);
    in_file.read(reinterpret_cast<char*>(h_in),
                 sizeof(float)*num_samples);
    in_file.close();

    cudaMemcpy(d_in, h_in, sizeof(float) * num_samples,
               cudaMemcpyHostToDevice);

    gpufilter::recursive_1d<0,true,ORDER>(h_in, num_samples, iir_weights);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes,
                                   d_in, d_out, rec_op, num_samples);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes,
                                   d_in, d_out, rec_op, num_samples);
    
    cudaMemcpy(h_out, d_out, sizeof(float) * num_samples,
               cudaMemcpyDeviceToHost);

    float max_abs_err, max_rel_err;
    gpufilter::check_cpu_reference(h_in, h_out, num_samples,
                                   max_abs_err, max_rel_err);

    GpuTimer cub_timer;
    cub_timer.Start();
    for (int i = 0; i < num_repeats; ++i) {
        cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes,
                                       d_in, d_out, rec_op, num_samples);
    }
    cub_timer.Stop();
    double elapsed_sec = cub_timer.ElapsedMillis() / (double)1000.;

    std::cout << std::fixed
              << (num_samples*num_repeats)/(elapsed_sec*1024*1024)
              << std::flush;

    std::cout << " " << std::scientific << max_abs_err << " "
              << std::scientific << max_rel_err << "\n";

    if (h_in) delete [] h_in;
    if (h_out) delete [] h_out;
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    if (d_temp_storage) cudaFree(d_temp_storage);

    return EXIT_SUCCESS;

}
