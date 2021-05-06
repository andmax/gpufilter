/**
 *  @file simple1Gis.cu
 *  @brief Simple IIR filter using CUB on 1 Gibi samples
 *  @author Andre Maximo
 *  @date Jan, 2018
 *  @copyright The MIT License
 */

#include <cstdlib>

#include <iostream>

#include <cub/device/device_scan.cuh>
#include <test/test_util.h>

#include <util/util.h>
#include <util/gaussian.h>
#include <util/recfilter.h>
#include <util/alg0_xd_cpu.h>

#define ORDER 1 // default filter order r=1

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

    // CUB works until sample 18 then it fails for no apparent reason
    // CUB run time is a little bit deceitful because of the 1st dry-run
    // CUB run time is bigger than memcpy (45 x 42 Gis/s for 1Gis)

    float gaussian_sigma = 4.f;
    gpufilter::Vector<float, ORDER+1> iir_weights;
    gpufilter::weights(gaussian_sigma, iir_weights);
    std::cout << "Recursive filter weights: " << iir_weights << "\n";

    RecOp<float> rec_op(iir_weights[0], iir_weights[1]);

    long int num_samples = 1 << 30; // 1Gi = 2^30

    std::cout << "Allocating 2x (I/O) " << num_samples << " (1Gi) samples"
              << " in both the CPU and the GPU..." << std::flush;

    float *h_in = new float[num_samples];
    float *h_out = new float[num_samples];

    float *d_in = NULL;
    float *d_out = NULL;
    
    cudaMalloc(&d_in, sizeof(float) * num_samples);
    cudaMalloc(&d_out, sizeof(float) * num_samples);

    std::cout << " done!\nGenerating random data..." << std::flush;

    // generating random data on h_in
    srand( 1234 );
    for (int i = 1; i < num_samples; ++i)
        h_in[i] = rand() / (float)RAND_MAX;
    // first element has to be zero due to cub inclusive scan limitation
    h_in[0] = 0.; // consider it a zero border for order 1 iir filter
    
    std::cout << " done!\nCopying random array to the GPU..." << std::flush;

    cudaMemcpy(d_in, h_in, sizeof(float) * num_samples, cudaMemcpyHostToDevice);

    std::cout << " done!\nComputing IIR fwd in-place in the CPU..." << std::flush;
    
    gpufilter::recursive_1d<0,true,ORDER>(h_in, num_samples, iir_weights);

    std::cout << " done!\nComputing IIR fwd in-place in the GPU..." << std::flush;

    // cub dry-run: determining temporary storage
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, rec_op, num_samples);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // cub run
    cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, rec_op, num_samples);
    
    std::cout << " done!\nCopying result array to the CPU..." << std::flush;

    cudaMemcpy(h_out, d_out, sizeof(float) * num_samples, cudaMemcpyDeviceToHost);

    std::cout << " done!\nChecking CPU vs. GPU results (gpufilter)..." << std::flush;

    // verify gpu solution against cpu reference
    float max_abs_err, max_rel_err;
    gpufilter::check_cpu_reference(h_in, h_out, num_samples, max_abs_err, max_rel_err);

    long int num_repeats = 100;

    std::cout << " done!\nRunning for timing measurements ("
              << num_repeats << ")..." << std::flush;

    GpuTimer cub_timer;
    cub_timer.Start();
    for (int i = 0; i < num_repeats; ++i) {
        cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, rec_op, num_samples);
    }
    cub_timer.Stop();
    double elapsed_sec = cub_timer.ElapsedMillis() / (double)1000.;

    std::cout << " done!\nTiming result: "
              << std::fixed << (num_samples*num_repeats)/(elapsed_sec*1024*1024)
              << "Mis/s\n";

    std::cout << "Comparing results (cub)... " << std::flush;

    int compare = CompareResults(h_out, h_in, num_samples);

    std::cout << " = compare result = " << (compare ? "FAIL" : "PASS") << "\n";
    if (compare) { // if compare fails
        gpufilter::print_array(h_out, 32, "gpu inc. scan (32):");
        gpufilter::print_array(h_in, 32, "cpu reference (32):");
    }

    std::cout << "Error result (max-absolute-error) (max-relative-error):";
    std::cout << " " << std::scientific << max_abs_err << " "
              << std::scientific << max_rel_err << "\n";

    std::cout << "Deallocating 2x memory for 1Gis (1 Gibi samples)"
              << " in both the CPU and the GPU.\n";

    if (h_in) delete [] h_in;
    if (h_out) delete [] h_out;
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    if (d_temp_storage) cudaFree(d_temp_storage);

    return 0;

}
