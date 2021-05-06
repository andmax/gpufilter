/**
 *  @file simple8s.cu
 *  @brief Simple IIR filter using CUB on 8 samples
 *  @author Andre Maximo
 *  @date Jan, 2018
 *  @copyright The MIT License
 */

#include <iostream>

#include <cub/device/device_scan.cuh>

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
    
    float gaussian_sigma = 4.f;
    gpufilter::Vector<float, ORDER+1> iir_weights;
    gpufilter::weights(gaussian_sigma, iir_weights);
    std::cout << "Recursive filter weights: " << iir_weights << "\n";

    RecOp<float> rec_op(iir_weights[0], iir_weights[1]);

    int num_samples = 8;
    float h_in[8] = { 0., 8., 6., 7., 5., 3., 0., 9. };

    gpufilter::print_array(h_in, num_samples, "h_in");

    float *h_out = new float[num_samples];

    memcpy(h_out, h_in, sizeof(float) * num_samples);

    gpufilter::recursive_1d<0,true,ORDER>(h_out, num_samples, iir_weights);

    gpufilter::print_array(h_out, num_samples, "cpu reference:");

    float *d_in = NULL;
    float *d_out = NULL;
    
    cudaMalloc(&d_in, sizeof(float) * num_samples);
    cudaMalloc(&d_out, sizeof(float) * num_samples);

    cudaMemcpy(d_in, h_in, sizeof(float) * num_samples, cudaMemcpyHostToDevice);

    // cub dry-run: determining temporary storage
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, rec_op, num_samples);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // cub run
    cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, rec_op, num_samples);

    cudaMemcpy(h_out, d_out, sizeof(float) * num_samples, cudaMemcpyDeviceToHost);

    gpufilter::print_array(h_out, num_samples, "gpu inc. scan:");

    if (h_out) delete [] h_out;
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    if (d_temp_storage) cudaFree(d_temp_storage);

    return 0;

}
