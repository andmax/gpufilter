/**
 *  @file simple7s.cu
 *  @brief Simple IIR filter using CUB on 7 samples
 *  @author Andre Maximo
 *  @date Jan, 2018
 *  @copyright The MIT License
 */

#include <iostream>

#include <cub/device/device_scan.cuh>

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

template <typename T>
void print_array(const T a[], const int& n, const char* t) {
    std::cout << t << std::flush;
    for (int i = 0; i < n; ++i) {
        std::cout << " " << a[i] << std::flush;
    }
    std::cout << "\n";
}

int main(int argc, char** argv) {
    
    int num_samples = 7;
    float h_in[7] = { 8., 6., 7., 5., 3., 0., 9. };
    float h_out[7];
    float *d_in = NULL;
    float *d_out = NULL;
    RecOp<float> rec_op(0.5, 0.5);

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

    print_array(h_in, num_samples, "h_in");
    print_array(h_out, num_samples, "h_out");
    
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    if (d_temp_storage) cudaFree(d_temp_storage);

    return 0;
    
}
