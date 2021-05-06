/**
 *  @file gen_random_samples.cpp
 *  @brief Generates random array of samples
 *  @author Andre Maximo
 *  @date Jan, 2018
 *  @copyright The MIT License
 */
    
#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <typeinfo>

#ifdef DOUBLE
typedef long int T;
#else
typedef float T;
#endif


int main() {

    long int num_samples = 1 << 30;

    std::cout << "Allocating " << num_samples << " (1Gi) samples"
              << " of " << typeid(T).name() << " type as an"
              << " array in the CPU..." << std::flush;

    T *h_in = new T[num_samples];

    std::cout << " done!\nGenerating mersenne-twister-random data..."
              << std::flush;

    std::mt19937 mt_rand; // mersenne twister random generator

    mt_rand.seed(1234);

    // generating mersenne-twister-random data with fixed seed on h_in
    for (int i = 1; i < num_samples; ++i) {
#ifdef DOUBLE
        h_in[i] = mt_rand();
#else
        h_in[i] = mt_rand() / (T)mt_rand.max();
#endif
    }

    // first element has to be zero due to cub inclusive scan limitation
    h_in[0] = 0;

    std::cout << " done!\nArray [0,32):" << std::flush;

    for (int i = 0; i < 32; ++i) {
        std::cout << " " << h_in[i] << std::flush;
    }

#ifdef DOUBLE
    std::string random_array_fn = "../bin/random_array_double.bin";
#else
    std::string random_array_fn = "../bin/random_array.bin";
#endif

    std::cout << "\nWriting array to: " << random_array_fn
              << " ..." << std::flush;

    std::ofstream out_file(random_array_fn, std::ios::binary);

    out_file.write(reinterpret_cast<char*>(h_in),
                   sizeof(T)*num_samples);

    out_file.close();

    std::cout << " done!\n";

    return 0;

}
