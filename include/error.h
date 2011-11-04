/**
 *  @file error.h
 *  @brief Device Error Management definition
 *  @author Rodolfo Lima
 *  @author Andre Maximo
 *  @date February, 2011
 */

#ifndef ERROR_H
#define ERROR_H

//== INCLUDES =================================================================

#include <sstream>
#include <iostream>
#include <stdexcept>

#include <cuda_runtime.h>

//== NAMESPACES ===============================================================

namespace gpufilter {

//=== IMPLEMENTATION ==========================================================

/**
 *  @ingroup utils
 *  @{
 */

/**
 *  @brief Check error in device
 *
 *  This function checks if there is a device error.
 *
 *  @param[in] msg Message to appear in case there is a device error
 */
inline void check_cuda_error( const std::string &msg ) {
    cudaError_t err = cudaGetLastError();
    if( err != cudaSuccess ) {
        if( msg.empty() )
            throw std::runtime_error(cudaGetErrorString(err));
        else {
            std::stringstream ss;
            ss << msg << ": " << cudaGetErrorString(err);
            throw std::runtime_error(ss.str().c_str());
        }
    }
}

/**
 *  @brief Print device computation errors
 *
 *  This function prints values computed wrong by the device (GPU)
 *  considering a threshold (1e-5).  This function is for debug only.
 *
 *  @param[in] ref Reference (CPU) values
 *  @param[in] res Result (GPU) values
 *  @param[in] w Image width
 *  @param[in] h Image height
 *  @param[in] thrld Error threshold (default: 1e-5)
 *  @tparam T1 Values type used in the GPU
 *  @tparam T2 Values type used in the CPU
 */
template< class T1, class T2 >
void print_errors( const T1 *ref,
                   const T2 *res,
                   const int& w,
                   const int& h,
                   const T1& thrld = (T1)1e-5 ) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            T1 a = (T1)(res[i*w+j]) - ref[i*w+j];
            if( a < (T1)0 ) a = -a;
            if( a > thrld )
                std::cout << "Row: " << i << " Col: " << j << " Err: " << a
                          << " Ref: " << ref[i*w+j] << " Res: " << res[i*w+j] << "\n";
        }
    }
}

/**
 *  @overload
 *
 *  @param[in] ref Reference (CPU) values
 *  @param[in] res Result (GPU) values
 *  @param[in] ne Number of elements in vector
 *  @param[in] thrld Error threshold (default: 1e-5)
 *  @tparam T1 Values type used in the GPU
 *  @tparam T2 Values type used in the CPU
 */
template< class T1, class T2 >
void print_errors( const T1 *ref,
                   const T2 *res,
                   const int& ne,
                   const T1& thrld = (T1)1e-5 ) {
    for (int i = 0; i < ne; i++) {
        T1 a = (T1)(res[i]) - ref[i];
        if( a < (T1)0 ) a = -a;
        if( a > thrld )
            std::cout << "Elem: " << i << " Err: " << a
                      << " Ref: " << ref[i] << " Res: " << res[i] << "\n";
    }
}

/**
 *  @brief Print device vector values
 *
 *  This function prints values from a vector stored in the
 *  GPU.  This function is for debug only.
 *
 *  @param[in] res Result (GPU) values
 *  @param[in] ne Number of elements in vector
 *  @tparam T Values type used in the GPU
 */
template< class T >
void print_vector( const T *res,
                   const int& ne ) {
    std::cout << res[ 0 ] << std::flush;
    for (int i = 1; i < ne; i++)
        std::cout << " " << res[ i ] << std::flush;
    std::cout << "\n";
}

/**
 *  @}
 */

//=============================================================================
} // namespace gpufilter
//=============================================================================
#endif // ERROR_H
//=============================================================================
//vi: ai sw=4 ts=4
