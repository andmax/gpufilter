/**
 *  @file alloc.h
 *  @brief Device Memory Allocator definition
 *  @author Rodolfo Lima
 *  @date February, 2011
 */

#ifndef ALLOC_H
#define ALLOC_H

//== INCLUDES =================================================================

#include <sstream>
#include <stdexcept>

#include <error.h>

//== NAMESPACES ===============================================================

namespace gpufilter {

//== IMPLEMENTATION ===========================================================

/**
 *  @ingroup utils
 *  @{
 */

/**
 *  @brief Allocates a new memory space in the GPU
 *
 *  This function allocates device (GPU) memory space.
 *
 *  @param[in] elements Number of elements to allocate
 *  @return Pointer to the device memory allocated
 *  @tparam T Memory values type
 */
template< class T >
T *cuda_new( size_t elements ) {
    T *ptr = 0;

    cudaMalloc((void **)&ptr, elements*sizeof(T));
    check_cuda_error("Memory allocation error");
    if( ptr == 0 )
        throw std::runtime_error("Memory allocation error");

    return ptr;
}

/**
 *  @brief Deallocates a memory space in the GPU
 *
 *  This function deletes device (GPU) memory space.
 *
 *  @param[in] d_ptr Device pointer (in the GPU memory)
 *  @tparam T Memory values type
 */
template< class T >
void cuda_delete( T *d_ptr ) {
    cudaFree((void *)d_ptr);
    check_cuda_error("Error freeing memory");
}

/**
 *  @}
 */

//=============================================================================
} // namespace gpufilter
//=============================================================================
#endif // ALLOC_H
//=============================================================================
//vi: ai sw=4 ts=4
