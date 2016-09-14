/**
 *  @file alloc.h
 *  @brief Device memory allocator definition and implementation
 *  @author Rodolfo Lima
 *  @author Andre Maximo
 *  @date February, 2011
 *  @copyright The MIT License
 */

#ifndef ALLOC_H
#define ALLOC_H

//== INCLUDES ==================================================================

#include <sstream>
#include <stdexcept>

#include "error.h"

//== NAMESPACES ================================================================

namespace gpufilter {

//== IMPLEMENTATION ============================================================

/**
 *  @ingroup utils
 *  @brief Allocates a new memory space in the GPU
 *
 *  This function allocates device (GPU) memory space.
 *
 *  @param[in] elements Number of elements to allocate
 *  @return Pointer to the device memory allocated
 *  @tparam T Memory values type
 */
template< class T >
T *cuda_new( const size_t& elements ) {
    T *ptr = 0;

    cudaError_t e = cudaMalloc((void **)&ptr, elements*sizeof(T));
    if( (e != cudaSuccess) || (ptr == 0) )
        throw std::runtime_error("Memory allocation error");
    check_cuda_error("Memory allocation error");

    return ptr;
}

/**
 *  @ingroup utils
 *  @brief Allocates a new memory space in the GPU
 *
 *  This function allocates device (GPU) memory space.
 *
 *  @param[in] elements Number of elements to allocate
 *  @return Pointer to the device memory allocated
 *  @tparam T Memory values type
 */
template< class T >
T *cuda_new( size_t& pitch,
             const size_t& width,
             const size_t& height ) {
    T *ptr = 0;

    cudaError_t e = cudaMallocPitch((void **)&ptr, &pitch, width*sizeof(T), height);
    check_cuda_error("Memory allocation error");
    if( (e != cudaSuccess) || (ptr == 0) )
        throw std::runtime_error("Memory allocation error");

    return ptr;
}

/**
 *  @ingroup utils
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

//== CLASS DEFINITION ==========================================================

/**
 *  @struct cuda_deleter alloc.h
 *  @ingroup utils
 *  @brief CUDA deleter struct to encapsulate cuda_delete()
 */
struct cuda_deleter
{
    // @brief delete operator functor
    template <class T>
    void operator()(T *ptr) const {
        cuda_delete(ptr);
    }
};

/**
 *  @class cuda_allocator alloc.h
 *  @ingroup utils
 *  @brief CUDA standard allocator to encapsulate cuda_new() and cuda_delete()
 *  @tparam T Memory values type
 */
template <class T>
class cuda_allocator : public std::allocator<T>
{

public:

    typedef typename std::allocator<T>::pointer pointer;
    typedef typename std::allocator<T>::size_type size_type;

    // @brief allocate n elements
    pointer allocate(size_type n,
                     std::allocator<void>::const_pointer hint=0) {
        return cuda_new<T>(n);
    }

    // @brief deallocate a pointer
    void deallocate(pointer ptr,
                    size_type n) {
        cuda_delete(ptr);
    }

    // @brief null construct
    void construct(pointer ptr,
                   const T &val) {
        // do nothing
    }

    // @brief null destroy
    void destroy(pointer ptr) {
        // do nothing
    }
};

//==============================================================================
} // namespace gpufilter
//==============================================================================
#endif // ALLOC_H
//==============================================================================
