/**
 *  @file dvector.h
 *  @brief Device Vector Class definition
 *  @author Rodolfo Lima
 *  @date February, 2011
 */

#ifndef DVECTOR_H
#define DVECTOR_H

//== INCLUDES =================================================================

#include <vector>

#include <alloc.h>

//== NAMESPACES ===============================================================

namespace gpufilter {

//== CLASS DEFINITION =========================================================

/**
 *  @class dvector dvector.h
 *  @ingroup utils
 *  @brief Device Vector class
 *
 *  Device vector is a STL-based vector in the GPU memory.
 *
 *  @tparam T Device vector values type
 */
template< class T >
class dvector {

public:

    /**
     *  Constructor
     *  @param[in] that Host (STL) Vector data (non-converted) to be copied into this object
     */
    explicit dvector( const std::vector<T>& that ) : m_size(0), m_capacity(0), m_data(0) {
        *this = that;
    }

    /**
     *  Constructor
     *  @param[in] data Vector data to be copied into this object
     *  @param[in] size Vector data size
     */
    dvector( const T *data,
             const size_t& size ) : m_size(0), m_capacity(0), m_data(0) {
        copy_from( data, size );
    }

    /**
     *  Constructor
     *  @param[in] data Vector data in 2D to be copied into this object
     *  @param[in] h_data Height of the vector data
     *  @param[in] w_data Width of the vector data
     *  @param[in] h Height of the vector data in device memory
     *  @param[in] w Width of the vector data in device memory
     */
    dvector( const T *data,
             const size_t& h_data,
             const size_t& w_data,
             const size_t& h,
             const size_t& w ) : m_size(0), m_capacity(0), m_data(0) {
        copy_from( data, h_data, w_data, h, w );
    }

    /**
     *  Copy Constructor
     *  @param[in] that Copy that object to this object
     */
    dvector( const dvector<T>& that ) : m_size(0), m_capacity(0), m_data(0) {
        *this = that;
    }

    /**
     *  Default Constructor
     *  @param[in] size Vector data size
     */
    dvector( const size_t& size = 0 ) : m_size(0), m_capacity(0), m_data(0) {
        resize(size);
    }

    /**
     *  Destructor
     */
    ~dvector() {
        cuda_delete(m_data);
        m_data = 0;
        m_capacity = 0;
        m_size = 0;
    }

    /**
     *  @brief Resize this vector
     *  @param[in] size The new vector size
     */
    void resize( const size_t& size ) {
        if( size > m_capacity )
        {
            cuda_delete(m_data);
            m_data = 0;
            m_capacity = 0;
            m_size = 0;

            m_data = cuda_new<T>(size);
            m_capacity = size;
            m_size = size;
        }
        else
            m_size = size;
    }

    /**
     *  @brief Clear this vector
     */
    void clear() {
        m_size = 0;
    }

    /**
     *  @brief Read/write operator
     *  @param[in] idx Index of vector value
     *  @return Vector value at index
     */
    T operator [] ( const int& idx ) const {
        T value;
        cudaMemcpy(&value, data()+idx, sizeof(T), cudaMemcpyDeviceToHost);
        return value;
    }

    /**
     *  @brief Assign operator
     *  @param[in] that Device vector to copy from
     *  @return This device vector with assigned values
     */
    dvector& operator = ( const dvector<T>& that ) {
        resize(that.size());
        cudaMemcpy(data(), that.data(), size()*sizeof(T), cudaMemcpyDeviceToDevice);
        cuda_error("Error during memcpy from device to device");
        return *this;
    }

    /**
     *  @brief Assign operator
     *  @param[in] that Host (STL) Vector to copy from
     *  @return This device vector with assigned values
     */
    dvector& operator = ( const std::vector<T>& that ) {
        resize(that.size());
        cudaMemcpy(data(), &that[0], size()*sizeof(T), cudaMemcpyHostToDevice);
        cuda_error("Error during memcpy from host to device");
        return *this;
    }

    /**
     *  @brief Copy values from this vector to a host (CPU) vector
     *  @param[out] data Host Vector to copy values to
     *  @param[in] s Maximum number of elements to copy
     */
    void copy_to( T *data,
                  const size_t& s ) const {
        cudaMemcpy(data, this->data(), std::min(size(),s)*sizeof(T), cudaMemcpyDeviceToHost);
        cuda_error("Error during memcpy from device to host");
    }

    /**
     *  @brief Copy values from this vector in 2D to a host (CPU) vector
     *  @param[out] data Host Vector to copy values to
     *  @param[in] h Height of the vector data in device memory
     *  @param[in] w Width of the vector data in device memory
     *  @param[in] h_data Height of the vector data to copy values to
     *  @param[in] w_data Width of the vector data to copy values to
     */
    void copy_to( T *data,
                  const size_t& h,
                  const size_t& w,
                  const size_t& h_data,
                  const size_t& w_data ) const {
        cudaMemcpy2D(data, w_data*sizeof(T), this->data(), w*sizeof(T), w_data*sizeof(T), h_data, cudaMemcpyDeviceToHost);
        cuda_error("Error during memcpy from device to host");
    }

    /**
     *  @brief Copy values from a host (CPU) vector to this vector
     *  @param[in] data Vector data to be copied into this object
     *  @param[in] size Vector data size
     */
    void copy_from( const T *data,
                    const size_t& size ) {
        resize(size);
        cudaMemcpy(this->data(), const_cast<T *>(data),
                   size*sizeof(T), cudaMemcpyHostToDevice);
        cuda_error("Error during memcpy from host to device");
    }

    /**
     *  @brief Copy values from a 2D host (CPU) vector to this vector
     *  @param[in] data Vector data in 2D to be copied into this object
     *  @param[in] h_data Height of the vector data
     *  @param[in] w_data Width of the vector data
     *  @param[in] h Height of the vector data in device memory
     *  @param[in] w Width of the vector data in device memory
     */
    void copy_from( const T *data,
                    const size_t& h_data,
                    const size_t& w_data,
                    const size_t& h,
                    const size_t& w ) {
        resize(w*h);
        cudaMemcpy2D(this->data(), w*sizeof(T),
                     const_cast<T *>(data), w_data*sizeof(T), w_data*sizeof(T),
                     h_data, cudaMemcpyHostToDevice);
        cuda_error("Error during memcpy from host to device");
    }

    /**
     *  @brief Check if this vector is empty
     *  @return True if this vector is empty
     */
    bool empty() const { return size()==0; }

    /**
     *  @brief Size of this vector
     *  @return Vector size
     */
    size_t size() const { return m_size; }

    /**
     *  @brief Data in this vector
     *  @return Vector data
     */
    T *data() { return m_data; }

    /**
     *  @overload const T *data() const
     *  @return Constant vector data
     */
    const T *data() const { return m_data; }

    /**
     *  @brief Get last element of the vector
     *  @return Last element of this vector
     */
    T back() const { return operator[](size()-1); }

    /**
     *  @brief Address access operator
     *  @return Pointer to vector data
     */
    operator T* () { return data(); }

    /**
     *  @brief Address access operator
     *  @return Constant pointer to vector data
     */
    operator const T* () const { return data(); }

    /**
     *  @brief Swap vector values
     *  @param[in,out] a Vector to be swapped
     *  @param[in,out] b Vector to be swapped
     */
    friend void swap( dvector<T>& a,
                      dvector<T>& b ) {
        std::swap(a.m_data, b.m_data);
        std::swap(a.m_size, b.m_size);
        std::swap(a.m_capacity, b.m_capacity);
    }

private:

    T *m_data; ///< Vector data
    size_t m_size, m_capacity; ///< Vector size and capacity

};

//== IMPLEMENTATION ===========================================================

/**
 *  @relates dvector
 *  @brief Copy to the CPU a vector in the GPU
 *
 *  This function copies a device vector (GPU) to a host vector (CPU).
 *
 *  @param[in] d_vec Pointer to the device vector (in the GPU memory)
 *  @param[in] len Length of the device vector
 *  @return Host vector (in the CPU memory) as a STL vector
 *  @tparam T Vector values type
 */
template< class T >
std::vector<T> to_cpu( const T *d_vec,
                       unsigned len ) {
    std::vector<T> out;
    out.resize(len);

    cudaMemcpy(&out[0], d_vec, len*sizeof(T), cudaMemcpyDeviceToHost);
    cuda_error("Error during memcpy from device to host");

    return out;
}

/**
 *  @relates dvector
 *  @overload
 *
 *  @param[in] v Device vector (in the GPU memory)
 *  @return Host vector (in the CPU memory) as a STL vector
 *  @tparam T Vector values type
 */
template< class T >
std::vector<T> to_cpu( const dvector<T>& v ) {
    return to_cpu(v.data(), v.size());
}

//=============================================================================
} // namespace gpufilter
//=============================================================================
#endif // DVECTOR_H
//=============================================================================
//vi: ai sw=4 ts=4
