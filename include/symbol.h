/**
 *  @file symbol.h
 *  @ingroup utils
 *  @brief Device Symbol Management definition
 *  @author Rodolfo Lima
 *  @date February, 2011
 */

#ifndef SYMBOL_H
#define SYMBOL_H

//== INCLUDES =================================================================

#include <string>
#include <vector>

#include <error.h>

//=== IMPLEMENTATION ==========================================================

/**
 *  @brief Copy value(s) to symbol
 *
 *  @param[in] name Name of the symbol in device (GPU)
 *  @param[in] value Value to copy to symbol
 *  @tparam T Value type used in the GPU
 */
template< class T >
void copy_to_symbol( const std::string &name,
                     const T &value ) {
    size_t size_storage;
    cudaGetSymbolSize(&size_storage, name.c_str());
    check_cuda_error("Invalid symbol '"+name+"'");

    if( sizeof(T) > size_storage )
        throw std::runtime_error("'"+name+"'"+" storage overflow");

    cudaMemcpyToSymbol(name.c_str(), &value, sizeof(T), 0, cudaMemcpyHostToDevice);
    check_cuda_error("Error copying '"+name+"' buffer to device");
}

/**
 *  @overload
 *
 *  @param[in] name Name of the symbol in device (GPU)
 *  @param[in] value Value to copy to symbol
 */
inline void copy_to_symbol( const std::string &name,
                            unsigned long value ) {
    copy_to_symbol(name, (unsigned int)value);
}

/**
 *  @overload
 *
 *  @param[in] name Name of the symbol in device (GPU)
 *  @param[in] value Value to copy to symbol
 */
inline void copy_to_symbol( const std::string &name,
                            long value ) {
    copy_to_symbol(name, (int)value);
}

/**
 *  @overload
 *
 *  @param[in] name Name of the symbol in device (GPU)
 *  @param[in] size_name Name of the symbol storing the size of the items to be copied
 *  @param[in] items Host (STL) Vector values
 *  @tparam T Vector values type used in the CPU and the GPU
 */
template< class T >
void copy_to_symbol( const std::string &name,
                     const std::string &size_name,
                     const std::vector<T> &items ) {
    size_t size_storage;
    cudaGetSymbolSize(&size_storage, name.c_str());
    check_cuda_error("Invalid symbol '"+name+"'");

    size_t size = items.size()*sizeof(T);

    if( size > size_storage )
        throw std::runtime_error("'"+name+"'"+" storage overflow");

    cudaMemcpyToSymbol(name.c_str(),&items[0], size, 0,
                       cudaMemcpyHostToDevice);
    check_cuda_error("Error copying '"+name+"' buffer to device");

    if( !size_name.empty() )
        copy_to_symbol(size_name.c_str(), items.size());
}

/**
 *  @overload
 *
 *  @param[in] name Name of the symbol in device (GPU)
 *  @param[in] items Host (STL) Vector values
 *  @tparam T Vector values type used in the CPU and the GPU
 */
template< class T >
void copy_to_symbol( const std::string &name,
                     const std::vector<T> &items ) {
    copy_to_symbol(name, "", items);
}

//=============================================================================
#endif // SYMBOL_H
//=============================================================================
//vi: ai sw=4 ts=4
