/**
 *  @file linalg.h
 *  @brief Linear algebra definition
 *  @author Rodolfo Lima
 *  @author Andre Maximo
 *  @date December, 2011
 *  @copyright The MIT License
 */

#ifndef LINALG_H
#define LINALG_H

//== INCLUDES ==================================================================

#include <cassert>

#include <vector>

//== DEFINES ===================================================================

#if !defined(HOSTDEV)
#   if defined(__CUDA_ARCH__)
#       define HOSTDEV __host__ __device__
#   else
#       define HOSTDEV
#   endif
#endif

//== NAMESPACES ================================================================

namespace gpufilter {

//== CLASS DEFINITION ==========================================================

/**
 *  @class Vector linalg.h
 *  @ingroup utils
 *  @brief Vector class
 *
 *  Vector class to represent special small vectors, such as the
 *  vector of filter weights \f$w\f$ used in forward and reverse
 *  filter computations, described in the paper ([NehabMaximo:2016] cited in
 *  alg6()).
 *
 *  @tparam T Vector value type
 *  @tparam N Number of elements
 */
template <class T, int N>
class Vector {

public:

    /**
     *  @brief Convert a vector of this class to an stl vector
     *  @return STL Vector
     */
    std::vector<T> to_vector() const;

    /**
     *  @brief Get size (number of elements) of this vector
     *  @return Vector size
     */
    HOSTDEV int size() const { return N; }

    /**
     *  @brief Access (constant) operator
     *  @param[in] i Position to access
     *  @return Value (constant reference) at given position
     */
    HOSTDEV const T &operator[](int i) const;

    /**
     *  @brief Access operator
     *  @param[in] i Position to access
     *  @return Value at given position
     */
    HOSTDEV T &operator[](int i);

    /**
     *  @brief Pointer-access operator (constant)
     *  @return Pointer to the first element of this vector
     */
    HOSTDEV operator const T *() const { return &m_data[0]; }

    /**
     *  @brief Pointer-access operator
     *  @return Pointer to the first element of this vector
     */
    HOSTDEV operator T *() { return &m_data[0]; }

    /**
     *  @brief Get a sub-vector of this vector
     *  @param[in] beg Position to start getting values to the end
     *  @return Vector of value from given index to the end
     *  @tparam R Size of the sub-vector to be returned
     */
    template <int R>
    HOSTDEV Vector<T,R> subv(int beg) const
    {
        assert(beg+R <= N);
        Vector<T,R> v;
        for(int i=beg; i<beg+R; ++i)
            v[i-beg] = m_data[i];
        return v;
    }

private:

    T m_data[N]; ///< Vector values

};

//== CLASS DEFINITION ==========================================================

/**
 *  @class Matrix linalg.h
 *  @ingroup utils
 *  @brief Matrix class
 *
 *  Matrix class to represent special small matrices, such as
 *  \f$A_{FB}\f$ and \f$A_{RB}\f$ described in the paper
 *  ([NehabMaximo:2016] cited in alg6()).
 *
 *  @tparam T Matrix value type
 *  @tparam M Number of rows
 *  @tparam N Number of columns (default: N=M)
 */
template <class T, int M, int N=M>
class Matrix {

public:

    /**
     *  @brief Get number of rows of this matrix
     *  @return Number of rows
     */
    HOSTDEV int rows() const { return M; }

    /**
     *  @brief Get number of columns of this matrix
     *  @return Number of columns
     */
    HOSTDEV int cols() const { return N; }

    /**
     *  @brief Access (constant) operator
     *  @param[in] i Row of the matrix to access
     *  @return Vector (constant) of the corresponding row
     */
    HOSTDEV const Vector<T,N> &operator[](int i) const;

    /**
     *  @brief Access operator
     *  @param[in] i Row of the matrix to access
     *  @return Vector of the corresponding row
     */
    HOSTDEV Vector<T,N> &operator[](int i);

    /**
     *  @brief Get column j of this matrix
     *  @param[in] j Index of column to get
     *  @return Column j as a vector
     */
    HOSTDEV Vector<T,M> col(int j) const;

    /**
     *  @brief Set column j of this matrix
     *  @param[in] j Index of column to set
     *  @param[in] c Vector to place in matrix column
     */
    HOSTDEV void set_col(int j, const Vector<T,M> &c);

    /**
     *  @overload void set_col(int j, const T *c)
     */
    HOSTDEV void set_col(int j, const T *c);

private:

    Vector<T,N> m_data[M]; ///< Matrix values

};

//==============================================================================
} // namespace gpufilter
//==============================================================================

//== IMPLEMENTATION ============================================================

#include "linalg.hpp"

//==============================================================================
#endif // LINALG_H
//==============================================================================
