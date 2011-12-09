/**
 *  @file util.h
 *  @brief Matrix and vector utility classes to facilitate head/tail operations
 *  @author Rodolfo Lima
 *  @date December, 2011
 */

#ifndef UTIL_H
#define UTIL_H

//== INCLUDES =================================================================

#include <cassert>
#include <iostream>

//== NAMESPACES ===============================================================

namespace gpufilter {

//== CLASS DEFINITION =========================================================

/**
 *  @class Matrix util.h
 *  @ingroup utils
 *  @brief Matrix class
 *
 *  Matrix class to represent special small matrices, such as
 *  \f$A_{FB}\f$ and \f$A_{RB}\f$ described in the paper ([Nehab:2011]
 *  cited in alg5()).
 *
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam T Matrix value type
 */
template <int M, int N=M, class T=float>
class Matrix {

    /**
     *  @struct row_proxy
     *  @brief Row proxy to access matrix rows
     *
     *  Proxies are used to enable checking row index range (hopefully
     *  the compiler will optimize them out).
     */
    struct row_proxy
    {
        /**
         *  Constructor
         *  @param[in] row Pointer to the matrix row array
         */
        row_proxy( T *row ) : m_row(row) { }

        /**
         *  @brief Construct an STL-based vector from the matrix row
         *  @return STL-based vector
         */
        std::vector<T> to_vector() const {
            return std::vector<T>(m_row, m_row+N);
        }

        /**
         *  @brief Access operator
         *  @param[in] j Position to access
         *  @return Value at given position
         */
        T& operator [] ( int j ) const {
            assert(j>=0 && j<N);
            return m_row[j];
        }

        /**
         *  @brief Address (constant) operator
         *  @return Pointer to matrix row array
         */
        T *operator & () { return m_row; }

        /**
         *  @brief Address (constant) operator
         *  @return Constant pointer to matrix row array
         */
        const T *operator & () const { return m_row; }

    private:

        T *m_row; ///< Row array pointer

    };

    /**
     *  @struct const_row_proxy
     *  @brief Row proxy (constant) to access matrix rows
     *
     *  Proxies are used to enable checking row index range (hopefully
     *  the compiler will optimize them out).
     */
    struct const_row_proxy
    {
        /**
         *  Constructor
         *  @param[in] row Constant pointer to the matrix row array
         */
        const_row_proxy( const T *row ) : m_row(row) { }

        /**
         *  @brief Construct an STL-based vector from the matrix row
         *  @return STL-based vector
         */
        std::vector<T> to_vector() const {
            return std::vector<T>(m_row, m_row+N);
        }

        /**
         *  @brief Access (constant) operator
         *  @param[in] j Position to access
         *  @return Value (constant reference) at given position
         */
        const T& operator [] ( int j ) const {
            assert(j>=0 && j<N);
            return m_row[j];
        }

    private:

        const T *m_row; ///< Row array constant pointer

    };

public:

    /**
     *  @brief Access (constant) operator
     *  @param[in] i Row of the matrix to access
     *  @return Array (constant) row proxy
     */
    const_row_proxy operator [] ( int i ) const {
        assert(i >= 0 && i < M);
        return const_row_proxy(m_data[i]);
    }

    /**
     *  @brief Access operator
     *  @param[in] i Row of the matrix to access
     *  @return Array row proxy
     */
    row_proxy operator [] ( int i ) {
        assert(i >= 0 && i < M);
        return row_proxy(m_data[i]);
    }

    /**
     *  @brief Output stream operator
     *  @param[out] out Output stream
     *  @param[in] m Matrix to output values from
     *  @return Output stream
     */
    friend std::ostream& operator << ( std::ostream& out,
                                       const Matrix& m ) {
        out << '[';
        for(int i=0; i<M; ++i) {
            for(int j=0; j<N; ++j) {
                out << m[i][j];
                if(j < N-1) out << ',';
            }
            if(i < M-1) out << ";\n";
        }
        return out << ']';
    }

    /**
     *  @brief Multiply operator
     *  @param[in] rhs Right-hand-side matrix to multiply
     *  @return Resulting matrix from multiplication
     *  @tparam P Number of rows of other matrix
     *  @tparam Q Number of columns of other matrix
     */
    template <int P, int Q>
    Matrix<M,Q,T> operator * ( const Matrix<P,Q,T>& rhs ) const {
        assert(N==P);
        Matrix<M,Q,T> r;
        for(int i=0; i<M; ++i) {
            for(int j=0; j<Q; ++j) {
                r[i][j] = m_data[i][0]*rhs[0][j];
                for(int k=1; k<N; ++k)
                    r[i][j] += m_data[i][k]*rhs[k][j];
            }
        }
        return r;
    }

    /**
     *  @brief Multiply-then-assign operator
     *  @param[in] val Scalar value to multilpy matrix to
     *  @return Resulting matrix from multiplication
     */
    Matrix& operator *= ( const T& val ) {
        for(int i=0; i<M; ++i)
            for(int j=0; j<N; ++j)
                m_data[i][j] *= val;
        return *this;
    }

    /**
     *  @brief Multiply operator
     *  @param[in] m Matrix to multiply
     *  @param[in] val Scalar value to multiply matrix to
     *  @return Resulting matrix from multiplication
     */
    friend Matrix operator * ( const Matrix& m,
                               const T& val ) {
        return Matrix(m) *= val;
    }

    /**
     *  @brief Multiply operator
     *  @param[in] val Scalar value to multiply matrix to
     *  @param[in] m Matrix to multiply
     *  @return Resulting matrix from multiplication
     */
    friend Matrix operator * ( const T& val,
                               const Matrix &m ) {
        return operator*(m,val);
    }

private:

    T m_data[M][N]; ///< Matrix values

};

/**
 *  @class Vector util.h
 *  @ingroup utils
 *  @brief Vector class
 *
 *  Vector class to represent special small vectors, such as the
 *  vector of filter weights \f$a_k\f$ used in forward and reverse
 *  filter computation (see equations 1 and 3 in [Nehab:2011] cited in
 *  alg5()).
 *
 *  @tparam M Number of elements
 *  @tparam T Vector value type
 */
template <int M, class T=float>
class Vector {
    
public:

    /**
     *  Constructor
     */
    Vector() { for(int i=0; i<M; ++i) m_data[i] = (T)0; }

    /**
     *  Constructor
     *  @param[in] m Matrix with 1 column to initiate this vector
     */
    Vector( const Matrix<M,1,T>& m ) {
        for (int i=0; i<M; ++i)
            m_data[i] = m[i][0];
    }

    /**
     *  @brief Access operator
     *  @param[in] i Position to access
     *  @return Value at given position
     */
    T& operator [] ( int i ) {
        assert(i >= 0 && i < M);
        return m_data[i];
    }

    /**
     *  @brief Access (constant) operator
     *  @param[in] i Position to access
     *  @return Value (constant reference) at given position
     */
    const T& operator [] ( int i ) const {
        assert(i >= 0 && i < M);
        return m_data[i];
    }

    /**
     *  @brief Output stream operator
     *  @param[out] out Output stream
     *  @param[in] v Vector to output values from
     *  @return Output stream
     */
    friend std::ostream& operator << ( std::ostream& out,
                                       const Vector& v ) {
        out << '[';
        for(int i=0; i<M; ++i) {
            out << v[i];
            if(i < M-1) out << ',';
        }
        return out << ']';
    }

private:

    T m_data[M]; ///< Vector values

};

/**
 *  @relates Matrix
 *  @brief Instantiate a matrix with zeros
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam T Matrix value type
 */
template <int M, int N, class T>
Matrix<M,N,T> zeros() {
    Matrix<M,N,T> mat;
    std::fill(&mat[0][0], &mat[M-1][N-1]+1, T());
    return mat; // I'm hoping that RVO will kick in
}

/**
 *  @relates Matrix
 *  @brief Instantiate an identity matrix
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam T Matrix value type
 */
template <int M, int N, class T>
Matrix<M,N,T> identity() {
    Matrix<M,N,T> mat;
    for(int i=0; i<M; ++i)
        for(int j=0; j<N; ++j)
            mat[i][j] = i==j ? 1 : 0;
    return mat;
}

/**
 *  @relates Matrix
 *  @brief Instantiate the transposed version of a given matrix
 *  @param[in] m Given matrix
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam T Matrix value type
 */
template <int M, int N, class T>
Matrix<N,M,T> transp( const Matrix<M,N,T>& m ) {
    Matrix<N,M,T> tm;
    for(int i=0; i<M; ++i)
        for(int j=0; j<N; ++j)
            tm[j][i] = m[i][j];
    return tm;
}

/**
 *  @relates Matrix
 *  @brief Computes the \a forward operator on matrices
 *
 *  Computes the matrix resulting from applying the \a forward
 *  operator \f$F\f$ (causal filter) given a prologue \f$R \times N\f$
 *  matrix \f$p\f$ (i.e. initial conditions) and an input \f$M \times
 *  N\f$ matrix \f$b\f$ (where M is the number of rows \f$h\f$ and N
 *  is the number of columns \f$w\f$ as described in section 2 of
 *  [Nehab:2011] cited in alg5()).  The resulting matrix is \f$M
 *  \times N\f$ and it has the same size as the input \f$b\f$.
 *
 *  @param[in] p Prologue \f$R \times N\f$ matrix (\f$R\f$ is the filter order)
 *  @param[in] b Input \f$M \times N\f$ matrix
 *  @param[in] w Filter weights with \f$R+1\f$ size (\f$R\f$ feedback coefficients)
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Number of feedback coefficients
 *  @tparam T Matrix value type
 */
template <int M, int N, int R, class T>
Matrix<M,N,T> fwd( const Matrix<R,N,T> &p,
                   const Matrix<M,N,T> &b,
                   const Vector<R+1,T> &w ) {
    Matrix<M,N,T> fb;
    for(int j=0; j<N; ++j) {
        for(int i=0; i<M; ++i) {
            fb[i][j] = b[i][j]*w[0];
            for(int k=1; k<R+1; ++k) {
                if(i-k < 0) fb[i][j] -= p[R+i-k][j]*w[k]; // use data from prologue
                else fb[i][j] -= fb[i-k][j]*w[k];
            }
        }
    }
    return fb;
}

/**
 *  @relates Matrix
 *  @brief Computes the \a forward-transposed operator on matrices
 *
 *  Computes the matrix resulting from applying the \a
 *  forward-transposed operator \f$F^{T}\f$ (causal filter on rows)
 *  given a prologue \f$M \times R\f$ matrix \f$p^{T}\f$ (i.e. initial
 *  conditions) and an input \f$M \times N\f$ matrix \f$b\f$ (where M
 *  is the number of rows \f$h\f$ and N is the number of columns
 *  \f$w\f$ as described in section 2 of [Nehab:2011] cited in
 *  alg5()).  The resulting matrix is \f$M \times N\f$ and it has the
 *  same size as the input \f$b\f$.
 *
 *  @param[in] pT Prologue transposed \f$M \times R\f$ matrix (\f$R\f$ is the filter order)
 *  @param[in] b Input \f$M \times N\f$ matrix
 *  @param[in] w Filter weights with \f$R+1\f$ size (\f$R\f$ feedback coefficients)
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Number of feedback coefficients
 *  @tparam T Matrix value type
 */
template <int M, int N, int R, class T>
Matrix<M,N,T> fwdT( const Matrix<M,R,T>& pT,
                    const Matrix<M,N,T>& b,
                    const Vector<R+1,T>& w ) {
    return transp(fwd(transp(pT), transp(b), w));
}

/**
 *  @relates Matrix
 *  @brief Computes the \a reverse operator on matrices
 *
 *  Computes the matrix resulting from applying the \a reverse
 *  operator \f$R\f$ (anticausal filter) given an epilogue \f$R \times
 *  N\f$ matrix \f$e\f$ (i.e. initial conditions) and an input \f$M
 *  \times N\f$ matrix \f$b\f$ (where M is the number of rows \f$h\f$
 *  and N is the number of columns \f$w\f$ as described in section 2
 *  of [Nehab:2011] cited in alg5()).  The resulting matrix is \f$M
 *  \times N\f$ and it has the same size as the input \f$b\f$.
 *
 *  @param[in] b Input \f$M \times N\f$ matrix
 *  @param[in] e Epilogue \f$R \times N\f$ matrix (\f$R\f$ is the filter order)
 *  @param[in] w Filter weights with \f$R+1\f$ size (\f$R\f$ feedback coefficients)
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Number of feedback coefficients
 *  @tparam T Matrix value type
 */
template <int M, int N, int R, class T>
Matrix<M,N,T> rev( const Matrix<M,N,T>& b,
                   const Matrix<R,N,T>& e, 
                   const Vector<R+1,T>& w ) {
    Matrix<M,N,T> rb;
    for(int j=0; j<N; ++j) {
        for(int i=M-1; i>=0; --i) {
            rb[i][j] = b[i][j]*w[0];
            for(int k=1; k<R+1; ++k) {
                if(i+k >= M) rb[i][j] -= e[i+k-M][j]*w[k]; // use data from epilogue
                else rb[i][j] -= rb[i+k][j]*w[k];
            }
        }
    }
    return rb;
}

/**
 *  @relates Matrix
 *  @brief Computes the \a reverse-transposed operator on matrices
 *
 *  Computes the matrix resulting from applying the \a
 *  reverse-transposed operator \f$R^{T}\f$ (anticausal filter on
 *  rows) given an epilogue \f$M \times R\f$ matrix \f$e^{T}\f$
 *  (i.e. initial conditions) and an input \f$M \times N\f$ matrix
 *  \f$b\f$ (where M is the number of rows \f$h\f$ and N is the number
 *  of columns \f$w\f$ as described in section 2 of [Nehab:2011] cited
 *  in alg5()).  The resulting matrix is \f$M \times N\f$ and it has
 *  the same size as the input \f$b\f$.
 *
 *  @param[in] b Input \f$M \times N\f$ matrix
 *  @param[in] eT Epilogue \f$M \times R\f$ matrix (\f$R\f$ is the filter order)
 *  @param[in] w Filter weights with \f$R+1\f$ size (\f$R\f$ feedback coefficients)
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Number of feedback coefficients
 *  @tparam T Matrix value type
 */
template <int M, int N, int R, class T>
Matrix<M,N,T> revT( const Matrix<M,N,T>& b,
                    const Matrix<M,R,T>& eT,
                    const Vector<R+1,T>& w ) {
    return transp(rev(transp(b), transp(eT), w));
}

/**
 *  @relates Matrix
 *  @brief Computes the \a head operator on matrices
 *
 *  Computes the matrix resulting from applying the \a head operator
 *  \f$H\f$ given an input \f$M \times N\f$ matrix \f$mat\f$.  The
 *  operator extracts the \f$R \times N\f$ submatrix in the same shape
 *  and position as the column-epilogue of the input matrix,
 *  considering filter order \f$R\f$ (see [Nehab:2011] cited in alg5()
 *  function).  The following image illustrates the concept:
 *
 *  @image html blocks-2d.png "2D Block Notation"
 *  @image latex blocks-2d.eps "2D Block Notation"
 *
 *  2D block notation showing a block and its boundary data from
 *  adjacent blocks.  Note that the column-epilogue \f$E_{m+1,n}\f$
 *  has to be extracted from the next block \f$B_{m+1,n}\f$ in column
 *  \f$n\f$ using the head operator on that block.
 *
 *  @param[in] mat Input \f$M \times N\f$ matrix
 *  @tparam R Number of rows to extract (feedback coefficients)
 *  @tparam M Number of rows in input matrix
 *  @tparam N Number of columns in input matrix
 *  @tparam T Matrix value type
 */
template <int R, int M, int N, class T>
Matrix<R,N,T> head( const Matrix<M,N,T>& mat ) {
    Matrix<R,N,T> h;
    for(int i=0; i<R; ++i)
        for(int j=0; j<N; ++j)
            h[i][j] = mat[i][j];
    return h;
}

/**
 *  @relates Matrix
 *  @brief Computes the \a head-transposed operator on matrices
 *
 *  Computes the matrix resulting from applying the \a head-transposed
 *  operator \f$H^{T}\f$ given an input \f$M \times N\f$ matrix
 *  \f$mat\f$.  The operator extracts the \f$M \times R\f$ submatrix
 *  in the same shape and position as the row-epilogue of the input
 *  matrix, considering filter order \f$R\f$ (see [Nehab:2011] cited
 *  in alg5() function).
 *
 *  Note that, as shown in figure in head() function, the row-epilogue
 *  \f$E^{T}_{m,n+1}\f$ has to be extracted from the next block
 *  \f$B_{m,n+1}\f$ in row \f$m\f$ using the head-transposed operator
 *  on that block.
 *
 *  @param[in] mat Input \f$M \times N\f$ matrix
 *  @tparam R Number of columns to extract (feedback coefficients)
 *  @tparam M Number of rows in input matrix
 *  @tparam N Number of columns in input matrix
 *  @tparam T Matrix value type
 */
template <int R, int M, int N, class T>
Matrix<M,R,T> headT( const Matrix<M,N,T>& mat ) {
    return transp(head(transp(mat)));
}

/**
 *  @relates Matrix
 *  @brief Computes the \a tail operator on matrices
 *
 *  Computes the matrix resulting from applying the \a tail operator
 *  \f$T\f$ given an input \f$M \times N\f$ matrix \f$mat\f$.  The
 *  operator extracts the \f$R \times N\f$ submatrix in the same shape
 *  and position as the column-prologue of the input matrix,
 *  considering filter order \f$R\f$ (see [Nehab:2011] cited in alg5()
 *  function).
 *
 *  Note that, as shown in figure in head() function, the
 *  column-prologue \f$P_{m-1,n}\f$ has to be extracted from the
 *  previous block \f$B_{m-1,n}\f$ in column \f$n\f$ using the tail
 *  operator on that block.
 *
 *  @param[in] mat Input \f$M \times N\f$ matrix
 *  @tparam R Number of rows to extract (feedback coefficients)
 *  @tparam M Number of rows in input matrix
 *  @tparam N Number of columns in input matrix
 *  @tparam T Matrix value type
 */
template <int R, int M, int N, class T>
Matrix<R,N,T> tail( const Matrix<M,N,T>& mat ) {
    Matrix<R,N,T> t;
    for(int i=0; i<R; ++i)
        for(int j=0; j<N; ++j)
            t[i][j] = mat[M-R+i][j];
    return t;
}

/**
 *  @relates Matrix
 *  @brief Computes the \a tail-transposed operator on matrices
 *
 *  Computes the matrix resulting from applying the \a tail-transposed
 *  operator \f$T^{T}\f$ given an input \f$M \times N\f$ matrix
 *  \f$mat\f$.  The operator extracts the \f$M \times R\f$ submatrix
 *  in the same shape and position as the row-prologue of the input
 *  matrix, considering filter order \f$R\f$ (see [Nehab:2011] cited
 *  in alg5() function).
 *
 *  Note that, as shown in figure in head() function, the row-prologue
 *  \f$P^{T}_{m,n-1}\f$ has to be extracted from the previous block
 *  \f$B_{m,n-1}\f$ in row \f$m\f$ using the tail-transposed operator
 *  on that block.
 *
 *  @param[in] mat Input \f$M \times N\f$ matrix
 *  @tparam R Number of rows to extract (feedback coefficients)
 *  @tparam M Number of rows in input matrix
 *  @tparam N Number of columns in input matrix
 *  @tparam T Matrix value type
 */
template <int R, int M, int N, class T>
Matrix<M,R,T> tailT( const Matrix<M,N,T>& mat ) {
    return transp(head(transp(mat)));
}

//=============================================================================
} // namespace gpufilter
//=============================================================================
#endif // UTIL_H
//=============================================================================
