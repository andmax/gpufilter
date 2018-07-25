/**
 *  @file recfilter.h
 *  @brief Recursive filtering functions definition and implementation
 *  @author Rodolfo Lima
 *  @author Andre Maximo
 *  @date December, 2011
 *  @copyright The MIT License
 */

#ifndef RECFILTER_H
#define RECFILTER_H

//== INCLUDES =================================================================

#include <iostream>
#include <cassert>

#include "linalg.h"

// Disclaimer: this is optimized for usability, not speed
// Note: everything is row-major (i.e. Vector represents a row in a matrix)

//== NAMESPACES ================================================================

namespace gpufilter {

//== IMPLEMENTATION ============================================================

/**
 *  @ingroup utils
 *  @brief Computes the base recursive filter operation
 *
 *  This function is to define if the base recursive filter operation
 *  is either an addition or subtraction.  For instance, the equation
 *  \f$y_k = x_k - \sum^r_{i=1}{d_i y_{k-i}} \f$ corresponds to a
 *  minus sign, or subtraction base operation.  This operatorion is
 *  described in section 1 of [NehabMaximo:2016] cited in alg6().  The value
 *  returned correspond to operating the first value against the
 *  second passed as arguments.
 *
 *  @param[in] x First value to be operated (left-hand side)
 *  @param[in] a Second value to be operated (right-hand side)
 *  @return Value returned from first (operation) second value
 *  @tparam T Value type
 */
template <class T>
HOSTDEV
T rec_op(const T &x, const T &a) {
    return x - a;
}

/**
 *  @ingroup utils
 *  @brief Computes the \a forward operator on a value
 *
 *  Computes the vector resulting from applying the \a forward
 *  operator \f$F\f$ (causal filter) given a prologue vector \f$p\f$
 *  (i.e. initial conditions), an input value \f$x\f$ and a vector of
 *  weights \f$w\f$.  This operator is described in section 3 of
 *  [NehabMaximo:2016] cited in alg6().  The value returned is the filtered
 *  value at the current filtering position.
 *
 *  @see rec_op()
 *  @param[in,out] p Prologue vector with \f$R\f$ size
 *  @param[in] x Input value (at the current filtering position)
 *  @param[in] w Filter weights with \f$R+1\f$ size
 *  @return Filtered value at the current filtering position
 *  @tparam T Value type
 *  @tparam R Filter order
 */
template <class T, int R>
HOSTDEV
T fwd(Vector<T,R> &p, T x, const Vector<T,R+1> &w) {
    T acc = rec_op(x,p[R-1]*w[1]);
#pragma unroll
    for(int k=R-1; k>=1; --k)
    {
        acc = rec_op(acc,p[R-1-k]*w[k+1]);
        p[R-1-k] = p[R-1-k+1];
    }
    return p[R-1] = acc;
}

/**
 *  @ingroup utils
 *  @brief Computes direct \a forward operator in-place on a value
 *
 *  This function is similar to fwd() but it also multiplies the 
 *  input value \f$x\f$ by the first weight \f$w\f$ corresponding
 *  to the feedforward (or direct) coefficient.
 *
 *  @see rec_op()
 *  @param[in,out] p Prologue vector with \f$R\f$ size
 *  @param[in] x Input value (at the current filtering position)
 *  @param[in] w Filter weights with \f$R+1\f$ size
 *  @return Filtered value at the current filtering position
 *  @tparam T Value type
 *  @tparam R Filter order
 */
template <class T, int R>
HOSTDEV
T fwdI( Vector<T,R> &p,
        T x,
        const Vector<T,R+1>& w ) {
    T acc = rec_op(x*w[0],p[R-1]*w[1]);
#pragma unroll
    for (int k=R-1; k>=1; --k) {
        acc = rec_op(acc,p[R-1-k]*w[k+1]);
        p[R-1-k] = p[R-1-k+1];
    }
    return p[R-1] = acc;
}

/**
 *  @ingroup utils
 *  @relates Vector
 *  @brief Computes the \a forward operator on vectors (in-place)
 *
 *  @see fwd()
 *  @param[in] p Prologue \f$R\f$ vector
 *  @param[in,out] b In(out)put \f$N\f$ vector
 *  @param[in] w Filter weights with \f$R+1\f$ size
 *  @tparam N Number of elements in the in(out)put vector
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int N, int R>
void fwd_inplace(const Vector<T,R> &_p, Vector<T,N> &b,
                 const Vector<T,R+1> &w) {
    Vector<T,R> p = _p;
    for(int j=0; j<b.size(); ++j)
        b[j] = fwd(p, w[0]*b[j], w);
}

/**
 *  @ingroup utils
 *  @relates Matrix
 *  @brief Computes the \a forward operator on matrices (in-place)
 *
 *  @see fwd()
 *  @param[in] p Prologue \f$M \times R\f$ matrix
 *  @param[in,out] b In(out)put \f$M \times N\f$ matrix
 *  @param[in] w Filter weights with \f$R+1\f$ size
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
void fwdD_inplace(const Matrix<T,M,R> &p, Matrix<T,M,N> &b, 
                  const Vector<T,R+1> &w) {
    for(int i=0; i<b.rows(); ++i)
        fwd_inplace(p[i], b[i], w);
}

/**
 *  @ingroup utils
 *  @relates Matrix
 *  @overload void fwd_inplace(const Matrix<T,M,R> &p, Matrix<T,M,N> &b, const Vector<T,R+1> &w)
 *  @brief Computes the \a forward operator on matrices (in-place)
 *
 *  @see fwd()
 *  @param[in] p Prologue \f$M \times R\f$ matrix
 *  @param[in,out] b In(out)put \f$M \times N\f$ matrix
 *  @param[in] w Filter weights with \f$R+1\f$ size
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
void fwd_inplace(const Matrix<T,M,R> &p, Matrix<T,M,N> &b, 
                 const Vector<T,R+1> &w) {
    fwdD_inplace(p,b,w);
}

/**
 *  @ingroup utils
 *  @relates Matrix
 *  @brief Computes the \a forward operator on matrices
 *
 *  @see fwd()
 *  @param[in] p Prologue \f$M \times R\f$ matrix
 *  @param[in] b Input \f$M \times N\f$ matrix
 *  @param[in] w Filter weights with \f$R+1\f$ size
 *  @return Output \f$M \times N\f$ matrix
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
Matrix<T,M,N> fwdD(const Matrix<T,M,R> &p, const Matrix<T,M,N> &b, 
                   const Vector<T,R+1> &w) {
    Matrix<T,M,N> fb = b;
    fwdD_inplace(p, fb, w);
    return fb;
}

/**
 *  @ingroup utils
 *  @relates Matrix
 *  @overload Matrix<T,M,N> fwd(const Matrix<T,M,R> &p, const Matrix<T,M,N> &b, const Vector<T,R+1> &w)
 *  @brief Computes the \a forward operator on matrices
 *
 *  @param[in] p Prologue \f$M \times R\f$ matrix
 *  @param[in] b Input \f$M \times N\f$ matrix
 *  @param[in] w Filter weights with \f$R+1\f$ size
 *  @return Output \f$M \times N\f$ matrix
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
Matrix<T,M,N> fwd(const Matrix<T,M,R> &p, const Matrix<T,M,N> &b, 
                  const Vector<T,R+1> &w) {
    return fwdD(p, b, w);
}

/**
 *  @ingroup utils
 *  @relates Matrix
 *  @brief Computes the \a forward-transposed operator on matrices
 *
 *  Computes the matrix resulting from applying the \a
 *  forward-transposed operator \f$F^T\f$ (causal filter on columns)
 *  given a prologue \f$R \times N\f$ matrix \f$p^T\f$ (i.e. initial
 *  conditions), an input \f$M \times N\f$ matrix \f$b\f$ (where M is
 *  the number of rows and N is the number of columns) and a vector of
 *  weights \f$w\f$.  This operator is described in section 3 of
 *  [NehabMaximo:2016] cited in alg6().  The resulting matrix has the same
 *  size of the input.
 *
 *  @see fwd()
 *  @param[in] pT Prologue transposed \f$R \times N\f$ matrix
 *  @param[in] b Input \f$M \times N\f$ matrix
 *  @param[in] w Filter weights with \f$R+1\f$ size
 *  @return Output \f$M \times N\f$ matrix
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
Matrix<T,M,N> fwdT(const Matrix<T,R,N> &pT,
                   const Matrix<T,M,N> &b,
                   const Vector<T,R+1> &w) {
    return transp(fwd(transp(pT), transp(b), w));
}

/**
 *  @ingroup utils
 *  @relates Matrix
 *  @overload Matrix<T,M,N> fwd(const Matrix<T,R,N> &pT, const Matrix<T,M,N> &b, const Vector<T,R+1> &w)
 *  @brief Computes the \a forward-transposed operator on matrices
 *
 *  @see fwdT()
 *  @param[in] pT Prologue \f$R \times N\f$ matrix
 *  @param[in] b Input \f$M \times N\f$ matrix
 *  @param[in] w Filter weights with \f$R+1\f$ size
 *  @return Output \f$M \times N\f$ matrix
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
Matrix<T,M,N> fwd(const Matrix<T,R,N> &pT, const Matrix<T,M,N> &b, 
                  const Vector<T,R+1> &w) {
    return fwdT(pT, b, w);
}

/**
 *  @ingroup utils
 *  @relates Matrix
 *  @brief Computes the \a forward-transposed operator on matrices (in-place)
 *
 *  @see fwdT()
 *  @param[in] pT Prologue \f$R \times N\f$ matrix
 *  @param[in,out] b In(out)put \f$M \times N\f$ matrix
 *  @param[in] w Filter weights with \f$R+1\f$ size
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
void fwdT_inplace(const Matrix<T,R,N> &pT, Matrix<T,M,N> &b, 
                  const Vector<T,R+1> &w) {
    b = fwdT(pT, b, w);
}

/**
 *  @ingroup utils
 *  @relates Matrix
 *  @overload void fwd_inplace(const Matrix<T,R,N> &p, Matrix<T,M,N> &b, const Vector<T,R+1> &w)
 *  @brief Computes the \a forward-transposed operator on matrices (in-place)
 *
 *  @see fwdT()
 *  @param[in] pT Prologue \f$R \times N\f$ matrix
 *  @param[in,out] b In(out)put \f$M \times N\f$ matrix
 *  @param[in] w Filter weights with \f$R+1\f$ size
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
void fwd_inplace(const Matrix<T,R,N> &pT, Matrix<T,M,N> &b, 
                 const Vector<T,R+1> &w) {
    fwdT_inplace(pT, b, w);
}

/**
 *  @ingroup utils
 *  @brief Computes the \a reverse operator on a value
 *
 *  Computes the vector resulting from applying the \a reverse
 *  operator \f$R\f$ (anticausal filter) given an epilogue vector \f$e\f$
 *  (i.e. initial conditions), an input value \f$x\f$ and a vector of
 *  weights \f$w\f$.  This operator is described in section 3 of
 *  [NehabMaximo:2016] cited in alg6().  The value returned is the filtered
 *  value at the current filtering position.
 *
 *  @see rec_op()
 *  @param[in] x Input value (at the current filtering position)
 *  @param[in,out] e Epilogue vector with \f$R\f$ size
 *  @param[in] w Filter weights with \f$R+1\f$ size
 *  @return Filtered value at the current filtering position
 *  @tparam T Value type
 *  @tparam R Filter order
 */
template <class T, int R>
HOSTDEV
T rev(T x, Vector<T,R> &e, const Vector<T,R+1> &w) {
    T acc = rec_op(x,e[0]*w[1]);
#pragma unroll
    for(int k=R-1; k>=1; --k)
    {
        acc = rec_op(acc,e[k]*w[k+1]);
        e[k] = e[k-1];
    }
    return e[0] = acc;
}

/**
 *  @ingroup utils
 *  @brief Computes direct \a reverse operator in-place on a value
 *
 *  This function is similar to rev() but it also multiples the
 *  input value \f$x\f$ by the first weight \f$w\f$ corresponding
 *  to the direct reverse coefficient.
 *
 *  @see rec_op()
 *  @param[in] x Input value (at the current filtering position)
 *  @param[in,out] e Epilogue vector with \f$R\f$ size
 *  @param[in] w Filter weights with \f$R+1\f$ size
 *  @return Filtered value at the current filtering position
 *  @tparam T Value type
 *  @tparam R Filter order
 */
template <class T, int R>
HOSTDEV
T revI( T x,
        Vector<T,R> &e,
        const Vector<T,R+1>& w ) {
    T acc = rec_op(x*w[0],e[0]*w[1]);
#pragma unroll
    for (int k=R-1; k>=1; --k) {
        acc = rec_op(acc,e[k]*w[k+1]);
        e[k] = e[k-1];
    }
    return e[0] = acc;
}

/**
 *  @ingroup utils
 *  @relates Vector
 *  @brief Computes the \a reverse operator on vectors (in-place)
 *  @see rev()
 *  @param[in,out] b In(out)put \f$N\f$ vector
 *  @param[in] e Epilogue \f$R\f$ vector
 *  @param[in] w Filter weights with \f$R+1\f$ size
 *  @tparam N Number of elements in the in(out)put vector
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int N, int R>
void rev_inplace(Vector<T,N> &b, const Vector<T,R> &_e,
                 const Vector<T,R+1> &w) {
    Vector<T,R> e = _e;
    for(int j=b.size()-1; j>=0; --j)
        b[j] = rev(w[0]*b[j], e, w);
}

/**
 *  @ingroup utils
 *  @relates Matrix
 *  @brief Computes the \a reverse operator on matrices (in-place)
 *
 *  @see rev()
 *  @param[in,out] b In(out)put \f$M \times N\f$ matrix
 *  @param[in] e Epilogue \f$M \times R\f$ matrix
 *  @param[in] w Filter weights with \f$R+1\f$ size
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
void revD_inplace(Matrix<T,M,N> &b, const Matrix<T,M,R> &e, 
                  const Vector<T,R+1> &w) {
    for(int i=0; i<b.rows(); ++i)
        rev_inplace(b[i], e[i], w);
}

/**
 *  @ingroup utils
 *  @relates Matrix
 *  @overload void rev_inplace(Matrix<T,M,N> &b, const Matrix<T,M,R> &e, const Vector<T,R+1> &w)
 *  @brief Computes the \a reverse operator on matrices (in-place)
 *
 *  @see rev()
 *  @param[in,out] b In(out)put \f$M \times N\f$ matrix
 *  @param[in] e Epilogue \f$M \times R\f$ matrix
 *  @param[in] w Filter weights with \f$R+1\f$ size
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
void rev_inplace(Matrix<T,M,N> &b, const Matrix<T,M,R> &e, 
                 const Vector<T,R+1> &w) {
    revD_inplace(b, e, w);
}

/**
 *  @ingroup utils
 *  @relates Matrix
 *  @brief Computes the \a reverse operator on matrices
 *
 *  @see rev()
 *  @param[in] b Input \f$M \times N\f$ matrix
 *  @param[in] e Epilogue \f$M \times R\f$ matrix
 *  @param[in] w Filter weights with \f$R+1\f$ size
 *  @return Output \f$M \times N\f$ matrix
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
Matrix<T,M,N> revD(const Matrix<T,M,N> &b, const Matrix<T,M,R> &e, 
                   const Vector<T,R+1> &w) {
    Matrix<T,M,N> rb = b;
    revD_inplace(rb, e, w);

    return rb;
}

/**
 *  @ingroup utils
 *  @relates Matrix
 *  @overload Matrix<T,M,N> rev(const Matrix<T,M,N> &b, const Matrix<T,M,R> &e, const Vector<T,R+1> &w)
 *  @brief Computes the \a reverse operator on matrices
 *
 *  @see rev()
 *  @param[in] b Input \f$M \times N\f$ matrix
 *  @param[in] e Epilogue \f$M \times R\f$ matrix
 *  @param[in] w Filter weights with \f$R+1\f$ size
 *  @return Output \f$M \times N\f$ matrix
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
Matrix<T,M,N> rev(const Matrix<T,M,N> &b, const Matrix<T,M,R> &e, 
                  const Vector<T,R+1> &w) {
    return revD(b, e, w);
}

/**
 *  @ingroup utils
 *  @relates Matrix
 *  @brief Computes the \a reverse-transposed operator on matrices
 *
 *  Computes the matrix resulting from applying the \a
 *  reverse-transposed operator \f$R^T\f$ (anticausal filter on
 *  columns) given an epilogue \f$R \times N\f$ matrix \f$e^T\f$
 *  (i.e. initial conditions), an input \f$M \times N\f$ matrix
 *  \f$b\f$ (where M is the number of rows and N is the number of
 *  columns) and a vector of weights \f$w\f$.  This operator is
 *  described in section 3 of [NehabMaximo:2016] cited in alg6().  The
 *  resulting matrix has the same size of the input.
 *
 *  @see rev()
 *  @param[in] b Input \f$M \times N\f$ matrix
 *  @param[in] eT Epilogue \f$R \times N\f$ matrix
 *  @param[in] w Filter weights with \f$R+1\f$ size
 *  @return Output \f$M \times N\f$ matrix
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
Matrix<T,M,N> revT(const Matrix<T,M,N> &b, const Matrix<T,R,N> &eT, 
                   const Vector<T,R+1> &w) {
    return transp(rev(transp(b), transp(eT), w));
}

/**
 *  @ingroup utils
 *  @relates Matrix
 *  @overload Matrix<T,M,N> rev(const Matrix<T,M,N> &b, const Matrix<T,R,N> &eT, const Vector<T,R+1> &w)
 *  @brief Computes the \a reverse-transposed operator on matrices
 *
 *  @see revT()
 *  @param[in] b Input \f$M \times N\f$ matrix
 *  @param[in] eT Epilogue \f$R \times N\f$ matrix
 *  @param[in] w Filter weights with \f$R+1\f$ size
 *  @return Output \f$M \times N\f$ matrix
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
Matrix<T,M,N> rev(const Matrix<T,M,N> &b, const Matrix<T,R,N> &eT, 
                  const Vector<T,R+1> &w) {
    return revT(b,eT,w);
}

/**
 *  @ingroup utils
 *  @relates Matrix
 *  @brief Computes the \a reverse-transposed operator on matrices (in-place)
 *
 *  @see revT()
 *  @param[in,out] b In(out)put \f$M \times N\f$ matrix
 *  @param[in] eT Epilogue \f$R \times N\f$ matrix
 *  @param[in] w Filter weights with \f$R+1\f$ size
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
void revT_inplace(Matrix<T,M,N> &b, const Matrix<T,R,N> &eT, 
                  const Vector<T,R+1> &w) {
    b = revT(b, eT, w);
}

/**
 *  @ingroup utils
 *  @relates Matrix
 *  @overload void rev_inplace(Matrix<T,M,N> &b, const Matrix<T,R,N> &p, const Vector<T,R+1> &w)
 *  @brief Computes the \a reverse-transposed operator on matrices (in-place)
 *
 *  @see revT()
 *  @param[in,out] b In(out)put \f$M \times N\f$ matrix
 *  @param[in] eT Epilogue \f$R \times N\f$ matrix
 *  @param[in] w Filter weights with \f$R+1\f$ size
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
void rev_inplace(Matrix<T,M,N> &b, const Matrix<T,R,N> &eT, 
                 const Vector<T,R+1> &w) {
    revT_inplace(b, eT, w);
}

/**
 *  @ingroup utils
 *  @relates Matrix
 *  @brief Computes the \a head operator on matrices
 *
 *  Computes the matrix resulting from applying the \a head operator
 *  \f$H\f$ on an input \f$M \times N\f$ matrix \f$mat\f$.  The
 *  operator extracts the \f$M \times R\f$ submatrix in the same shape
 *  and position as the row-epilogue of the input matrix.  There is an
 *  additional boolean flag for clamp (constant padding boundary
 *  condition) that extracts only the first column of the input
 *  matrix, independently from the number of columns to extract given
 *  by the filter order.  This operator is described in section 3 of
 *  [NehabMaximo:2016] cited in alg6().
 *
 *  @param[in] mat Input \f$M \times N\f$ matrix
 *  @param[in] clamp Extract for clamp boundary condition
 *  @return Extracted output \f$M \times R\f$ matrix
 *  @tparam M Number of rows in input matrix
 *  @tparam N Number of columns in input matrix
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <int R, int M, int N, class T>
Matrix<T,M,R> head(const Matrix<T,M,N> &mat, bool clamp=false) {
    assert(mat.cols() >= R);

    Matrix<T,M,R> h;
    for(int j=0; j<R; ++j)
        for(int i=0; i<mat.rows(); ++i)
            h[i][j] = mat[i][clamp?0:j];

    return h;
}

/**
 *  @ingroup utils
 *  @relates Matrix
 *  @brief Computes the \a head-transposed operator on matrices
 *
 *  Computes the matrix resulting from applying the \a head-transposed
 *  operator \f$HT\f$ on an input \f$M \times N\f$ matrix \f$mat\f$.
 *  The operator extracts the \f$R \times N\f$ submatrix in the same
 *  shape and position as the column-epilogue of the input matrix.
 *  There is an additional boolean flag for clamp (constant padding
 *  boundary condition) that extracts only the first row of the input
 *  matrix, independently from the number of rows to extract given by
 *  the filter order.  This operator is described in section 3 of
 *  [NehabMaximo:2016] cited in alg6().
 *
 *  @param[in] mat Input \f$M \times N\f$ matrix
 *  @param[in] clamp Extract for clamp boundary condition
 *  @return Extracted output \f$R \times N\f$ matrix
 *  @tparam M Number of rows in input matrix
 *  @tparam N Number of columns in input matrix
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <int R, int M, int N, class T>
Matrix<T,R,N> headT(const Matrix<T,M,N> &mat, bool clamp=false) {
    return transp(head<R>(transp(mat), clamp));
}

/**
 *  @ingroup utils
 *  @relates Matrix
 *  @brief Computes the \a tail operator on matrices
 *
 *  Computes the matrix resulting from applying the \a tail operator
 *  \f$T\f$ given an input \f$M \times N\f$ matrix \f$mat\f$.  The
 *  operator extracts the \f$M \times R\f$ submatrix in the same shape
 *  and position as the row-prologue of the input matrix.  There is an
 *  additional boolean flag for clamp (constant padding boundary
 *  condition) that extracts only the last column of the input matrix,
 *  independently from the number of columns to extract given by the
 *  filter order.  This operator is described in section 3 of
 *  [NehabMaximo:2016] cited in alg6().
 *
 *  @param[in] mat Input \f$M \times N\f$ matrix
 *  @param[in] clamp Extract for clamp boundary condition
 *  @return Extracted output \f$M \times R\f$ matrix
 *  @tparam M Number of rows in input matrix
 *  @tparam N Number of columns in input matrix
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <int R, int M, int N, class T>
Matrix<T,M,R> tail(const Matrix<T,M,N> &mat, bool clamp=false) {
    assert(mat.cols() >= R);

    Matrix<T,M,R> t;
    for(int j=0; j<R; ++j)
        for(int i=0; i<mat.rows(); ++i)
            t[i][j] = mat[i][clamp?mat.cols()-1:mat.cols()-R+j];

    return t;
}

/**
 *  @ingroup utils
 *  @relates Matrix
 *  @brief Computes the \a tail-transposed operator on matrices
 *
 *  Computes the matrix resulting from applying the \a tail-transposed
 *  operator \f$T^T\f$ given an input \f$M \times N\f$ matrix
 *  \f$mat\f$.  The operator extracts the \f$R \times N\f$ submatrix
 *  in the same shape and position as the column-prologue of the input
 *  matrix.  There is an additional boolean flag for clamp (constant
 *  padding boundary condition) that extracts only the last row of the
 *  input matrix, independently from the number of rows to extract
 *  given by the filter order.  This operator is described in section
 *  3 of [NehabMaximo:2016] cited in alg6().
 *
 *  @param[in] mat Input \f$M \times N\f$ matrix
 *  @param[in] clamp Extract for clamp boundary condition
 *  @return Extracted output \f$R \times N\f$ matrix
 *  @tparam M Number of rows in input matrix
 *  @tparam N Number of columns in input matrix
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <int R, int M, int N, class T>
Matrix<T,R,N> tailT(const Matrix<T,M,N> &mat, bool clamp=false) {
    return transp(tail<R>(transp(mat), clamp));
}

/**
 *  @ingroup utils
 *  @brief Computes the \a forward operator on an array
 *
 *  Computes in-place the array resulting from applying the \a forward
 *  operator \f$F\f$ (causal filter) considering the prologue vector
 *  \f$p\f$ (i.e. initial conditions) zero, given the input array
 *  \f$io\f$ and a vector of weights \f$w\f$.  The array is returned
 *  in the input array itself.
 *
 *  @see fwd()
 *  @param[in,out] io Input array (at the current filtering position)
 *  @param[in] n Number of elements in the array
 *  @param[in] w Filter weights with \f$R+1\f$ size
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <int R, typename T>
void iir_fwd_inplace( T *io,
                      const int& n,
                      const Vector<T,R+1>& w ) {
    Vector<T,R> p = zeros<T,R>();
    for (int i = 0; i < n; ++i) {
        io[i] = fwd(p, io[i]*w[0], w);
    }
}

/**
 *  @ingroup utils
 *  @brief Computes the \a forward and reverse operators on an array
 *
 *  Computes in-place the array resulting from applying the \a forward
 *  operator \f$F\f$ (causal filter) and \a reverse operator \f$R\f$
 *  (anti-causal filter) considering both the prologue and epilogue
 *  \f$p\f$  and \f$e\f$ (i.e. initial conditions) zero, given an
 *  input array \f$io\f$ and weights \f$w\f$.  The array is returned
 *  in the input array itself.
 *
 *  @see fwd()
 *  @param[in,out] io Input array (at the current filtering position)
 *  @param[in] n Number of elements in the array
 *  @param[in] w Filter weights with \f$R+1\f$ size
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <int R, typename T>
void iir_fwd_rev_inplace( T *io,
                          const int& n,
                          const Vector<T,R+1>& w ) {
    Vector<T,R> p = zeros<T,R>();
    for (int i = 0; i < n; ++i) {
        io[i] = fwd(p, io[i]*w[0], w);
    }
    Vector<T,R> e = zeros<T,R>();
    for (int i = n-1; i >= 0; --i) {
        io[i] = rev(io[i]*w[0], e, w);
    }
}


//==============================================================================
} // namespace gpufilter
//==============================================================================
#endif // RECFILTER_H
//==============================================================================
