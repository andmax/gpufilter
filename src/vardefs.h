/**
 *  @file vardefs.h
 *  @brief Auxiliary recursive filter functions using variable coefficients
 *  @author Andre Maximo
 *  @date Dec, 2012
 *  @copyright The MIT License
 */

#ifndef VARDEFS_H
#define VARDEFS_H

//== INCLUDES ==================================================================

#include <util/solve.h>

//== NAMESPACES ================================================================

namespace gpufilter {

//== CONSTANTS =================================================================

// Disclaimer: this is optimized for usability, not speed
// Note: everything is row-major (i.e. Vector represents a row in a matrix)

const float b0f=1.f, b0r=spline::pinv; // b0 coefficient for forward and reverse

//== IMPLEMENTATION ============================================================

/**
 *  @ingroup cpu
 *  @brief Build varying coefficients using [NehabHoppe:2011] method
 *  @see [NehabHoppe:2011] cited in nehab_hoppe_tr2011_sec6()
 *  @param[out] a The varying feedback coefficients
 *  @param[in] n The number of elements to filter
 */
template <class T>
bool build_coefficients( T *& a,
                         const int& n ) {
    if (n<spline::ln) return false;
    if (a) delete [] a;
    a = new T[n+1];
    if (!a) return false;
    a[0] = 0.f;
    for (int i=1; i<=spline::ln; ++i) a[i] = spline::l[i-1];
    for (int i=spline::ln+1; i<n; ++i) a[i] = spline::linf;
    a[n] = spline::vinv;
    return true;
}

/**
 *  @ingroup cpu
 *  @brief Compute [NehabHoppe:2011] recursive filter
 *  @see [NehabHoppe:2011] cited in nehab_hoppe_tr2011_sec6()
 *  @param[in,out] inout The in(out)put 2D image to filter
 *  @param[in] w Image width
 *  @param[in] h Image height
 *  @tparam T Image value type
 */
template< class T >
void nehab_hoppe_tr2011_recfilter( T *inout,
                                   const int& w,
                                   const int& h ) {
    T *aw=0, *ah=0;
    build_coefficients(aw, w);
    build_coefficients(ah, h);
    for (int y=0; y<h; ++y) {
        for (int x=0; x<w; ++x)
            inout[y*w+x] = b0f*inout[y*w+x]-aw[x]*(x==0?(T)0:inout[y*w+x-1]);
        for (int x=w-1; x>=0; --x)
            inout[y*w+x] = aw[x+1]*b0r*inout[y*w+x]-aw[x+1]*(x==w-1?(T)0:inout[y*w+x+1]);
    }

    for (int x=0; x<w; ++x) {
        for (int y=0; y<h; ++y)
            inout[y*w+x] = b0f*inout[y*w+x]-ah[y]*(y==0?(T)0:inout[(y-1)*w+x]);
        for (int y=h-1; y>=0; --y)
            inout[y*w+x] = ah[y+1]*b0r*inout[y*w+x]-ah[y+1]*(y==h-1?(T)0:inout[(y+1)*w+x]);
    }

    delete [] aw;
    delete [] ah;
}

/**
 *  @ingroup cpu
 *  @overload T fwd( Vector<T,R>& p, const T& x, const T& b0, const T *a )
 *  @brief Compute the \a forward operator on a value given varying coefficients
 *  @see [NehabMaximo:2016] cited in alg6()
 *  @see rec_op()
 *  @param[in,out] p Prologue vector with \f$R\f$ size
 *  @param[in] x Input value (at the current filtering position)
 *  @param[in] b0 Filter feedforward coefficient
 *  @param[in] a Filter feedback coefficients
 *  @return Filtered value at the current filtering position
 *  @tparam T Value type
 *  @tparam R Filter order
 */
template <class T, int R>
HOSTDEV
T fwd( Vector<T,R>& p,
       const T& x,
       const T& b0,
       const T *a ) {
    // fwd = fwdI / a only has feedback coefficients
    // assuming a pointer at the correct position
    T acc = rec_op(b0*x, a[0]*p[R-1]);
#pragma unroll
    for (int k=R-1; k>=1; --k) {
        acc = rec_op(acc, a[k]*p[R-1-k]);
        p[R-1-k] = p[R-1-k+1];
    }
    return p[R-1] = acc;
}

/**
 *  @ingroup cpu
 *  @relates Vector
 *  @overload void fwd_inplace( const Vector<T,R>& _p, Vector<T,N>& b, const T& b0, const T *a )
 *  @brief Compute the \a forward operator on vectors (in-place) given varying coefficients
 *  @see [NehabMaximo:2016] cited in alg6()
 *  @see fwd()
 *  @param[in,out] _p Prologue vector with \f$R\f$ size
 *  @param[in,out] b In(out)put \f$N\f$ vector
 *  @param[in] b0 Filter feedforward coefficient
 *  @param[in] a Filter feedback coefficients
 *  @tparam N Number of elements in the in(out)put vector
 *  @tparam T Value type
 *  @tparam R Filter order
 */
template <class T, int N, int R>
void fwd_inplace( const Vector<T,R>& _p,
                  Vector<T,N>& b,
                  const T& b0,
                  const T *a ) {
    Vector<T,R> p = _p;
#pragma unroll
    for(int j=0; j<N; ++j)
        b[j] = fwd(p, b[j], b0, a+j);
}

/**
 *  @ingroup cpu
 *  @relates Matrix
 *  @overload void fwdD_inplace( const Matrix<T,M,R>& p, Matrix<T,M,N>& b, const T& b0, const T *a)
 *  @brief Computes the \a forward operator on matrices (in-place) given varying coefficients
 *  @see [NehabMaximo:2016] cited in alg6()
 *  @see fwd()
 *  @param[in] p Prologue \f$M \times R\f$ matrix
 *  @param[in,out] b In(out)put \f$M \times N\f$ matrix
 *  @param[in] b0 Filter feedforward coefficient
 *  @param[in] a Filter feedback coefficients
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
void fwdD_inplace( const Matrix<T,M,R>& p,
                   Matrix<T,M,N>& b,
                   const T& b0,
                   const T *a) {
    // the coefficients b0 and a are the same for each row
#pragma unroll
    for(int i=0; i<M; ++i)
        fwd_inplace(p[i], b[i], b0, a);
}

/**
 *  @ingroup cpu
 *  @relates Matrix
 *  @overload void fwd_inplace(const Matrix<T,M,R> &p, Matrix<T,M,N> &b, const T& b0, const T *a)
 *  @brief Computes the \a forward operator on matrices (in-place) given varying coefficients
 *  @see [NehabMaximo:2016] cited in alg6()
 *  @see fwd()
 *  @param[in] p Prologue \f$M \times R\f$ matrix
 *  @param[in,out] b In(out)put \f$M \times N\f$ matrix
 *  @param[in] b0 Filter feedforward coefficient
 *  @param[in] a Filter feedback coefficients
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
void fwd_inplace( const Matrix<T,M,R>& p,
                  Matrix<T,M,N>& b,
                  const T& b0,
                  const T *a) {
    fwdD_inplace(p, b, b0, a);
}

/**
 *  @ingroup cpu
 *  @relates Matrix
 *  @overload Matrix<T,M,N> fwdD( const Matrix<T,M,R>& p, const Matrix<T,M,N>& b, const T& b0, const T *a )
 *  @brief Computes the \a forward operator on matrices given varying coefficients
 *  @see [NehabMaximo:2016] cited in alg6()
 *  @see fwd()
 *  @param[in] p Prologue \f$M \times R\f$ matrix
 *  @param[in] b Input \f$M \times N\f$ matrix
 *  @param[in] b0 Filter feedforward coefficient
 *  @param[in] a Filter feedback coefficients
 *  @return Output \f$M \times N\f$ matrix
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
Matrix<T,M,N> fwdD( const Matrix<T,M,R>& p,
                    const Matrix<T,M,N>& b, 
                    const T& b0,
                    const T *a ) {
    Matrix<T,M,N> fb = b;
    fwdD_inplace(p, fb, b0, a);
    return fb;
}

/**
 *  @ingroup cpu
 *  @relates Matrix
 *  @overload Matrix<T,M,N> fwd(const Matrix<T,M,R> &p, const Matrix<T,M,N> &b, const T& b0, const T *a )
 *  @brief Computes the \a forward operator on matrices given varying coefficients
 *  @see [NehabMaximo:2016] cited in alg6()
 *  @param[in] p Prologue \f$M \times R\f$ matrix
 *  @param[in] b Input \f$M \times N\f$ matrix
 *  @param[in] b0 Filter feedforward coefficient
 *  @param[in] a Filter feedback coefficients
 *  @return Output \f$M \times N\f$ matrix
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
Matrix<T,M,N> fwd( const Matrix<T,M,R>& p,
                   const Matrix<T,M,N>& b,
                   const T& b0,
                   const T *a ) {
    return fwdD(p, b, b0, a);
}

/**
 *  @ingroup cpu
 *  @relates Matrix
 *  @overload Matrix<T,M,N> fwdT( const Matrix<T,R,N>& pT, const Matrix<T,M,N>& b, const T& b0, const T *a )
 *  @brief Computes the \a forward-transposed operator on matrices given varying coefficients
 *  @see [NehabMaximo:2016] cited in alg6()
 *  @see fwd()
 *  @param[in] pT Prologue transposed \f$R \times N\f$ matrix
 *  @param[in] b Input \f$M \times N\f$ matrix
 *  @param[in] b0 Filter feedforward coefficient
 *  @param[in] a Filter feedback coefficients
 *  @return Output \f$M \times N\f$ matrix
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
Matrix<T,M,N> fwdT( const Matrix<T,R,N>& pT,
                    const Matrix<T,M,N>& b,
                    const T& b0,
                    const T *a ) {
    return transp(fwd(transp(pT), transp(b), b0, a));
}

/**
 *  @ingroup cpu
 *  @relates Matrix
 *  @overload Matrix<T,M,N> fwd(const Matrix<T,R,N> &pT, const Matrix<T,M,N> &b, const T& b0, const T *a )
 *  @brief Computes the \a forward-transposed operator on matrices given varying coefficients
 *  @see [NehabMaximo:2016] cited in alg6()
 *  @see fwdT()
 *  @param[in] pT Prologue \f$R \times N\f$ matrix
 *  @param[in] b Input \f$M \times N\f$ matrix
 *  @param[in] b0 Filter feedforward coefficient
 *  @param[in] a Filter feedback coefficients
 *  @return Output \f$M \times N\f$ matrix
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
Matrix<T,M,N> fwd( const Matrix<T,R,N>& pT,
                   const Matrix<T,M,N>& b,
                   const T& b0,
                   const T *a ) {
    return fwdT(pT, b, b0, a);
}

/**
 *  @ingroup cpu
 *  @relates Matrix
 *  @overload void fwdT_inplace( const Matrix<T,R,N>& p, Matrix<T,M,N>& b, const T& b0, const T *a )
 *  @brief Computes the \a forward-transposed operator on matrices (in-place) given varying coefficients
 *  @see [NehabMaximo:2016] cited in alg6()
 *  @see fwdT()
 *  @param[in] pT Prologue \f$R \times N\f$ matrix
 *  @param[in,out] b In(out)put \f$M \times N\f$ matrix
 *  @param[in] b0 Filter feedforward coefficient
 *  @param[in] a Filter feedback coefficients
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
void fwdT_inplace( const Matrix<T,R,N>& p,
                   Matrix<T,M,N>& b, 
                   const T& b0,
                   const T *a ) {
    b = fwdT(p, b, b0, a);
}

/**
 *  @ingroup cpu
 *  @relates Matrix
 *  @overload void fwd_inplace(const Matrix<T,R,N> &p, Matrix<T,M,N> &b, const T& b0, const T *a )
 *  @brief Computes the \a forward-transposed operator on matrices (in-place) given varying coefficients
 *  @see [NehabMaximo:2016] cited in alg6()
 *  @see fwdT()
 *  @param[in] pT Prologue \f$R \times N\f$ matrix
 *  @param[in,out] b In(out)put \f$M \times N\f$ matrix
 *  @param[in] b0 Filter feedforward coefficient
 *  @param[in] a Filter feedback coefficients
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
void fwd_inplace( const Matrix<T,R,N>& p,
                  Matrix<T,M,N>& b,
                  const T& b0,
                  const T *a ) {
    fwdT_inplace(p, b, b0, a);
}

/**
 *  @ingroup cpu
 *  @overload T rev( const T& x, Vector<T,R>& e, const T& b0, const T *a )
 *  @brief Computes the \a reverse operator on a value given varying coefficients
 *  @see [NehabMaximo:2016] cited in alg6()
 *  @see rec_op()
 *  @param[in] x Input value (at the current filtering position)
 *  @param[in,out] e Epilogue vector with \f$R\f$ size
 *  @param[in] b0 Filter feedforward coefficient
 *  @param[in] a Filter feedback coefficients
 *  @return Filtered value at the current filtering position
 *  @tparam T Value type
 *  @tparam R Filter order
 */
template <class T, int R>
HOSTDEV
T rev( const T& x,
       Vector<T,R>& e,
       const T& b0,
       const T *a ) {
    // rev = revI / a only has feedback coefficients
    // assuming a pointer at the correct position
    T acc = rec_op(a[0]*b0*x, a[0]*e[0]);
#pragma unroll
    for (int k=R-1; k>=1; --k) {
        acc = rec_op(acc, a[k]*e[k]);
        e[k] = e[k-1];
    }
    return e[0] = acc;
}

/**
 *  @ingroup cpu
 *  @overload void rev_inplace( Vector<T,N>& b, const Vector<T,R>& _e, const T& b0, const T *a )
 *  @relates Vector
 *  @brief Computes the \a reverse operator on vectors (in-place) given varying coefficients
 *  @see [NehabMaximo:2016] cited in alg6()
 *  @see rev()
 *  @param[in,out] b In(out)put \f$N\f$ vector
 *  @param[in] e Epilogue \f$R\f$ vector
 *  @param[in] b0 Filter feedforward coefficient
 *  @param[in] a Filter feedback coefficients
 *  @tparam N Number of elements in the in(out)put vector
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int N, int R>
void rev_inplace( Vector<T,N>& b,
                  const Vector<T,R>& _e,
                  const T& b0,
                  const T *a ) {
    Vector<T,R> e = _e;
#pragma unroll
    for(int j=N-1; j>=0; --j)
        b[j] = rev(b[j], e, b0, a+j);
}

/**
 *  @ingroup cpu
 *  @relates Matrix
 *  @overload void revD_inplace( Matrix<T,M,N>& b, const Matrix<T,M,R>& e, const T& b0, const T *a )
 *  @brief Computes the \a reverse operator on matrices (in-place) given varying coefficients
 *  @see [NehabMaximo:2016] cited in alg6()
 *  @see rev()
 *  @param[in,out] b In(out)put \f$M \times N\f$ matrix
 *  @param[in] e Epilogue \f$M \times R\f$ matrix
 *  @param[in] b0 Filter feedforward coefficient
 *  @param[in] a Filter feedback coefficients
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
void revD_inplace( Matrix<T,M,N>& b,
                   const Matrix<T,M,R>& e,
                   const T& b0,
                   const T *a ) {
    // the coefficients b0 and a are the same for each row
#pragma unroll
    for(int i=0; i<M; ++i)
        rev_inplace(b[i], e[i], b0, a);
}

/**
 *  @ingroup cpu
 *  @relates Matrix
 *  @overload void rev_inplace( Matrix<T,M,N>& b, const Matrix<T,M,R>& e, const T& b0, const T *a )
 *  @brief Computes the \a reverse operator on matrices (in-place) given varying coefficients
 *  @see [NehabMaximo:2016] cited in alg6()
 *  @see rev()
 *  @param[in,out] b In(out)put \f$M \times N\f$ matrix
 *  @param[in] e Epilogue \f$M \times R\f$ matrix
 *  @param[in] b0 Filter feedforward coefficient
 *  @param[in] a Filter feedback coefficients
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
void rev_inplace( Matrix<T,M,N>& b,
                  const Matrix<T,M,R>& e, 
                  const T& b0,
                  const T *a ) {
    revD_inplace(b, e, b0, a);
}

/**
 *  @ingroup cpu
 *  @relates Matrix
 *  @overload  Matrix<T,M,N> revD( const Matrix<T,M,N>& b, const Matrix<T,M,R>& e, const T& b0, const T *a )
 *  @brief Computes the \a reverse operator on matrices given varying coefficients
 *  @see [NehabMaximo:2016] cited in alg6()
 *  @see rev()
 *  @param[in] b Input \f$M \times N\f$ matrix
 *  @param[in] e Epilogue \f$M \times R\f$ matrix
 *  @param[in] b0 Filter feedforward coefficient
 *  @param[in] a Filter feedback coefficients
 *  @return Output \f$M \times N\f$ matrix
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
Matrix<T,M,N> revD( const Matrix<T,M,N>& b,
                    const Matrix<T,M,R>& e,
                    const T& b0,
                    const T *a ) {
    Matrix<T,M,N> rb = b;
    revD_inplace(rb, e, b0, a);
    return rb;
}

/**
 *  @ingroup cpu
 *  @relates Matrix
 *  @overload Matrix<T,M,N> rev(const Matrix<T,M,N> &b, const Matrix<T,M,R> &e, const T& b0, const T *a )
 *  @brief Computes the \a reverse operator on matrices given varying coefficients
 *  @see [NehabMaximo:2016] cited in alg6()
 *  @see rev()
 *  @param[in] b Input \f$M \times N\f$ matrix
 *  @param[in] e Epilogue \f$M \times R\f$ matrix
 *  @param[in] b0 Filter feedforward coefficient
 *  @param[in] a Filter feedback coefficients
 *  @return Output \f$M \times N\f$ matrix
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
Matrix<T,M,N> rev( const Matrix<T,M,N>& b,
                   const Matrix<T,M,R>& e,
                   const T& b0,
                   const T *a ) {
    return revD(b, e, b0, a);
}

/**
 *  @ingroup cpu
 *  @relates Matrix
 *  @overload Matrix<T,M,N> revT( const Matrix<T,M,N>& b, const Matrix<T,R,N>& eT, const T& b0, const T *a )
 *  @brief Computes the \a reverse-transposed operator on matrices given varying coefficients
 *  @see [NehabMaximo:2016] cited in alg6()
 *  @see revT()
 *  @param[in] b Input \f$M \times N\f$ matrix
 *  @param[in] eT Epilogue \f$R \times N\f$ matrix
 *  @param[in] b0 Filter feedforward coefficient
 *  @param[in] a Filter feedback coefficients
 *  @return Output \f$M \times N\f$ matrix
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
Matrix<T,M,N> revT( const Matrix<T,M,N>& b,
                    const Matrix<T,R,N>& eT,
                    const T& b0,
                    const T *a ) {
    return transp(rev(transp(b), transp(eT), b0, a));
}

/**
 *  @ingroup cpu
 *  @relates Matrix
 *  @overload Matrix<T,M,N> rev(const Matrix<T,M,N> &b, const Matrix<T,R,N> &eT
 *  @brief Computes the \a reverse-transposed operator on matrices given varying coefficients
 *  @see [NehabMaximo:2016] cited in alg6()
 *  @see revT()
 *  @param[in] b Input \f$M \times N\f$ matrix
 *  @param[in] eT Epilogue \f$R \times N\f$ matrix
 *  @param[in] b0 Filter feedforward coefficient
 *  @param[in] a Filter feedback coefficients
 *  @return Output \f$M \times N\f$ matrix
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
Matrix<T,M,N> rev( const Matrix<T,M,N>& b,
                   const Matrix<T,R,N>& eT,
                   const T& b0,
                   const T *a ) {
    return revT(b, eT, b0, a);
}

/**
 *  @ingroup cpu
 *  @relates Matrix
 *  @overload void revT_inplace( Matrix<T,M,N>& b, const Matrix<T,R,N>& eT, const T& b0, const T *a )
 *  @brief Computes the \a reverse-transposed operator on matrices (in-place) given varying coefficients
 *  @see [NehabMaximo:2016] cited in alg6()
 *  @see revT()
 *  @param[in,out] b In(out)put \f$M \times N\f$ matrix
 *  @param[in] eT Epilogue \f$R \times N\f$ matrix
 *  @param[in] b0 Filter feedforward coefficient
 *  @param[in] a Filter feedback coefficients
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
void revT_inplace( Matrix<T,M,N>& b,
                   const Matrix<T,R,N>& eT,
                   const T& b0,
                   const T *a ) {
    b = revT(b, eT, b0, a);
}

/**
 *  @ingroup cpu
 *  @relates Matrix
 *  @overload void rev_inplace(Matrix<T,M,N> &b, const Matrix<T,R,N> &p, const T& b0, const T *a )
 *  @brief Computes the \a reverse-transposed operator on matrices (in-place) given varying coefficients
 *  @see [NehabMaximo:2016] cited in alg6()
 *  @see revT()
 *  @param[in,out] b In(out)put \f$M \times N\f$ matrix
 *  @param[in] eT Epilogue \f$R \times N\f$ matrix
 *  @param[in] b0 Filter feedforward coefficient
 *  @param[in] a Filter feedback coefficients
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Filter order
 *  @tparam T Value type
 */
template <class T, int M, int N, int R>
void rev_inplace( Matrix<T,M,N>& b,
                  const Matrix<T,R,N>& eT,
                  const T& b0,
                  const T *a ) {
    revT_inplace(b, eT, b0, a);
}

//==============================================================================
} // namespace gpufilter
//==============================================================================
#endif // VARDEFS_H
//==============================================================================
