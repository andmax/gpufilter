/**
 *  @file conv.h
 *  @brief Convolutions utilities definition
 *  @author Andre Maximo
 *  @date Jun, 2014
 */

#ifndef CONV_H
#define CONV_H

#include <util/linalg.h>


template <class T>
HOSTDEV
T conv_op(const T &x, const T &a)
{
    return x + a;
}

// convolution -----------------------------------------------------

// convolution 1 matrix (left) on prologues
template <class T, int M, int N>
Matrix<T,M,N> conv1p(const Vector<T,R+1>& ck)
{
    Matrix<T,M,N> m;
    for(int i=0; i<M; ++i) {
        for(int j=0; j<N; ++j) {
            int k = i+j+1;
            if (k>R)
                m[i][j] = (T)0;
            else
                m[i][j] = ck[k];
        }
    }
    return m;
}

// convolution 2 matrix (right) on epilogues
template <class T, int M, int N>
Matrix<T,M,N> conv2e(const Vector<T,R+1>& ck)
{
    Matrix<T,M,N> m;
    for(int i=M-1; i>=0; --i) {
        for(int j=0; j<N; ++j) {
            int k = M-1-i+j+1;
            if (k>R)
                m[i][j] = (T)0;
            else
                m[i][j] = ck[k];
        }
    }
    return m;
}

// convolution 1 matrix (left) on blocks
template <class T, int M, int N>
Matrix<T,M,M> conv1b(const Vector<T,N+1>& ck)
{
    Matrix<T,M,M> m = zeros<float,M,M>();
    for(int i=0; i<M; ++i) {
        for(int j=0; j<=N; ++j) {
            int k = i+(-1)*j;
            if (k>=0 && k<M)
                m[i][k] = ck[j];
        }
    }
    return m;
}

// convolution 2 matrix (right) on blocks
template <class T, int M, int N>
Matrix<T,M,M> conv2b(const Vector<T,N+1>& ck)
{
    Matrix<T,M,M> m = zeros<float,M,M>();
    for(int i=0; i<M; ++i) {
        for(int j=0; j<=N; ++j) {
            int k = i+(+1)*j;
            if (k>=0 && k<M)
                m[i][k] = ck[j];
        }
    }
    return m;
}

// conv1 ---------------------------------------------------------------

template <class T, int R>
T conv1(const Vector<T,R> &p, T x, const Vector<T,R+1> &ck)
{
    x = x * ck[0];
#pragma unroll
    for(int i=0; i<R; ++i)
        x = conv_op(x, p[i]*ck[i+1]);
    return x;
}

template <class T, int N, int R>
void conv1_inplace(const Vector<T,R> &_p, Vector<T,N> &b, const Vector<T,R+1> &ck)
{
    Vector<T,R> p = _p;
    for(int j=0; j<b.size(); ++j) {
        T x = conv1(p, b[j], ck);
        for (int k=R-1; k>=1; --k)
            p[k] = p[k-1];
        p[0] = b[j];
        b[j] = x;
    }
}

template <class T, int M, int N, int R>
void conv1_inplace(const Matrix<T,M,R> &p, Matrix<T,M,N> &b, 
                   const Vector<T,R+1> &ck)
{
    for(int i=0; i<b.rows(); ++i)
        conv1_inplace(p[i], b[i], ck);
}

template <class T, int M, int N, int R>
Matrix<T,M,N> conv1(const Matrix<T,M,R> &p, const Matrix<T,M,N> &b,
                    const Vector<T,R+1> &ck)
{
    Matrix<T,M,N> cb = b;
    conv1_inplace(p, cb, ck);
    return cb;
}

// conv2 ---------------------------------------------------------------

template <class T, int R>
T conv2(T x, const Vector<T,R> &e, const Vector<T,R+1> &ck)
{
    x = x * ck[0];
#pragma unroll
    for(int i=0; i<R; ++i)
        x = conv_op(x, e[i]*ck[i+1]);
    return x;
}

template <class T, int N, int R>
void conv2_inplace(Vector<T,N> &b, const Vector<T,R> &_e, const Vector<T,R+1> &ck)
{
    Vector<T,R> e = _e;
    for(int j=b.size()-1; j>=0; --j) {
        T x = conv2(b[j], e, ck);
        for (int k=R-1; k>=1; --k)
            e[k] = e[k-1];
        e[0] = b[j];
        b[j] = x;
    }
}

template <class T, int M, int N, int R>
void conv2_inplace(Matrix<T,M,N> &b, const Matrix<T,M,R> &e,
                   const Vector<T,R+1> &ck)
{
    for(int i=0; i<b.rows(); ++i)
        conv2_inplace(b[i], e[i], ck);
}

template <class T, int M, int N, int R>
Matrix<T,M,N> conv2(const Matrix<T,M,N> &b, const Matrix<T,M,R> &e,
                    const Vector<T,R+1> &ck)
{
    Matrix<T,M,N> cb = b;
    conv2_inplace(cb, e, ck);
    return cb;
}

#endif // CONV_H
