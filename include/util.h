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

template <int M, int N=M, class T=float>
class Matrix
{
    // we use proxies to be able to check row index range,
    // hopefully the compiler will optimize them out.
    struct row_proxy
    {
        row_proxy(T *row) : m_row(row) {}
        T &operator[](int j) const
        {
            assert(j>=0 && j<N);
            return m_row[j];
        }
        std::vector<T> to_vector() const
        {
            return std::vector<T>(m_row, m_row+N);
        }
        T *operator &() { return m_row; }
        const T *operator &() const { return m_row; }
    private:
        T *m_row;
    };
    struct const_row_proxy
    {
        const_row_proxy(const T *row) : m_row(row) {}
        const T &operator[](int j) const
        {
            assert(j>=0 && j<N);
            return m_row[j];
        }
        std::vector<T> to_vector() const
        {
            return std::vector<T>(m_row, m_row+N);
        }
    private:
        const T *m_row;
    };
public:
    const_row_proxy operator[](int i) const
    {
        assert(i >= 0 && i < M);
        return const_row_proxy(m_data[i]);
    }

    row_proxy operator[](int i)
    {
        assert(i >= 0 && i < M);
        return row_proxy(m_data[i]);
    }

    friend std::ostream &operator<<(std::ostream &out, const Matrix &m)
    {
        out << '[';
        for(int i=0; i<M; ++i)
        {
            for(int j=0; j<N; ++j)
            {
                out << m[i][j];
                if(j < N-1)
                    out << ',';
            }
            if(i < M-1)
                out << ";\n";
        }
        return out << ']';
    }

    template <int P, int Q>
    Matrix<M,Q,T> operator*(const Matrix<P,Q,T> &rhs) const
    {
        assert(N==P);

        Matrix<M,Q,T> r;
        for(int i=0; i<M; ++i)
        {
            for(int j=0; j<Q; ++j)
            {
                r[i][j] = m_data[i][0]*rhs[0][j];
                for(int k=1; k<N; ++k)
                    r[i][j] += m_data[i][k]*rhs[k][j];
            }
        }
        return r;
    }

    Matrix &operator*=(T val)
    {
        for(int i=0; i<M; ++i)
            for(int j=0; j<N; ++j)
                m_data[i][j] *= val;
        return *this;
    }

    friend Matrix operator*(const Matrix &m, T val)
    {
        return Matrix(m) *= val;
    }

    friend Matrix operator*(T val, const Matrix &m)
    {
        return operator*(m,val);
    }

private:
    T m_data[M][N];
};

template <int M, class T=float>
class Vector 
{
public:

    Vector() { for(int i=0; i<M; ++i) m_data[i] = (T)0; }

    Vector(const Matrix<M,1,T> &m)
    {
        for(int i=0; i<M; ++i)
            m_data[i] = m[i][0];
    }

    const T &operator[](int i) const
    {
        assert(i >= 0 && i < M);
        return m_data[i];
    }

    T &operator[](int i)
    {
        assert(i >= 0 && i < M);
        return m_data[i];
    }

    friend std::ostream &operator<<(std::ostream &out, const Vector &v)
    {
        out << '[';
        for(int i=0; i<M; ++i)
        {
            out << v[i];
            if(i < M-1)
                out << ',';
        }
        return out << ']';
    }

private:
    T m_data[M];
};

template <int M, int N, class T>
Matrix<M,N,T> zeros()
{
    Matrix<M,N,T> mat;
    std::fill(&mat[0][0], &mat[M-1][N-1]+1, T());
    return mat; // I'm hoping that RVO will kick in
}

template <int M, int N, class T>
Matrix<M,N,T> identity()
{
    Matrix<M,N,T> mat;

    for(int i=0; i<M; ++i)
        for(int j=0; j<N; ++j)
            mat[i][j] = i==j ? 1 : 0;

    return mat;
}

template <int M, int N, class T>
Matrix<N,M,T> transp(const Matrix<M,N,T> &m)
{
    Matrix<N,M,T> tm;
    for(int i=0; i<M; ++i)
        for(int j=0; j<N; ++j)
            tm[j][i] = m[i][j];

    return tm;
}


template <int M, int N, int R, class T>
Matrix<M,N,T> fwd(const Matrix<R,N,T> &p, const Matrix<M,N,T> &b, 
              const Vector<R+1,T> &w)
{
    Matrix<M,N,T> fb;
    for(int j=0; j<N; ++j)
    {
        for(int i=0; i<M; ++i)
        {
            fb[i][j] = b[i][j]*w[0];

            for(int k=1; k<R+1; ++k)
            {
                if(i-k < 0)
                    fb[i][j] += p[R+i-k][j]*w[k]; // use data from prologue
                else
                    fb[i][j] += fb[i-k][j]*w[k];
            }
        }
    }

    return fb;
}

template <int M, int N, int R, class T>
Matrix<M,N,T> rev(const Matrix<M,N,T> &b, const Matrix<R,N,T> &e, 
              const Vector<R+1,T> &w)
{
    Matrix<M,N,T> rb;
    for(int j=0; j<N; ++j)
    {
        for(int i=M-1; i>=0; --i)
        {
            rb[i][j] = b[i][j]*w[0];

            for(int k=1; k<R+1; ++k)
            {
                if(i+k >= M)
                    rb[i][j] += e[i+k-M][j]*w[k]; // use data from epilogue
                else
                    rb[i][j] += rb[i+k][j]*w[k];
            }
        }
    }

    return rb;
}

template <int R, int M, int N, class T>
Matrix<R,N,T> head(const Matrix<M,N,T> &mat)
{
    Matrix<R,N,T> h;
    for(int i=0; i<R; ++i)
        for(int j=0; j<N; ++j)
            h[i][j] = mat[i][j];

    return h;
}

template <int R, int M, int N, class T>
Matrix<R,N,T> tail(const Matrix<M,N,T> &mat)
{
    Matrix<R,N,T> t;
    for(int i=0; i<R; ++i)
        for(int j=0; j<N; ++j)
            t[i][j] = mat[M-R+i][j];

    return t;
}

template <int M, int N, int R, class T>
Matrix<M,N,T> fwdT(const Matrix<M,R,T> &pT, const Matrix<M,N,T> &b, 
                  const Vector<R+1,T> &w)
{
    return transp(fwd(transp(pT), transp(b), w));
}

template <int M, int N, int R, class T>
Matrix<M,N,T> revT(const Matrix<M,N,T> &b, const Matrix<M,R,T> &eT, 
                  const Vector<R+1,T> &w)
{
    return transp(rev(transp(b), transp(eT), w));
}

template <int R, int M, int N, class T>
Matrix<M,R,T> headT(const Matrix<M,N,T> &mat)
{
    return transp(head(transp(mat)));
}

template <int R, int M, int N, class T>
Matrix<M,R,T> tailT(const Matrix<M,N,T> &mat)
{
    return transp(head(transp(mat)));
}

//=============================================================================
} // namespace gpufilter
//=============================================================================
#endif // UTIL_H
//=============================================================================
