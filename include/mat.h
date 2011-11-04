/**
 *  @file mat.h
 *  @brief Simple Matrix Class definition
 *  @author Diego Nehab
 *  @date October, 2010
 */

#ifndef MAT_H
#define MAT_H

//== NAMESPACES ===============================================================

namespace gpufilter {

//== CLASS DEFINITION =========================================================

/**
 *  @ingroup utils
 *  @{
 */

/**
 *  @class mat2 mat.h
 *  @brief Simple matrix type for second-order recursive filtering computations
 *
 *  This class is a simple 2 by 2 matrix with basic operations and
 *  static creation of special matrices.
 *
 *  @tparam T Matrix values type
 */
template< class T >
class mat2 {

public:

    /**
     *  Constructor
     *  @param[in] a Value in matrix at {0,0}
     *  @param[in] b Value in matrix at {0,1}
     *  @param[in] c Value in matrix at {1,0}
     *  @param[in] d Value in matrix at {1,1}
     */
    mat2( const T& a,
          const T& b,
          const T& c,
          const T& d ) {
        m[0][0] = a; m[0][1] = b; m[1][0] = c; m[1][1] = d;
    }

    /**
     *  @brief Minus operator
     *  @param[in] A Left matrix (to be subtracted)
     *  @param[in] B Right matrix (to subtract)
     *  @return The result matrix
     */
    friend mat2<T> operator - ( const mat2<T>& A,
                                const mat2<T>& B ) {
        T a = A.m[0][0] - B.m[0][0];
        T b = A.m[0][1] - B.m[0][1];
        T c = A.m[1][0] - B.m[1][0];
        T d = A.m[1][1] - B.m[1][1];
        return mat2<T>(a, b, c, d);
    }

    /**
     *  @brief Multiply operator
     *  @param[in] A Left matrix (this matrix times)
     *  @param[in] B Right matrix (the other matrix)
     *  @return The result matrix
     */
    friend mat2<T> operator * ( const mat2<T>& A,
                                const mat2<T>& B ) {
        T a = A.m[0][0] * B.m[0][0] + A.m[0][1] * B.m[1][0];
        T b = A.m[0][0] * B.m[0][1] + A.m[0][1] * B.m[1][1];
        T c = A.m[1][0] * B.m[0][0] + A.m[1][1] * B.m[1][0];
        T d = A.m[1][0] * B.m[0][1] + A.m[1][1] * B.m[1][1];
        return mat2<T>(a, b, c, d);
    }

    /**
     *  @brief Multiply by a 2-dimensional vector and return it
     *  @param[in] M Left matrix (this matrix times)
     *  @param[in,out] v0 The first value of the vector
     *  @param[in,out] v1 The second value of the vector
     */
    friend void mul( const mat2<T>& M,
                     T& v0,
                     T& v1 ) {
        T r0 = M.m[0][0]*v0 + M.m[0][1]*v1;
        T r1 = M.m[1][0]*v0 + M.m[1][1]*v1;
        v0 = r0; v1 = r1;
    }

    /**
     *  @brief Matrix \f$C\f$ of Coefficients
     *  @param[in] b0 Feedforward coefficient
     *  @param[in] a1 Feedback first-order coefficient
     *  @param[in] a2 Feedback second-order coefficient
     *  @return The matrix \f$C\f$
     */
    inline static mat2<T> mC( const T& b0,
                              const T& a1,
                              const T& a2 ) {
        (void) b0;
        T a = -a2;
        T b = -a1;
        T c = a1*a2;
        T d = p2(a1)-a2;
        return mat2<T>(a, b, c, d);
    }

    /**
     *  @brief Matrix \f$\bar{C}\f$
     *  @param[in] b0 Feedforward coefficient
     *  @param[in] a1 Feedback first-order coefficient
     *  @param[in] a2 Feedback second-order coefficient
     *  @return The matrix \f$\bar{C}\f$
     */
    inline static mat2<T> mCbar( const T& b0,
                                 const T& a1,
                                 const T& a2 ) {
        (void) a2;
        T a = b0;
        T b = (T)0;
        T c = -a1*b0;
        T d = b0;
        return mat2<T>(a, b, c, d);
    }

    /**
     *  @brief Matrix \f$C_\infty\f$
     *  @param[in] b0 Feedforward coefficient
     *  @param[in] a1 Feedback first-order coefficient
     *  @param[in] a2 Feedback second-order coefficient
     *  @return The matrix \f$C_\infty\f$
     */
    inline static mat2<T> mCinf( const T& b0,
                                 const T& a1,
                                 const T& a2 ) {
        (void) b0;
        T inv_d = (T)1/(p2((T)1+a2)-p2(a1));
        T a = ((T)1-p2(a1)+a2)*inv_d;
        T b = (-a1)*inv_d;
        T c = (a1*a2)*inv_d;
        T d = ((T)1+a2)*inv_d;
        return mat2<T>(a, b, c, d);
    }

    /**
     *  @brief Matrix \f$Q = C_\infty * \bar{C}\f$
     *  @param[in] b0 Feedforward coefficient
     *  @param[in] a1 Feedback first-order coefficient
     *  @param[in] a2 Feedback second-order coefficient
     *  @return The matrix \f$Q\f$
     */
    inline static mat2<T> mQ( const T& b0,
                              const T& a1,
                              const T& a2 ) {
        return mCinf(b0, a1, a2) * mCbar(b0, a1, a2);
    }

    /**
     *  @brief Matrix \f$A\f$ of coefficients
     *  @param[in] b0 Feedforward coefficient
     *  @param[in] a1 Feedback first-order coefficient
     *  @param[in] a2 Feedback second-order coefficient
     *  @return The matrix \f$A\f$
     */
    inline static mat2<T> mA( const T& b0,
                              const T& a1,
                              const T& a2 ) {
        (void) b0;
        T a = p2(a1)-a2;
        T b = a1*a2;
        T c = -a1;
        T d = -a2;
        return mat2<T>(a, b, c, d);
    }

    /**
     *  @brief Matrix \f$\bar{A}\f$
     *  @param[in] b0 Feedforward coefficient
     *  @param[in] a1 Feedback first-order coefficient
     *  @param[in] a2 Feedback second-order coefficient
     *  @return The matrix \f$\bar{A}\f$
     */
    inline static mat2<T> mAbar( const T& b0,
                                 const T& a1,
                                 const T& a2 ) {
        (void) a2;
        T a = b0;
        T b = -a1*b0;
        T c = (T)0;
        T d = b0;
        return mat2<T>(a, b, c, d);
    }

    /**
     *  @brief Matrix \f$A_\infty\f$
     *  @param[in] b0 Feedforward coefficient
     *  @param[in] a1 Feedback first-order coefficient
     *  @param[in] a2 Feedback second-order coefficient
     *  @return The matrix \f$A_\infty\f$
     */
    inline static mat2<T> mAinf( const T& b0,
                                 const T& a1,
                                 const T& a2 ) {
        (void) b0;
        const T inv_d = (T)1/(p2((T)1+a2)-p2(a1));
        T a = ((T)1+a2)*inv_d;
        T b = (a1*a2)*inv_d;
        T c = (-a1)*inv_d;
        T d = ((T)1-p2(a1)+a2)*inv_d;
        return mat2(a, b, c, d);
    }

    /**
     *  @brief Matrix \f$S_\infty\f$
     *  @param[in] b0 Feedforward coefficient
     *  @param[in] a1 Feedback first-order coefficient
     *  @param[in] a2 Feedback second-order coefficient
     *  @return The matrix \f$S_\infty\f$
     */
    inline static mat2<T> mSinf( const T& b0,
                                 const T& a1,
                                 const T& a2 ) {
        (void) b0;
        const T inv_d1 = (T)1/((p2(a2)-(T)1)*(p2(p2(a2)-1)-p4(a1)+4.f*p2(a1)*a2));
        const T inv_d2 = (T)1/((a2+(T)1)*(p2(p2(a2)-(T)1)-p4(a1)+4.f*p2(a1)*a2));
        T a = (p4(a1)-p2(p2(a2)-(T)1)-p2(a1)*a2*(3.f+a2-p2(a2)+p3(a2)))*inv_d1;
        T b = -a1*(p2(a1)-a2+p3(a2))*inv_d2;
        T c = a1*a2*((T)1+a2*(p2(a1)-a2))*inv_d2;
        T d = (p4(a1)+p2(a1)*(a2+(T)1)*((a2-(T)2)*a2-(T)1)-p2(p2(a2)-(T)1))*inv_d1;
        return mat2<T>(a, b, c, d);
    }

    /**
     *  @brief Matrix \f$N = S_\infty * C\f$
     *  @param[in] b0 Feedforward coefficient
     *  @param[in] a1 Feedback first-order coefficient
     *  @param[in] a2 Feedback second-order coefficient
     *  @return The matrix \f$N\f$
     */
    inline static mat2<T> mN( const T& b0,
                              const T& a1,
                              const T& a2 ) {
        return mSinf(b0, a1, a2) * mC(b0, a1, a2);
    }

    /**
     *  @brief Matrix \f$M = ( A_\infty * \bar{A} - S_\infty * C ) * C_\infty * \bar{C}\f$
     *  @param[in] b0 Feedforward coefficient
     *  @param[in] a1 Feedback first-order coefficient
     *  @param[in] a2 Feedback second-order coefficient
     *  @return The matrix \f$M\f$
     */
    inline static mat2<T> mM( const T& b0,
                              const T& a1,
                              const T& a2 ) {
        mat2<T> Ainf = mAinf(b0,a1,a2);
        mat2<T> Abar = mAbar(b0,a1,a2);
        mat2<T> Cinf = mCinf(b0,a1,a2);
        mat2<T> Cbar = mCbar(b0,a1,a2);
        mat2<T> C = mC(b0,a1,a2);
        mat2<T> Sinf = mSinf(b0,a1,a2);
        return ( Ainf * Abar - Sinf * C ) * Cinf * Cbar;
    }

private:

    /**
     *  @brief Compute power of 2
     *  @param[in] f Value to compute power of
     *  @return Power of 2
     */
    inline static T p2( const T& f) { return f*f; }

    /**
     *  @brief Compute power of 3
     *  @param[in] f Value to compute power of
     *  @return Power of 3
     */
    inline static T p3( const T& f) { return f*f*f; }

    /**
     *  @brief Compute power of 4
     *  @param[in] f Value to compute power of
     *  @return Power of 4
     */
    inline static T p4( const T& f) { T ff = f*f; return ff*ff; }

    T m[2][2]; ///< Matrix values
};

/**
 *  @}
 */

//=============================================================================
} // namespace gpufilter
//=============================================================================
#endif // MAT_H
//=============================================================================
