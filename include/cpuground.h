/**
 *  @file cpuground.h
 *  @ingroup cpu
 *  @brief CPU Groundtruth Recursive Filtering functions
 *  @author Diego Nehab
 *  @author Andre Maximo
 *  @date October, 2010
 */

#ifndef CPUFILTER_H
#define CPUFILTER_H

//== INCLUDES =================================================================

#include <cmath>

#include <mat.h>

//== NAMESPACES ===============================================================

namespace gpufilter {

//=== IMPLEMENTATION ==========================================================

/**
 *  @brief Compute first-order recursive filtering forward with zero-border
 *
 *  Given an input 2D image compute a first-order recursive filtering
 *  on its rows and columns with only causal filter.  The filter is
 *  computed using a feedforward coefficient, i.e. a weight on the
 *  current element, and a feedback coefficient, i.e. a weight on the
 *  previous element.  The initial condition is zero-border.  When the
 *  feedforward coefficient is 1 and the feedback coefficient is -1
 *  this filter becomes the summed-area table computation.  The
 *  computation is done sequentially in a naïve single-core CPU
 *  fashion.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] h Height of the input image
 *  @param[in] w Width of the input image
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @tparam T Image value type
 */
template< class T >
void rf_0( T *inout,
           const int& h,
           const int& w,
           const T& b0,
           const T& a1 ) {
    T p = (T)0;
	for (int x = 0; x < w; x++) {
        p = inout[x]*b0 - p*a1;
        p = p*b0;
        inout[x] = p;
    }
	for (int y = 1; y < h; y++) {
		p = inout[y*w]*b0;
		inout[y*w] = inout[y*w]*b0 - inout[(y-1)*w]*a1;
		for (int x = 1; x < w; x++) {
			p = inout[y*w+x]*b0 - p*a1;
			inout[y*w+x] = p*b0 - inout[(y-1)*w+x]*a1;
		}
	}

}

/** 
 *  @brief Compute the Summed-area Table of a matrix
 *
 *  Given an input grid compute its Summed-Area Table (SAT) by
 *  applying a first-order recursive filters forward using zero-border
 *  initial conditions.
 *
 *  @param[in,out] in The grid (or 2D image) to compute the SAT
 *  @param[in] hin Height of the input image
 *  @param[in] win Width of the input image
 *  @tparam T Image value type
 */
template< class T >
void sat( T *in,
          const int& hin,
          const int& win ) {
    rf_0(in, hin, win, (T)1, (T)-1);
}
/** @example example_sat.cc
 *
 *  This is an example of how to use the @see sat function.
 *
 *  @see cpuground.h
 */

/** 
 *  @brief Compute first-order recursive filtering on columns forward and reverse with zero-border
 *
 *  Given an input 2D image compute a first-order recursive filtering
 *  on its columns with a causal-anticausal filter pair.  The filter
 *  is computed using a feedforward coefficient, i.e. a weight on the
 *  current element, and a feedback coefficient, i.e. a weight on the
 *  previous element.  The initial condition is zero-border.  The
 *  computation is done sequentially in a naïve single-core CPU
 *  fashion.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] h Height of the input image
 *  @param[in] w Width of the input image
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @tparam T Image value type
 */
template< class T >
void rcfr_0( T *inout,
             const int& h,
             const int& w,
             const T& b0,
             const T& a1 ) {
    for (int j = 0; j < w; j++) {
        T p = (T)0;
        for (int i = 0; i < h; i++) {
            p = inout[i*w+j]*b0 - p*a1;
            inout[i*w+j] = p; 
        }
        p = (T)0;
        for (int i = h-1; i >= 0; i--) {
            p = inout[i*w+j]*b0 - p*a1;
            inout[i*w+j] = p; 
        }
    }
}

/** 
 *  @brief Compute first-order recursive filtering on rows forward and reverse with zero-border
 *
 *  Given an input 2D image compute a first-order recursive filtering
 *  on its rows with a causal-anticausal filter pair.  The filter is
 *  computed using a feedforward coefficient, i.e. a weight on the
 *  current element, and a feedback coefficient, i.e. a weight on the
 *  previous element.  The initial condition is zero-border.  The
 *  computation is done sequentially in a naïve single-core CPU
 *  fashion.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] h Height of the input image
 *  @param[in] w Width of the input image
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @param[in] ac Anticausal filter flag (default: true)
 *  @tparam T Image value type
 */
template< class T >
void rrfr_0( T *inout,
             const int& h,
             const int& w,
             const T& b0,
             const T& a1 ) {
    for (int i = 0; i < h; i++) {
        T p = (T)0;
        for (int j = 0; j < w; j++) {
            p = inout[i*w+j]*b0 - p*a1;
            inout[i*w+j] = p; 
        }
        p = (T)0;
        for (int j = w-1; j >= 0; j--) {
            p = inout[i*w+j]*b0 - p*a1;
            inout[i*w+j] = p; 
        }
    }
}

/** 
 *  @brief Compute first-order recursive filtering with zero-border
 *
 *  Given an input 2D image compute a first-order recursive filtering
 *  on its columns and rows with a causal-anticausal filter pair.  The
 *  filter is computed using a feedforward coefficient, i.e. a weight
 *  on the current element, and a feedback coefficient, i.e. a weight
 *  on the previous element.  The initial condition is zero-border.
 *  The computation is done sequentially in a naïve single-core CPU
 *  fashion.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] h Height of the input image
 *  @param[in] w Width of the input image
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @tparam T Image value type
 */
template< class T >
void r_0( T *inout,
          const int& h,
          const int& w,
          const T& b0,
          const T& a1 ) {
    rcfr_0(inout, h, w, b0, a1);
    rrfr_0(inout, h, w, b0, a1);
}

/** 
 *  @brief Compute second-order recursive filtering on columns forward and reverse with zero-border
 *
 *  Given an input 2D image compute a second-order recursive filtering
 *  on its columns with a causal-anticausal filter pair.  The filter
 *  is computed using a feedforward coefficient, i.e. a weight on the
 *  current element, and two feedback coefficients, i.e. weights on
 *  the previous two elements.  The initial condition is zero-border.
 *  The computation is done sequentially in a naïve single-core CPU
 *  fashion.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] h Height of the input image
 *  @param[in] w Width of the input image
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @param[in] a2 Feedback second-order coefficient
 *  @tparam T Image value type
 */
template< class T >
void rcfr_0( T *inout,
             const int& h,
             const int& w,
             const T& b0,
             const T& a1,
             const T& a2 ) {
    for (int j = 0; j < w; j++) {
        T p = (T)0;
        T pp = p;
        for (int i = 0; i < h; i++) {
            T o = inout[i*w+j]*b0 - p*a1 - pp*a2;
            pp = p;
            p = inout[i*w+j] = o; 
        }
        p = (T)0;
        pp = p;
        for (int i = h-1; i >= 0; i--) {
            T o = inout[i*w+j]*b0 - p*a1 - pp*a2;
            pp = p;
            p = inout[i*w+j] = o; 
        }
    }
}

/** 
 *  @brief Compute second-order recursive filtering on rows forward and reverse with zero-border
 *
 *  Given an input 2D image compute a second-order recursive filtering
 *  on its rows with a causal-anticausal filter pair.  The filter is
 *  computed using a feedforward coefficient, i.e. a weight on the
 *  current element, and two feedback coefficients, i.e. weights on
 *  the previous two elements.  The initial condition is zero-border.
 *  The computation is done sequentially in a naïve single-core CPU
 *  fashion.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] h Height of the input image
 *  @param[in] w Width of the input image
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @param[in] a2 Feedback second-order coefficient
 *  @tparam T Image value type
 */
template< class T >
void rrfr_0( T *inout,
             const int& h,
             const int& w,
             const T& b0,
             const T& a1,
             const T& a2 ) {
    for (int i = 0; i < h; i++) {
        T p = (T)0;
        T pp = p;
        for (int j = 0; j < w; j++) {
            T o = inout[i*w+j]*b0 - p*a1 - pp*a2;
            pp = p;
            p = inout[i*w+j] = o; 
        }
        p = (T)0;
        pp = p;
        for (int j = w-1; j >= 0; j--) {
            T o = inout[i*w+j]*b0 - p*a1 - pp*a2;
            pp = p;
            p = inout[i*w+j] = o; 
        }
    }
}

/** 
 *  @brief Compute second-order recursive filtering with zero-border
 *
 *  Given an input 2D image compute a second-order recursive filtering
 *  on its columns and rows with a causal-anticausal filter pair.  The
 *  filter is computed using a feedforward coefficient, i.e. a weight
 *  on the current element, and two feedback coefficients,
 *  i.e. weights on the previous two elements.  The initial condition
 *  is zero-border.  The computation is done sequentially in a naïve
 *  single-core CPU fashion.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] h Height of the input image
 *  @param[in] w Width of the input image
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @param[in] a2 Feedback second-order coefficient
 *  @tparam T Image value type
 */
template< class T >
void r_0( T *inout,
          const int& h,
          const int& w,
          const T& b0,
          const T& a1,
          const T& a2 ) {
    rcfr_0(inout, h, w, b0, a1, a2);
    rrfr_0(inout, h, w, b0, a1, a2);
}

/** 
 *  @brief Compute first-order recursive filtering on columns forward and reverse with clamp-to-border
 *
 *  Given an input 2D image compute a first-order recursive filtering
 *  on its columns with a causal-anticausal filter pair.  The filter
 *  is computed using a feedforward coefficient, i.e. a weight on the
 *  current element, and a feedback coefficient, i.e. a weight on the
 *  previous element.  The initial condition is clamp-to-border.  The
 *  computation is done sequentially in a naïve single-core CPU
 *  fashion.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] h Height of the input image
 *  @param[in] w Width of the input image
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @tparam T Image value type
 */
template< class T >
void rcfr_c( T *inout,
             const int& h,
             const int& w,
             const T& b0,
             const T& a1 ) {
    const T q = b0/((T)1+a1);
    const T n = -a1/((T)1-a1*a1);
    const T m = (q - n)*q;
    for (int j = 0; j < w; j++) {
        T l = inout[(h-1)*w+j];
        T p = q*inout[j];
        for (int i = 0; i < h; i++) {
            p = inout[i*w+j]*b0 - p*a1;
            inout[i*w+j] = p; 
        }
        p = m*l + n*p;
        for (int i = h-1; i >= 0; i--) {
            p = inout[i*w+j]*b0 - p*a1;
            inout[i*w+j] = p; 
        }
    }
}

/** 
 *  @brief Compute first-order recursive filtering on rows forward and reverse with clamp-to-border
 *
 *  Given an input 2D image compute a first-order recursive filtering
 *  on its rows with a causal-anticausal filter pair.  The filter is
 *  computed using a feedforward coefficient, i.e. a weight on the
 *  current element, and a feedback coefficient, i.e. a weight on the
 *  previous element.  The initial condition is clamp-to-border.  The
 *  computation is done sequentially in a naïve single-core CPU
 *  fashion.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] h Height of the input image
 *  @param[in] w Width of the input image
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @tparam T Image value type
 */
template< class T >
void rrfr_c( T *inout,
             const int& h,
             const int& w,
             const T& b0,
             const T& a1 ) {
    const T q = b0/((T)1+a1);
    const T n = -a1/((T)1-a1*a1);
    const T m = (q - n)*q;
    for (int i = 0; i < h; i++) {
        T l = inout[(i*w)+w-1];
        T p = q*inout[i*w];
        for (int j = 0; j < w; j++) {
            p = inout[i*w+j]*b0 - p*a1;
            inout[i*w+j] = p; 
        }
        p = m*l + n*p;
        for (int j = w-1; j >= 0; j--) {
            p = inout[i*w+j]*b0 - p*a1;
            inout[i*w+j] = p; 
        }
    }
}

/** 
 *  @brief Compute first-order recursive filtering with clamp-to-border
 *
 *  Given an input 2D image compute a first-order recursive filtering
 *  on its columns and rows with a causal-anticausal filter pair.  The
 *  filter is computed using a feedforward coefficient, i.e. a weight
 *  on the current element, and a feedback coefficient, i.e. a weight
 *  on the previous element.  The initial condition is
 *  clamp-to-border.  The computation is done sequentially in a naïve
 *  single-core CPU fashion.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] h Height of the input image
 *  @param[in] w Width of the input image
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @tparam T Image value type
 */
template< class T >
void r_c( T *inout,
          const int& h,
          const int& w,
          const T& b0,
          const T& a1 ) {
    rcfr_c(inout, h, w, b0, a1);
    rrfr_c(inout, h, w, b0, a1);
}

/** 
 *  @brief Compute second-order recursive filtering on columns forward and reverse with clamp-to-border
 *
 *  Given an input 2D image compute a second-order recursive filtering
 *  on its columns with a causal-anticausal filter pair.  The filter
 *  is computed using a feedforward coefficient, i.e. a weight on the
 *  current element, and two feedback coefficients, i.e. weights on
 *  the previous two elements.  The initial condition is
 *  clamp-to-border.  The computation is done sequentially in a naïve
 *  single-core CPU fashion.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] h Height of the input image
 *  @param[in] w Width of the input image
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @param[in] a2 Feedback second-order coefficient
 *  @tparam T Image value type
 */
template< class T >
void rcfr_c( T *inout,
             const int& h,
             const int& w,
             const T& b0,
             const T& a1,
             const T& a2 ) {
    mat2<T> Q = mat2<T>::mQ(b0, a1, a2);
    mat2<T> M = mat2<T>::mM(b0, a1, a2);
    mat2<T> N = mat2<T>::mN(b0, a1, a2);
    for (int j = 0; j < w; j++) {
        T l = inout[(h-1)*w+j];
        T ll = l;
        T p = inout[j];
        T pp = p;
        mul(Q, pp, p);
        for (int i = 0; i < h; i++) {
            T o = inout[i*w+j]*b0 - p*a1 - pp*a2;
            pp = p;
            p = inout[i*w+j] = o; 
        }
        mul(M, ll, l);
        mul(N, pp, p);
        pp += ll;
        p += l;
        for (int i = h-1; i >= 0; i--) {
            T o = inout[i*w+j]*b0 - p*a1 - pp*a2;
            pp = p;
            p = inout[i*w+j] = o; 
        }
    }
}

/** 
 *  @brief Compute second-order recursive filtering on rows forward and reverse with clamp-to-border
 *
 *  Given an input 2D image compute a second-order recursive filtering
 *  on its rows with a causal-anticausal filter pair.  The filter is
 *  computed using a feedforward coefficient, i.e. a weight on the
 *  current element, and two feedback coefficients, i.e. weights on
 *  the previous two elements.  The initial condition is
 *  clamp-to-border.  The computation is done sequentially in a naïve
 *  single-core CPU fashion.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] h Height of the input image
 *  @param[in] w Width of the input image
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @param[in] a2 Feedback second-order coefficient
 *  @tparam T Image value type
 */
template< class T >
void rrfr_c( T *inout,
             const int& h,
             const int& w,
             const T& b0,
             const T& a1,
             const T& a2 ) {
    mat2<T> Q = mat2<T>::mQ(b0, a1, a2);
    mat2<T> M = mat2<T>::mM(b0, a1, a2);
    mat2<T> N = mat2<T>::mN(b0, a1, a2);
    for (int i = 0; i < h; i++) {
        T l = inout[i*w+w-1];
        T ll = l;
        T p = inout[i*w];
        T pp = p;
        mul(Q, pp, p);
        for (int j = 0; j < w; j++) {
            T o = inout[i*w+j]*b0 - p*a1 - pp*a2;
            pp = p;
            p = inout[i*w+j] = o; 
        }
        mul(M, ll, l);
        mul(N, pp, p);
        pp += ll;
        p += l;
        for (int j = w-1; j >= 0; j--) {
            T o = inout[i*w+j]*b0 - p*a1 - pp*a2;
            pp = p;
            p = inout[i*w+j] = o; 
        }
    }
}

/** 
 *  @brief Compute second-order recursive filtering with clamp-to-border
 *
 *  Given an input 2D image compute a second-order recursive filtering
 *  on its columns and rows with a causal-anticausal filter pair.  The
 *  filter is computed using a feedforward coefficient, i.e. a weight
 *  on the current element, and two feedback coefficients,
 *  i.e. weights on the previous two elements.  The initial condition
 *  is clamp-to-border.  The computation is done sequentially in a
 *  naïve single-core CPU fashion.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] h Height of the input image
 *  @param[in] w Width of the input image
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @param[in] a2 Feedback second-order coefficient
 *  @tparam T Image value type
 */
template< class T >
void r_c( T *inout,
          const int& h,
          const int& w,
          const T& b0,
          const T& a1,
          const T& a2 ) {
    rcfr_c(inout, h, w, b0, a1, a2);
    rrfr_c(inout, h, w, b0, a1, a2);
}

/** 
 *  @brief Gaussian blur an image
 *
 *  Given an input 2D image compute the Gaussian blur of it by
 *  applying a sequence of recursive filters using clamp-to-border
 *  initial conditions.
 *
 *  @param[in,out] in The 2D image to compute Gaussian blur
 *  @param[in] hin Height of the input image
 *  @param[in] win Width of the input image
 *  @param[in] depth Depth of the input image (color channels)
 *  @tparam T Image value type
 */
template< class T >
void gaussian( T **in,
               const int& hin,
               const int& win,
               const int& depth,
               const T& s ) {
    T b10, a11;
    weights1(s, b10, a11);
    T b20, a21, a22;
    weights2(s, b20, a21, a22);
    for (int c = 0; c < depth; c++) {
        r_c(in[c], hin, win, b10, a11);
        r_c(in[c], hin, win, b20, a21, a22);
    }
}

/** 
 *  @brief Compute the Bicubic B-Spline interpolation of an image
 *
 *  Given an input 2D image compute the Bicubic B-Spline interpolation
 *  of it by applying a first-order recursive filters using
 *  clamp-to-border initial conditions.
 *
 *  @param[in,out] in The 2D image to compute the Bicubic B-Spline interpolation
 *  @param[in] hin Height of the input image
 *  @param[in] win Width of the input image
 *  @param[in] depth Depth of the input image (color channels)
 *  @tparam T Image value type
 */
template< class T >
void bspline3i( T **in,
                const int& hin,
                const int& win,
                const int& depth ) {
    const T alpha = (T)2 - sqrt((T)3);
    for (int c = 0; c < depth; c++) {
        r_c(in[c], hin, win, (T)1+alpha, alpha);
    }
}
/** @example app_recursive_cpu.cc
 *
 *  This is an application example of the @see gaussian and @see
 *  bspline3i functions usage.
 *
 *  @see cpuground.h
 */

//=============================================================================
} // namespace gpufilter
//=============================================================================
#endif // CPUFILTER_H
//=============================================================================
