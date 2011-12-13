/**
 *  @file cpuground.h
 *  @brief CPU Groundtruth Recursive Filtering functions
 *  @author Diego Nehab
 *  @author Andre Maximo
 *  @date October, 2010
 */

#ifndef CPUFILTER_H
#define CPUFILTER_H

//== INCLUDES =================================================================

#include <cmath>

#include <gpudefs.h>
#include <extension.h>

//== NAMESPACES ===============================================================

namespace gpufilter {

//== IMPLEMENTATION ===========================================================

/**
 *  @ingroup cpu
 *  @brief Compute first-order recursive filtering on columns forward and reverse
 *
 *  Given an input 2D image compute a first-order recursive filtering
 *  on its columns with a causal-anticausal filter pair.  The filter
 *  is computed using a feedforward coefficient, i.e. a weight on the
 *  current element, and a feedback coefficient, i.e. a weight on the
 *  previous element.  The initial condition can be zero, clamp,
 *  repeat or mirror.  The computation is done sequentially in a naïve
 *  single-core CPU fashion.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] h Height of the input image
 *  @param[in] w Width of the input image
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @param[in] ff Forward-only (ignore anticausal filter) flag
 *  @param[in] ext Extension (in pixels) to consider outside image
 *  @param[in] ic Initial condition (for outside access)
 *  @tparam T Image value type
 */
template< class T >
void rcfr( T *inout,
           const int& h,
           const int& w,
           const T& b0,
           const T& a1,
           const bool& ff = false,
           const int& ext = 0,
           const initcond& ic = zero ) {
    for (int j = 0; j < w; j++, inout += 1) {
        int i;
        // p=0 is not initial condition based, it is due
        // to the filter order outside image + extension
        T p = (T)0;
        T c;
        for (i = -ext; i < h+ext; i++) {
            c = lookat( inout, i, h, ic, w );
            p = c*b0 - p*a1;
            if( i >= 0 and i < h ) // in image range
                inout[i*w] = p;
        }
        if( ff ) continue;
        p = (T)0;
        for (i--; i >= 0; i--) {
            c = lookat( inout, i, h, ic, w );
            p = c*b0 - p*a1;
            if( i >= 0 and i < h ) // in image range
                inout[i*w] = p;
        }
    }
}

/**
 *  @ingroup cpu
 *  @brief Compute first-order recursive filtering on rows forward and reverse
 *
 *  Given an input 2D image compute a first-order recursive filtering
 *  on its rows with a causal-anticausal filter pair.  The filter is
 *  computed using a feedforward coefficient, i.e. a weight on the
 *  current element, and a feedback coefficient, i.e. a weight on the
 *  previous element.  The initial condition can be zero, clamp,
 *  repeat or mirror.  The computation is done sequentially in a naïve
 *  single-core CPU fashion.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] h Height of the input image
 *  @param[in] w Width of the input image
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @param[in] ff Forward-only (ignore anticausal filter) flag
 *  @param[in] ext Extension (in pixels) to consider outside image
 *  @param[in] ic Initial condition (for outside access)
 *  @tparam T Image value type
 */
template< class T >
void rrfr( T *inout,
           const int& h,
           const int& w,
           const T& b0,
           const T& a1,
           const bool& ff = false,
           const int& ext = 0,
           const initcond& ic = zero ) {
    for (int i = 0; i < h; i++, inout += w) {
        int j;
        // p=0 is not initial condition based, it is due
        // to the filter order outside image + extension
        T p = (T)0;
        T c;
        for (j = -ext; j < w+ext; j++) {
            c = lookat( inout, j, w, ic );
            p = c*b0 - p*a1;
            if( j >= 0 and j < w ) // in image range
                inout[j] = p;
        }
        if( ff ) continue;
        p = (T)0;
        for (j--; j >= 0; j--) {
            c = lookat( inout, j, w, ic );
            p = c*b0 - p*a1;
            if( j >= 0 and j < w ) // in image range
                inout[j] = p;
        }
    }
}
/**
 *  @example example_r1.cc
 *
 *  This is an example of how to use the rrfr() function in the CPU.
 *
 *  @see cpuground.h
 */

/**
 *  @ingroup cpu
 *  @brief Compute first-order recursive filtering
 *
 *  Given an input 2D image compute a first-order recursive filtering
 *  on its columns and rows with a causal-anticausal filter pair.  The
 *  filter is computed using a feedforward coefficient, i.e. a weight
 *  on the current element, and a feedback coefficient, i.e. a weight
 *  on the previous element.  The initial condition can be zero,
 *  clamp, repeat or mirror.  The computation is done sequentially in
 *  a naïve single-core CPU fashion.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] h Height of the input image
 *  @param[in] w Width of the input image
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @param[in] ff Forward-only (ignore anticausal filter) flag
 *  @param[in] ext Extension (in pixels) to consider outside image
 *  @param[in] ic Initial condition (for outside access)
 *  @tparam T Image value type
 */
template< class T >
void r( T *inout,
        const int& h,
        const int& w,
        const T& b0,
        const T& a1,
        const bool& ff = false,
        const int& ext = 0,
        const initcond& ic = zero ) {
    rcfr(inout, h, w, b0, a1, ff, ext, ic);
    rrfr(inout, h, w, b0, a1, ff, ext, ic);
}

/**
 *  @ingroup cpu
 *  @overload
 *  @brief Compute first-order recursive filtering
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] h Height of the input image
 *  @param[in] w Width of the input image
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @param[in] ic Initial condition (for outside access)
 *  @param[in] ext Extension (in pixels) to consider outside image
 *  @param[in] ff Forward-only (ignore anticausal filter) flag
 *  @tparam T Image value type
 */
template< class T >
void r( T *inout,
        const int& h,
        const int& w,
        const T& b0,
        const T& a1,
        const initcond& ic,
        const int& ext = 0,
        const bool& ff = false ) {
    rcfr(inout, h, w, b0, a1, ff, ext, ic);
    rrfr(inout, h, w, b0, a1, ff, ext, ic);
}

/**
 *  @ingroup cpu
 *  @brief Compute second-order recursive filtering on columns forward and reverse
 *
 *  Given an input 2D image compute a second-order recursive filtering
 *  on its columns with a causal-anticausal filter pair.  The filter
 *  is computed using a feedforward coefficient, i.e. a weight on the
 *  current element, and two feedback coefficients, i.e. weights on
 *  the previous two elements.  The initial condition can be zero,
 *  clamp, repeat or mirror.  The computation is done sequentially in
 *  a naïve single-core CPU fashion.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] h Height of the input image
 *  @param[in] w Width of the input image
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @param[in] a2 Feedback second-order coefficient
 *  @param[in] ff Forward-only (ignore anticausal filter) flag
 *  @param[in] ext Extension (in pixels) to consider outside image
 *  @param[in] ic Initial condition (for outside access)
 *  @tparam T Image value type
 */
template< class T >
void rcfr( T *inout,
           const int& h,
           const int& w,
           const T& b0,
           const T& a1,
           const T& a2,
           const bool& ff = false,
           const int& ext = 0,
           const initcond& ic = zero ) {
    for (int j = 0; j < w; j++, inout += 1) {
        int i;
        // pp=p=0 is not initial condition based, it is due
        // to the filter order outside image + extension
        T pp = (T)0;
        T p = (T)0;
        T c;
        for (i = -ext; i < h+ext; i++) {
            c = lookat( inout, i, h, ic, w );
            c = c*b0 - p*a1 - pp*a2;
            pp = p;
            p = c;
            if( i >= 0 and i < h ) // in image range
                inout[i*w] = p;
        }
        if( ff ) continue;
        pp = p = (T)0;
        for (i--; i >= 0; i--) {
            c = lookat( inout, i, h, ic, w );
            c = c*b0 - p*a1 - pp*a2;
            pp = p;
            p = c;
            if( i >= 0 and i < h ) // in image range
                inout[i*w] = p;
        }
    }
}

/**
 *  @ingroup cpu
 *  @brief Compute second-order recursive filtering on rows forward and reverse
 *
 *  Given an input 2D image compute a second-order recursive filtering
 *  on its rows with a causal-anticausal filter pair.  The filter is
 *  computed using a feedforward coefficient, i.e. a weight on the
 *  current element, and two feedback coefficients, i.e. weights on
 *  the previous two elements.  The initial condition can be zero,
 *  clamp, repeat or mirror.  The computation is done sequentially in
 *  a naïve single-core CPU fashion.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] h Height of the input image
 *  @param[in] w Width of the input image
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @param[in] a2 Feedback second-order coefficient
 *  @param[in] ff Forward-only (ignore anticausal filter) flag
 *  @param[in] ext Extension (in pixels) to consider outside image
 *  @param[in] ic Initial condition (for outside access)
 *  @tparam T Image value type
 */
template< class T >
void rrfr( T *inout,
           const int& h,
           const int& w,
           const T& b0,
           const T& a1,
           const T& a2,
           const bool& ff = false,
           const int& ext = 0,
           const initcond& ic = zero ) {
    for (int i = 0; i < h; i++, inout += w) {
        int j;
        // pp=p=0 is not initial condition based, it is due
        // to the filter order outside image + extension
        T pp = (T)0;
        T p = (T)0;
        T c;
        for (j = -ext; j < w+ext; j++) {
            c = lookat( inout, j, w, ic );
            c = c*b0 - p*a1 - pp*a2;
            pp = p;
            p = c;
            if( j >= 0 and j < w ) // in image range
                inout[j] = p;
        }
        if( ff ) continue;
        pp = p = (T)0;
        for (j--; j >= 0; j--) {
            c = lookat( inout, j, w, ic );
            c = c*b0 - p*a1 - pp*a2;
            pp = p;
            p = c;
            if( j >= 0 and j < w ) // in image range
                inout[j] = p;
        }
    }
}

/**
 *  @ingroup cpu
 *  @brief Compute second-order recursive filtering
 *
 *  Given an input 2D image compute a second-order recursive filtering
 *  on its columns and rows with a causal-anticausal filter pair.  The
 *  filter is computed using a feedforward coefficient, i.e. a weight
 *  on the current element, and two feedback coefficients,
 *  i.e. weights on the previous two elements.  The initial condition
 *  can be zero, clamp, repeat or mirror.  The computation is done
 *  sequentially in a naïve single-core CPU fashion.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] h Height of the input image
 *  @param[in] w Width of the input image
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @param[in] a2 Feedback second-order coefficient
 *  @param[in] ff Forward-only (ignore anticausal filter) flag
 *  @param[in] ext Extension (in pixels) to consider outside image
 *  @param[in] ic Initial condition (for outside access)
 *  @tparam T Image value type
 */
template< class T >
void r( T *inout,
        const int& h,
        const int& w,
        const T& b0,
        const T& a1,
        const T& a2,
        const bool& ff = false,
        const int& ext = 0,
        const initcond& ic = zero ) {
    rcfr(inout, h, w, b0, a1, a2, ff, ext, ic);
    rrfr(inout, h, w, b0, a1, a2, ff, ext, ic);
}

/**
 *  @ingroup api_cpu
 *  @brief Compute the Summed-area Table of an image in the CPU
 *
 *  Given an input 2D image compute its Summed-Area Table (SAT) by
 *  applying a first-order recursive filters forward using zero-border
 *  initial conditions.
 *
 *  @param[in,out] in The 2D image to compute the SAT
 *  @param[in] hin Height of the input image
 *  @param[in] win Width of the input image
 *  @tparam T Image value type
 */
template< class T >
void sat_cpu( T *in,
              const int& hin,
              const int& win ) {
    r(in, hin, win, (T)1, (T)-1, true);
}
/**
 *  @example example_sat1.cc
 *
 *  This is an example of how to use the sat_cpu() function in the
 *  CPU.
 *
 *  @see cpuground.h
 */

/**
 *  @ingroup cpu
 *  @overload
 *  @brief Compute second-order recursive filtering
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] h Height of the input image
 *  @param[in] w Width of the input image
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @param[in] a2 Feedback second-order coefficient
 *  @param[in] ic Initial condition (for outside access)
 *  @param[in] ext Extension (in pixels) to consider outside image
 *  @param[in] ff Forward-only (ignore anticausal filter) flag
 *  @tparam T Image value type
 */
template< class T >
void r( T *inout,
        const int& h,
        const int& w,
        const T& b0,
        const T& a1,
        const T& a2,
        const initcond& ic,
        const int& ext = 0,
        const bool& ff = false ) {
    rcfr(inout, h, w, b0, a1, a2, ff, ext, ic);
    rrfr(inout, h, w, b0, a1, a2, ff, ext, ic);
}

/**
 *  @ingroup api_cpu
 *  @brief Gaussian blur an image in the CPU
 *
 *  Given an input 2D image compute the Gaussian blur of it by
 *  applying a sequence of recursive filters using clamp-to-border
 *  initial conditions.
 *
 *  @param[in,out] in The 2D image to compute Gaussian blur
 *  @param[in] hin Height of the input image
 *  @param[in] win Width of the input image
 *  @param[in] depth Depth of the input image (color channels)
 *  @param[in] s Sigma support of Gaussian blur computation
 *  @param[in] ic Initial condition (for outside access) (default clamp)
 *  @param[in] ext Extension (in pixels) to consider outside image (default block-size)
 *  @tparam T Image value type
 */
template< class T >
void gaussian_cpu( T **in,
                   const int& hin,
                   const int& win,
                   const int& depth,
                   const T& s,
                   const initcond& ic = clamp,
                   const int& ext = WS ) {
    T b10, a11, b20, a21, a22;
    weights1(s, b10, a11);
    weights2(s, b20, a21, a22);
    for (int c = 0; c < depth; c++) {
        r(in[c], hin, win, b10, a11, ic, ext);
        r(in[c], hin, win, b20, a21, a22, ic, ext);
    }
}
/**
 *  @example app_recursive.cc
 *
 *  This is an application example of how to use the gaussian_cpu()
 *  function and bspline3i_cpu() function in the CPU; as well as the
 *  gaussian_gpu() function and bspline3i_gpu() function in the GPU.
 *
 *  @see cpuground.h
 */

/**
 *  @ingroup api_cpu
 *  @overload
 *  @brief Gaussian blur a single-channel image in the CPU
 *
 *  @param[in,out] in The single-channel 2D image to compute Gaussian blur
 *  @param[in] hin Height of the input image
 *  @param[in] win Width of the input image
 *  @param[in] s Sigma support of Gaussian blur computation
 *  @param[in] ic Initial condition (for outside access) (default clamp)
 *  @param[in] ext Extension (in pixels) to consider outside image (default block-size)
 *  @tparam T Image value type
 */
template< class T >
void gaussian_cpu( T *in,
                   const int& hin,
                   const int& win,
                   const T& s,
                   const initcond& ic = clamp,
                   const int& ext = WS ) {
    T b10, a11, b20, a21, a22;
    weights1(s, b10, a11);
    weights2(s, b20, a21, a22);
    r(in, hin, win, b10, a11, ic, ext);
    r(in, hin, win, b20, a21, a22, ic, ext);
}
/**
 *  @example example_gauss.cc
 *
 *  This is an example of how to use the gaussian_cpu() function in
 *  the CPU and the gaussian_gpu() function in the GPU.
 *
 *  @see cpuground.h
 */

/**
 *  @ingroup api_cpu
 *  @brief Compute the Bicubic B-Spline interpolation of an image in the CPU
 *
 *  Given an input 2D image compute the Bicubic B-Spline interpolation
 *  of it by applying a first-order recursive filter using
 *  clamp-to-border initial conditions.
 *
 *  @param[in,out] in The 2D image to compute the Bicubic B-Spline interpolation
 *  @param[in] hin Height of the input image
 *  @param[in] win Width of the input image
 *  @param[in] depth Depth of the input image (color channels)
 *  @param[in] ic Initial condition (for outside access) (default mirror)
 *  @param[in] ext Extension (in pixels) to consider outside image (default block-size)
 *  @tparam T Image value type
 */
template< class T >
void bspline3i_cpu( T **in,
                    const int& hin,
                    const int& win,
                    const int& depth,
                    const initcond& ic = mirror,
                    const int& ext = WS ) {
    const T alpha = (T)2 - sqrt((T)3);
    for (int c = 0; c < depth; c++) {
        r(in[c], hin, win, (T)1+alpha, alpha, ic, ext);
    }
}

/**
 *  @ingroup api_cpu
 *  @overload
 *  @brief Compute the Bicubic B-Spline interpolation of a single-channel image in the CPU
 *
 *  @param[in,out] in The single-channel 2D image to compute the Bicubic B-Spline interpolation
 *  @param[in] hin Height of the input image
 *  @param[in] win Width of the input image
 *  @param[in] ic Initial condition (for outside access) (default mirror)
 *  @param[in] ext Extension (in pixels) to consider outside image (default block-size)
 *  @tparam T Image value type
 */
template< class T >
void bspline3i_cpu( T *in,
                    const int& hin,
                    const int& win,
                    const initcond& ic = mirror,
                    const int& ext = WS ) {
    const T alpha = (T)2 - sqrt((T)3);
    r(in, hin, win, (T)1+alpha, alpha, ic, ext);
}
/**
 *  @example example_bspline.cc
 *
 *  This is an example of how to use the bspline3i_cpu() function in
 *  the CPU and the bspline3i_gpu() function in the GPU.
 *
 *  @see cpuground.h
 */

//=============================================================================
} // namespace gpufilter
//=============================================================================
#endif // CPUFILTER_H
//=============================================================================
