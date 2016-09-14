/**
 *  @file alg0_cpu.h
 *  @brief Algorithm 0 in the CPU
 *  @author Andre Maximo
 *  @date May, 2012
 *  @copyright The MIT License
 */

#ifndef ALG0_CPU_H
#define ALG0_CPU_H

//== INCLUDES ===================================================================

#include <util/image.h>
#include <util/linalg.h>
#include <util/recfilter.h>
#include <util/solve.h>

//== DEFINES ===================================================================

#define BS 32 ///< Block size (defines b x b block size)

//== NAMESPACES ================================================================

namespace gpufilter {

//=== IMPLEMENTATION ============================================================

/**
 *  @ingroup cpu
 *  @brief Compute R-order recursive filtering on rows forward
 *
 *  Given an input 2D image compute an R-order recursive filtering on
 *  its rows with a causal filter.  The filter is computed using a
 *  feedforward coefficient, i.e. a weight on the current element, and
 *  a feedback coefficient, i.e. a weight on the previous element.
 *  The computation is done sequentially in a naïve single-core CPU
 *  fashion, and it servers only for reference.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] w Width of the input image
 *  @param[in] h Height of the input image
 *  @param[in] weights Filter weights (feedforward and feedforward coefficients)
 *  @tparam R Filter order
 *  @tparam T Image value type
 */
template <int R, class T>
void recursive_rows_fwd( T *inout,
                         const int& w, const int& h,
                         const Vector<T,R+1> &weights ) {

    for (int y = 0; y < h; y++) {

        Vector<T,R> p = zeros<T,R>();

        for (int x = 0; x < w; x++)
            inout[y*w+x] = fwd(p, inout[y*w+x]*weights[0], weights);

    }

}

/**
 *  @ingroup cpu
 *  @brief Compute R-order recursive filtering on rows reverse
 *
 *  Given an input 2D image compute an R-order recursive filtering on
 *  its rows with an anticausal filter.  The filter is computed using a
 *  feedforward coefficient, i.e. a weight on the current element, and
 *  a feedback coefficient, i.e. a weight on the previous element.
 *  The computation is done sequentially in a naïve single-core CPU
 *  fashion, and it servers only for reference.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] w Width of the input image
 *  @param[in] h Height of the input image
 *  @param[in] weights Filter weights (feedforward and feedforward coefficients)
 *  @tparam R Filter order
 *  @tparam T Image value type
 */
template <int R, class T>
void recursive_rows_rev( T *inout,
                         const int& w, const int& h,
                         const Vector<T,R+1> &weights ) {

    for (int y = 0; y < h; y++) {

        Vector<T,R> e = zeros<T,R>();

        for (int x = w-1; x >= 0; x--)
            inout[y*w+x] = rev(inout[y*w+x]*weights[0], e, weights);

    }

}

/**
 *  @ingroup cpu
 *  @brief Compute R-order recursive filtering on columns forward
 *
 *  Given an input 2D image compute an R-order recursive filtering on
 *  its columns with a causal filter.  The filter is computed using a
 *  feedforward coefficient, i.e. a weight on the current element, and
 *  a feedback coefficient, i.e. a weight on the previous element.
 *  The computation is done sequentially in a naïve single-core CPU
 *  fashion, and it servers only for reference.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] w Width of the input image
 *  @param[in] h Height of the input image
 *  @param[in] weights Filter weights (feedforward and feedforward coefficients)
 *  @tparam R Filter order
 *  @tparam T Image value type
 */
template <int R, class T>
void recursive_cols_fwd( T *inout,
                         const int& w, const int& h,
                         const Vector<T,R+1> &weights ) {

    for (int x = 0; x < w; x++) {

        Vector<T,R> p = zeros<T,R>();

        for (int y = 0; y < h; y++)
            inout[y*w+x] = fwd(p, inout[y*w+x]*weights[0], weights);

    }

}

/**
 *  @ingroup cpu
 *  @brief Compute R-order recursive filtering on columns reverse
 *
 *  Given an input 2D image compute an R-order recursive filtering on
 *  its columns with an anticausal filter.  The filter is computed
 *  using a feedforward coefficient, i.e. a weight on the current
 *  element, and a feedback coefficient, i.e. a weight on the previous
 *  element.  The computation is done sequentially in a naïve
 *  single-core CPU fashion, and it servers only for reference.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] w Width of the input image
 *  @param[in] h Height of the input image
 *  @param[in] weights Filter weights (feedforward and feedforward coefficients)
 *  @tparam R Filter order
 *  @tparam T Image value type
 */
template <int R, class T>
void recursive_cols_rev( T *inout,
                         const int& w, const int& h,
                         const Vector<T,R+1> &weights ) {

    for (int x = 0; x < w; x++) {

        Vector<T,R> e = zeros<T,R>();

        for (int y = h-1; y >= 0; y--)
            inout[y*w+x] = rev(inout[y*w+x]*weights[0], e, weights);

    }

}

/**
 *  @ingroup api_cpu
 *  @brief Compute algorithm 0 in the CPU
 *
 *  Algorithm 0 is the naïve single-core CPU computation of an R-order
 *  recursive filtering, and it servers only for reference.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] width Image width
 *  @param[in] height Image height
 *  @param[in] weights Filter weights (feedforward and feedforward coefficients)
 *  @param[in] border Number of border blocks (32x32) outside image
 *  @param[in] btype Border type (either zero, clamp, repeat or reflect)
 *  @param[in] st The solve type (other attempts to solve with borders)
 *  @tparam R Filter order
 */
template <int R>
void alg0_cpu( float *inout,
               int width, int height, 
               const Vector<float, R+1> &weights,
               int border=0,
               BorderType border_type=CLAMP_TO_ZERO,
               spline::solve_type st=spline::traditional ) {

    int border_left, border_top, border_right, border_bottom;

    calc_borders(&border_left, &border_top, &border_right, &border_bottom, 
                 width, height, border);

    float *eimg = extend_image(inout, width, height, 
                               border_top, border_left, 
                               border_bottom, border_right,
                               border_type);

    int nw = width+border_left+border_right,
        nh = height+border_top+border_bottom;

    assert(nw%32 == 0);
    assert(nh%32 == 0);

    if (R>1 || st==spline::traditional) {
        recursive_rows_fwd<R>(eimg, nw, nh, weights);
        recursive_rows_rev<R>(eimg, nw, nh, weights);
        recursive_cols_fwd<R>(eimg, nw, nh, weights);
        recursive_cols_rev<R>(eimg, nw, nh, weights);
    } else if (st==spline::unser313) {
        spline::unser_etal_pami1991_3_13(eimg, nw, nh);
    } else if (st==spline::unser316) {
        spline::unser_etal_pami1991_3_16(eimg, nw, nh);
    } else if (st==spline::unser318) {
        spline::unser_etal_pami1991_3_18(eimg, nw, nh);
    } else if (st==spline::nehab6) {
        spline::nehab_hoppe_tr2011_sec6(eimg, nw, nh);
    }

    crop_image(inout, eimg, width, height, 
               border_top, border_left, 
               border_bottom, border_right);

    delete [] eimg;

}

//==============================================================================
} // namespace gpufilter
//==============================================================================
#endif // ALG0_CPU_H
//==============================================================================
