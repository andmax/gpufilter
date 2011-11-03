/**
 *  @file goyfilter.h
 *  @brief GPU-Efficient Recursive Filtering and Summed-Area Tables definitions
 *  @author Diego Nehab
 *  @author Andre Maximo
 *  @author Rodolfo Lima
 *  @date October, 2010
 *  @date September, 2011
 *  @copyright The MIT License
 */

#ifndef GPUFILTER_H
#define GPUFILTER_H

/*! @mainpage GPU-Efficient Recursive Filtering and Summed-Area Tables

\section introduction Introduction

The gpufilter project is a set of C for CUDA functions to compute
recursive filters and summed-area tables in GPUs.  This project
presents a new algorithmic framework for parallel evaluation.  It
partitions the image into 2D blocks, with a small band of data
buffered along each block perimeter.  A remarkable result is that the
image data is read only twice and written just once, independent of
image size, and thus total memory bandwidth is reduced even compared
to the traditional serial algorithm.  The project is based on the
paper: "GPU-Efficient Recursive Filtering and Summed-Area Tables" by
Diego Nehab, André Maximo, Rodolfo S. Lima and Hugues Hoppe.

\section structure Functions Structure

Naming conventions are: c_ constant; t_ texture; g_ global memory; s_
shared memory; d_ device pointer; a_ cuda-array; p_ template
parameter; f_ surface.

\section usage How to use

The gpufilter project provides close-to-metal CUDA functions as well
as high-level C++ functions to access the main GPU algorithms.

\section download How to get it

The source code of the zsig library is available under the <em>MIT
License</em>, refer to the COPYING file for more details.

\section acknowledgments Acknowledgments

This work has been funded in part by a post-doctoral scholarship from
CNPq and by an INST grant from FAPERJ.

\section credits Credits

The people involved in the gpufilter project are listed below:

\par
\htmlonly <a href="http://www.impa.br/~diego" target="_blank">Diego Nehab</a> \endhtmlonly
\latexonly \href{http://www.impa.br/~diego}{Diego Nehab} \endlatexonly
\htmlonly <a href="http://www.impa.br/~andmax" target="_blank">André Maximo</a> \endhtmlonly
\latexonly \href{http://www.impa.br/~andmax}{Andre Maximo} \endlatexonly
\htmlonly <a href="http://www.rodsoft.org" target="_blank">Rodolfo S. Lima</a> \endhtmlonly
\latexonly \href{http://www.rodsoft.org}{Rodolfo S. Lima} \endlatexonly
\htmlonly <a href="http://research.microsoft.com/en-us/um/people/hoppe" target="_blank">Hugues Hoppe</a> \endhtmlonly
\latexonly \href{http://research.microsoft.com/en-us/um/people/hoppe}{Hugues Hoppe} \endlatexonly

*/

/**
 *  @defgroup utils Utility classes and functions to CPU/GPU computations
 */

/**
 *  @defgroup cpu Functions to compute recursive filtering in the CPU
 */

//== INCLUDES =================================================================

#include <cmath>
#include <complex>

//== NAMESPACES ===============================================================

namespace gpufilter {

//=== IMPLEMENTATION ==========================================================

/** @ingroup utils
 *  @brief Compute recursive filtering scaling factor
 *
 *  Compute the scaling factor of the recursive filter representing a
 *  true Gaussian filter convolution with arbitrary support sigma.
 *
 *  @see [vanVliet:1998] cited in @ref weights2
 *  @param[in] s Sigma support of the true Gaussian filter
 *  @return Scaling factor q of the recursive filter approximation
 *  @tparam T Sigma value type
 */
template< class T >
T qs( const T& s ) {
    return (T)0.00399341 + (T)0.4715161 * s;
}

/** @ingroup utils
 *  @brief Rescale poles of the recursive filtering z-transform
 *
 *  Given a complex-valued pole on |z|=1 ROC of the recursive filter
 *  z-transform compute a rescaled pole representing a true Gaussian
 *  filter convolution with arbitrary support sigma.
 *
 *  @see [vanVliet:1998] cited in @ref weights2
 *  @param[in] d Complex-valued pole of a stable recursive filter
 *  @param[in] s Sigma support of the true Gaussian filter
 *  @return Rescaled complex-valued pole of the recursive filter approximation
 *  @tparam T Sigma value type
 */
template< class T >
std::complex<T> ds( const std::complex<T>& d,
                    const T& s ) {
    T q = qs(s);
    return std::polar(std::pow(std::abs(d),(T)1/q), std::arg(d)/q);
}

/** @ingroup utils
 *  @brief Rescale poles in the real-axis of the recursive filtering z-transform
 *
 *  Given a real pole on |z|=1 ROC of the recursive filter z-transform
 *  compute a rescaled pole representing a true Gaussian filter
 *  convolution with arbitrary support sigma.
 *
 *  @see [vanVliet:1998] cited in @ref weights2
 *  @param[in] d Real pole of a stable recursive filter
 *  @param[in] s Sigma support of the true Gaussian filter
 *  @return Rescaled real pole of the recursive filter approximation
 *  @tparam T Sigma value type
 */
template< class T >
T ds( const T& d,
      const T& s ) {
    return std::pow(d, (T)1/qs(s));
}

/** @ingroup utils
 *  @brief Compute first-order weights
 *
 *  Given a Gaussian sigma value compute the feedforward and feedback
 *  first-order coefficients.
 *
 *  @see [vanVliet:1998] cited in @ref weights2
 *  @param[in] s Gaussian sigma
 *  @param[out] b0 Feedforward coefficient
 *  @param[out] a1 Feedback first-order coefficient
 *  @tparam T1 Sigma value type
 *  @tparam T2 Coefficients value type
 */
template< class T1, class T2 >
void weights1( const T1& s,
               T2& b0,
               T2& a1 ) {
    const T1 d3 = (T1)1.86543;
    T1 d = ds(d3, s);
    b0 = static_cast<T2>(-((T1)1-d)/d);
    a1 = static_cast<T2>((T1)-1/d);
}

/** @ingroup utils
 *  @brief Compute first- and second-order weights
 *
 *  Given a Gaussian sigma value compute the feedforward and feedback
 *  first- and second-order coefficients.  Refer to the following
 *  paper for more information:
 *  @verbatim
@inproceedings{vanVliet:1998,
  author = {L. J. {van Vliet} and I. T. Young and P. W. Verbeek},
  title = {Recursive {Gaussian} derivative filters},
  booktitle = {Proceedings of the 14th International Conference on Pattern Recognition},
  year = {1998},
  pages = {509--514 (v. 1)}
}   @endverbatim
 *
 *  @param[in] s Gaussian sigma
 *  @param[out] b0 Feedforward coefficient
 *  @param[out] a1 Feedback first-order coefficient
 *  @param[out] a2 Feedback second-order coefficient
 *  @tparam T1 Sigma value type
 *  @tparam T2 Coefficients value type
 */
template< class T1, class T2 >
void weights2( const T1& s,
               T2& b0,
               T2& a1,
               T2& a2 ) {
    const std::complex<T1> d1((T1)1.41650, (T1)1.00829);
    std::complex<T1> d = ds(d1, s);
    T1 n2 = std::abs(d); 
    n2 *= n2;
    T1 re = std::real(d);
    b0 = static_cast<T2>(((T1)1-(T1)2*re+n2)/n2);
    a1 = static_cast<T2>((T1)-2*re/n2);
    a2 = static_cast<T2>((T1)1/n2);
}

//=============================================================================
} // namespace gpufilter
//=============================================================================
#endif // GPUFILTER_H
//=============================================================================
