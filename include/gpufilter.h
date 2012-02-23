/**
 *  @file gpufilter.h
 *  @brief GPU-Efficient Recursive Filtering and Summed-Area Tables definitions
 *  @author Diego Nehab
 *  @author Andre Maximo
 *  @author Rodolfo Lima
 *  @date October, 2010
 *  @date December, 2011
 *  @note The MIT License
 */

#ifndef GPUFILTER_H
#define GPUFILTER_H

//== MAIN DOCUMENTATION =======================================================

/**
 *  @defgroup utils Utility classes and functions
 */

/**
 *  @defgroup api_cpu API for CPU functions
 */

/**
 *  @defgroup cpu CPU Computation functions
 *  @ingroup api_cpu
 */

/**
 *  @defgroup api_gpu API for GPU functions
 */

/**
 *  @defgroup gpu GPU Computation functions
 *  @ingroup api_gpu
 */

/**
 *  @mainpage gpufilter Library

\htmlonly <font color="red"><i><b>The source code in this project is
still under development (for cleaning and better
documentation).</b></i></font> \endhtmlonly

\section introduction Introduction

The GPU-Efficient Recursive Filtering and Summed-Area Tables
(::gpufilter) project is a set of <em>C for CUDA</em> functions to
compute recursive filters and summed-area tables in GPUs (see
illustrative figure in algSAT()).  This project presents a new
algorithmic framework for parallel evaluation.  It partitions the
image into 2D blocks, with a small band of data buffered along each
block perimeter (see figure in head() for more details).  A remarkable
result is that the image data is read only twice and written just
once, independent of image size, and thus total memory bandwidth is
reduced even compared to the traditional serial algorithm.

The ::gpufilter project is based on the paper: <b>"GPU-Efficient
Recursive Filtering and Summed-Area Tables"</b> by <b>Diego Nehab</b>,
<b>André Maximo</b>, <b>Rodolfo S. Lima</b> and <b>Hugues Hoppe</b>.

\section usage How to use

The ::gpufilter project provides a list of low-level CUDA kernel
functions (written in <em>C for CUDA</em>) in the @ref gpu module.
These kernel functions implement the algorithms described in the
paper.  On top of those CUDA kernel functions, there are several
high-level C++ functions in the @ref api_gpu module to access them.

For comparison purposes, the project also provides @ref cpu to perform
the same recursive filters in the CPU, an easy way to assess the
difference between traditional serial algorithms and their massively
parallel counterparts.  On top of those CPU functions, there are
several high-level C++ functions in the @ref api_cpu module to access
them.

Finally the project includes @ref utils to (1) facilitate the
communication between the CPU and the GPU, (2) measure computation
timings, (3) approximate Gaussian filter convolution by third-order
recursive filter with cascade first- and second-order filters.

\section download How to get it

The source code of the ::gpufilter library is available under the
<em>MIT License</em>, refer to the COPYING file for more details.

The ::gpufilter library can be downloaded following the link:

\htmlonly <a href="https://code.google.com/p/gpufilter" target="_blank">Google Code: code.google.com/p/gpufilter</a> \endhtmlonly
\latexonly \href{https://code.google.com/p/gpufilter}{Google Code: code.google.com/p/gpufilter} \endlatexonly

Alternatively, the source code can be downloaded directly from the
repository using the following command:

$ git clone https://code.google.com/p/gpufilter

Or a static (non-version control) instance of the source code can be
downloaded from the following link:

\htmlonly <a href="http://www.impa.br/~andmax/libs/gpufilter.zip" target="_blank">http://www.impa.br/~andmax/libs/gpufilter.zip</a> \endhtmlonly
\latexonly \href{http://www.impa.br/~andmax/libs/gpufilter.zip}{http://www.impa.br/~andmax/libs/gpufilter.zip} \endlatexonly

For more information about the project visit the main project page at:

\htmlonly <a href="http://www.impa.br/~diego/projects/NehEtAl11" target="_blank">Project Page: www.impa.br/~diego/projects/NehEtAl11</a> \endhtmlonly
\latexonly \href{http://www.impa.br/~diego/projects/NehEtAl11}{Project Page: www.impa.br/~diego/projects/NehEtAl11} \endlatexonly

\section acknowledgments Acknowledgments

This work has been funded in part by a post-doctoral scholarship from
CNPq (Brazilian National Counsel of Technological and Scientific
Development) and by an INST grant from FAPERJ (Rio de Janeiro
Foundation for Research).

\section credits Credits

The people involved in the ::gpufilter project are listed below:

\par
\htmlonly <a href="http://www.impa.br/~diego" target="_blank">Diego Nehab</a> \endhtmlonly
\latexonly \href{http://www.impa.br/~diego}{Diego Nehab} \endlatexonly

\par
\htmlonly <a href="http://www.impa.br/~andmax" target="_blank">André Maximo</a> \endhtmlonly
\latexonly \href{http://www.impa.br/~andmax}{Andre Maximo} \endlatexonly

\par
\htmlonly <a href="http://www.rodsoft.org" target="_blank">Rodolfo S. Lima</a> \endhtmlonly
\latexonly \href{http://www.rodsoft.org}{Rodolfo S. Lima} \endlatexonly

\par
\htmlonly <a href="http://research.microsoft.com/en-us/um/people/hoppe" target="_blank">Hugues Hoppe</a> \endhtmlonly
\latexonly \href{http://research.microsoft.com/en-us/um/people/hoppe}{Hugues Hoppe} \endlatexonly

*/

//== INCLUDES =================================================================

#include <cmath>
#include <complex>

#include <dvector.h>
#include <extension.h>

//== NAMESPACES ===============================================================

/**
 *  @namespace gpufilter
 *  @brief Main namespace for the gpufilter library
 */
namespace gpufilter {

//== IMPLEMENTATION ===========================================================

/**
 *  @ingroup utils
 *  @brief Compute recursive filtering scaling factor
 *
 *  Compute the scaling factor of the recursive filter representing a
 *  true Gaussian filter convolution with arbitrary support sigma.
 *
 *  @see [vanVliet:1998] cited in weights2()
 *  @param[in] s Sigma support of the true Gaussian filter
 *  @return Scaling factor q of the recursive filter approximation
 *  @tparam T Sigma value type
 */
template< class T >
T qs( const T& s ) {
    return (T)0.00399341 + (T)0.4715161 * s;
}

/**
 *  @ingroup utils
 *  @brief Rescale poles of the recursive filtering z-transform
 *
 *  Given a complex-valued pole on |z|=1 ROC of the recursive filter
 *  z-transform compute a rescaled pole representing a true Gaussian
 *  filter convolution with arbitrary support sigma.
 *
 *  @see [vanVliet:1998] cited in weights2()
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

/**
 *  @ingroup utils
 *  @brief Rescale poles in the real-axis of the recursive filtering z-transform
 *
 *  Given a real pole on |z|=1 ROC of the recursive filter z-transform
 *  compute a rescaled pole representing a true Gaussian filter
 *  convolution with arbitrary support sigma.
 *
 *  @see [vanVliet:1998] cited in weights2()
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

/**
 *  @ingroup utils
 *  @brief Compute first-order weights
 *
 *  Given a Gaussian sigma value compute the feedforward and feedback
 *  first-order coefficients.
 *
 *  @see [vanVliet:1998] cited in weights2()
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

/**
 *  @ingroup utils
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

//== EXTERNS ==================================================================

//-- SAT ----------------------------------------------------------------------

/**
 *  @ingroup api_gpu
 *  @brief Prepare for Algorithm SAT
 *
 *  This function prepares the data structures used by the summed-area
 *  table (SAT) algorithm of an input 2D image.
 *
 *  The algorithm SAT is discussed in depth in our paper (see
 *  [Nehab:2011] in alg5() function) and it is implemented in algSAT()
 *  function.
 *
 *  @param[out] d_inout The in/output 2D image to be allocated and copied to device memory
 *  @param[out] d_ybar The \f$P_{m,n}(\bar{Y})\f$ to be allocated in device memory
 *  @param[out] d_vhat The \f$P^T_{m,n}(\hat{V})\f$ to be allocated in device memory
 *  @param[out] d_ysum The \f$s(P_{m,n}(Y))\f$ to be allocated in device memory
 *  @param[out] cg_img Computation grid for SAT Stage 1 and 4
 *  @param[out] cg_ybar Computation grid for SAT Stage 2
 *  @param[out] cg_vhat Computation grid for SAT Stage 3
 *  @param[out] h_out Height (multiple of 32) of the output image
 *  @param[out] w_out Width (multiple of 32) of the output image
 *  @param[in] h_in The input 2D image to compute SAT in host memory
 *  @param[in] h Image height
 *  @param[in] w Image width
 */
extern
void prepare_algSAT( dvector<float>& d_inout,
                     dvector<float>& d_ybar,
                     dvector<float>& d_vhat,
                     dvector<float>& d_ysum,
                     dim3& cg_img,
                     dim3& cg_ybar,
                     dim3& cg_vhat,
                     int& h_out,
                     int& w_out,
                     const float *h_in,
                     const int& h,
                     const int& w );
/**
 *  @example example_sat3.cc
 *
 *  This is an example of how to use the prepare_algSAT() function and
 *  algSAT() function in the GPU.
 *
 *  @see gpufilter.h
 */

/**
 *  @ingroup api_gpu
 *  @overload
 *  @brief Compute Algorithm SAT
 *
 *  @note The pre-allocated device memory should match the values in
 *  the computational grids.
 *
 *  @see Base algSAT() function and [Nehab:2011] cited in alg5() function
 *  @param[out] d_out The output 2D image allocated in device memory
 *  @param[in] d_in The input 2D image allocated in device memory
 *  @param[out] d_ybar The \f$P_{m,n}(\bar{Y})\f$ allocated in device memory
 *  @param[out] d_vhat The \f$P^T_{m,n}(\hat{V})\f$ allocated in device memory
 *  @param[out] d_ysum The \f$s(P_{m,n}(Y))\f$ allocated in device memory
 *  @param[in] cg_img Computation grid for SAT Stage 1 and 4
 *  @param[in] cg_ybar Computation grid for SAT Stage 2
 *  @param[in] cg_vhat Computation grid for SAT Stage 3
 */
extern
void algSAT( dvector<float>& d_out,
             const dvector<float>& d_in,
             dvector<float>& d_ybar,
             dvector<float>& d_vhat,
             dvector<float>& d_ysum,
             const dim3& cg_img,
             const dim3& cg_ybar,
             const dim3& cg_vhat );

/**
 *  @ingroup api_gpu
 *  @overload
 *  @brief Compute Algorithm SAT
 *
 *  @note The pre-allocated device memory should match the values in
 *  the computational grids.
 *
 *  @see Base algSAT() function and [Nehab:2011] cited in alg5() function
 *  @param[in,out] d_inout The in/output 2D image allocated in device memory
 *  @param[out] d_ybar The \f$P_{m,n}(\bar{Y})\f$ allocated in device memory
 *  @param[out] d_vhat The \f$P^T_{m,n}(\hat{V})\f$ allocated in device memory
 *  @param[out] d_ysum The \f$s(P_{m,n}(Y))\f$ allocated in device memory
 *  @param[in] cg_img Computation grid for SAT Stage 1 and 4
 *  @param[in] cg_ybar Computation grid for SAT Stage 2
 *  @param[in] cg_vhat Computation grid for SAT Stage 3
 */
extern
void algSAT( dvector<float>& d_inout,
             dvector<float>& d_ybar,
             dvector<float>& d_vhat,
             dvector<float>& d_ysum,
             const dim3& cg_img,
             const dim3& cg_ybar,
             const dim3& cg_vhat );

/**
 *  @ingroup api_gpu
 *  @brief Compute Algorithm SAT
 *
 *  This function computes the summed-area table (SAT) of an input 2D
 *  image using algorithm SAT.
 *
 *  The algorithm SAT is discussed in depth in our paper (see
 *  [Nehab:2011] in alg5() function) where the following image
 *  illustrates the process:
 *
 *  @image html sat-stages.png "Illustration of Algorithm SAT"
 *  @image latex sat-stages.eps "Illustration of Algorithm SAT" width=\textwidth
 *
 *  Overlapped summed-area table computation according to algorithm
 *  SAT.  Stage S.1 reads the input (in gray) then computes and stores
 *  incomplete prologues \f$P_{m,n}(\bar{Y})\f$ (in red) and
 *  \f$P^T_{m,n}(\hat{V})\f$ (in blue).  Stage S.2 completes prologues
 *  \f$P_{m,n}(Y)\f$ and computes scalars
 *  \f$s\big(P_{m-1,n}(Y)\big)\f$ (in yellow).  Stage S.3 completes
 *  prologues \f$P^T_{m,n}(V)\f$. Finally, stage S.4 reads the input
 *  and completed prologues, then computes and stores the final
 *  summed-area table.
 *
 *  @note For performance purposes (in CUDA kernels implementation)
 *  this function works better in multiples of 32 in each dimension.
 *
 *  @param[in,out] inout The in/output 2D image to compute SAT
 *  @param[in] h Image height
 *  @param[in] w Image width
 */
extern
void algSAT( float *inout,
             const int& h,
             const int& w );
/**
 *  @example example_sat2.cc
 *
 *  This is an example of how to use the algSAT() function in the GPU.
 *
 *  @see gpufilter.h
 */

//-- Alg4 ---------------------------------------------------------------------


extern
void prepare_alg4( dvector<float>& d_out,
                   dvector<float>& d_transp_out,
                   int& transp_out_height,
                   cudaArray *& a_in,
                   dvector<float2>& d_transp_pybar,
                   dvector<float2>& d_transp_ezhat,
                   dvector<float2>& d_pubar,
                   dvector<float2>& d_evhat,
                   dim3& cg_img,
                   const float *h_in,
                   const int& h,
                   const int& w,
                   const float& b0,
                   const float& a1,
                   const float& a2,
                   const initcond& ic = zero,
                   const int& extb = 0 );



extern
void alg4( dvector<float>& d_out,
           dvector<float>& d_transp_out,
           int& transp_out_height,
           const int& h,
           const int& w,
           const cudaArray *a_in,
           dvector<float2>& d_transp_pybar,
           dvector<float2>& d_transp_ezhat,
           dvector<float2>& d_pubar,
           dvector<float2>& d_evhat,
           const dim3& cg_img );


/**
 *  @ingroup api_gpu
 *  @brief Compute Algorithm 4 (second-order)
 *
 *  This function computes second-order recursive filtering with given
 *  feedback and feedforward coefficients of an input 2D image using
 *  algorithm \f$4_2\f$.
 *
 *  The algorithm 4 is discussed in depth in our paper ([Nehab:2011]
 *  cited in alg5() function).
 *
 *  @note For performance purposes (in CUDA kernels implementation)
 *  this function only works with \f$64^2\f$ minimum image resolution,
 *  and only in multiples of 64 in each dimension.
 *
 *  @param[in,out] h_inout The in/output 2D image to compute recursive filtering
 *  @param[in] h Image height
 *  @param[in] w Image width
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @param[in] a2 Feedback second-order coefficient
 *  @param[in] ic Initial condition (for outside access) (default zero)
 *  @param[in] extb Extension (in blocks) to consider outside image (default 0)
 */
extern
void alg4( float *h_inout,
           const int& h,
           const int& w,
           const float& b0,
           const float& a1,
           const float& a2,
           const initcond& ic = zero,
           const int& extb = 0 );
/**
 *  @example example_r4.cc
 *
 *  This is an example of how to use the alg4() function in the GPU
 *  and the r() function in the CPU, as well as the
 *  gpufilter::scoped_timer_stop class.
 *
 *  @see gpufilter.h
 */

//-- Alg5 ---------------------------------------------------------------------

/**
 *  @ingroup api_gpu
 *  @brief Prepare for Algorithm 5
 *
 *  This function prepares the data structures used by the recursive
 *  filtering algorithm 5 of an input 2D image.
 *
 *  The algorithm 5 is discussed in depth in our paper (see
 *  [Nehab:2011] in alg5() function) and it is implemented in alg5()
 *  function.
 *
 *  @param[out] d_out The output 2D image to be allocated in device memory
 *  @param[out] a_in The input 2D image as cudaArray to be allocated and copied to device memory
 *  @param[out] d_transp_pybar The \f$P_{m,n}(\bar{Y})\f$ to be allocated in device memory
 *  @param[out] d_transp_ezhat The \f$E_{m,n}(\hat{Z})\f$ to be allocated in device memory
 *  @param[out] d_ptucheck The \f$P^T_{m,n}(\check{U})\f$ to be allocated in device memory
 *  @param[out] d_etvtilde The \f$E^T_{m,n}(\tilde{U})\f$ to be allocated in device memory
 *  @param[out] cg_img Computation grid for algorithm 5
 *  @param[in] h_in The input 2D image to compute algorithm 5 in host memory
 *  @param[in] h Image height
 *  @param[in] w Image width
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback coefficient
 *  @param[in] ic Initial condition (for outside access) (default zero)
 *  @param[in] extb Extension (in blocks) to consider outside image (default 0)
 */
extern
void prepare_alg5( dvector<float>& d_out,
                   cudaArray *& a_in,
                   dvector<float>& d_transp_pybar,
                   dvector<float>& d_transp_ezhat,
                   dvector<float>& d_ptucheck,
                   dvector<float>& d_etvtilde,
                   dim3& cg_img,
                   const float *h_in,
                   const int& h,
                   const int& w,
                   const float& b0,
                   const float& a1,
                   const initcond& ic = zero,
                   const int& extb = 0 );
/**
 *  @example example_r3.cc
 *
 *  This is an example of how to use the prepare_alg5() function and
 *  alg5() function in the GPU.
 *
 *  @see gpufilter.h
 */

/**
 *  @ingroup api_gpu
 *  @overload
 *  @brief Compute Algorithm 5 (first-order)
 *
 *  @param[out] d_out The output 2D image allocated in device memory
 *  @param[in] a_in The input 2D image allocated in device memory as cudaArray
 *  @param[out] d_transp_pybar The \f$P_{m,n}(\bar{Y})\f$ allocated in device memory
 *  @param[out] d_transp_ezhat The \f$E_{m,n}(\hat{Z})\f$ allocated in device memory
 *  @param[out] d_ptucheck The \f$P^T_{m,n}(\check{U})\f$ allocated in device memory
 *  @param[out] d_etvtilde The \f$E^T_{m,n}(\tilde{U})\f$ allocated in device memory
 *  @param[in] cg_img Computation grid for algorithm 5
 */
extern
void alg5( dvector<float>& d_out,
           const cudaArray *a_in,
           dvector<float>& d_transp_pybar,
           dvector<float>& d_transp_ezhat,
           dvector<float>& d_ptucheck,
           dvector<float>& d_etvtilde,
           const dim3& cg_img );

/**
 *  @ingroup api_gpu
 *  @brief Compute Algorithm 5 (first-order)
 *
 *  This function computes first-order recursive filtering with given
 *  feedback and feedforward coefficients of an input 2D image using
 *  algorithm \f$5_1\f$.
 *
 *  The algorithm 5 is discussed in depth in our paper:
 *
 *  @verbatim
@inproceedings{Nehab:2011,
  title = {{GPU}-{E}fficient {R}ecursive {F}iltering and {S}ummed-{A}rea {T}ables},
  author = {{N}ehab, {D}. and {M}aximo, {A}. and {L}ima, {R}. {S}. and {H}oppe, {H}.},
  journal = {{ACM} {T}ransactions on {G}raphics ({P}roceedings of the {ACM} {SIGGRAPH} {A}sia 2011)},
  year = {2011},
  volume = {30},
  number = {6},
  doi = {},
  publisher = {ACM},
  address = {{N}ew {Y}ork, {NY}, {USA}}
}   @endverbatim
 *
 *  @param[in,out] h_inout The in/output 2D image to compute recursive filtering in host memory
 *  @param[in] h Image height
 *  @param[in] w Image width
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback coefficient
 *  @param[in] ic Initial condition (for outside access) (default zero)
 *  @param[in] extb Extension (in blocks) to consider outside image (default 0)
 */
extern
void alg5( float *h_inout,
           const int& h,
           const int& w,
           const float& b0,
           const float& a1,
           const initcond& ic = zero,
           const int& extb = 0 );
/**
 *  @example example_r2.cc
 *
 *  This is an example of how to use the alg5() function in the GPU
 *  and the r() function in the CPU, as well as the
 *  gpufilter::scoped_timer_stop class.
 *
 *  @see gpufilter.h
 */

//-- Gaussian -----------------------------------------------------------------

/**
 *  @ingroup api_gpu
 *  @brief Gaussian blur an image in the GPU
 *
 *  Given an input single-channel 2D image compute the Gaussian blur
 *  of it by applying a first-order recursive filter (using alg5())
 *  followed by a second-order recursive filter (using alg4()) and
 *  zero-border initial condition.
 *
 *  @param[in,out] inout The 2D image to compute Gaussian blur
 *  @param[in] h Height of the input image
 *  @param[in] w Width of the input image
 *  @param[in] d Depth of the input image (color channels)
 *  @param[in] s Sigma support of Gaussian blur computation
 *  @param[in] ic Initial condition (for outside access) (default clamp)
 *  @param[in] extb Extension (in blocks) to consider outside image (default 1 block)
 */
extern
void gaussian_gpu( float **inout,
                   const int& h,
                   const int& w,
                   const int& d,
                   const float& s,
                   const initcond& ic = clamp,
                   const int& extb = 1 );

/**
 *  @ingroup api_gpu
 *  @overload
 *  @brief Gaussian blur a single-channel image in the GPU
 *
 *  @param[in,out] inout The single-channel 2D image to compute Gaussian blur
 *  @param[in] h Height of the input image
 *  @param[in] w Width of the input image
 *  @param[in] s Sigma support of Gaussian blur computation
 *  @param[in] ic Initial condition (for outside access) (default clamp)
 *  @param[in] extb Extension (in blocks) to consider outside image (default 1 block)
 */
extern
void gaussian_gpu( float *inout,
                   const int& h,
                   const int& w,
                   const float& s,
                   const initcond& ic = clamp,
                   const int& extb = 1 );

//-- BSpline ------------------------------------------------------------------

/**
 *  @ingroup api_gpu
 *  @brief Compute the Bicubic B-Spline interpolation of an image in the GPU
 *
 *  Given an input 2D image compute the Bicubic B-Spline interpolation
 *  of it by applying a first-order recursive filter using zero-border
 *  initial conditions.
 *
 *  @param[in,out] inout The 2D image to compute the Bicubic B-Spline interpolation
 *  @param[in] h Height of the input image
 *  @param[in] w Width of the input image
 *  @param[in] d Depth of the input image (color channels)
 *  @param[in] ic Initial condition (for outside access) (default mirror)
 *  @param[in] extb Extension (in blocks) to consider outside image (default 1 block)
 */
extern
void bspline3i_gpu( float **inout,
                    const int& h,
                    const int& w,
                    const int& d,
                    const initcond& ic = mirror,
                    const int& extb = 1 );

/**
 *  @ingroup api_gpu
 *  @overload
 *  @brief Compute the Bicubic B-Spline interpolation of a single-channel image in the GPU
 *
 *  @param[in,out] inout The single-channel 2D image to compute the Bicubic B-Spline interpolation
 *  @param[in] h Height of the input image
 *  @param[in] w Width of the input image
 *  @param[in] ic Initial condition (for outside access) (default mirror)
 *  @param[in] extb Extension (in blocks) to consider outside image (default 1 block)
 */
extern
void bspline3i_gpu( float *inout,
                    const int& h,
                    const int& w,
                    const initcond& ic = mirror,
                    const int& extb = 1 );

//=============================================================================
} // namespace gpufilter
//=============================================================================
#endif // GPUFILTER_H
//=============================================================================
