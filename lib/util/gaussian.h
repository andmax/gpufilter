/**
 *  @file gaussian.h
 *  @brief Gaussian blur via recursive filtering definition
 *  @author Rodolfo Lima
 *  @author Andre Maximo
 *  @date December, 2011
 *  @copyright The MIT License
 */

#ifndef GAUSSIAN_H
#define GAUSSIAN_H

//== INCLUDES ==================================================================

#include <cstdlib> // for rand

#include "linalg.h"

//== NAMESPACES ================================================================

namespace gpufilter {

//== DEFINITIONS ===============================================================

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
 *  @param[out] w Weights with feedforward and feedback first-order coefficients
 */
void weights(double s, Vector<float,3> &w);

/**
 *  @ingroup utils
 *  @overload void weights(double s, Vector<float,2> &w)
 *  @brief Compute first-order weights
 */
void weights(double s, Vector<float,2> &w);

/**
 *  @ingroup utils
 *  @overload void weights(double s, Vector<float,4> &w)
 *  @brief Compute third-order weights
 */
void weights(double s, Vector<float,4> &w);

//== IMPLEMENTATION ============================================================

/**
 *  @ingroup utils
 *  @overload void weights(double s, Vector<float,R> &w)
 *  @brief Compute random weights for any order (fourth or greater)
 *  @tparam R Filter order
 */
template <int R>
void weights(double s, Vector<float,R> &w) {

    float sum = 0;
    for(int i=0; i<R; ++i)
        sum += w[i] = rand()/(float)RAND_MAX;

    for(int i=0; i<R; ++i)
        w[i] /= sum;
}

//==============================================================================
} // namespace gpufilter
//==============================================================================
#endif // GAUSSIAN_H
//==============================================================================
