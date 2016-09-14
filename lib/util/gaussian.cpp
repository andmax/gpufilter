/**
 *  @file gaussian.cpp
 *  @brief Gaussian blur via recursive filtering implementation
 *  @author Rodolfo Lima
 *  @author Andre Maximo
 *  @date December, 2011
 *  @copyright The MIT License
 */

//== INCLUDES ==================================================================

#include <complex>

#include "recfilter.h"
#include "gaussian.h"

//== NAMESPACES ================================================================

namespace gpufilter {

//== DEFINITIONS ===============================================================

using std::complex;
using std::polar;

typedef std::complex<double> dcomplex;

//== IMPLEMENTATION ============================================================

namespace { // unnamed namespace

const dcomplex d1(1.41650, 1.00829);
const double d3(1.86543);

/**
 *  @ingroup utils
 *  @brief Compute recursive filtering scaling factor
 *
 *  Compute the scaling factor of the recursive filter representing a
 *  true Gaussian filter convolution with arbitrary support sigma.
 *
 *  @see [vanVliet:1998] cited in weights()
 *  @param[in] s Sigma support of the true Gaussian filter
 *  @return Scaling factor q of the recursive filter approximation
 *  @tparam T Sigma value type
 */
double qs(double s) {
    return .00399341 + .4715161*s;
}

/**
 *  @ingroup utils
 *  @brief Rescale poles in the real-axis of the recursive filtering z-transform
 *
 *  Given a real pole on |z|=1 ROC of the recursive filter z-transform
 *  compute a rescaled pole representing a true Gaussian filter
 *  convolution with arbitrary support sigma.
 *
 *  @see [vanVliet:1998] cited in weights()
 *  @param[in] d Real pole of a stable recursive filter
 *  @param[in] s Sigma support of the true Gaussian filter
 *  @return Rescaled real pole of the recursive filter approximation
 */
double ds(double d, double s) {
    return pow(d, 1.0/qs(s));
}

/**
 *  @ingroup utils
 *  @overload dcomplex ds(dcomplex d, double s)
 *  @brief Rescale poles of the recursive filtering z-transform
 *
 *  Given a complex-valued pole on |z|=1 ROC of the recursive filter
 *  z-transform compute a rescaled pole representing a true Gaussian
 *  filter convolution with arbitrary support sigma.
 *
 *  @see [vanVliet:1998] cited in weights()
 *  @param[in] d Complex-valued pole of a stable recursive filter
 *  @param[in] s Sigma support of the true Gaussian filter
 *  @return Rescaled complex-valued pole of the recursive filter approximation
 */
dcomplex ds(dcomplex d, double s) {
    double q = qs(s);
    return std::polar(pow(abs(d),1.0/q), arg(d)/q);
}

} // unnamed namespace

/**
 *  @ingroup utils
 *  @brief Returns the sign of the base recursive filter operation
 *  @see rec_op()
 *  @return -1 for minus sign, and +1 for plus sign
 */
int rec_op_sign() {
    return (rec_op(1,1)==0) ? -1 : 1;
}

//-- Weights -------------------------------------------------------------------

void weights(double s, Vector<float,2> &w) 
{
    double d = ds(d3, s);

    int sign = rec_op_sign();

    w[0] = static_cast<float>(-(1.0-d)/d);
    w[1] = sign*static_cast<float>(1.0/d);
}

void weights(double s, Vector<float,3> &w) 
{
    dcomplex d = ds(d1, s);
    double n2 = abs(d); 
    n2 *= n2;
    double re = real(d);

    int sign = rec_op_sign();

    w[0] = static_cast<float>((1-2*re+n2)/n2);
    w[1] = sign*static_cast<float>(2*re/n2);
    w[2] = sign*static_cast<float>(-1/n2);
}

void weights(double s, Vector<float,4> &w) 
{
    Vector<float, 2> w1;
    Vector<float, 3> w2;
    weights(s, w1);
    weights(s, w2);

    // computing N=3 based on first and second orders
    // z_i = (b10 b20) x_i
    //     - (a11 + a21) z_{i-1} 
    //     - (a11 a21 + a22) z_{i-2}
    //     - (a11 a22) z_{i-3}
    // thx Diego :)
    w[0] = w1[0] * w2[0];
    w[1] = w1[1] + w2[1];
    w[2] = w1[1] * w2[1] + w2[2];
    w[3] = w1[1] * w2[2];
}

//==============================================================================
} // namespace gpufilter
//==============================================================================
