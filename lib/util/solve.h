/**
 *  @file solve.h
 *  @brief Solve tridiagonal spline definition and implementation
 *  @author Andre Maximo
 *  @date Dec, 2012
 *  @copyright The MIT License
 */

#ifndef SOLVE_H
#define SOLVE_H

//== NAMESPACES ================================================================

/**
 *  @ingroup utils
 *  @brief The namespace spline includes other attempts to solve with borders
 */
namespace spline {

//=== IMPLEMENTATION ===========================================================

const float l[] = { 0.2f, 0.26315789f, 0.26760563f, 0.26792453f,
                    0.26794742f, 0.26794907f, 0.26794918f, 0.26794919f };
const int ln = sizeof(l)/sizeof(l[0]);
const float p = 1/6.f, q = 4*p, v = 4.73205078f;
const float linf = l[ln-1], pinv = 1/p, vinv = 1/v, pinv_v = pinv*vinv;

// unser_etal_pami1991 (3.13) and (3.16) define a cascaded (sum) and
// parallel (mul) solutions for recursive filtering
const float alpha = sqrt(3)-2, mb0=-6*alpha,
    sb0 = mb0/(1-alpha*alpha), b1=alpha;

// nehab_etal_tog2011 (introduction) and (1) define recursive
// operation with subtraction of previous elements
const float w0=sqrt(mb0), w1=-b1;

enum solve_type {
    traditional = 0, unser313, unser316, unser318, nehab6
};

/**
 *  @ingroup utils
 *  @brief Compute [UnserEtAl:1991] section 3.13
 *
 *  Three attempts were made by [Unser et al. 1991] in their sections
 *  3.13, 3.16 and 3.18, by obtaining the exact anticausal
 *  initialization for the cascaded decomposition of cubic B-spline
 *  interpolation filters in the even-periodic extension case (causal
 *  initialization is still approximated by padding).  This function
 *  implements the section 3.13.  The reference for the 1991 article
 *  is:
 *
 *  @verbatim
UNSER, M., ALDROUBI, A., and EDEN, M. 1991. Fast B-spline
transforms for continuous image representation and interpolation.
IEEE Transactions on Pattern Analysis and Machine Intelligence,
13(3):277–285.
    @endverbatim
 *
 *  @param[in,out] inout The in(out)put 2D image to filter
 *  @param[in] w Image width
 *  @param[in] h Image height
 *  @tparam T Image value type
 */
template <class T>
void unser_etal_pami1991_3_13( T *inout,
                               const int& w,
                               const int& h ) {
    for (int y=0; y<h; ++y) {
        for (int x=1; x<w; ++x)
            inout[y*w+x] += b1*inout[y*w+x-1];
        for (int x=w-2; x>=0; --x)
            inout[y*w+x] += b1*inout[y*w+x+1];
        for (int x=0; x<w; ++x)
            inout[y*w+x] *= mb0;
    }
    for (int x=0; x<w; ++x) {
        for (int y=1; y<h; ++y)
            inout[y*w+x] += b1*inout[(y-1)*w+x];
        for (int y=h-2; y>=0; --y)
            inout[y*w+x] += b1*inout[(y+1)*w+x];
        for (int y=0; y<h; ++y)
            inout[y*w+x] *= mb0;
    }
}

/**
 *  @brief Compute [UnserEtAl:1991] section 3.16
 *
 *  @see [UnserEtAl:1991] cited in unser_etal_pami1991_3_13()
 *  @param[in,out] inout The in(out)put 2D image to filter
 *  @param[in] w Image width
 *  @param[in] h Image height
 *  @tparam T Image value type
 */
template <class T>
void unser_etal_pami1991_3_16( T *inout,
                               const int& w,
                               const int& h ) {
    T *cplus = new T[w*h], *cminus = new T[w*h];
    for (int y=0; y<h; ++y) {
        cplus[y*w+0] = inout[y*w+0];
        for (int x=1; x<w; ++x)
            cplus[y*w+x] = inout[y*w+x]+b1*cplus[y*w+x-1];
        cminus[y*w+w-1] = inout[y*w+w-1];
        for (int x=w-2; x>=0; --x)
            cminus[y*w+x] = inout[y*w+x]+b1*cminus[y*w+x+1];
        for (int x=0; x<w; ++x)
            inout[y*w+x] = sb0*(cplus[y*w+x]+cminus[y*w+x]-inout[y*w+x]);
    }
    for (int x=0; x<w; ++x) {
        cplus[x] = inout[x];
        for (int y=1; y<h; ++y)
            cplus[y*w+x] = inout[y*w+x]+b1*cplus[(y-1)*w+x];
        cminus[(h-1)*w+x] = inout[(h-1)*w+x];
        for (int y=h-2; y>=0; --y)
            cminus[y*w+x] = inout[y*w+x]+b1*cminus[(y+1)*w+x];
        for (int y=0; y<h; ++y)
            inout[y*w+x] = sb0*(cplus[y*w+x]+cminus[y*w+x]-inout[y*w+x]);
    }
    delete [] cplus;
    delete [] cminus;
}

/**
 *  @brief Compute [UnserEtAl:1991] section 3.18
 *
 *  There is a mistake in the section, it is not minus alpha in the
 *  first equation.
 *
 *  @see [UnserEtAl:1991] cited in unser_etal_pami1991_3_13()
 *  @param[in,out] inout The in(out)put 2D image to filter
 *  @param[in] w Image width
 *  @param[in] h Image height
 *  @tparam T Image value type
 */
template <class T>
void unser_etal_pami1991_3_18( T *inout,
                               const int& w,
                               const int& h ) {
    T *dplus = new T[w*h];
    for (int y=0; y<h; ++y) {
        dplus[y*w+0] = 6*inout[y*w+0];
        for (int x=1; x<w; ++x)
            dplus[y*w+x] = 6*inout[y*w+x]+alpha*dplus[y*w+x-1];
        inout[y*w+w-1] = (-alpha/(1-alpha*alpha))*(2*dplus[y*w+w-1]-6*inout[y*w+w-1]);
        for (int x=w-2; x>=0; --x)
            inout[y*w+x] = alpha*(inout[y*w+x+1]-dplus[y*w+x]);
    }
    for (int x=0; x<w; ++x) {
        dplus[x] = 6*inout[x];
        for (int y=1; y<h; ++y)
            dplus[y*w+x] = 6*inout[y*w+x]+alpha*dplus[(y-1)*w+x];
        inout[(h-1)*w+x] = (-alpha/(1-alpha*alpha))*(2*dplus[(h-1)*w+x]-6*inout[(h-1)*w+x]);
        for (int y=h-2; y>=0; --y)
            inout[y*w+x] = alpha*(inout[(y+1)*w+x]-dplus[y*w+x]);
    }
    delete [] dplus;
}

/**
 *  @ingroup utils
 *  @brief Compute [NehabHoppe:2011] section 6
 *
 *  Another attempt done by [NehabHoppe:2011] in section 6, explores
 *  the fast convergence of filter coefficients, that is proportional
 *  to the impulse response decays.  This attempt is implemented in
 *  this function; and also it is the base for the algorithm 5
 *  modified to support varying filter coefficients: alg5varc.  The
 *  reference for the 2011 article is:
 *
 *  @verbatim
NEHAB, D. and HOPPE, H. 2011. Generalized sampling in computer
graphics. IMPA Tech. Report E022/2011 & Microsoft Research
MSR-TR-2011-16, February 2011.
    @endverbatim
 *
 *  @param[in,out] inout The in(out)put 2D image to filter
 *  @param[in] w Image width
 *  @param[in] h Image height
 *  @tparam T Image value type
 */
template <class T>
void nehab_hoppe_tr2011_sec6( T *inout,
                              const int& w,
                              const int& h ) {
    for (int y=0; y<h; ++y) {
        for (int x=1; x<=ln; ++x)
            inout[y*w+x] -= l[x-1]*inout[y*w+x-1];
        for (int x=ln+1; x<w; ++x)
            inout[y*w+x] -= linf*inout[y*w+x-1];
        inout[y*w+w-1] = pinv_v*inout[y*w+w-1];
        for (int x=w-2; x>=ln; --x)
            inout[y*w+x] = linf*(pinv*inout[y*w+x]-inout[y*w+x+1]);
        for (int x=ln-1; x>=0; --x)
            inout[y*w+x] = l[x]*(pinv*inout[y*w+x]-inout[y*w+x+1]);
    }
    for (int x=0; x<w; ++x) {
        for (int y=1; y<=ln; ++y)
            inout[y*w+x] -= l[y-1]*inout[(y-1)*w+x];
        for (int y=ln+1; y<h; ++y)
            inout[y*w+x] -= linf*inout[(y-1)*w+x];
        inout[(h-1)*w+x] = pinv_v*inout[(h-1)*w+x];
        for (int y=h-2; y>=ln; --y)
            inout[y*w+x] = linf*(pinv*inout[y*w+x]-inout[(y+1)*w+x]);
        for (int y=ln-1; y>=0; --y)
            inout[y*w+x] = l[y]*(pinv*inout[y*w+x]-inout[(y+1)*w+x]);
    }
}

//==============================================================================
} // namespace spline
//==============================================================================
#endif // SOLVE_H
//==============================================================================
