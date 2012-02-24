/**
 *  GPU definitions (implementation)
 *  @author Andre Maximo
 *  @author Rodolfo Lima
 *  @date March, 2011
 */

//== INCLUDES =================================================================

#include <vector>
#include <complex>

#include <util.h>
#include <symbol.h>

#include <gpudefs.h>

//== NAMESPACES ===============================================================

namespace gpufilter {

//== IMPLEMENTATION ===========================================================

__host__
void calc_alg_setup( alg_setup& algs,
                     const int& w,
                     const int& h ) {

    algs.width = w;
    algs.height = h;
    algs.m_size = (w+WS-1)/WS;
    algs.n_size = (h+WS-1)/WS;
    algs.last_m = algs.m_size-1;
    algs.last_n = algs.n_size-1;
    algs.border = 0;
    algs.carry_width = algs.m_size*WS;
    algs.carry_height = algs.n_size*WS;
    algs.carry_height = h;
    algs.inv_width = 1.f/(float)w;
    algs.inv_height = 1.f/(float)h;

}

__host__
void calc_alg_setup( alg_setup& algs,
                     const int& w,
                     const int& h,
                     const int& extb ) {

    int bleft, btop, bright, bbottom;
    calc_borders( bleft, btop, bright, bbottom, w, h, extb );

    algs.width = w;
    algs.height = h;
    algs.m_size = (w+bleft+bright+WS-1)/WS;
    algs.n_size = (h+btop+bbottom+WS-1)/WS;
    algs.last_m = (bleft+w-1)/WS;
    algs.last_n = (btop+h-1)/WS;
    algs.border = extb;
    algs.carry_width = algs.m_size*WS;
    algs.carry_height = algs.n_size*WS;
    algs.inv_width = 1.f/(float)w;
    algs.inv_height = 1.f/(float)h;

}

__host__
void up_alg_setup( const alg_setup& algs ) {

	copy_to_symbol("c_width", algs.width);
    copy_to_symbol("c_height", algs.height);
    copy_to_symbol("c_m_size", algs.m_size);
	copy_to_symbol("c_n_size", algs.n_size);
    copy_to_symbol("c_last_m", algs.last_m);
    copy_to_symbol("c_last_n", algs.last_n);
    copy_to_symbol("c_border", algs.border);
	copy_to_symbol("c_carry_width", algs.carry_width);
    copy_to_symbol("c_carry_height", algs.carry_height);
    copy_to_symbol("c_inv_width", algs.inv_width);
    copy_to_symbol("c_inv_height", algs.inv_height);

}

__host__
void up_constants_coefficients1( const float& b0,
                                 const float& a1 ) {

    copy_to_symbol("c_b0", b0);
    copy_to_symbol("c_a1", a1);
    copy_to_symbol("c_inv_b0", 1.f/b0);

    const int B = WS, R = 1;

    Vector<float,R+1> w;
    w[0] = b0;
    w[1] = a1;

    Matrix<float,R,R> Ir = identity<float,R,R>();
    Matrix<float,B,R> Zbr = zeros<float,B,R>();
    Matrix<float,R,B> Zrb = zeros<float,R,B>();
    Matrix<float,B,B> Ib = identity<float,B,B>();

    Matrix<float,R,B> AFP_T = fwd(Ir, Zrb, w),
                      ARE_T = rev(Zrb, Ir, w);
    Matrix<float,B,B> AFB_T = fwd(Zbr, Ib, w),
                      ARB_T = rev(Ib, Zbr, w);

    Matrix<float,R,R> AbF_T = tail<R>(AFP_T),
                      AbR_T = head<R>(ARE_T),
                      AbF = transp(AbF_T),
                      AbR = transp(AbR_T),
                      HARB_AFP_T = AFP_T*head<R>(ARB_T),
                      HARB_AFP = transp(HARB_AFP_T);
    Matrix<float,R,B> ARB_AFP_T = AFP_T*ARB_T,
                      TAFB = transp(tail<R>(AFB_T)),
                      HARB_AFB = transp(AFB_T*head<R>(ARB_T));

    copy_to_symbol("c_AbF", AbF);
    copy_to_symbol("c_AbR", AbR);
    copy_to_symbol("c_HARB_AFP", HARB_AFP);

    copy_to_symbol("c_ARE", ARE_T);
    copy_to_symbol("c_ARB_AFP_T", ARB_AFP_T);
    copy_to_symbol("c_TAFB", TAFB);
    copy_to_symbol("c_HARB_AFB", HARB_AFB);

}

__host__
void up_constants_coefficients2( const float& b0,
                                 const float& a1,
                                 const float& a2 ) {

    copy_to_symbol("c_b0", b0);
    copy_to_symbol("c_a1", a1);
    copy_to_symbol("c_a2", a2);
    copy_to_symbol("c_inv_b0", 1.f/b0);

    const int B = WS, R = 2;

    Vector<float,R+1> w;
    w[0] = b0;
    w[1] = a1;
    w[2] = a2;

    Matrix<float,R,R> Ir = identity<float,R,R>();
    Matrix<float,B,R> Zbr = zeros<float,B,R>();
    Matrix<float,R,B> Zrb = zeros<float,R,B>();
    Matrix<float,B,B> Ib = identity<float,B,B>();

    Matrix<float,R,B> AFP_T = fwd(Ir, Zrb, w),
                      ARE_T = rev(Zrb, Ir, w);
    Matrix<float,B,B> AFB_T = fwd(Zbr, Ib, w),
                      ARB_T = rev(Ib, Zbr, w);

    Matrix<float,R,R> AbF_T = tail<R>(AFP_T),
                      AbR_T = head<R>(ARE_T),
                      HARB_AFP_T = AFP_T*head<R>(ARB_T);

    copy_to_symbol("c_AbF2", AbF_T);
    copy_to_symbol("c_AbR2", AbR_T);
    copy_to_symbol("c_AFP_HARB", HARB_AFP_T);

}

//=============================================================================
} // namespace gpufilter
//=============================================================================
