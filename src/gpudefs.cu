/**
 *  GPU definitions (implementation)
 *  @author Andre Maximo
 *  @author Rodolfo Lima
 *  @date March, 2011
 */

#include <vector>
#include <complex>

#include <util.h>
#include <symbol.h>

#include <gpudefs.h>

//== NAMESPACES ===============================================================

namespace gpufilter {

//== IMPLEMENTATION ===========================================================

__host__
void calc_borders( int& left,
                   int& top,
                   int& right,
                   int& bottom,
                   const int& h,
                   const int& w,
                   const int& extb ) {

    left = extb*WS;
    top = extb*WS;

    if( extb > 0 ) {

        right = (extb+1)*WS-(w%WS);
        bottom = (extb+1)*WS-(h%WS);

    } else {

        right = WS-(w%WS);
        if( right == WS ) right = 0;
        bottom = WS-(h%WS);
        if( bottom == WS ) bottom = 0;

    }

}

__host__
void up_constants_sizes( dim3& g,
                         const int& h,
                         const int& w ) {

    g.x = (w+WS-1)/WS; // m_size
    g.y = (h+WS-1)/WS; // n_size

	copy_to_symbol("c_width", w);
    copy_to_symbol("c_height", h);

    copy_to_symbol("c_m_size", g.x);
	copy_to_symbol("c_n_size", g.y);

}

__host__
void up_constants_sizes( dim3& g,
                         int& ext_h,
                         int& ext_w,
                         const int& h,
                         const int& w,
                         const int& extb ) {

    int bleft, btop, bright, bbottom;
    calc_borders( bleft, btop, bright, bbottom, h, w, extb );

    g.x = (w+bleft+bright+WS-1)/WS; // m_size
    g.y = (h+btop+bbottom+WS-1)/WS; // n_size

	copy_to_symbol("c_width", w);
    copy_to_symbol("c_height", h);

    copy_to_symbol("c_m_size", g.x);
	copy_to_symbol("c_n_size", g.y);

    copy_to_symbol("c_inv_width", 1.f / w);
    copy_to_symbol("c_inv_height", 1.f / h);

    copy_to_symbol("c_last_m", (bleft+w-1)/WS);
    copy_to_symbol("c_last_n", (btop+h-1)/WS);

    ext_w = g.x*WS;
    ext_h = g.y*WS;

	copy_to_symbol("c_carry_width", ext_w);
    copy_to_symbol("c_carry_height", ext_h);

    copy_to_symbol("c_border", extb);

}

/*
template <class T>
__host__
void mul( T R[2][2],
          const T A[2][2],
          const T B[2][2] ) {

    T aux[2][2];
    aux[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0];
    aux[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1];

    aux[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0];
    aux[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1];

    R[0][0] = aux[0][0];
    R[0][1] = aux[0][1];
    R[1][0] = aux[1][0];
    R[1][1] = aux[1][1];

}

__host__
void calc_forward_matrix( float T[2][2],
                          float n,
                          float L,
                          float M ) {
    if(n == 1)
    {
        T[0][0] = 0;
        T[0][1] = 1;
        T[1][0] = -L;
        T[1][1] = -M;
        return;
    }

    std::complex<float> delta = sqrt(std::complex<float>(M*M-4*L));

    std::complex<float> S[2][2] = {{1,1},
                     {-(delta+M)/2.f, (delta-M)/2.f}},
           iS[2][2] = {{(delta-M)/(2.f*delta), -1.f/delta},
                       {(delta+M)/(2.f*delta), 1.f/delta}};

    std::complex<float> LB[2][2] = {{-(delta+M)/2.f, 0},
                      {0,(delta-M)/2.f}};

    LB[0][0] = pow(LB[0][0], n);
    LB[1][1] = pow(LB[1][1], n);

    std::complex<float> cT[2][2];
    mul(cT, S, LB);
    mul(cT, cT, iS);

    T[0][0] = real(cT[0][0]);
    T[0][1] = real(cT[0][1]);
    T[1][0] = real(cT[1][0]);
    T[1][1] = real(cT[1][1]);

}

__host__
void calc_reverse_matrix( float T[2][2],
                          float n,
                          float L,
                          float N ) {

    if(n == 1)
    {
        T[0][0] = -L*N;
        T[0][1] = -L;
        T[1][0] = 1;
        T[1][1] = 0;
        return;
    }

    std::complex<float> delta = sqrt(std::complex<float>(L*L*N*N)-4*L);

    std::complex<float> S[2][2] = {{1,1},
                                   {(delta-L*N)/(2*L), -(delta+L*N)/(2*L)}},
        iS[2][2] = {{(delta+L*N)/(2.f*delta), L/delta},
                    {(delta-L*N)/(2.f*delta), -L/delta}};
                        
    std::complex<float> LB[2][2] = {{-(delta+L*N)/2.f, 0},
                       {0, (delta-L*N)/2.f}};

    LB[0][0] = pow(LB[0][0], n);
    LB[1][1] = pow(LB[1][1], n);

    std::complex<float> cT[2][2];
    mul(cT, S, LB);
    mul(cT, cT, iS);

    T[0][0] = real(cT[0][0]);
    T[0][1] = real(cT[0][1]);
    T[1][0] = real(cT[1][0]);
    T[1][1] = real(cT[1][1]);

}

__host__
void calc_forward_reverse_matrix( float T[2][2],
                                  int n,
                                  float L,
                                  float M,
                                  float N ) {

    using std::swap;

    float block_raw[WS+4], *block = block_raw+2;

    block[-2] = 1;
    block[-1] = 0;

    block[n] = block[n+1] = 0;

    for(int i=0; i<2; ++i)
    {
        for(int j=0; j<n; ++j)
            block[j] = -L*block[j-2] - M*block[j-1];

        for(int j=n-1; j>=0; --j)
            block[j] = (block[j] - block[j+1]*N - block[j+2])*L;

        T[0][i] = block[0];
        T[1][i] = block[1];

        swap(block[-1], block[-2]); // [0, 1]
    }

}
*/

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
