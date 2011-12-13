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

void up_constants_sizes( dim3& g,
                         const int& h,
                         const int& w,
                         const int& p ) {

    g.x = (w+WS-1)/WS;
    g.y = (h+WS-1)/WS;

    copy_to_symbol("c_height", h);
	copy_to_symbol("c_width", w);
	copy_to_symbol("c_img_pitch", (p==0)?w:p);
    copy_to_symbol("c_m_size", g.y);
	copy_to_symbol("c_n_size", g.x);

}

void up_constants_texture( const int& h,
                           const int& w ) {

    copy_to_symbol("c_tex_height", 1.f / (float)h);
    copy_to_symbol("c_tex_width", 1.f / (float)w);

}

template <class T>
void mul(T R[2][2], const T A[2][2], const T B[2][2])
{
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

void calc_forward_matrix(float T[2][2], float n, float L, float M)
{
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

void calc_reverse_matrix(float T[2][2], float n, float L, float N)
{
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

void calc_forward_reverse_matrix(float T[2][2], int n, float L, float M, float N)
{
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

void up_constants_coefficients1( const float& b0,
                                 const float& a1 )
{
    copy_to_symbol("c_b0", b0);
    copy_to_symbol("c_a1", a1);
    copy_to_symbol("c_1_b0", 1.f/b0);

    const int B = 32, R = 1;

    Vector<R+1> w;
    w[0] = b0;
    w[1] = a1;

    Matrix<R,R> Ir = identity<R,R,float>();
    Matrix<B,R> Zbr = zeros<B,R,float>();
    Matrix<R,B> Zrb = zeros<R,B,float>();
    Matrix<B,B> Ib = identity<B,B,float>();

    Matrix<B,R> AFP = fwd(Ir, Zbr, w),
                ARE = rev(Zbr, Ir, w);
    Matrix<B,B> AFB = fwd(Zrb, Ib, w),
                ARB = rev(Ib, Zrb, w);

    Matrix<R,R> AbF = tail<R>(AFP),
                AbR = head<R>(ARE),
                HARB_AFP = head<R>(ARB)*AFP;
    Matrix<R,B> ARE_T = transp(ARE),
                ARB_AFP_T = transp(ARB*AFP),
                TAFB = tail<R>(AFB),
                HARB_AFB = head<R>(ARB)*AFB;

    copy_to_symbol("c_AbF", AbF[0][0]);
    copy_to_symbol("c_AbR", AbR[0][0]);
    copy_to_symbol("c_HARB_AFP", HARB_AFP[0][0]);
    copy_to_symbol("c_HARB_AFP_T", HARB_AFP[0][0]);

    copy_to_symbol("c_ARE_T", ARE_T[0].to_vector());
    copy_to_symbol("c_ARB_AFP_T", ARB_AFP_T[0].to_vector());
    copy_to_symbol("c_TAFB", TAFB[0].to_vector());
    copy_to_symbol("c_HARB_AFB", HARB_AFB[0].to_vector());
}

void up_constants_coefficients2( const float& b0,
                                 const float& a1,
                                 const float& a2 )
{
    const float Linf = a2, Ninf = a1/a2, Minf = a1, iR = b0*b0*b0*b0/Linf/Linf;

    copy_to_symbol("c_iR2", iR);
    copy_to_symbol("c_Linf2", Linf);
    copy_to_symbol("c_Minf", Minf);
    copy_to_symbol("c_Ninf", Ninf);

    copy_to_symbol("c_Llast2", Linf);

    float T[2][2];
    calc_forward_matrix(T, WS, Linf, Minf);
    copy_to_symbol("c_Af",std::vector<float>(&T[0][0], &T[0][0]+4));

    calc_reverse_matrix(T, WS, Linf, Ninf);
    copy_to_symbol("c_Ar",std::vector<float>(&T[0][0], &T[0][0]+4));

    calc_forward_reverse_matrix(T, WS, Linf, Minf, Ninf);
    copy_to_symbol("c_Arf",std::vector<float>(&T[0][0], &T[0][0]+4));
}

//=============================================================================
} // namespace gpufilter
//=============================================================================
