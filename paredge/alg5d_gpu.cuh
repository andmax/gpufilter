/**
 *  @file alg5d_gpu.cuh
 *  @brief Algorithm 5 in the GPU (Deriche version) with borders
 *  @author Andre Maximo
 *  @date Jun, 2014
 */

#ifndef ALG5D_GPU_CUH
#define ALG5D_GPU_CUH

//=== IMPLEMENTATION ============================================================

// Alg5 (Deriche) GPU -----------------------------------------------------------

// Collect carries --------------------------------------------------------------

template <int R>
__global__ __launch_bounds__(WS*NWC, NBCW)
void alg5d_collect_carries( Matrix<float,R,WS> *g_py1bar,
                            Matrix<float,R,WS> *g_pty1bar,
                            Matrix<float,R,WS> *g_ety1bar,
                            Matrix<float,R,WS> *g_ey2bar,
                            Matrix<float,R,WS> *g_pty2bar,
                            Matrix<float,R,WS> *g_ety2bar,
                            Matrix<float,R,WS> *g_ptu11hat,
                            Matrix<float,R,WS> *g_etu12hat,
                            Matrix<float,R,WS> *g_etu22hat,
                            Matrix<float,R,WS> *g_ptu21hat,
                            Matrix<float,R,WS> *g_px,
                            Matrix<float,R,WS> *g_ex,
                            float inv_width, float inv_height,
                            int m_size, int n_size ) {

    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n = blockIdx.y;

    __shared__ Matrix<float,WS,WS+1> block;
    read_block<NWC>(block, m, n, inv_width, inv_height);
    __syncthreads();

    float x[32]; // 32 regs per thread

    if (ty==0) { // 1 warp forward computing y1

#pragma unroll
        for (int i=0; i<32; ++i)
            x[i] = block[tx][i];

        float xp = 0.f; // previous x
        Vector<float,R> py1; // current P(Y_1)
        Vector<float,R> px; // this block P(X)

        // for now, no boundary condition (zero)

#pragma unroll
        for (int i=0; i<R; ++i)
            px[i] = x[WS-1-i];

#pragma unroll // calculate py1bar, scan left -> right
        for (int j=0; j<WS; ++j) {
            float xc = x[j]; // current x
            x[j] = c_coefs[0]*xc + c_coefs[1]*xp;
            x[j] = fwdI(py1, x[j], c_weights);
            xp = xc;
        }

        g_py1bar[n*(m_size+1)+m+1].set_col(tx, py1);
        g_px[n*(m_size+1)+m+1].set_col(tx, px);

    } else if (ty==1) { // 1 warp reverse computing y2

#pragma unroll
        for (int i=0; i<32; ++i)
            x[i] = block[tx][i];

        float xn = 0.f, xa = 0.f; // next and after next x
        Vector<float,R> ey2; // current E(Y_2)
        Vector<float,R> ex; // this block E(X)

#pragma unroll
        for (int i=0; i<R; ++i)
            ex[i] = x[i];

#pragma unroll // calculate ey2bar, scan right -> left
        for (int j=WS-1; j>=0; --j) {
            float xc = x[j]; // current x
            x[j] = c_coefs[2]*xn + c_coefs[3]*xa;
            x[j] = revI(x[j], ey2, c_weights);
            xa = xn;
            xn = xc;
        }

        g_ey2bar[n*(m_size+1)+m].set_col(tx, ey2);
        g_ex[n*(m_size+1)+m].set_col(tx, ex);

    }

    __syncthreads();

    if (ty==0) { // save y1 in smem

#pragma unroll
        for (int i=0; i<32; ++i)
            block[tx][i] = x[i];

    }

    __syncthreads();

    if (ty==0) { //  1 warp forward-down computing u11

#pragma unroll // transpose regs
        for (int i=0; i<32; ++i)
            x[i] = block[i][tx];

        float xp = 0.f; // previous x
        Vector<float,R> ptu11; // current Pt(U_11)
        Vector<float,R> pty1; // this block Pt(Y1)

#pragma unroll
        for (int i=0; i<R; ++i)
            pty1[i] = x[WS-1-i];

#pragma unroll // calculate pu11hat, scan top -> bottom
        for (int j=0; j<WS; ++j) {
            float xc = x[j]; // current x
            x[j] = c_coefs[0]*xc + c_coefs[1]*xp;
            x[j] = fwdI(ptu11, x[j], c_weights);
            xp = xc;
        }

        g_ptu11hat[m*(n_size+1)+n+1].set_col(tx, ptu11);
        g_pty1bar[m*(n_size+1)+n+1].set_col(tx, pty1);

    } else if (ty==2) { //  1 warp forward-up computing u12

#pragma unroll // transpose regs
        for (int i=0; i<32; ++i)
            x[i] = block[i][tx];
        
        float xn = 0.f, xa = 0.f; // next and after next x
        Vector<float,R> etu12; // current Et(U_12)
        Vector<float,R> ety1; // this block Et(Y1)

#pragma unroll
        for (int i=0; i<R; ++i)
            ety1[i] = x[i];

#pragma unroll // calculate eu12hat, scan bottom -> top
        for (int j=WS-1; j>=0; --j) {
            float xc = x[j]; // current x
            x[j] = c_coefs[2]*xn + c_coefs[3]*xa;
            x[j] = revI(x[j], etu12, c_weights);
            xa = xn;
            xn = xc;
        }

        g_etu12hat[m*(n_size+1)+n].set_col(tx, etu12);
        g_ety1bar[m*(n_size+1)+n].set_col(tx, ety1);

    }

    __syncthreads();

    if (ty==1) { // save y2 in smem

#pragma unroll
        for (int i=0; i<32; ++i)
            block[tx][i] = x[i];

    }

    __syncthreads();

    if (ty==1) { //  1 warp reverse-up computing u22

#pragma unroll // transpose regs
        for (int i=0; i<32; ++i)
            x[i] = block[i][tx];

        float xn = 0.f, xa = 0.f; // next and after next x
        Vector<float,R> etu22; // current Et(U_22)
        Vector<float,R> ety2; // this block Et(Y2)

#pragma unroll
        for (int i=0; i<R; ++i)
            ety2[i] = x[i];

#pragma unroll // calculate eu12hat, scan bottom -> top
        for (int j=WS-1; j>=0; --j) {
            float xc = x[j]; // current x
            x[j] = c_coefs[2]*xn + c_coefs[3]*xa;
            x[j] = revI(x[j], etu22, c_weights);
            xa = xn;
            xn = xc;
        }

        g_etu22hat[m*(n_size+1)+n].set_col(tx, etu22);
        g_ety2bar[m*(n_size+1)+n].set_col(tx, ety2);

    } else if (ty==3) { //  1 warp reverse-down computing u21

#pragma unroll // transpose regs
        for (int i=0; i<32; ++i)
            x[i] = block[i][tx];

        float xp = 0.f; // previous x
        Vector<float,R> ptu21; // current Pt(U_21)
        Vector<float,R> pty2; // this block Pt(Y2)

#pragma unroll
        for (int i=0; i<R; ++i)
            pty2[i] = x[WS-1-i];

#pragma unroll // calculate pu11hat, scan top -> bottom
        for (int j=0; j<WS; ++j) {
            float xc = x[j]; // current x
            x[j] = c_coefs[0]*xc + c_coefs[1]*xp;
            x[j] = fwdI(ptu21, x[j], c_weights);
            xp = xc;
        }

        g_ptu21hat[m*(n_size+1)+n+1].set_col(tx, ptu21);
        g_pty2bar[m*(n_size+1)+n+1].set_col(tx, pty2);

    }

}

// Adjust carries rows ----------------------------------------------------------

template <int R>
__global__ __launch_bounds__(WS*NWA, NBA)
void alg5d_adjust_carries_rows( Matrix<float,R,WS> *g_py1bar,
                                Matrix<float,R,WS> *g_ey2bar,
                                Matrix<float,R,WS> *g_px,
                                Matrix<float,R,WS> *g_ex,
                                int m_size ) {

    int tx = threadIdx.x, ty = threadIdx.y, m, n = blockIdx.y;
    Matrix<float,R,WS> *gpy1bar, *gey2bar;
    Vector<float,R> py1, ey2, py1bar, ey2bar;
    __shared__ Matrix<float,R,WS> spy1ey2bar[NWA];

    Matrix<float,R,WS> *gpx, *gex;
    Vector<float,R> px, ex;
    __shared__ Matrix<float,R,WS> spxex[NWA];

    // P(y1bar) -> P(y1) processing ---------------------------------------------
    m = 0;
    gpy1bar = (Matrix<float,R,WS> *)&g_py1bar[n*(m_size+1)+m+ty+1][0][tx];
    gpx = (Matrix<float,R,WS> *)&g_px[n*(m_size+1)+m+ty][0][tx];
    py1 = zeros<float,R>(); // zero boundary condition

    for (; m < m_size; m += NWA) { // for all image blocks

        if (m+ty < m_size) { // using smem as cache
            spy1ey2bar[ty].set_col(tx, gpy1bar->col(0));
            spxex[ty].set_col(tx, gpx->col(0));
        }

        __syncthreads(); // wait to load smem

        if (ty==0) {
#pragma unroll // adjust py1bar left -> right
            for (int w = 0; w < NWA; ++w) {

                py1bar = spy1ey2bar[w].col(tx);
                px = spxex[w].col(tx);

                py1 = py1bar + py1 * c_AbF_T + px * c_TAFB_AC1P_T;

                spy1ey2bar[w].set_col(tx, py1);

            }
        }

        __syncthreads(); // wait to store result in gmem

        if (m+ty < m_size) gpy1bar->set_col(0, spy1ey2bar[ty].col(tx));

        gpy1bar += NWA;
        gpx += NWA;

    }

    // E(y2bar) -> E(y2) processing ---------------------------------------------
    m = m_size-1;
    gey2bar = (Matrix<float,R,WS> *)&g_ey2bar[n*(m_size+1)+m-ty][0][tx];
    gex = (Matrix<float,R,WS> *)&g_ex[n*(m_size+1)+m+1-ty][0][tx];
    ey2 = zeros<float,R>();

    for (; m >= 0; m -= NWA) { // for all image blocks

        if (m-ty >= 0) { // using smem as cache
            spy1ey2bar[ty].set_col(tx, gey2bar->col(0));
            spxex[ty].set_col(tx, gex->col(0));
        }

        __syncthreads(); // wait to load smem

        if (ty==0) {
#pragma unroll // adjust ey2bar right -> left
            for (int w = 0; w < NWA; ++w) {

                ey2bar = spy1ey2bar[w].col(tx);
                ex = spxex[w].col(tx);

                ey2 = ey2bar + ey2 * c_AbR_T + ex * c_HARB_AC2E_T;

                spy1ey2bar[w].set_col(tx, ey2);

            }
        }

        __syncthreads(); // wait to store result in gmem

        if (m-ty >= 0) gey2bar->set_col(0, spy1ey2bar[ty].col(tx));

        gey2bar -= NWA;
        gex -= NWA;

    }

}

// Adjust carries columns -------------------------------------------------------

template <int R>
__global__ __launch_bounds__(WS*NWA, NBAC)
void alg5d_adjust_carries_cols( Matrix<float,R,WS> *g_py1,
                                Matrix<float,R,WS> *g_pty1bar,
                                Matrix<float,R,WS> *g_ety1bar,
                                Matrix<float,R,WS> *g_ey2,
                                Matrix<float,R,WS> *g_pty2bar,
                                Matrix<float,R,WS> *g_ety2bar,
                                Matrix<float,R,WS> *g_ptu11hat,
                                Matrix<float,R,WS> *g_etu12hat,
                                Matrix<float,R,WS> *g_etu22hat,
                                Matrix<float,R,WS> *g_ptu21hat,
                                Matrix<float,R,WS> *g_px,
                                Matrix<float,R,WS> *g_ex,
                                int m_size, int n_size ) {

    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n;
    Matrix<float,R,WS> *gptu11hat, *gptu21hat;//, *getu12hat, *getu22hat;
    Vector<float,R> ptu11hat, ptu21hat, ptu1hat, etu2hat;
    Vector<float,R> pty1bar, pty2bar;
    Vector<float,R> ptz, etz, ptu1, etu2;

    Matrix<float,R,WS> *gpty1bar, *gpty2bar;//, *gety1bar, *gety2bar;
    Matrix<float,R,WS> *gpx, *gex, *gpy1, *gey2;
    Vector<float,R> px, ex, py1, ey2;

    __shared__ Matrix<float,R,WS> spx[NWA], sex[NWA], spy1[NWA], sey2[NWA];
    __shared__ Matrix<float,R,WS> spt1[NWA], spt2[NWA], set1[NWA], set2[NWA];

    // store (pty1bar, pty2bar) or (ptu11hat, ptu21hat) in (spt1, spt2)
    // store (ety1bar, ety2bar) or (etu22hat, etu12hat) in (set1, set2)
    // store ptu1 in ptu11hat and etu2 in etu22hat

    // Pt(u11hat) -> Pt(u1) processing ------------------------------------------
    n = 0;
    gptu11hat = (Matrix<float,R,WS> *)&g_ptu11hat[m*(n_size+1)+n+1+ty][0][tx];
    gptu21hat = (Matrix<float,R,WS> *)&g_ptu21hat[m*(n_size+1)+n+1+ty][0][tx];
    gpty1bar = (Matrix<float,R,WS> *)&g_pty1bar[m*(n_size+1)+n+1+ty][0][tx];
    gpty2bar = (Matrix<float,R,WS> *)&g_pty2bar[m*(n_size+1)+n+1+ty][0][tx];
    gpx = (Matrix<float,R,WS> *)&g_px[(n+ty)*(m_size+1)+m+0][0][tx];
    gex = (Matrix<float,R,WS> *)&g_ex[(n+ty)*(m_size+1)+m+1][0][tx];
    gpy1 = (Matrix<float,R,WS> *)&g_py1[(n+ty)*(m_size+1)+m+0][0][tx];
    gey2 = (Matrix<float,R,WS> *)&g_ey2[(n+ty)*(m_size+1)+m+1][0][tx];

    ptu1 = ptz = zeros<float,R>(); // zero boundary condition

    for (; n < n_size; n += NWA) { // for all image blocks

        if (n+ty < n_size) { // using smem as cache
            spt1[ty].set_col(tx, gptu11hat->col(0));
            spt2[ty].set_col(tx, gptu21hat->col(0));
            spy1[ty].set_col(tx, gpy1->col(0));
            sey2[ty].set_col(tx, gey2->col(0));
            spx[ty].set_col(tx, gpx->col(0));
            sex[ty].set_col(tx, gex->col(0));
        }

        __syncthreads(); // wait to load smem

        if (ty == 0) {
#pragma unroll // adjust ptucheck top -> bottom
            for (int w = 0; w < NWA; ++w) {

                ptu11hat = spt1[w].col(tx);
                ptu21hat = spt2[w].col(tx);
                py1 = spy1[w].col(tx);
                ey2 = sey2[w].col(tx);
                px = spx[w].col(tx);
                ex = sex[w].col(tx);

                ptz = pty1bar + pty2bar + ey2 * c_ARE_T.col(tx) + py1 * c_AFP_T.col(tx);

                ptu1hat = ptu11hat + ptu21hat;
                fixpet(ptu1hat, c_TAFB_AC1B, c_AFP_T, py1);
                fixpet(ptu1hat, c_TAFB_AC1B, c_ARE_T, ey2);
                fixpet(ptu1hat, c_TAFB_AC1B, c_AC1P_AFB_T, px);
                fixpet(ptu1hat, c_TAFB_AC1B, c_AC2E_ARB_T, ex);

                ptu1 = ptu1hat + ptu1 * c_AbF_T + ptz * c_TAFB_AC1P_T;

                spt1[w].set_col(tx, ptu1);

            }
        }

        __syncthreads(); // wait to store result in gmem

        if (n+ty < n_size) gptu11hat->set_col(0, spt1[ty].col(tx));

        gptu11hat += NWA;
        //gpx, gex...
        gpy1 += NWA*(m_size+1);
        gey2 += NWA*(m_size+1);

    }
/*
    // Et(vtilde) -> Et(v) processing -------------------------------------------
    n = n_size-1;
    getvtilde = (Matrix<float,R,WS> *)&g_etvtilde[m*(n_size+1)+n-ty][0][tx];
    gptucheck = (Matrix<float,R,WS> *)&g_ptucheck[m*(n_size+1)+n-ty][0][tx];
    gpy = (Matrix<float,R,WS> *)&g_py[(n-ty)*(m_size+1)+m][0][tx];
    gez = (Matrix<float,R,WS> *)&g_ez[(n-ty)*(m_size+1)+m+1][0][tx];
    etv = zeros<float,R>();

    for (; n >= 0; n -= NWA) { // for all image blocks

        if (n-ty >= 0) { // using smem as cache
            setvtilde[ty].set_col(tx, getvtilde->col(0));
            sptucheck[ty].set_col(tx, gptucheck->col(0));
            spy[ty].set_col(tx, gpy->col(0));
            sez[ty].set_col(tx, gez->col(0));
        }

        __syncthreads(); // wait to load smem

        if (ty == 0) {
#pragma unroll // adjust etvtilde bottom -> top
            for (int w = 0; w < NWA; ++w) {

                etvtilde = setvtilde[w].col(tx);
                ptu = sptucheck[w].col(tx);
                py = spy[w].col(tx);
                ez = sez[w].col(tx);

                etv = etvtilde + etv * c_AbR_T + ptu * c_HARB_AFP_T;

                fixpet(etv, c_HARB_AFB, c_ARE_T, ez);
                fixpet(etv, c_HARB_AFB, c_ARB_AFP_T, py);

                setvtilde[w].set_col(tx, etv);

            }
        }

        __syncthreads(); // wait to store result in gmem

        if (n-ty >= 0) getvtilde->set_col(0, setvtilde[ty].col(tx));

        getvtilde -= NWA;
        gptucheck -= NWA;
        gpy -= NWA*(m_size+1);
        gez -= NWA*(m_size+1);

    }
*/
}

// Write results ----------------------------------------------------------------

template <int R>
__global__ __launch_bounds__(WS*NWW, NBCW)
void alg5d_write_results( float *g_out,
                          const Matrix<float,R,WS> *g_py1,
                          const Matrix<float,R,WS> *g_ey2,
                          const Matrix<float,R,WS> *g_ptu1,
                          const Matrix<float,R,WS> *g_etu2,
                          const Matrix<float,R,WS> *g_px,
                          const Matrix<float,R,WS> *g_ex,
                          const Matrix<float,R,WS> *g_ptz,
                          const Matrix<float,R,WS> *g_etz,
                          float inv_width, float inv_height,
                          int m_size, int n_size,
                          int out_stride ) {

    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x, n = blockIdx.y;

    __shared__ Matrix<float,WS,WS+1> block;
    read_block<NWW>(block, m, n, inv_width, inv_height);
    __syncthreads();

    float x[32]; // 32 regs per thread

    Matrix<float,R,WS> 
        *py1m1 = (Matrix<float,R,WS>*)&g_py1[n*(m_size+1)+m][0][tx],
        *ey2m1 = (Matrix<float,R,WS>*)&g_ey2[n*(m_size+1)+m+1][0][tx],
        *ptu1m1 = (Matrix<float,R,WS>*)&g_ptu1[m*(n_size+1)+n][0][tx],
        *etu2m1 = (Matrix<float,R,WS>*)&g_etu2[m*(n_size+1)+n+1][0][tx];

    Matrix<float,R,WS>
        *pxm1 = (Matrix<float,R,WS>*)&g_px[n*(m_size+1)+m][0][tx],
        *exm1 = (Matrix<float,R,WS>*)&g_ex[n*(m_size+1)+m+1][0][tx],
        *ptzm1 = (Matrix<float,R,WS>*)&g_ptz[m*(n_size+1)+n][0][tx],
        *etzm1 = (Matrix<float,R,WS>*)&g_etz[m*(n_size+1)+n+1][0][tx];

    if (ty==0) { // 1 warp forward computing y1

#pragma unroll
        for (int i=0; i<32; ++i)
            x[i] = block[tx][i];

        float xp; // previous x
        Vector<float,R> py1 = py1m1->col(0); // current P(Y_1)
        Vector<float,R> px = pxm1->col(0); // previous block P(X)

        xp = px[0];

#pragma unroll // calculate block, scan left -> right
        for (int j=0; j<WS; ++j) {
            float xc = x[j]; // current x
            x[j] = c_coefs[0]*xc + c_coefs[1]*xp;
            x[j] = fwdI(py1, x[j], c_weights);
            xp = xc;
        }

    } else if (ty==1) { // 1 warp reverse computing y2

#pragma unroll
        for (int i=0; i<32; ++i)
            x[i] = block[tx][i];

        float xn, xa; // next and after next x
        Vector<float,R> ey2 = ey2m1->col(0); // current E(Y_2)
        Vector<float,R> ex = exm1->col(0); // next block E(X)

        xn = ex[0];
        if (R==1) xa = xn;
        else xa = ex[1];

#pragma unroll // calculate block, scan right -> left
        for (int j=WS-1; j>=0; --j) {
            float xc = x[j]; // current x
            x[j] = c_coefs[2]*xn + c_coefs[3]*xa;
            x[j] = revI(x[j], ey2, c_weights);
            xa = xn;
            xn = xc;
        }

    }

    // preparing for adding y1 and y2

    __syncthreads(); // guarantee no smem usage

    if (ty==1) {

#pragma unroll
        for (int i=0; i<32; ++i)
            block[tx][i] = x[i]; // save partial result in smem

    }

    __syncthreads(); // guarantee warp 0 can read smem

    if (ty==0) { // now add

#pragma unroll
        for (int i=0; i<32; ++i)
            block[tx][i] = x[i] + block[tx][i]; // smem now has z

    }

    __syncthreads(); // guarantee no smem usage

    if (ty==0) { // 1 warp down computing u1

#pragma unroll // transpose regs
        for (int i=0; i<32; ++i)
            x[i] = block[i][tx];

        float xp; // previous x
        Vector<float,R> ptu1 = ptu1m1->col(0); // current Pt(U_1)
        Vector<float,R> ptz = pxm1->col(0); // previous block Pt(Z)

        xp = ptz[0];

#pragma unroll // calculate block, scan top -> bottom
        for (int j=0; j<WS; ++j) {
            float xc = x[j]; // current x
            x[j] = c_coefs[0]*xc + c_coefs[1]*xp;
            x[j] = fwdI(ptu1, x[j], c_weights);
            xp = xc;
        }

    } else if (ty==1) { // 1 warp up computing u2

#pragma unroll // transpose regs
        for (int i=0; i<32; ++i)
            x[i] = block[i][tx];

        float xn, xa; // next and after next x
        Vector<float,R> etu2 = etu2m1->col(0); // current Et(U_2)
        Vector<float,R> etz = etzm1->col(0); // next block Et(Z)

        xn = etz[0];
        if (R==1) xa = xn;
        else xa = etz[1];

#pragma unroll // calculate block, scan bottom -> top
        for (int j=WS-1; j>=0; --j) {
            float xc = x[j]; // current x
            x[j] = c_coefs[2]*xn + c_coefs[3]*xa;
            x[j] = revI(x[j], etu2, c_weights);
            xa = xn;
            xn = xc;
        }

    }

    // preparing for adding u1 and u2

    __syncthreads(); // guarantee no smem usage

    if (ty==1) {

#pragma unroll
        for (int i=0; i<32; ++i)
            block[i][tx] = x[i]; // save partial result in smem

    }

    __syncthreads(); // guarantee warp 0 can read smem

    if (ty==0) { // now add

#pragma unroll
        for (int i=0; i<32; ++i)
            x[i] += block[i][tx]; // regs now has v

        g_out += ((n+1)*WS-1)*out_stride + m*WS + tx;
#pragma unroll // write block
        for (int i=0; i<WS; ++i, g_out-=out_stride) {
            *g_out = x[WS-1-i];
        }

    }

}

// Alg5 Deriche in the GPU ------------------------------------------------------

__host__
void alg5d_gpu( float *h_img, int width, int height, 
                float sigma, int order ) {

    const int B = WS;
    if (R != 2) { std::cout << "R must be 2!\n"; return; }

    Vector<float, 6> coefs; // [a0-a3, coefp, coefn]
    Vector<float, R+1> w; // weights
    w[0] = 1.f;
    
    computeFilterParams(coefs[0], coefs[1], coefs[2], coefs[3],
                        w[1], w[2], coefs[4], coefs[5],
                        sigma, order);

    Vector<float, R+1> ck1; // convolution kernel 1 (plus)
    ck1[0] = coefs[0]; ck1[1] = coefs[1]; ck1[2] = 0.f;
    Vector<float, R+1> ck2; // convolution kernel 2 (minus)
    ck2[0] = 0.f; ck2[1] = coefs[2]; ck2[2] = coefs[3];

    // pre-compute basic alg5 Deriche matrices
    Matrix<float,R,R> Ir = identity<float,R,R>();
    Matrix<float,B,R> Zbr = zeros<float,B,R>();
    Matrix<float,R,B> Zrb = zeros<float,R,B>();
    Matrix<float,B,B> Ib = identity<float,B,B>();

    Matrix<float,R,B> AFP_T = fwd(Ir, Zrb, w), ARE_T = rev(Zrb, Ir, w);
    Matrix<float,B,B> AFB_T = fwd(Zbr, Ib, w), ARB_T = rev(Ib, Zbr, w);
    Matrix<float,R,R> AbF_T = tail<R>(AFP_T), AbR_T = head<R>(ARE_T);
    Matrix<float,B,R> TAFB_T = tail<R>(AFB_T), HARB_T = head<R>(ARB_T);

    Matrix<float,R,B> AC1P_T = transp(conv1p<float,B,R>(ck1));
    Matrix<float,R,B> AC2E_T = transp(conv2e<float,B,R>(ck2));
    Matrix<float,B,B> AC1B_T = transp(conv1b<float,B,R>(ck1));
    Matrix<float,B,B> AC2B_T = transp(conv2b<float,B,R>(ck2));

    Matrix<float,R,R> TAFB_AC1P_T = AC1P_T*TAFB_T;
    Matrix<float,R,R> HARB_AC2E_T = AC2E_T*HARB_T;

    Matrix<float,R,B> TAFB_AC1B = transp(AC1B_T*TAFB_T);
    Matrix<float,R,B> HARB_AC2B = transp(AC2B_T*HARB_T);

    Matrix<float,R,B> AC1P_AFB_T = AC1P_T*AFB_T;
    Matrix<float,R,B> AC2E_ARB_T = AC2E_T*ARB_T;

    int m_size = (width+WS-1)/WS, n_size = (height+WS-1)/WS;

    // upload to the GPU
    copy_to_symbol(c_weights, w);
    copy_to_symbol(c_coefs, coefs);

    copy_to_symbol(c_AbF_T, AbF_T);
    copy_to_symbol(c_AbR_T, AbR_T);

    copy_to_symbol(c_AFP_T, AFP_T);
    copy_to_symbol(c_ARE_T, ARE_T);

    copy_to_symbol(c_TAFB_AC1P_T, TAFB_AC1P_T);
    copy_to_symbol(c_HARB_AC2E_T, HARB_AC2E_T);

    copy_to_symbol(c_TAFB_AC1B, TAFB_AC1B);
    copy_to_symbol(c_HARB_AC2B, HARB_AC2B);

    copy_to_symbol(c_AC1P_AFB_T, AC1P_AFB_T);
    copy_to_symbol(c_AC2E_ARB_T, AC2E_ARB_T);

    float inv_width = 1.f/width, inv_height = 1.f/height;

    cudaArray *a_in;
    cudaChannelFormatDesc ccd = cudaCreateChannelDesc<float>();
    cudaMallocArray(&a_in, &ccd, width, height);
    cudaMemcpyToArray(a_in, 0, 0, h_img, width*height*sizeof(float),
                      cudaMemcpyHostToDevice);

    t_in.normalized = true;
    t_in.filterMode = cudaFilterModePoint;
    t_in.addressMode[0] = t_in.addressMode[1] = cudaAddressModeBorder;

    // naming convention: stride in elements and pitch in bytes
    // to achieve at the same time coalescing and L2-cache colision avoidance
    // the transposed and regular output images have stride of maximum size
    // (4Ki) + 1 block (WS), considering up to 4096x4096 input images
    int stride_img = MAXSIZE+WS;
    dvector<float> d_img(height*stride_img);

    // +1 padding is important even in zero-border to avoid if's in kernels
    dvector< Matrix<float,R,B> >
        d_py1bar((m_size+1)*n_size), d_ey2bar((m_size+1)*n_size);
    dvector< Matrix<float,R,B> >
        d_pty1bar((n_size+1)*m_size), d_ety1bar((n_size+1)*m_size),
        d_ptu11hat((n_size+1)*m_size), d_etu12hat((n_size+1)*m_size);
    dvector< Matrix<float,R,B> >
        d_pty2bar((n_size+1)*m_size), d_ety2bar((n_size+1)*m_size),
        d_ptu21hat((n_size+1)*m_size), d_etu22hat((n_size+1)*m_size);
    dvector< Matrix<float,R,B> >
        d_px((m_size+1)*n_size), d_ex((m_size+1)*n_size),
        d_ptz((n_size+1)*m_size), d_etz((n_size+1)*m_size);

    d_py1bar.fillzero();
    d_ey2bar.fillzero();
    d_pty1bar.fillzero();
    d_ety1bar.fillzero();
    d_ptu11hat.fillzero();
    d_etu12hat.fillzero();
    d_pty2bar.fillzero();
    d_ety2bar.fillzero();
    d_ptu21hat.fillzero();
    d_etu22hat.fillzero();

    d_px.fillzero();
    d_ex.fillzero();
    d_ptz.fillzero();
    d_etz.fillzero();

#if RUNTIMES>1
    base_timer &timer_total = timers.gpu_add("alg5d_gpu", width*height*RUNTIMES, "iP");

    for(int r=0; r<RUNTIMES; ++r) {

        cudaBindTextureToArray(t_in, a_in);

        alg5d_collect_carries<<< dim3(m_size, n_size), dim3(WS, NWC) >>>
            ( &d_py1bar, &d_pty1bar, &d_ety1bar,
              &d_ey2bar, &d_pty2bar, &d_ety2bar,
              &d_ptu11hat, &d_etu12hat, &d_etu22hat, &d_ptu21hat,
              &d_px, &d_ex,
              inv_width, inv_height, m_size, n_size );

        alg5d_adjust_carries_rows<<< dim3(1, n_size), dim3(WS, NWA) >>>
            ( &d_py1bar, &d_ey2bar, &d_px, &d_ex, m_size );

        // alg5d_adjust_carries_cols<<< dim3(m_size, 1), dim3(WS, NWA) >>>
        //     ( &d_py1bar, &d_pty1bar, &d_ety1bar,
        //       &d_ey2bar, &d_pty2bar, &d_ety2bar,
        //       &d_ptu11hat, &d_etu12hat, &d_etu22hat, &d_ptu21hat,
        //       &d_px, &d_ex,
        //       m_size, n_size );

        alg5d_write_results<<< dim3(m_size, n_size), dim3(WS, NWW) >>>
            ( d_img, &d_py1bar, &d_ey2bar, &d_ptu11hat, &d_etu22hat,
              &d_px, &d_ex, &d_ptz, &d_etz,
              inv_width, inv_height, m_size, n_size, stride_img );

        cudaUnbindTexture(t_in);

    }

    timer_total.stop();

    std::cout << std::fixed << timer_total.data_size()/(timer_total.elapsed()*1024*1024) << " " << std::flush;
#else
    base_timer &timer_total = timers.gpu_add("alg5d_gpu", width*height, "iP");
    base_timer *timer;

    timer = &timers.gpu_add("collect-carries");

    cudaBindTextureToArray(t_in, a_in);

    alg5d_collect_carries<<< dim3(m_size, n_size), dim3(WS, NWC) >>>
        ( &d_py1bar, &d_pty1bar, &d_ety1bar,
          &d_ey2bar, &d_pty2bar, &d_ety2bar,
          &d_ptu11hat, &d_etu12hat, &d_etu22hat, &d_ptu21hat,
          &d_px, &d_ex,
          inv_width, inv_height, m_size, n_size );

    timer->stop(); timer = &timers.gpu_add("adjust-carries-rows");

    alg5d_adjust_carries_rows<<< dim3(1, n_size), dim3(WS, NWA) >>>
        ( &d_py1bar, &d_ey2bar, &d_px, &d_ex, m_size );

    timer->stop(); timer = &timers.gpu_add("adjust-carries-columns");

    // alg5d_adjust_carries_cols<<< dim3(m_size, 1), dim3(WS, NWA) >>>
    //     ( &d_py1bar, &d_pty1bar, &d_ety1bar,
    //       &d_ey2bar, &d_pty2bar, &d_ety2bar,
    //       &d_ptu11hat, &d_etu12hat, &d_etu22hat, &d_ptu21hat,
    //       &d_px, &d_ex,
    //       m_size, n_size );

    timer->stop(); timer = &timers.gpu_add("write-block");

    alg5d_write_results<<< dim3(m_size, n_size), dim3(WS, NWW) >>>
        ( d_img, &d_py1bar, &d_ey2bar, &d_ptu11hat, &d_etu22hat,
          &d_px, &d_ex, &d_ptz, &d_etz,
          inv_width, inv_height, m_size, n_size, stride_img );

    timer->stop();

    cudaUnbindTexture(t_in);

    timer_total.stop();
    timers.flush();
#endif
    cudaMemcpy2D(h_img, width*sizeof(float), d_img, stride_img*sizeof(float), width*sizeof(float), height, cudaMemcpyDeviceToHost);

}

#endif // ALG5D_GPU_CUH
