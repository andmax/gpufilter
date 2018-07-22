/**
 *  @file rg.cuh
 *  @brief Recursive Gaussian Deriche original CUDA sample
 *  @author Andre Maximo
 *  @date May, 2014
 *  @copyright GE Proprietary
 */

/**
 * @brief Recursive Gaussian Deriche filter
 * @param[in] id Input data
 * @param[out] od Output data
 * @param[in] w Image width
 * @param[in] h Image height
 * @param[in] a0-a3, b1, b2, coefp, coefn - filter parameters
*/

__global__ void
d_rg0(float *id, float *od, int w, int h,
      float a0, float a1, float a2, float a3,
      float b1, float b2, float coefp, float coefn) {

    int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x, bdx = blockDim.x, bdy = blockDim.y;
    int bdxy = bdx*bdy;
    unsigned int x = bx*(bdxy) + ty*bdx + tx;

    if (x >= w) return;

    id += x;    // advance pointers to correct column
    od += x;

    // forward pass
    float xp = 0.f;  // previous input
    float yp = 0.f;  // previous output
    float yb = 0.f;  // previous output by 2
#if CLAMPTOEDGE
    xp = *id;
    yb = coefp*xp;
    yp = yb;
#endif

    for (int y = 0; y < h; y++) {
        float xc = *id;
        float yc = a0*xc + a1*xp - b1*yp - b2*yb;
        *od = yc;
        id += w;
        od += w;    // move to next row
        xp = xc;
        yb = yp;
        yp = yc;
    }

    // reset pointers to point to last element in column
    id -= w;
    od -= w;

    // reverse pass
    // ensures response is symmetrical
    float xn = 0.f;
    float xa = 0.f;
    float yn = 0.f;
    float ya = 0.f;
#if CLAMPTOEDGE
    xn = xa = *id;
    yn = coefn*xn;
    ya = yn;
#endif

    for (int y = h-1; y >= 0; y--) {
        float xc = *id;
        float yc = a2*xn + a3*xa - b1*yn - b2*ya;
        xa = xn;
        xn = xc;
        ya = yn;
        yn = yc;
        *od = *od + yc;
        id -= w;
        od -= w;  // move to previous row
    }
}

__global__ void
d_rg1(float *id, float *od, float *td, int w, int h,
      float a0, float a1, float a2, float a3,
      float b1, float b2, float coefp, float coefn) {

    int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x, bdx = blockDim.x, bdy = blockDim.y;
    int bdxy2 = bdx*bdy/2, bdy2 = bdy/2, tyby2 = ty % bdy2;
    unsigned int x = bx*bdxy2 + tyby2*bdx + tx;

    if (x >= w) return;

    id += x;    // advance pointers to correct column
    od += x;
    td += x;

    if (ty < bdy2) {

        // forward pass
        float xp = 0.f;  // previous input
        float yp = 0.f;  // previous output
        float yb = 0.f;  // previous output by 2
#if CLAMPTOEDGE
        xp = *id;
        yb = coefp*xp;
        yp = yb;
#endif

        for (int y = 0; y < h; y++) {
            float xc = *id;
            float yc = a0*xc + a1*xp - b1*yp - b2*yb;
            *od = yc;
            id += w;
            od += w;    // move to next row
            xp = xc;
            yb = yp;
            yp = yc;
        }

        od -= w;
        td += w*(h-1);

    } else {

        id += w*(h-1);
        td += w*(h-1); // advance pointers to last row

        // reverse pass
        // DOEST NOT ensures response is symmetrical
        float xn = 0.f;
        float xa = 0.f;
        float yn = 0.f;
        float ya = 0.f;
#if CLAMPTOEDGE
        xn = xa = *id;
        yn = coefn*xn;
        ya = yn;
#endif

        for (int y = h-1; y >= 0; y--) {
            float xc = *id;
            float yc = a2*xn + a3*xa - b1*yn - b2*ya;
            xa = xn;
            xn = xc;
            ya = yn;
            yn = yc;
            *td = yc;
            id -= w;
            td -= w;  // move to previous row
        }

        td += w;

    }

    __syncthreads();

    if (ty < bdy2) {

        for (int y = 0; y < (h+1)/2; y++) {
            *od = *od + *td;
            od -= w;
            td -= w;
        }

    } else {

        for (int y = 0; y < h/2; y++) {
            *od = *od + *td;
            od += w;
            td += w;
        }

    }

}
