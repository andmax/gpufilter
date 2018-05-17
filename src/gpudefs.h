/**
 *  @file gpudefs.h
 *  @brief Auxiliary GPU definition and implementation
 *  @author Andre Maximo
 *  @date Nov, 2012
 *  @copyright The MIT License
 */

#ifndef GPUDEFS_H
#define GPUDEFS_H

//== GLOBAL-SCOPE DEFINITIONS ==================================================

#define WS 32 ///< Warp size (defines b x b block size where b = WS)

/**
 *  @note Compute shared memory usage by number of blocks (nb), order
 *  (r) and number of carries (nc):
 *  @verbatim
def sm(nb, r, nc):
    return nb*r*nc*32*4;
    @endverbatim
 *
 */
#define NWA 8 ///< # of warps adjusting carries
#define NWAC 4 ///< # of warps adjusting carries cols
#define NWARC 4 ///< # of warps adjusting carries rows+cols
#define NBA 8 ///< # of blocks adjusting carries
#define NBARC 16 ///< # of blocks adjusting carries rows+cols
#define NWC 5 ///< # of warps collect carries
#define NWW 5 ///< # of warps write results
#define NBCW 11 ///< # of blocks collect carries / write results

//== NAMESPACES ================================================================

namespace gpufilter {

//== CONSTANTS =================================================================

// basics
__constant__ int c_border;
__constant__ Vector<float,ORDER+1> c_weights;

// alg3 and alg4
__constant__ Matrix<float,ORDER,ORDER> c_AbF_T, c_AbR_T, c_HARB_AFP_T;

// alg5
__constant__ Matrix<float,ORDER,WS> c_ARE_T, c_ARB_AFP_T, c_TAFB, c_HARB_AFB;

// sat
__constant__ Matrix<float,ORDER,WS> c_AFP_T;

// clamp
__constant__ Matrix<float,ORDER,ORDER> c_AbarFIArF_T, c_ArFSRRF_T,
    c_AbarFIArFAbarRIArRArFSRRF_T;

// repeat
__constant__ Matrix<float,ORDER,ORDER> c_IAwF_T, c_IAwR_T, c_IAhF_T, c_IAhR_T;

// reflect
__constant__ Matrix<float,ORDER,ORDER>
    c_IArFVAbarFV_T, c_IArFVAbarFVAwF_T, c_IArFVAbarFVAhF_T,
    c_HARwAFPIArFVAbarFVAwFAwR_T,
    c_HARhAFPIArFVAbarFVAhFAhR_T,
    c_IArFVAbarFVAwRHARwAFPIArFVAbarFVAwFAwR_T,
    c_IArFVAbarFVAhRHARhAFPIArFVAbarFVAhFAhR_T;

texture<float, cudaTextureType2D, cudaReadModeElementType> t_in;

//== IMPLEMENTATION ============================================================

// Auxiliary functions ---------------------------------------------------------

template <int W, int V>
__device__
void read_block( Matrix<float,WS,V>& block,
                 const int& m, const int& n,
                 const float& inv_width,
                 const float& inv_height ) {
    int tx = threadIdx.x, ty = threadIdx.y;
    float tu = (m*WS+tx+.5f)*inv_width,
          tv = (n*WS+ty+.5f)*inv_height;
    float (*bdata)[V] = (float (*)[V]) &block[ty][tx];
#pragma unroll
    for (int i=0; i<WS-(WS%W); i+=W) {
        **bdata = tex2D(t_in, tu, tv);
        bdata += W;
        tv += W*inv_height;
    }
    if (ty < WS%W) {
        **bdata = tex2D(t_in, tu, tv);
    }
}

template <class T, int R>
__device__ 
Vector<T,R> mad( Matrix<T,R,WS>& r,
                 const Vector<T,R>& a, 
                 const Matrix<T,R,R>& b ) {
#pragma unroll
    for (int j=0; j<R; ++j) {
        T acc = *r[j];
#pragma unroll
        for (int i=0; i<R; ++i)
            acc += a[i]*b[i][j];
        *r[j] = acc;
    }
    return r.col(0);
}

template <class T, int R>
__device__ 
void mad( Matrix<T,R,WS>& r,
          const Matrix<T,R,WS>& a, 
          const Matrix<T,R,R>& b) {
#pragma unroll
    for (int j=0; j<R; ++j) {
        T acc = *r[j];
#pragma unroll
        for (int i=0; i<R; ++i)
            acc += *a[i]*b[i][j];
        *r[j] = acc;
    }
}

template <class T, int R>
__device__ 
Vector<T,R> mad( Matrix<T,R,WS> &r,
                 const Matrix<T,R,R> &a,
                 const Vector<T,R> &b ) {
#pragma unroll
    for (int i=0; i<R; ++i) {
        T acc = *r[i];
#pragma unroll
        for (int j=0; j<R; ++j)
            acc += a[i][j]*b[j];
        *r[i] = acc;
    }
    return r.col(0);
}

template <class T, int R>
__device__ 
void mad( Matrix<T,R,WS> &r,
          const Matrix<T,R,R> &a,
          const Matrix<T,R,WS> &b ) {
#pragma unroll
    for (int i=0; i<R; ++i) {
        T acc = *r[i];
#pragma unroll
        for (int j=0; j<R; ++j)
            acc += a[i][j]* *b[j];
        *r[i] = acc;
    }
}

template <class T, int R>
__device__ 
void mad( Matrix<T,R,WS> &r,
          const Matrix<T,R,WS> &a,
          const Matrix<T,R,WS> &b,
          const Matrix<T,R,WS> &c,
          volatile T (*block_RD)[WS/2+WS+1] ) {
    int tx = threadIdx.x, ty = threadIdx.y;
    Matrix<T,R,R> rint;
#pragma unroll
    for (int i=0; i<R; ++i) {
#pragma unroll
        for (int j=0; j<R; ++j) {
            block_RD[ty][tx] = a[i][tx] * *b[j];
            block_RD[ty][tx] += block_RD[ty][tx-1];
            block_RD[ty][tx] += block_RD[ty][tx-2];
            block_RD[ty][tx] += block_RD[ty][tx-4];
            block_RD[ty][tx] += block_RD[ty][tx-8];
            block_RD[ty][tx] += block_RD[ty][tx-16];
            rint[i][j] = block_RD[ty][WS-1];
        }
    }
    mad(r, rint, (const Matrix<T,R,WS> &)c[0][tx]);
}

template <class T, int R>
__device__ // fix Pt or Et giving P or E and fixing matrices a and b
void fixpet( Vector<T,R>& pet,
             const Matrix<float,R,WS>& a,
             const Matrix<float,R,WS>& b,
             const Vector<T,R>& pe ) {
    int tx = threadIdx.x; // one-warp computing (ty==0)
    // pt||et += B_T * ( p||e * A_T )
#pragma unroll // computing corner cols south||north west||east
    for (int i = 0; i < R; ++i) {
#pragma unroll // computing corner rows south||north west||east
        for (int j = 0; j < R; ++j) {
            float v = a[i][tx] * pe[j];
#pragma unroll // recursive doubling by shuffle
            for (int k = 1; k < WS; k *= 2) {
                float p = __shfl_up(v, k);
                if (tx >= k)
                    v += p;
            }
            pet[i] += b[j][tx] * __shfl(v, WS-1);
        }
    }
}

template <class T, int R>
__device__ // fix Pt or Et giving P or E and fixing matrices a and b
void fixpet( Vector<T,R>& pet,
             const Vector<float,R>& a,
             const Vector<float,R>& b,
             const Vector<T,R>& pe ) {
    int tx = threadIdx.x; // one-warp computing (ty==0)
    // pt||et += B_T * ( p||e * A_T )
#pragma unroll // computing corner cols south||north west||east
    for (int i = 0; i < R; ++i) {
#pragma unroll // computing corner rows south||north west||east
        for (int j = 0; j < R; ++j) {
            float v = a[i] * pe[j];
#pragma unroll // recursive doubling by shuffle
            for (int k = 1; k < WS; k *= 2) {
                float p = __shfl_up(v, k);
                if (tx >= k)
                    v += p;
            }
            pet[i] += b[j] * __shfl(v, WS-1);
        }
    }
}

template <class T, int R>
__device__ // fix Pt or Et giving P or E and fixing matrix b for clamp
void mad_clamp( Vector<T,R> &pet,
                const Vector<T,R>& pe,
                const Matrix<T,R,WS>& b ) {
    int tx = threadIdx.x; // one-warp computing (ty==0)
    // pt||et += p||e * B_T
#pragma unroll // rows are the same in clamp
    for (int i = 0; i < R; ++i)
#pragma unroll // Pt or Et contains the repeated rows
        for (int j = 0; j < R; ++j)
            pet[i] += pe[j] * b[j][tx];
}

template <class T, int R>
__device__ // fix Pt or Et giving P or E and fixing matrix b for clamp
void mad_clamp( Vector<T,R> &pet,
                const Vector<T,R>& pe,
                const Vector<T,R>& b ) {
    // pt||et += p||e * B_T
#pragma unroll // rows are the same in clamp
    for (int i = 0; i < R; ++i)
#pragma unroll // Pt or Et contains the repeated rows
        for (int j = 0; j < R; ++j)
            pet[i] += pe[j] * b[j];
}

//==============================================================================
} // namespace gpufilter
//==============================================================================
#endif // GPUDEFS_H
//==============================================================================
