/**
 *  @file app_recursive_gpu.cc
 *  @brief Example of an application of recursive filtering in the GPU
 *  @author Andre Maximo
 *  @date November, 2011
 */


#include <cstdio>
#include <cstddef>
#include <cstdarg>
#include <cstdlib>

#include "cv.h"
#include "highgui.h"

#include <gpufilter.h>
#include <cpuground.h>

//-----------------------------------------------------------------------------
// Aborts and prints an error message
void errorf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    fflush(stderr);
    exit(1);
}


//-----------------------------------------------------------------------------
// Clamp values between 0.f and 1.f 
static inline float clamp(float f) {
    return f > 1.f? 1.f: (f < 0.f? 0.f: f);
}

// Main
int main( int argc, char** argv ) {

    const char *file_in = "../images/cactus-512.png", *file_out = "cactus-gauss.png"; 
    const float sigma = 4.f;

    printf("Loading input image '%s'\n", file_in);

    IplImage *in_img = cvLoadImage(file_in, CV_LOAD_IMAGE_UNCHANGED);

    if( !in_img )
        errorf("Unable to load image '%s'", file_in);

    int w_in = in_img->width, h_in = in_img->height, depth = in_img->nChannels;

    printf("Image is %dx%dx%d\n", w_in, h_in, depth);

    printf("Flattening input image\n");

    float **flat_in = new float*[depth];

    for (int c = 0; c < depth; c++)
        flat_in[c] = new float[w_in*h_in];

    if( !flat_in ) 
        errorf("Out of memory!");

    for (int c = 0; c < depth; c++) {

        cvSetImageCOI(in_img, c+1);

        IplImage *ch_img = cvCreateImage(cvSize(w_in, h_in), in_img->depth, 1);

        cvCopy(in_img, ch_img);

        IplImage *uc_img = cvCreateImage(cvSize(w_in, h_in), IPL_DEPTH_8U, 1);

        cvConvertImage(ch_img, uc_img);

        for (int i = 0; i < h_in; ++i)
            for (int j = 0; j < w_in; ++j)
                flat_in[c][i*w_in+j] = ((unsigned char *)(uc_img->imageData + i*uc_img->widthStep))[j]/255.f;

        cvReleaseImage(&ch_img);
        cvReleaseImage(&uc_img);

    }

    for (int c = 0; c < depth; c++)
        gpufilter::gaussian_gpu( flat_in[c], h_in, w_in, sigma );

    printf("Packing output image\n");

    IplImage *out_img = cvCreateImage(cvSize(w_in, h_in), IPL_DEPTH_8U, depth);

    for (int c = 0; c < depth; c++)
        for (int i = 0; i < h_in; ++i)
            for (int j = 0; j < w_in; ++j)
                ((unsigned char *)(out_img->imageData + i*out_img->widthStep))[j*depth + c] = (unsigned char) (floorf(clamp(flat_in[c][i*w_in+j])*255.f+0.5f));

    printf("Saving output image '%s'\n", file_out);

    if( !cvSaveImage(file_out, out_img) )
        errorf("Could not save: %s\n", file_out);

    cvReleaseImage(&in_img);
    cvReleaseImage(&out_img);

    return 0;

}
