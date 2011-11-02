
#include <cstdlib>

#include "cv.h"
#include "highgui.h"

#include <gpufilter.h>
#include <cpufilter.h>

// Runs a given filter
int main(int argc, char *argv[]) {
    if( argc < 3 )
        errorf(
            "Usage:\n"
            "  recursive [options] <input image> <output image>\n"
            "where options are:\n"
            "  -filter:<method>\n"
            "    gaussian\n"
            "    bspline3i\n"
            "  -sigma:<value> (for Gaussian)\n"
            );

    // process options
    const char *filter = "gaussian";
    const char *file_in = NULL, *file_out = NULL; 

    float sigma = 1.0;
    int end = 0;

    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "-filter:", sizeof("-filter:")-1) == 0) {
            filter = argv[i]+sizeof("-filter:")-1;
        } else if (sscanf(argv[i], "-sigma:%f%n", &sigma, &end) == 1) {
            if (argv[i][end] != '\0' || sigma < 0.f) 
                errorf("Invalid argument '%s'", argv[i]);
        } else if (argv[i][0] == '-') {
            errorf("Unknown option '%s'", argv[i]);
        } else {
            if (!file_in) file_in = argv[i];
            else if (!file_out) file_out = argv[i];
            else errorf("Too many arguments");
        }
    }

    if( !file_in || !file_out )
        errorf("Not enough arguments!");

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
        cvSsetImageCOI(in_img, c+1);
        IplImage *ch_img = cvCreateImage(cvSize(w_in, h_in), in_img->depth, 1);
        cvCopy(in_img, ch_img);
        IplImage *fl_img = cvCreateImage(cvSize(w_in, h_in), IPL_DEPTH_32F, 1);
        cvConvertImage(ch_img, fl_img);
        for (int i = 0; i < h_in; ++i)
            for (int j = 0; j < w_in; ++j)
                flat_in[c][i*w+j] = ((float *)(fl_img->imageData + i*fl_img->widthStep))[j];
        cvReleaseImage(&ch_img);
        cvReleaseImage(&fl_img);
    }

    if( strcmp(filter, "gaussian") == 0 ) {
        printf("Applying filter gaussian (sigma = %g)\n", sigma);
        gaussian(flat_in, h_in, w_in, depth, sigma);
    } else if (strcmp(filter, "bspline3i") == 0) {
        printf("Applying filter bspline3i\n");
        bspline3i(flat_in, h_in, w_in, depth);
    } else {
        errorf("Unknown method '%s'", filter);
    }

    printf("Packing output image\n");

    IplImage *out_img = cvCreateImage(cvSize(w_in, h_in), IPL_DEPTH_32F, in_img->nChannels);

    for (int c = 0; c < depth; c++)
        for (int i = 0; i < h_in; ++i)
            for (int j = 0; j < w_in; ++j)
                ((float *)(out_img->imageData + i*out_img->widthStep))[j*depth + c] = flat_in[c][i*w+j];

    printf("Saving output image '%s'\n", file_out);

    if( !cvSaveImage(file_out, out_img) )
        errorf("Could not save: %s\n", file_out);

    cvReleaseImage(&in_img);
    cvReleaseImage(&out_img);

    return 0;
}
