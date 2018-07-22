#pragma warning(disable:4819)

/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
    Recursive Gaussian filter
    sgreen 8/1/08

    This code sample implements a Gaussian blur using Deriche's recursive method:
    http://citeseer.ist.psu.edu/deriche93recursively.html

    This is similar to the box filter sample in the SDK, but it uses the previous
    outputs of the filter as well as the previous inputs. This is also known as an
    IIR (infinite impulse response) filter, since its response to an input impulse
    can last forever.

    The main advantage of this method is that the execution time is independent of
    the filter width.

    The GPU processes columns of the image in parallel. To avoid uncoalesced reads
    for the row pass we transpose the image and then transpose it back again
    afterwards.

    The implementation is based on code from the CImg library:
    http://cimg.sourceforge.net/
    Thanks to David Tschumperl and all the CImg contributors!
*/

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA includes and interop headers
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>      // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda_gl.h>   // includes cuda_runtime_api.h

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "recursiveGaussian.h"

#define MAX(a,b) ((a > b) ? a : b)

#define USE_SIMPLE_FILTER 0

#define MAX_EPSILON_ERROR 5.0f
#define THRESHOLD  0.15f

// Define the files that are to be save and the reference images for validation
const char *sOriginal[] =
{
    "lena_10.ppm",
    "lena_14.ppm",
    "lena_18.ppm",
    "lena_22.ppm",
    NULL
};

const char *sReference[] =
{
    "ref_10.ppm",
    "ref_14.ppm",
    "ref_18.ppm",
    "ref_22.ppm",
    NULL
};

char *image_filename = (char*)"lena.ppm";
//char *image_filename = (char*)"calhouse.ppm";
float sigma = 4.0f;
int order = 0;
int nthreads = 256;  // number of threads per block

unsigned int width, height;
unsigned int *h_img = NULL;
unsigned int *d_img = NULL;
unsigned int *d_temp = NULL;

float *h_gimg = NULL;
float *d_gimg = NULL;
float *d_gtemp = NULL;

GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint texid = 0;   // texture

StopWatchInterface *timer = 0;

// Auto-Verification Code
const int frameCheckNumber = 4;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
uint frameCount = 0;

int *pArgc = NULL;
char **pArgv = NULL;

bool runBenchmark = true;

const char *sSDKsample = "CUDA Recursive Gaussian";

extern "C"
void gaussianFilterRGBA(unsigned int *d_src, unsigned int *d_dest, unsigned int *d_temp, int width, int height, float sigma, int order, int nthreads);

extern "C"
void rg0(float *d_src, float *d_dest, float *d_temp, int width, int height, float sigma, int order, int nthreads);

extern "C"
void rg1(float *d_src, float *d_dest, float *d_temp, int width, int height, float sigma, int order, int nthreads);

extern "C"
float rgbaIntToFloat1(uint c);

extern "C"
void convert_rgbaIntToFloat(uint *id, float *od0, float *od1, float *od2, float *od3, int w, int h);

extern "C"
void convert_rgbaFloatToUchar(uchar *od, float *id0, float *id1, float *id2, float *id3, int w, int h);

extern "C"
void runAlg5f4(float *id, int width, int height, float sigma);

extern "C"
void runAlg4pd(float *id, int width, int height, float sigma, int order);

extern "C"
void runAlg4d(float *id, int width, int height, float sigma, int order);

extern "C"
void runAlg5d(float *id, int width, int height, float sigma, int order);

void cleanup();

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "%s (sigma=%4.2f): %3.1f fps", sSDKsample, sigma, ifps);

        glutSetWindowTitle(fps);
        fpsCount = 0;

        fpsLimit = MAX(ifps, 1.f);
        sdkResetTimer(&timer);
    }
}

// display results using OpenGL
void display()
{
    sdkStartTimer(&timer);

    // execute filter, writing results to pbo
    unsigned int *d_result;
    checkCudaErrors(cudaGLMapBufferObject((void **)&d_result, pbo));
    gaussianFilterRGBA(d_img, d_result, d_temp, width, height, sigma, order, nthreads);
    checkCudaErrors(cudaGLUnmapBufferObject(pbo));

    // load texture from pbo
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_2D, texid);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);

    glBegin(GL_QUADS);
    glTexCoord2f(0, 1);
    glVertex2f(0, 0);
    glTexCoord2f(1, 1);
    glVertex2f(1, 0);
    glTexCoord2f(1, 0);
    glVertex2f(1, 1);
    glTexCoord2f(0, 0);
    glVertex2f(0, 1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glutSwapBuffers();

    sdkStopTimer(&timer);

    computeFPS();
}

void idle()
{
    glutPostRedisplay();
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    checkCudaErrors(cudaFree(d_img));
    checkCudaErrors(cudaFree(d_temp));
    checkCudaErrors(cudaFree(d_gimg));
    checkCudaErrors(cudaFree(d_gtemp));

    delete [] h_img;
    delete [] h_gimg;

    if (!runBenchmark)
    {
        if (pbo)
        {
            checkCudaErrors(cudaGLUnregisterBufferObject(pbo));
            glDeleteBuffersARB(1, &pbo);
        }

        if (texid)
        {
            glDeleteTextures(1, &texid);
        }
    }
}

void keyboard(uchar key, int x, int y)
{
    switch (key)
    {
        case 27:
            exit(EXIT_SUCCESS);
            break;

        case '=':
            sigma+=0.1f;
            break;

        case '-':
            sigma-=0.1f;

            if (sigma < 0.0)
            {
                sigma = 0.0f;
            }

            break;

        case '+':
            sigma+=1.0f;
            break;

        case '_':
            sigma-=1.0f;

            if (sigma < 0.0)
            {
                sigma = 0.0f;
            }

            break;

        case '0':
            order = 0;
            break;

        case '1':
            order = 1;
            sigma = 0.5f;
            break;

        case '2':
            order = 2;
            sigma = 0.5f;
            break;

        default:
            break;
    }

    printf("sigma = %f\n", sigma);
    glutPostRedisplay();
}

void reshape(int x, int y)
{
    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void initCudaBuffers()
{
    unsigned int size = width * height * sizeof(unsigned int);
    unsigned int gsize = width * height * sizeof(float); // size for gray image

    // allocate device memory
    checkCudaErrors(cudaMalloc((void **) &d_img, size));
    checkCudaErrors(cudaMalloc((void **) &d_temp, size));
    checkCudaErrors(cudaMalloc((void **) &d_gimg, gsize));
    checkCudaErrors(cudaMalloc((void **) &d_gtemp, gsize));

    checkCudaErrors(cudaMemcpy(d_img, h_img, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_gimg, h_gimg, gsize, cudaMemcpyHostToDevice));

    sdkCreateTimer(&timer);
}


void initGLBuffers()
{
    // create pixel buffer object to store final image
    glGenBuffersARB(1, &pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, h_img, GL_STREAM_DRAW_ARB);

    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    checkCudaErrors(cudaGLRegisterBufferObject(pbo));

    // create texture for display
    glGenTextures(1, &texid);
    glBindTexture(GL_TEXTURE_2D, texid);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow(sSDKsample);
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    printf("Press '+' and '-' to change filter width\n");
    printf("0, 1, 2 - change filter order\n");

    glewInit();

    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object"))
    {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(EXIT_FAILURE);
    }
}

void
benchmark(void)
{
    // allocate memory for result
    float *d_gresult;
    unsigned int size = width * height * sizeof(float);
    checkCudaErrors(cudaMalloc((void **) &d_gresult, size));

    // execute the kernel (RUNTIMES number of times)
    rg0(d_gimg, d_gresult, d_gtemp, width, height, sigma, order, nthreads);

    float *rg0_img = new float[width*height];
    checkCudaErrors(cudaMemcpy(rg0_img, d_gresult, width*height*sizeof(float), cudaMemcpyDeviceToHost));

    rg1(d_gimg, d_gresult, d_gtemp, width, height, sigma, order, nthreads);

    float *rg1_img = new float[width*height];
    checkCudaErrors(cudaMemcpy(rg1_img, d_gresult, width*height*sizeof(float), cudaMemcpyDeviceToHost));

    // algs
    int ne = width*height;
    float me = 0.f, mre = 0.f;

    std::vector< float > van_img(ne), ap4_img(ne), ag4_img(ne), ag5_img(ne);
    for (uint i = 0; i < ne; ++i)
        van_img[i] = ap4_img[i] = ag4_img[i] = ag5_img[i] = h_gimg[i];

    runAlg5f4(&van_img[0], width, height, sigma);

    runAlg4pd(&ap4_img[0], width, height, sigma, order);

    runAlg4d(&ag4_img[0], width, height, sigma, order);

    //runAlg5d(&ag5_img[0], width, height, sigma, order);

    static char *rg0fn = (char*)"rg0.pgm";
    static char *rg1fn = (char*)"rg1.pgm";
    static char *vanfn = (char*)"van.pgm";
    static char *ap4fn = (char*)"ap4.pgm";
    static char *ag4fn = (char*)"ag4.pgm";

    sdkSavePGM(rg0fn, rg0_img, width, height);
    sdkSavePGM(rg1fn, rg1_img, width, height);
    sdkSavePGM(vanfn, &van_img[0], width, height);
    sdkSavePGM(ap4fn, &ap4_img[0], width, height);
    sdkSavePGM(ag4fn, &ag4_img[0], width, height);

#if RUNTIMES==1
    std::cout << " --- Checking computations ---\n";
    check_cpu_reference( rg0_img, rg1_img, ne, me, mre );
    std::cout << " rg0 x rg1: me = " << me << " mre = " << mre << "\n";
    check_cpu_reference( rg0_img, &van_img[0], ne, me, mre );
    std::cout << " rg0 x alg5f4: me = " << me << " mre = " << mre << "\n";
    check_cpu_reference( rg0_img, &ap4_img[0], ne, me, mre );
    std::cout << " rg0 x alg4pd: me = " << me << " mre = " << mre << "\n";
    check_cpu_reference( rg0_img, &ag4_img[0], ne, me, mre );
    std::cout << " rg0 x alg4d: me = " << me << " mre = " << mre << "\n";
#endif

    delete [] rg0_img;
    delete [] rg1_img;

    checkCudaErrors(cudaFree(d_gresult));
}

void
colorAlg(void)
{
    int ne = width*height;
    std::vector< float > *alg_img = new std::vector< float >[4];

    for (uint i = 0; i < 4; ++i)
    {
        alg_img[i].resize(ne);
    }

    convert_rgbaIntToFloat(h_img, &alg_img[0][0], &alg_img[1][0],
                           &alg_img[2][0], &alg_img[3][0], width, height);

    for (uint i = 0; i < ne; ++i)
    {
        runAlg5f4(&alg_img[i][0], width, height, sigma);
        std::cout << "\n";
    }

    uchar *a_result = new uchar[ne*4];
    convert_rgbaFloatToUchar(a_result, &alg_img[0][0], &alg_img[1][0],
                           &alg_img[2][0], &alg_img[3][0], width, height);

    static char *algppmfn = (char*)"alg.ppm";
    sdkSavePPM4ub(algppmfn, a_result, width, height);

    delete [] alg_img;
}

bool
runSingleTest(const char *ref_file, const char *exec_path)
{
    // allocate memory for result
    int nTotalErrors = 0;
    unsigned int *d_result;
    unsigned int size = width * height * sizeof(unsigned int);
    checkCudaErrors(cudaMalloc((void **) &d_result, size));

    // warm-up
    gaussianFilterRGBA(d_img, d_result, d_temp, width, height, sigma, order, nthreads);

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStartTimer(&timer);

    gaussianFilterRGBA(d_img, d_result, d_temp, width, height, sigma, order, nthreads);
    checkCudaErrors(cudaDeviceSynchronize());
    getLastCudaError("Kernel execution failed");
    sdkStopTimer(&timer);

    uchar *h_result = (uchar *)malloc(width*height*4);
    checkCudaErrors(cudaMemcpy(h_result, d_result, width*height*4, cudaMemcpyDeviceToHost));

    char dump_file[1024];
    sprintf(dump_file, "lena_%02d.ppm", (int)sigma);
    sdkSavePPM4ub(dump_file, h_result, width, height);

    if (!sdkComparePPM(dump_file, sdkFindFilePath(ref_file, exec_path), MAX_EPSILON_ERROR, THRESHOLD, false))
    {
        nTotalErrors++;
    }

    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    printf("%.2f Mpixels/sec\n", (width*height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);

    checkCudaErrors(cudaFree(d_result));
    free(h_result);

    printf("Summary: %d errors!\n", nTotalErrors);

    printf(nTotalErrors == 0 ? "Test passed\n": "Test failed!\n");
    return(nTotalErrors == 0);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    pArgc = &argc;
    pArgv = argv;
    char *ref_file = NULL;

#if RUNTIMES==1
    printf("%s Starting...\n\n", sSDKsample);
#endif

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    if (argc > 1)
    {
        if (checkCmdLineFlag(argc, (const char **)argv, "file"))
        {
            getCmdLineArgumentString(argc, (const char **)argv, "file", &ref_file);
            fpsLimit = frameCheckNumber;
        }
    }

    // Get the path of the filename
    char *filename;

    if (getCmdLineArgumentString(argc, (const char **) argv, "image", &filename))
    {
        image_filename = filename;
    }

    // load image or random image
    char *image_path = sdkFindFilePath(image_filename, argv[0]);

    if (image_path == NULL)
    {
        fprintf(stderr, "Error unable to find and load image file: '%s'\n", image_filename);
        exit(EXIT_FAILURE);
    }

    bool randimg = false;

    if (checkCmdLineFlag(argc, (const char **)argv, "width"))
    {
        width = getCmdLineArgumentInt(argc, (const char **) argv, "width");
        randimg = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "height"))
    {
        height = getCmdLineArgumentInt(argc, (const char **) argv, "height");
        randimg = true;
    }

    if (randimg)
    {
        image_path = (char *)"rand";
        h_img = new unsigned int[width*height*4];
        srand(1234);
        for (int i = 0; i < width*height*4; ++i)
            h_img[i] = (unsigned int)(rand() % 256);
    } else
    {
        sdkLoadPPM4ub(image_path, (uchar **)&h_img, &width, &height);
    }

    if (!h_img)
    {
        printf("Error unable to generate image file: '%s'\n", image_path);
        exit(EXIT_FAILURE);
    }

#if RUNTIMES==1
    printf("Loaded '%s', %d x %d pixels\n", image_path, width, height);
#endif

    if (checkCmdLineFlag(argc, (const char **)argv, "threads"))
    {
        //nthreads = getCmdLineArgumentInt(argc, (const char **) argv, "threads");
        printf("Error unable to change number of threads!'\n");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "sigma"))
    {
        sigma = getCmdLineArgumentFloat(argc, (const char **) argv, "sigma");
    }

#if RUNTIMES==1
    printf("Sigma: %f\n", sigma);
#endif

    if (checkCmdLineFlag(argc, (const char **)argv, "order"))
    {
        order = getCmdLineArgumentInt(argc, (const char **) argv, "order");
    }

#if RUNTIMES==1
    printf("Filter order: %d\n", order);
#endif

    if (checkCmdLineFlag(argc, (const char **)argv, "benchmark"))
    {
        runBenchmark = (bool)getCmdLineArgumentInt(argc, (const char **) argv, "benchmark");
    }

#if RUNTIMES==1
    printf("Run benchmark: %d\n", (benchmark?1:0));
#endif

    h_gimg = new float[width*height]; // allocate gray image

    for (uint i=0; i<width*height; ++i)
        h_gimg[i] = rgbaIntToFloat1(h_img[i]);

    static char *gfn = (char*)"gray.pgm";
    sdkSavePGM(gfn, h_gimg, width, height);

#if RUNTIMES==1
    printf("Gray image converted: %s\n", gfn);
#endif

    int device;
    struct cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    if (!strncmp("Tesla", prop.name, 5))
    {
        printf("Tesla card detected, running the test in benchmark mode (no OpenGL display)\n");
        runBenchmark = true;
    }

    // Benchmark or AutoTest mode detected, no OpenGL
    if (runBenchmark == true || ref_file != NULL)
    {
#if RUNTIMES==1
        findCudaDevice(argc, (const char **)argv);
#endif
    }
    else
    {
        // First initialize OpenGL context, so we can properly set the GL for CUDA.
        // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
        initGL(&argc, argv);
        findCudaGLDevice(argc, (const char **)argv);
    }

    initCudaBuffers();

    if (ref_file)
    {
        printf("(Automated Testing)\n");
        bool testPassed = runSingleTest(ref_file, argv[0]);

        cleanup();
        cudaDeviceReset();

        exit(testPassed ? EXIT_SUCCESS : EXIT_FAILURE);
    }

    if (runBenchmark)
    {
#if RUNTIMES==1
        printf("(Run Benchmark)\n");
#endif
        benchmark();

        cleanup();
        cudaDeviceReset();

        exit(EXIT_SUCCESS);
    }

    initGLBuffers();
    //atexit(cleanup);
    glutMainLoop();

    cleanup();
    cudaDeviceReset();

    exit(EXIT_SUCCESS);
}
