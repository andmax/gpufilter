/**
 *  @file cpudefs.h
 *  @brief Auxiliary CPU definition and implementation
 *  @author Andre Maximo
 *  @date Jan, 2012
 *  @copyright The MIT License
 */

#ifndef CPUDEFS_H
#define CPUDEFS_H

//== IMPLEMENTATION ============================================================

/**
 *  @ingroup api_cpu
 *  @brief Show the default usage scenario
 *  @param[in] argv0 The first program argument is its name
 */
void usage(char *argv0) {
    std::cout << APPNAME << " Usage: " << argv0
              << " [width height run-times border-type border-blocks]\n";
    std::cout << APPNAME << " Where: run-times = number of runs "
              << "for performance measurements\n";
    std::cout << APPNAME << " Where: border-type =  "
              << "0: Zero  1: Clamp  2: Repeat  3: Reflect\n";
    std::cout << APPNAME << " Where: border-blocks = number of blocks "
              << "outside input image (padding)\n";
    std::cout << APPNAME << " Where: zero number of blocks means to use "
              << "the infinite extension\n";
}

/**
 *  @ingroup api_cpu
 *  @brief Return all the initial variables setup for default applications
 *  @param[out] width Image width
 *  @param[out] height Image height
 *  @param[out] runtimes Number of run times (1 for debug and 1000 for performance measurements)
 *  @param[out] btype Border type (either zero, clamp, repeat or reflect)
 *  @param[out] border Number of border blocks (32x32) outside image
 *  @param[out] cpu_img Random image to be filtered in the CPU
 *  @param[out] gpu_img Random image to be filtered in the GPU
 *  @param[out] w Filter weights (feedforward and feedback coefficients)
 *  @param[out] a0border Number of border blocks for algorithm 0 (reference)
 *  @param[out] me Maximum error to check the results versus reference
 *  @param[out] mre Maximum relative error to check the results versus reference
 *  @param[out] argc Number of arguments passed to the application
 *  @param[out] argv Arguments passed to the application
 */
void initial_setup(int& width, int& height, int& runtimes,
                   gpufilter::BorderType& btype, int& border,
                   std::vector<float>& cpu_img, std::vector<float>& gpu_img,
                   gpufilter::Vector<float, ORDER+1>& w,
                   int& a0border,
                   float& me, float& mre,
                   int& argc, char **argv) {

    width = height = 1024;
    runtimes = 1; // # of run times (1 for debug; 1000 for performance)
    btype = gpufilter::CLAMP_TO_ZERO;
    border = 0;

    if ((argc != 1 && argc != 6)
        || (argc==6 && (sscanf(argv[1], "%d", &width) != 1 ||
                        sscanf(argv[2], "%d", &height) != 1 ||
                        sscanf(argv[3], "%d", &runtimes) != 1 ||
                        sscanf(argv[4], "%d", (int *) &btype) != 1 ||
                        sscanf(argv[5], "%d", &border) != 1))) {
        std::cerr << APPNAME << " Bad arguments!\n";
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    if ((btype != gpufilter::CLAMP_TO_ZERO
         && btype != gpufilter::CLAMP_TO_EDGE
         && btype != gpufilter::REPEAT
         && btype != gpufilter::REFLECT)
        || (width <= 0 || height <= 0 || runtimes <= 0 || border < 0)) {
        std::cout << APPNAME << " Wrong arguments!\n";
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    cpu_img.resize(width*height);
    gpu_img.resize(width*height);

    srand( 1234 );
    for (int i = 0; i < width*height; ++i)
        gpu_img[i] = cpu_img[i] = rand() / (float)RAND_MAX;

    double sigma = 4.0; // width / 6.0;
    gpufilter::weights(sigma, w);

    a0border = (width+63)/64; // up to half-image border in blocks
    if (btype == gpufilter::CLAMP_TO_ZERO)
        a0border = 0;
    if (a0border < border)
        a0border = border;

    me = mre = 0.f; // maximum error and maximum relative error
}

/**
 *  @ingroup api_cpu
 *  @brief Print the application setup information
 *  @param[in] width Image width
 *  @param[in] height Image height
 *  @param[in] btype Border type (either zero, clamp, repeat or reflect)
 *  @param[in] border Number of border blocks (32x32) outside image
 *  @param[in] a0border Number of border blocks for algorithm 0 (reference)
 *  @param[in] w Filter weights (feedforward and feedback coefficients)
 */
void print_info(const int& width, const int& height,
                const gpufilter::BorderType& btype,
                const int& border, const int& a0border,
                const gpufilter::Vector<float, ORDER+1>& w) {
    std::cout << APPNAME << " Size: " << width << " x " << height
              << "  Order: " << ORDER << "  Run-times: 1\n";
    std::cout << APPNAME << " Border type: "
              << (btype==gpufilter::CLAMP_TO_ZERO?"zero":
                  (btype==gpufilter::CLAMP_TO_EDGE?"clamp":
                   (btype==gpufilter::REPEAT?"repeat":
                    (btype==gpufilter::REFLECT?"reflect":
                     "undefined")))) << "\n";
    std::cout << APPNAME << " Weights: " << w << "\n";
    std::cout << APPNAME << " Border for the reference (alg0): "
              << a0border << "\n";
    std::cout << APPNAME << " Border for the result: "
              << border << "\n";
    std::cout << APPNAME << " (1) Runs the reference in the CPU (ref)\n";
    std::cout << APPNAME << " (2) Runs the algorithm in the GPU (res)\n";
    std::cout << APPNAME << " (3) Checks computations (ref x res)\n";
}

//==============================================================================
#endif // CPUDEFS_H
//==============================================================================
