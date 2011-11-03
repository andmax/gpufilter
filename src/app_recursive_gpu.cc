
// Main
int main( int argc, char** argv ) {

    int width = 1024, height = 1024;

    int ne = width * height; // number of elements

    std::cout << "Initializing input image with random values\n";

    std::vector< float > h_in_img( ne );

    srand( 1234 );
    for (int i = 0; i < ne; ++i)
        h_in_img[i] = rand() / (float)RAND_MAX;

    std::vector< float > h_ref_img( h_in_img );

    float sigma = 5.f;
    float a1, b1;
    weights1(sigma, &a1, &b1);

    float a2, b2, c2;
    weights2(sigma, &a2, &b2, &c2);

    std::cout << "Computing in the CPU\n";

    recursive_rows_forward( &h_ref_img[0], height, width, a1, b1 );
    recursive_rows_backward( &h_ref_img[0], height, width, a1, b1 );
    recursive_columns_forward( &h_ref_img[0], height, width, a1, b1 );
    recursive_columns_backward( &h_ref_img[0], height, width, a1, b1 );

    std::cout << "Computing in the GPU\n";

    recursive_filtering_5_1( &h_in_img[0], width, height, a1, b1 );

    std::cout << "Checking result\n";

    float me = 0.f, mre = 0.f; // maximum error and maximum relative error

    check_cpu_reference( &h_ref_img[0], &h_in_img[0], ne, me, mre );

    std::cout << "me = " << me << " mre = " << mre << "\n";

    print_errors( &h_ref_img[0], &h_in_img[0], width, height, .0001f );

    return 0;

}
