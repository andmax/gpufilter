# gpufilter

gpufilter stands for GPU Recursive Filtering.  The goal of this project
is to provide the baseline code in C/C++/CUDA for computing the fastest
boundary-aware recursive filtering using the GPU (graphics processing unit).
The *fastest* means fastest to date (as of 2016) and *boundary
awareness* means closed-form exact (no approximations), i.e.,
the idea is to compute the exact initial feedbacks needed for
recursive filtering infinite input extensions.

Please keep in mind that this code is just to check the performance
and accuracy of the recursive filtering algorithms on a 2D random
image of 32bit floats.  Nevertheless, the code can be specialized
for 1D or 3D inputs, and for reading any given input.

## Papers

The project contains the mathematical derivations and algorithms described
in the following papers:

+ [GPU-Efficient Recursive Filtering and Summed Area Tables](http://dx.doi.org/10.1145/2024156.2024210)
+ [Parallel Recursive Filtering of Infinite Input Extensions](http://dx.doi.org/10.1145/2980179.2980222)

## Getting started

The code is in C/C++/CUDA and has been tested in an NVIDIA GTX Titan.
Jupyter notebooks are included for explanations purposes only.
The following sequence of commands assume a working computer environment
with CMake and the CUDA SDK (see prerequisites below).  It compiles
all the algorithms for testing in a range of recursive filter orders.
The total compilation time may take **tens of minutes** to complete.

```
mkdir build
cd build
cmake ..
make
```

It may be the case (depending on your environment) that the *cmake*
command line must be properly configured (for instance to access
the proper host compiler):

```
cmake -DCUDA_HOST_COMPILER=/usr/bin/g++ ..
```

Or that you need to change the *sm_61* (for Pascal) architecture to
another target architecture of your choice in the root
[CMakeLists.txt] (CMakeLists.txt) file.

To run the algorithms after compiling, execute:

```
src/algN_R
```

replacing N for the desired algorithm (3-6) and R for the desired
filter order (1-5).  There are also three extra algorithms that can be
called by:

```
src/alg5f4
src/alg5varc
src/sat
```

where the first is the algorithm 5 fusioned with 4,
and the second is the algorithm 5 with varying coefficients.

### Prerequisities

The project has been successfully compiled and tested using
the following environment:

+ Ubuntu 16.04
+ CUDA 8.0
+ gcc/g++ 5.4.0
+ CMake 3.5.1
+ Python 2.7.6
+ Pandas 0.18.1

## Running the tests

First compile all the algorithms (see getting started above)
to then be able to run (this may take **hours** to finish):

```
cd ../scripts
mkdir results
sh runall.sh
```

The bash scripts use a python script (in scripts/) to compute the
average of the performance results.  The python script depends on the
Pandas library, that can be installed via:

```
pip install pandas
```

## Authors

The authors of this project is listed in the [AUTHORS](AUTHORS) file.

## License

This project is licensed under the MIT License - see the [COPYING](COPYING)
file for details.

