# gpufilter

gpufilter stands for GPU Recursive Filtering (paper 1).  Additionally,
filtering can also be regarded as Prefix Scans or Sums and Summed-area
Tables (paper 1) or Integral Images.  The main goal of this project is
to provide the baseline code in C/C++/CUDA for computing the fastest
boundary-aware recursive filtering (paper 3) running on the GPU
(graphics processing unit), i.e. a massively-parallel processor (paper
2).  The *fastest* means fastest to date (as the last published paper)
and *boundary awareness* means closed-form exact (i.e. no
approximations).  The idea of boundary awareness is to compute the
exact initial feedbacks needed for recursive filtering infinite input
extensions (paper 3).

Please keep in mind that this code is just to check the performance
and accuracy of the recursive filtering algorithms on a 2D random
image of 32bit floats.  Nevertheless, the code can be specialized for
1D or 3D inputs, and for reading any given input and data type.

## Papers

The project contains the mathematical derivations and algorithms described
in the following papers:

1. [GPU-Efficient Recursive Filtering and Summed Area Tables](http://dx.doi.org/10.1145/2024156.2024210)
2. [Efficient finite impulse response filters in massively-parallel recursive systems](https://doi.org/10.1007/s11554-015-0510-x)
3. [Parallel Recursive Filtering of Infinite Input Extensions](http://dx.doi.org/10.1145/2980179.2980222)

## Getting started

The code is in C/C++/CUDA and has been tested in an NVIDIA GTX Titan
X.  Jupyter notebooks are included for explanations purposes only.
The following sequence of commands assume a working computer
environment with CMake and the CUDA SDK (see prerequisites below).  It
compiles all the algorithms for testing in a range of recursive filter
orders.  The total compilation time may take **tens of minutes** to
complete.

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
another target architecture matching your GPU hardware in the root
[CMakeLists.txt](CMakeLists.txt) file.

To run the algorithms (paper 1 and 3) after compiling, execute:

```
src/algN_R
```

replacing N for the desired algorithm (3-6) and R for the desired
filter order (1-5).  There are also three extra algorithms (paper 1
and 3) that can be called by:

```
src/alg5f4
src/alg5varc
src/sat
```

where the first is the algorithm 5 fusioned with 4, the second is the
algorithm 5 with varying coefficients, and the third is the
summed-area table algorithm.

There are also other algorithms related to parallel interconnection of
causal and anticausal filters, extending to both recursive and
non-recursive filtering (paper 2).  Since they are not fully
integrated with the main source code, they are stored here for
reference purposes only:

```
paredge/
```

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
