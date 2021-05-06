# recxd :: Recursive filtering on x-Dimensions

The idea is to start with 1D and support 3D by combining (gpufilter)
2D with 1D.

## Suggested run order

1. mkdir build
2. cd build
3. cmake ..
4. make
5. src/gen_random_samples
6. cd ..
7. Download CUB to ../cub
8. cd cub
9. make sm=610 all
10. cd ../plr
11. make
12. cd ../sam
13. Follow instruction in README.md
14. cd ..
15. ./run_all.sh

## Files and directories

* bib/ bibliographic references (e.g. pdfs of papers)

* bin/ (will be created by gen_random_samples) binary generated and
shared by programs (e.g. random_data.bin with 1D array of random
samples)

* build/ (should be created) out-of-source cmake build files,
  run the following commands inside:
  $ cmake ..
  $ make

* cub/ cub (nvidia's) iir implementation, run inside:
  $ make sm=610 all

* plr/ parallel linear recurrences (Martin Butcher's) project
  implementation, run the following commands inside it:
  $ mkdir bin
  $ make

* src/ contain this project source main files, it depends on
  the previous gpufilter project on git-hub

* run.py: run script to run a command multiple times
  (e.g.) $ python run.py build/src/alg3 1000 bin/random_array.bin

* CMakefiles: main cmake file

* README.md: this read-me file

## Authors

* Andre Maximo
* Diego Nehab

## License

This project is licensed under the MIT License.
