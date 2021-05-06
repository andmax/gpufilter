# SAM :: Sepideh Annie Martin's algorithm

## How to compile SAM based on:

https://userweb.cs.txstate.edu/~burtscher/research/SAM/

## Run this sequence:

nvcc -Wno-deprecated-declarations -O3 -arch=sm_61 SAMinstaller1.1.cu -o SAMinstaller

./SAMinstaller

nvcc -Wno-deprecated-declarations -O3 -arch=sm_61 testSAM1.1.cu -o testSAM
