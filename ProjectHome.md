The GPU-Efficient Recursive Filtering and Summed-Area Table (**gpufilter**) project is a set of _C for CUDA_ functions to compute recursive filters and summed-area tables in GPUs. This project presents a new algorithmic framework for parallel evaluation. It partitions the image into 2D blocks, with a small band of data buffered along each block perimeter. A remarkable result is that the image data is read only twice and written just once, independent of image size, and thus total memory bandwidth is reduced even compared to the traditional serial algorithm.

The gpufilter project is based on the paper: **"GPU-Efficient Recursive Filtering and Summed-Area Tables"** by **Diego Nehab**, **André Maximo**, **Rodolfo S. Lima** and **Hugues Hoppe**. The paper is available at:

http://dx.doi.org/10.1145/2070781.2024210

The documentation of this project is available at:

http://www.impa.br/~andmax/gpufilter/index.html

And other external links include:

Project page: http://www.impa.br/~diego/projects/NehEtAl11

Video: http://www.youtube.com/watch?v=RxlP31vdiA0