/*
SAM installer v1.1: This code installs the SAM template header file and
autotunes it to the primary GPU in the system. To learn more about SAM, see
http://cs.txstate.edu/~burtscher/papers/pldi16.pdf.

Copyright (c) 2016, Texas State University. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted for academic, research, experimental, or personal use provided
that the following conditions are met:

   * Redistributions of source code must retain the above copyright notice,
     this list of conditions, and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions, and the following disclaimer in the documentation
     and/or other materials provided with the distribution.
   * Neither the name of Texas State University nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

For all other uses, please contact the Office for Commercialization and Industry
Relations at Texas State University <http://www.txstate.edu/ocir/>.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Version 1.1 (2016/3/13):
 - works with SAM 1.1

Authors: Sepideh Maleki and Martin Burtscher
*/


#include <cstdio>
#include <cassert>
#include "sam_pre1.1.h"

template <typename T>
__host__ __device__ T sum(T a, T b)
{
  return a + b;
}

struct GPUTimer
{
  cudaEvent_t beg, end;

  GPUTimer()
  {
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
  }

  ~GPUTimer()
  {
    cudaEventDestroy(beg);
    cudaEventDestroy(end);
  }

  void start()
  {
    cudaEventRecord(beg, 0);
  }

  double stop()
  {
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float ms;
    cudaEventElapsedTime(&ms, beg, end);
    return 0.001 * ms;
  }
};

static char* readPreHeader(size_t &size)
{
  FILE* f = fopen("sam_pre1.1.h", "rt");
  if (f == NULL) {
    fprintf(stderr, "ERROR: Could not find or open sam_pre1.1.h in current directory.\n");
    exit(-1);
  }
  fseek(f, 0, SEEK_END);
  size = ftell(f);
  if (size <= 0) {
    fprintf(stderr, "ERROR: sam_pre1.1.h appears to be empty.\n");
    exit(-1);
  }
  char* buf = new char[size];
  fseek(f, 0, SEEK_SET);
  size_t len = fread(buf, sizeof(char), size, f);
  if (len != size) {
    fprintf(stderr, "ERROR: Could not read sam_pre1.1.h in current directory.\n");
    exit(-1);
  }
  fclose(f);
  return buf;
}

static size_t checkGPU()
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
    fprintf(stderr, "ERROR: There is no CUDA capable device.\n");
    exit(-1);
  }
  if (deviceProp.major < 3) {
    fprintf(stderr, "ERROR: Need at least compute capability 3.0.\n");
    exit(-1);
  }
  if (32 < deviceProp.multiProcessorCount) {
    fprintf(stderr, "ERROR: GPUs with more than 32 SMs are not supported.\n");
    exit(-1);
  }
  if (SMs != deviceProp.multiProcessorCount) {
    size_t size;
    char* buf = readPreHeader(size);

    int i = 6;
    while ((i < size) && ((buf[i - 6] != 'S') || (buf[i - 5] != 'M') || (buf[i - 4] != 's') || (buf[i - 3] != ' ') || (buf[i - 2] != '=') || (buf[i - 1] != ' '))) {
      i++;
    }
    if (i > size - 3) {
      fprintf(stderr, "ERROR: sam_pre1.1.h does not contain 'SMs = ...' line.\n");
      exit(-1);
    }
    if (deviceProp.multiProcessorCount >= 10) {
      buf[i] = (deviceProp.multiProcessorCount / 10) + '0';
    } else {
      buf[i] = ' ';
    }
    buf[i + 1] = (deviceProp.multiProcessorCount % 10) + '0';

    FILE* f = fopen("sam_pre1.1.h", "wb");
    if (f == NULL) {
      fprintf(stderr, "ERROR: Could not write sam_pre1.1.h in current directory.\n");
      exit(-1);
    }
    size_t len = fwrite(buf, sizeof(char), size, f);
    if (len != size) {
      fprintf(stderr, "ERROR: Could not write sam_pre1.1.h in current directory.\n");
      exit(-1);
    }
    fclose(f);

    printf("INFO: Set SMs to %d. Please recompile and rerun the installer.\n", deviceProp.multiProcessorCount);
    exit(-1);
  }
  printf("using %s\n", deviceProp.name);
  return deviceProp.totalGlobalMem;
}

template <typename T, int factor, int dim, int order, T (*op)(T, T)>
static double measure(const int items)
{
  const size_t size = items * sizeof(T);

  T *ginput, *goutput;
  cudaMalloc(&ginput, size);
  cudaMalloc(&goutput, size);

  GPUTimer timer;
  timer.start();
  rSAM<T, factor, dim, order, op>(ginput, goutput, items);
  double runtime = timer.stop();

  if (cudaSuccess != cudaGetLastError()) {
    runtime = INFINITY;
  }

  cudaFree(ginput);  cudaFree(goutput);

  return runtime;
}

static const int MAXF = 20;
#define half(fac) if (fac <= MAXF) rt[fac] += measure<mytype, fac, dim, order, sum<mytype> >(hitems)
#define full(fac) if (fac <= MAXF) rt[fac] += measure<mytype, fac, dim, order, sum<mytype> >(items)

int main(void)
{
  printf("SAM Installer (%s)\n", __FILE__);
  printf("Copyright (c) 2016 Texas State University\n");

  size_t globmem = checkGPU();

  globmem /= 1024 * 1024;
  size_t toptest = 1;
  while (toptest * 17 < globmem) {
    toptest *= 2;
  }
  if (toptest > 1024) toptest = 1024;

  size_t size;
  char* buf = readPreHeader(size);

  FILE* f = fopen("sam.h", "wt");
  if (f == NULL) {
    fprintf(stderr, "ERROR: Could not write sam.h in current directory.\n");
    exit(-1);
  }
  fprintf(f, "#ifndef TXSTATE_CS_ECL_SAM\n");
  fprintf(f, "#define TXSTATE_CS_ECL_SAM\n");
  fprintf(f, "\n");
  size_t len = fwrite(buf, sizeof(char), size, f);
  if (len != size) {
    fprintf(stderr, "ERROR: Could not write sam.h in current directory.\n");
    exit(-1);
  }
  fprintf(f, "#define runSAM(fac) rSAM<T, fac, dim, order, op>(ginput, goutput, items)\n");
  fprintf(f, "\n");
  fprintf(f, "template <typename T, int dim, int order, T (*op)(T, T)>\n");
  fprintf(f, "static void SAM(const T * const __restrict__ ginput, T * const __restrict__ goutput, const int items)\n");
  fprintf(f, "{\n");

  fprintf(f, "  if (sizeof(T) <= 4) {\n");
  printf("This will take a while...\n");

  {
    fprintf(f, "    if ((dim == 1) && (order == 1)) {\n");
    const int dim = 1;
    const int order = 1;
    typedef int mytype;
    printf("tuning:  dim = 1  order = 1  int type\n");

    int bf[19], it[19], top = 0;
    for (int items = 2048; items < 1024 * 1024 * toptest / sizeof(mytype) * 4; items *= 2) {
      double rt[MAXF + 1];
      for (int i = 0; i <= MAXF; i++) rt[i] = 0;
      int hitems = items / 4 * 3;

      half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
      half(11);  half(12);  half(13);  half(14);  half(15);  half(16);  half(17);  half(18);  half(19);  half(20);
      full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);
      full(11);  full(12);  full(13);  full(14);  full(15);  full(16);  full(17);  full(18);  full(19);  full(20);

      half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
      half(11);  half(12);  half(13);  half(14);  half(15);  half(16);  half(17);  half(18);  half(19);  half(20);
      full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);
      full(11);  full(12);  full(13);  full(14);  full(15);  full(16);  full(17);  full(18);  full(19);  full(20);

      half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
      half(11);  half(12);  half(13);  half(14);  half(15);  half(16);  half(17);  half(18);  half(19);  half(20);
      full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);
      full(11);  full(12);  full(13);  full(14);  full(15);  full(16);  full(17);  full(18);  full(19);  full(20);

      double btime = rt[1];
      int bfactor = 1;
      for (int i = 2; i <= MAXF; i++) {
        if (rt[i] < btime) {
          btime = rt[i];
          bfactor = i;
        }
      }
      bf[top] = bfactor;
      it[top] = items;
      top++;
      assert(top <= 19);
    }
    assert(top > 0);
    fprintf(f, "      if (items <= 1024) runSAM(1);\n");
    for (int i = 0; i < top - 1; i++) {
      if (bf[i] != bf[i + 1]) {
        fprintf(f, "      else if (items <= %d) runSAM(%d);\n", it[i], bf[i]);
      }
    }
    fprintf(f, "      else runSAM(%d);\n", bf[top - 1]);
    fprintf(f, "    }\n");
  }

  {
    fprintf(f, "    if ((dim == 1) && (order > 1)) {\n");
    const int dim = 1;
    typedef int mytype;
    printf("tuning:  dim = 1  order > 1  int type\n");

    int bf[19], it[19], top = 0;
    for (int items = 2048; items < 1024 * 1024 * toptest / sizeof(mytype) * 4; items *= 2) {
      double rt[MAXF + 1];
      for (int i = 0; i <= MAXF; i++) rt[i] = 0;
      int hitems = items / 4 * 3;

      {
        const int order = 2;
        half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
        half(11);  half(12);  half(13);  half(14);  half(15);  half(16);  half(17);  half(18);  half(19);  half(20);
        full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);
        full(11);  full(12);  full(13);  full(14);  full(15);  full(16);  full(17);  full(18);  full(19);  full(20);
      }

      {
        const int order = 3;
        half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
        half(11);  half(12);  half(13);  half(14);  half(15);  half(16);  half(17);  half(18);  half(19);  half(20);
        full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);
        full(11);  full(12);  full(13);  full(14);  full(15);  full(16);  full(17);  full(18);  full(19);  full(20);
      }

      {
        const int order = 4;
        half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
        half(11);  half(12);  half(13);  half(14);  half(15);  half(16);  half(17);  half(18);  half(19);  half(20);
        full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);
        full(11);  full(12);  full(13);  full(14);  full(15);  full(16);  full(17);  full(18);  full(19);  full(20);
      }

      {
        const int order = 5;
        half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
        half(11);  half(12);  half(13);  half(14);  half(15);  half(16);  half(17);  half(18);  half(19);  half(20);
        full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);
        full(11);  full(12);  full(13);  full(14);  full(15);  full(16);  full(17);  full(18);  full(19);  full(20);
      }

      double btime = rt[1];
      int bfactor = 1;
      for (int i = 2; i <= MAXF; i++) {
        if (rt[i] < btime) {
          btime = rt[i];
          bfactor = i;
        }
      }
      bf[top] = bfactor;
      it[top] = items;
      top++;
      assert(top <= 19);
    }
    assert(top > 0);
    fprintf(f, "      if (items <= 1024) runSAM(1);\n");
    for (int i = 0; i < top - 1; i++) {
      if (bf[i] != bf[i + 1]) {
        fprintf(f, "      else if (items <= %d) runSAM(%d);\n", it[i], bf[i]);
      }
    }
    fprintf(f, "      else runSAM(%d);\n", bf[top - 1]);
    fprintf(f, "    }\n");
  }

  {
    fprintf(f, "    if ((dim > 1) && (order == 1)) {\n");

    const int order = 1;
    typedef int mytype;
    printf("tuning:  dim > 1  order = 1  int type\n");

    int bf[19], it[19], top = 0;
    for (int items = 2048; items < 1024 * 1024 * toptest / sizeof(mytype) * 4; items *= 2) {
      double rt[MAXF + 1];
      for (int i = 0; i <= MAXF; i++) rt[i] = 0;
      int hitems = items / 4 * 3;

      {
        const int dim = 2;
        half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
        half(11);  half(12);  half(13);  half(14);  half(15);  half(16);  half(17);  half(18);  half(19);  half(20);
        full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);
        full(11);  full(12);  full(13);  full(14);  full(15);  full(16);  full(17);  full(18);  full(19);  full(20);
      }

      {
        const int dim = 3;
        half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
        half(11);  half(12);  half(13);  half(14);  half(15);  half(16);  half(17);  half(18);  half(19);  half(20);
        full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);
        full(11);  full(12);  full(13);  full(14);  full(15);  full(16);  full(17);  full(18);  full(19);  full(20);
      }

      {
        const int dim = 4;
        half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
        half(11);  half(12);  half(13);  half(14);  half(15);  half(16);  half(17);  half(18);  half(19);  half(20);
        full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);
        full(11);  full(12);  full(13);  full(14);  full(15);  full(16);  full(17);  full(18);  full(19);  full(20);
      }

      {
        const int dim = 5;
        half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
        half(11);  half(12);  half(13);  half(14);  half(15);  half(16);  half(17);  half(18);  half(19);  half(20);
        full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);
        full(11);  full(12);  full(13);  full(14);  full(15);  full(16);  full(17);  full(18);  full(19);  full(20);
      }

      double btime = rt[1];
      int bfactor = 1;
      for (int i = 2; i <= MAXF; i++) {
        if (rt[i] < btime) {
          btime = rt[i];
          bfactor = i;
        }
      }
      bf[top] = bfactor;
      it[top] = items;
      top++;
      assert(top <= 19);
    }
    assert(top > 0);
    fprintf(f, "      if (items <= 1024) runSAM(1);\n");
    for (int i = 0; i < top - 1; i++) {
      if (bf[i] != bf[i + 1]) {
        fprintf(f, "      else if (items <= %d) runSAM(%d);\n", it[i], bf[i]);
      }
    }
    fprintf(f, "      else runSAM(%d);\n", bf[top - 1]);
    fprintf(f, "    }\n");
  }

  {
    fprintf(f, "    if ((dim > 1) && (order > 1)) {\n");

    typedef int mytype;
    printf("tuning:  dim > 1  order > 1  int type\n");

    int bf[19], it[19], top = 0;
    for (int items = 2048; items < 1024 * 1024 * toptest / sizeof(mytype) * 4; items *= 2) {
      double rt[MAXF + 1];
      for (int i = 0; i <= MAXF; i++) rt[i] = 0;
      int hitems = items / 4 * 3;

      {
        const int dim = 2;
        const int order = 2;
        half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
        half(11);  half(12);  half(13);  half(14);  half(15);  half(16);  half(17);  half(18);  half(19);  half(20);
        full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);
        full(11);  full(12);  full(13);  full(14);  full(15);  full(16);  full(17);  full(18);  full(19);  full(20);
      }

      {
        const int dim = 3;
        const int order = 2;
        half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
        half(11);  half(12);  half(13);  half(14);  half(15);  half(16);  half(17);  half(18);  half(19);  half(20);
        full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);
        full(11);  full(12);  full(13);  full(14);  full(15);  full(16);  full(17);  full(18);  full(19);  full(20);
      }

      {
        const int dim = 2;
        const int order = 3;
        half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
        half(11);  half(12);  half(13);  half(14);  half(15);  half(16);  half(17);  half(18);  half(19);  half(20);
        full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);
        full(11);  full(12);  full(13);  full(14);  full(15);  full(16);  full(17);  full(18);  full(19);  full(20);
      }

      {
        const int dim = 3;
        const int order = 3;
        half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
        half(11);  half(12);  half(13);  half(14);  half(15);  half(16);  half(17);  half(18);  half(19);  half(20);
        full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);
        full(11);  full(12);  full(13);  full(14);  full(15);  full(16);  full(17);  full(18);  full(19);  full(20);
      }

      double btime = rt[1];
      int bfactor = 1;
      for (int i = 2; i <= MAXF; i++) {
        if (rt[i] < btime) {
          btime = rt[i];
          bfactor = i;
        }
      }
      bf[top] = bfactor;
      it[top] = items;
      top++;
      assert(top <= 19);
    }
    assert(top > 0);
    fprintf(f, "      if (items <= 1024) runSAM(1);\n");
    for (int i = 0; i < top - 1; i++) {
      if (bf[i] != bf[i + 1]) {
        fprintf(f, "      else if (items <= %d) runSAM(%d);\n", it[i], bf[i]);
      }
    }
    fprintf(f, "      else runSAM(%d);\n", bf[top - 1]);
    fprintf(f, "    }\n");
  }

  fprintf(f, "  } else {\n");

  {
    fprintf(f, "    if ((dim == 1) && (order == 1)) {\n");
    const int dim = 1;
    const int order = 1;
    typedef long mytype;
    printf("tuning:  dim = 1  order = 1  long type\n");

    int bf[19], it[19], top = 0;
    for (int items = 2048; items < 1024 * 1024 * toptest / sizeof(mytype) * 4; items *= 2) {
      double rt[MAXF + 1];
      for (int i = 0; i <= MAXF; i++) rt[i] = 0;
      int hitems = items / 4 * 3;

      half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
      full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);

      half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
      full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);

      half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
      full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);

      double btime = rt[1];
      int bfactor = 1;
      for (int i = 1; i < 10; i++) {
        if (rt[i] < btime) {
          btime = rt[i];
          bfactor = i;
        }
      }
      bf[top] = bfactor;
      it[top] = items;
      top++;
      assert(top <= 19);
    }
    assert(top > 0);
    fprintf(f, "      if (items <= 1024) runSAM(1);\n");
    for (int i = 0; i < top - 1; i++) {
      if (bf[i] != bf[i + 1]) {
        fprintf(f, "      else if (items <= %d) runSAM(%d);\n", it[i], bf[i]);
      }
    }
    fprintf(f, "      else runSAM(%d);\n", bf[top - 1]);
    fprintf(f, "    }\n");
  }

  {
    fprintf(f, "    if ((dim == 1) && (order > 1)) {\n");

    const int dim = 1;
    typedef long mytype;
    printf("tuning:  dim = 1  order > 1  long type\n");

    int bf[19], it[19], top = 0;
    for (int items = 2048; items < 1024 * 1024 * toptest / sizeof(mytype) * 4; items *= 2) {
      double rt[MAXF + 1];
      for (int i = 0; i <= MAXF; i++) rt[i] = 0;
      int hitems = items / 4 * 3;

      {
        const int order = 2;
        half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
        full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);
      }

      {
        const int order = 3;
        half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
        full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);
      }

      {
        const int order = 4;
        half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
        full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);
      }

      {
        const int order = 5;
        half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
        full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);
      }

      double btime = rt[1];
      int bfactor = 1;
      for (int i = 1; i < 10; i++) {
        if (rt[i] < btime) {
          btime = rt[i];
          bfactor = i;
        }
      }
      bf[top] = bfactor;
      it[top] = items;
      top++;
      assert(top <= 19);
    }
    assert(top > 0);
    fprintf(f, "      if (items <= 1024) runSAM(1);\n");
    for (int i = 0; i < top - 1; i++) {
      if (bf[i] != bf[i + 1]) {
        fprintf(f, "      else if (items <= %d) runSAM(%d);\n", it[i], bf[i]);
      }
    }
    fprintf(f, "      else runSAM(%d);\n", bf[top - 1]);
    fprintf(f, "    }\n");
  }

  {
    fprintf(f, "    if ((dim > 1) && (order == 1)) {\n");
    const int order = 1;
    typedef long mytype;
    printf("tuning:  dim > 1  order = 1  long type\n");

    int bf[19], it[19], top = 0;
    for (int items = 2048; items < 1024 * 1024 * toptest / sizeof(mytype) * 4; items *= 2) {
      double rt[MAXF + 1];
      for (int i = 0; i <= MAXF; i++) rt[i] = 0;
      int hitems = items / 4 * 3;

      {
        const int dim = 2;
        half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
        full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);
      }

      {
        const int dim = 3;
        half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
        full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);
      }

      {
        const int dim = 4;
        half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
        full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);
      }

      {
        const int dim = 5;
        half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
        full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);
      }

      double btime = rt[1];
      int bfactor = 1;
      for (int i = 1; i < 10; i++) {
        if (rt[i] < btime) {
          btime = rt[i];
          bfactor = i;
        }
      }
      bf[top] = bfactor;
      it[top] = items;
      top++;
      assert(top <= 19);
    }
    assert(top > 0);
    fprintf(f, "      if (items <= 1024) runSAM(1);\n");
    for (int i = 0; i < top - 1; i++) {
      if (bf[i] != bf[i + 1]) {
        fprintf(f, "      else if (items <= %d) runSAM(%d);\n", it[i], bf[i]);
      }
    }
    fprintf(f, "      else runSAM(%d);\n", bf[top - 1]);
    fprintf(f, "    }\n");
  }

  {
    fprintf(f, "    if ((dim > 1) && (order > 1)) {\n");
    typedef long mytype;
    printf("tuning:  dim > 1  order > 1  long type\n");

    int bf[19], it[19], top = 0;
    for (int items = 2048; items < 1024 * 1024 * toptest / sizeof(mytype) * 4; items *= 2) {
      double rt[MAXF + 1];
      for (int i = 0; i <= MAXF; i++) rt[i] = 0;
      int hitems = items / 4 * 3;

      {
        const int dim = 2;
        const int order = 2;
        half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
        full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);
      }

      {
        const int dim = 3;
        const int order = 2;
        half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
        full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);
      }

      {
        const int dim = 2;
        const int order = 3;
        half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
        full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);
      }

      {
        const int dim = 3;
        const int order = 3;
        half(1);  half(2);  half(3);  half(4);  half(5);  half(6);  half(7);  half(8);  half(9);  half(10);
        full(1);  full(2);  full(3);  full(4);  full(5);  full(6);  full(7);  full(8);  full(9);  full(10);
      }

      double btime = rt[1];
      int bfactor = 1;
      for (int i = 1; i < 10; i++) {
        if (rt[i] < btime) {
          btime = rt[i];
          bfactor = i;
        }
      }
      bf[top] = bfactor;
      it[top] = items;
      top++;
      assert(top <= 19);
    }
    assert(top > 0);
    fprintf(f, "      if (items <= 1024) runSAM(1);\n");
    for (int i = 0; i < top - 1; i++) {
      if (bf[i] != bf[i + 1]) {
        fprintf(f, "      else if (items <= %d) runSAM(%d);\n", it[i], bf[i]);
      }
    }
    fprintf(f, "      else runSAM(%d);\n", bf[top - 1]);
    fprintf(f, "    }\n");
  }

  fprintf(f, "  }\n");
  fprintf(f, "}\n");
  fprintf(f, "\n");
  fprintf(f, "#endif\n");
  fclose(f);

  printf("done\n");
  return 0;
}
