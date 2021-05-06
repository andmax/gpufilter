#include <stdlib.h>
#include <stdio.h> 
#include <sys/time.h> 
#include <cub/cub.cuh> 
#include "cub/util_allocator.cuh"
#include "cub/device/device_scan.cuh"
#include <math.h>

typedef float T;
static const int order = 1;
double mebissec; // timing result in Mis/s
cub::CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

struct Matrix {
  T data[order][order];
  void initialize(T coef[order]) {
    for (int i = 0; i < order; i++) {
      data[i][0] = coef[i];
      for (int j = 1; j < order; j++) {
        if (j == (i + 1)) {
          data[i][j] = 1;
        } else {
          data[i][j] = 0;
        }
      }
    }
  }
};

struct Vector {
  T data[order];
  void initialize(T val) {
    data[0] = val;
    for (int i = 1; i < order; i++) {
      data[i] = 0;
    }
  }
};

struct Join {
  Matrix m;
  Vector v;
  void initialize(T coef[order], T val) {
    m.initialize(coef);
    v.initialize(val);
  }
};

struct JopJ
{
  __device__
  CUB_RUNTIME_FUNCTION __forceinline__
  Join operator()(const Join &a, const Join &b) const {
    Join d;
    d.v = b.v;
    for (int i = 0; i < order; i++) {
      for (int j = 0; j < order; j++) {
        d.v.data[i] += a.v.data[j] * b.m.data[j][i];
      }
    }

    Matrix c;
    for (int i = 0; i < order; i++) {
      for (int j = 0; j < order; j++) {
        c.data[i][j] = 0;
      }
    }
    for (int i = 0; i < order; i++) {
      for (int k = 0; k < order; k++) {
        for (int j = 0; j < order; j++) {
          c.data[i][j] += a.m.data[i][k] * b.m.data[k][j];
        }
      }
    }

    d.m = c;
    return d;
  }
};

static void cub_scan(const Join* const __restrict__ input, Join* const __restrict__ output, const long int len, const long int rep_cnt)
{
  void *temp_storage_d = NULL;
  size_t temp_storage_bytes = 0;
  JopJ join_op;   

  //struct timeval start, end;  
  cudaEvent_t start, stop;
  float elapsedTime;
  
  cub::DeviceScan::InclusiveScan(temp_storage_d, temp_storage_bytes, input, output, join_op, len);
  if (cudaSuccess != cudaMalloc((void **)&temp_storage_d, temp_storage_bytes)) {printf("could not allocate temp_storage_d\n"); exit(-1);}

  cub::DeviceScan::InclusiveScan(temp_storage_d, temp_storage_bytes, input, output, join_op, len);

  // start time
  //gettimeofday(&start, NULL);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // simulating fwd+rev by doubling iterations
  //for (long int i = 0; i < rep_cnt; i++) {
  for (long int i = 0; i < rep_cnt*2; i++) {
    cub::DeviceScan::InclusiveScan(temp_storage_d, temp_storage_bytes, input, output, join_op, len);
  }

  cudaDeviceSynchronize();
  
  // end time
  //gettimeofday(&end, NULL);
  //double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  double runtime = elapsedTime / 1000.f;

/*
  printf("runtime: %.6f s\n", runtime);
  printf("throughput: %.3f Melements/s\n\n", (rep_cnt * len * 0.000001) / runtime);
*/
  
  mebissec = (rep_cnt * len) / (runtime*1024*1024); // Mis/s

  if (temp_storage_d) cudaFree(temp_storage_d);
}

int main(int argc, char *argv[])
{
  long int input_size, rep_cnt;
  Join *in_d;
  Join *out_d;
  
  int arg_cnt = 3 + order; 
 
  if (argc != arg_cnt) {fprintf(stderr, "usage: %s  input_size  repeat count  %d coefficients\n", argv[0], order); exit(-1);}
  input_size = atol(argv[1]);
  if (input_size < 1) {fprintf(stderr, "The input_size must be at least 1.\n"); exit(-1);}
  rep_cnt = atol(argv[2]);
  if (rep_cnt < 1) {fprintf(stderr, "The repeat count must be at least 1.\n"); exit(-1);}

  // Allocating memory for real numbers, coefficients
  T* const coef = new T[order];
  for (int i = 0; i < order; i++) {
    coef[i] = atof(argv[i + 3]);
  }

  T* const data = new T[input_size];
  for (long int i = 0; i < input_size; i++) {
    data[i] = (i & 1) * 2 - 1;
  }
  
/*  
  printf("Solving linear recurrence of order %d for %ld terms with coefficients: ", order, input_size);
  for (int i = 0; i < order; i++) {
    printf("%f, ", coef[i]);
  }
  printf("\n");
*/

  Join* const in = new Join[input_size];
  for (long int i = 0; i < input_size; i++) {
    in[i].initialize(coef, data[i]);
  }
  
  CubDebugExit(g_allocator.DeviceAllocate((void**)&in_d, input_size * sizeof(Join))); 
  CubDebugExit(g_allocator.DeviceAllocate((void**)&out_d, input_size * sizeof(Join))); 

  if (cudaSuccess != cudaMemcpy(in_d, in, input_size * sizeof(Join), cudaMemcpyHostToDevice)) {
    printf("Copy from host to device failed for in_d\n"); exit(-1);
  }

  cub_scan(in_d, out_d, input_size, rep_cnt);
    
  // Copy the results back to the host 
  Join* const out = new Join[input_size];
  if (cudaSuccess != cudaMemcpy(out, out_d, input_size * sizeof(Join), cudaMemcpyDeviceToHost)) {
    printf("Copy from device to host failed for out_d\n"); exit(-1);
  }

  for (long int i = 1; i < input_size; i++) {
    for (int j = 0; j < order; j++) {
      if (i - 1 - j >= 0) {
        data[i] += coef[j] * data[i - 1 - j];
      }
    }
  }

  // Verifying the results
/*
  for (long int i = 0; i < input_size; i++) {
    if (data[i] != out[i].v.data[0]) {fprintf(stderr, "ERROR: mismatch at location %ld\n", i); exit(-1);}
  }
*/
  
  T max_abs_err = 0, max_rel_err = 0;
  for (long int i = 0; i < input_size; i++) {
    T a = data[i] - out[i].v.data[0];
    if( a < 0 ) a = -a;
    if( data[i] != 0 ) {
      T r = (data[i] < 0) ? -data[i] : data[i];
      T b = a / r;
      max_rel_err = b > max_rel_err ? b : max_rel_err;
    }
    max_abs_err = a > max_abs_err ? a : max_abs_err;
  }
  printf("%7.7f %e %e\n", mebissec, max_abs_err, max_rel_err);
  
  if (in_d) cudaFree(in_d);
  if (out_d) cudaFree(out_d);
  delete [] coef;
  delete [] data;
  delete [] in;
  delete [] out;

  return 0;
}
