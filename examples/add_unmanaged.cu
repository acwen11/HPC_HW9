#include <iostream>
#include <math.h>
#include <stdlib.h>

#define CUDA_ERROR(t_)                                            \
  do {                                                            \
       const long int  _err = t_;                                 \
       if (_err != cudaSuccess  )                                 \
       {                                                          \
         fprintf(stderr, "cuda function call %s failed\n", #t_);  \
         exit(-1);                                                \
       }                                                          \
    } while(0)



// Kernel function to add the elements of two arrays
__global__
void add(int n, double *x, double *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  double *dev_x, *dev_y;
  double *host_x, *host_y;

  CUDA_ERROR(cudaMalloc(&dev_x, N*sizeof(double)));
  CUDA_ERROR(cudaMalloc(&dev_y, N*sizeof(double)));

  host_x = (double *) malloc(N*sizeof(double));
  host_y = (double *) malloc(N*sizeof(double));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    host_x[i] = 1.0 * i;
    host_y[i] = 2.0 * i;
  }

  CUDA_ERROR(cudaMemcpy(dev_x, host_x, N*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_ERROR(cudaMemcpy(dev_y, host_y, N*sizeof(double), cudaMemcpyHostToDevice));

  int blockSize = 128;
//   int numBlocks = (N + blockSize - 1) / blockSize;
  int numBlocks = 128;
  // Run kernel on 1M elements on the GPU
  add<<<numBlocks, blockSize>>>(N, dev_x, dev_y);
  add<<<numBlocks, blockSize>>>(N, dev_x, dev_y);

  // Wait for GPU to finish before accessing on host
  CUDA_ERROR(cudaDeviceSynchronize());
  CUDA_ERROR(cudaMemcpy(host_y, dev_y,  N*sizeof(double), cudaMemcpyDeviceToHost));

  // Check for errors (all values should be 3.0 *  i)
  double maxError = 0.0;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(host_y[i]-4.0 * i));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  CUDA_ERROR(cudaFree(dev_x));
  CUDA_ERROR(cudaFree(dev_y));
  
  free(host_x);
  free(host_y);
  return 0;
}
