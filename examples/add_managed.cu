#include <iostream>
#include <math.h>
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
  double *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(double));
  cudaMallocManaged(&y, N*sizeof(double));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0 * i;
    y[i] = 2.0 * i;
  }

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  // Run kernel on 1M elements on the GPU
  add<<<numBlocks, blockSize>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0)
  double maxError = 0.0;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0 * i));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
