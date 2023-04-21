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


const int threadsPerBlock = 256;

// Kernel function to partially add the elements of an array
__global__
void sum_vector(int n, double *x, double *psum)
{
  // x : INPUT :  an array of doubles to be summed
  // psum : OUTPUT : an array containing the partial sum per block 


  // shared "cache" memory used by each block. This is shared by all
  // threads inside a block, but is unique to each block.
  __shared__ double cache[threadsPerBlock];

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int cacheIndex = threadIdx.x;

  // Not all threads have an associated data value (i.e., there are more threads
  // than elements in the array). Those threads just place zero in the cache.
  double temp = 0.0;
  for (int i = index; i < n; i += stride)
  {
    temp += x[i];
  }
  cache[cacheIndex]  = temp;

  __syncthreads();

  // start off with half the threads in a block summing two elements in the cache
  // then half those, etc, until only 1 thread has the total sum for this block
  int i = threadsPerBlock / 2;
  while ( i != 0)
  {
    if (cacheIndex < i)
    {
      cache[cacheIndex] += cache[cacheIndex + i];
    }
    __syncthreads();
    i /= 2;
  }

  // store the partial sum so the cpu can read it.
  if (cacheIndex == 0)
  {
    psum[blockIdx.x] = cache[0];
  }
}

int main(void)
{
  int N = 1<<20;
  double *dev_x, *dev_psum;
  double *host_x, *host_psum;

  const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

  CUDA_ERROR(cudaMalloc(&dev_x, N*sizeof(double)));
  CUDA_ERROR(cudaMalloc(&dev_psum, numBlocks*sizeof(double)));

  host_x = (double *) malloc(N*sizeof(double));
  host_psum = (double *) malloc(numBlocks*sizeof(double));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    host_x[i] = i+1;
  }

  CUDA_ERROR(cudaMemcpy(dev_x, host_x, N*sizeof(double), cudaMemcpyHostToDevice));


  // Run kernel on 1M elements on the GPU
  sum_vector<<<numBlocks, threadsPerBlock>>>(N, dev_x, dev_psum);

  CUDA_ERROR(cudaMemcpy(host_psum, dev_psum,  numBlocks*sizeof(double), cudaMemcpyDeviceToHost));

  double total = 0;
  for (int i = 0; i < numBlocks; i++)
  {
    total += host_psum[i];
  }

  #define TOTAL(n_) ((((long)(n_))*((long)(n_) + 1))/2)
  std::cout << (long int) total << " expected " << TOTAL(N) << " error " <<  total - TOTAL(N) << std::endl;

  // Free memory
  CUDA_ERROR(cudaFree(dev_x));
  CUDA_ERROR(cudaFree(dev_psum));
  
  free(host_x);
  free(host_psum);
  return 0;
}
