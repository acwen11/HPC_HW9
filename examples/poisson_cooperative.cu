#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>  // needed for timing
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Solving a 1-D Poisson problem using Jacobi iterations.
// This version of the code uses cooperative groups to synchronize
// over grids within the same kernel call.



typedef double gpu_fp;

static double get_time(void);

// get_time will return a double containing the current time in
// seconds.
static double get_time(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);

  return (tv.tv_sec) + 1.0e-6 * tv.tv_usec;
}


// Check error status of a CUDA call. Terminate if an error occurs.
#define CUDA_CHECK(t_)                                            \
  do {                                                            \
       const long int  _err = t_;                                 \
       if (_err != cudaSuccess  )                                 \
       {                                                          \
         fprintf(stderr, "cuda function call %s failed\n", #t_);  \
         exit(-1);                                                \
       }                                                          \
    } while(0)

struct GFS
{
  int n;
  gpu_fp dx2;
  gpu_fp idx2;
  gpu_fp *u;
  gpu_fp *u_new;
  gpu_fp *src;
  gpu_fp *error;
};


__global__
 void jacobi_iteration (struct GFS gfs)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  const int n = gfs.n;
  const gpu_fp dx2 = gfs.dx2;
  const gpu_fp idx2 = gfs.idx2;
  cg::grid_group grid = cg::this_grid();

  
  for (int it = 0; it  < 1000; it++)
  {
    for (int i = index ; i < n - 1; i+= stride)
    {
      if (i == 0) continue;
      gfs.u_new[i] =  (gfs.u[i+1] + gfs.u[i-1] - dx2 * gfs.src[i]) * 0.5 ;
    }

    grid.sync();

    for (int i = index ; i < n - 1; i+= stride)
    {
      if (i == 0) continue;
      gfs.u[i] =  gfs.u_new[i];
    }

    grid.sync();
  
  }
  for (int i = index ; i < n - 1; i+= stride)
  {
    if (i == 0) continue;
    gfs.error[i] = (gfs.u[i+1] + gfs.u[i-1] - 2 * gfs.u[i]) * idx2 - gfs.src[i];
    gfs.error[i] *= gfs.error[i] /n;
  }
}

static double l2_error(struct GFS gfs)
{
  const int n = gfs.n;
  const double *e = gfs.error;
  double sum = 0;
  for (int i = 1; i < n-1; i++)
  {
    sum += e[i];
  }
  return sqrt(sum);
}
int main(int argc, char **argv)
{
  const int N = 1<<10;

  struct GFS gfs_managed;

 // const gpu_fp t_final = atof(argv[1]);

  CUDA_CHECK(cudaMallocManaged(&(gfs_managed.u), N*sizeof(gpu_fp)));
  CUDA_CHECK(cudaMallocManaged(&(gfs_managed.u_new), N*sizeof(gpu_fp)));
  CUDA_CHECK(cudaMallocManaged(&(gfs_managed.src), N*sizeof(gpu_fp)));
  CUDA_CHECK(cudaMallocManaged(&(gfs_managed.error), N*sizeof(gpu_fp)));

  const gpu_fp dx = 1.0 / (N-1);
  gfs_managed.n = N;
  gfs_managed.dx2 = dx * dx;
  gfs_managed.idx2 = 1.0 / dx / dx;

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    const gpu_fp x = i * dx;
    gfs_managed.src[i] = exp(-(x-0.5)*(x-0.5)*50);
    gfs_managed.u[i] = 0;
    gfs_managed.u_new[i] = 0;
    gfs_managed.error[i] = 0;
  }

  // This will launch a grid that can maximally fill the GPU, on the default stream with kernel arguments
  int numBlocksPerSm = 0;
  // Number of threads my_kernel will be launched with
  int numThreads = 32;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, jacobi_iteration, numThreads, 0);
  // launch
  void *kernelArgs[] = { &gfs_managed };
  dim3 dimBlock(numThreads, 1, 1);
  dim3 dimGrid(deviceProp.multiProcessorCount*numBlocksPerSm, 1, 1);
  if (dimGrid.x > (N + numThreads - 1) / numThreads)
  {
    dimGrid.x = (N + numThreads - 1)/ numThreads;
  }

  const double t_start = get_time();
   
  
  printf("numBlocks = %d, blockSize = %d, totalThreads=%d, N=%d\n", dimGrid.x, dimBlock.x, dimGrid.x * dimBlock.x, N);
  
  int its = 0;
  for (; ; )
  {
    cudaLaunchCooperativeKernel((void*)jacobi_iteration, dimGrid, dimBlock, kernelArgs);
    its ++ ;
    CUDA_CHECK(cudaDeviceSynchronize());
    const double error  = l2_error(gfs_managed);
    printf("%d %20.16e\n", its, error);
    if (error < 1.0e-9) break;
  }


  // Wait for GPU to finish before accessing on host
  // Comment out to see what happens when synchronize isn't called
  CUDA_CHECK(cudaDeviceSynchronize());
  const double t_end = get_time();


   
  printf("%d iterations of grid size %d took %fs\n",its, N,  t_end - t_start);
  for (int i = 0; i < N; i++)
  {
    const double x = i * dx;
    printf("%20.16e %20.16e\n",x, gfs_managed.u[i]);
  }

  // Free memory
  CUDA_CHECK(cudaFree(gfs_managed.u));
  CUDA_CHECK(cudaFree(gfs_managed.u_new));
  CUDA_CHECK(cudaFree(gfs_managed.src));
  CUDA_CHECK(cudaFree(gfs_managed.error));
  CUDA_CHECK(cudaDeviceReset());
  
  return 0;
}

