#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>  // needed for timing

// Solving a 1-D Poisson problem using Jacobi iterations.
// This version of the code uses kernel level synchronization



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

static double l2_error(struct GFS gfs);

__global__
 void gauss_seidel (struct GFS gfs)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  //int stride = blockDim.x * gridDim.x;
  const int n = gfs.n;
  const gpu_fp dx2 = gfs.dx2;
  
  const int i = index;
  if (index < n-1 && index > 0)
  {
    gfs.u_new[i] =  (gfs.u[i+1] + gfs.u[i-1] - dx2 * gfs.src[i]) * 0.5 ;
  }
}

__global__
 void copy (struct GFS gfs)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  //int stride = blockDim.x * gridDim.x;
  const int n = gfs.n;
  
  const int i = index;
  if (index < n-1 && index > 0)
  {
    gfs.u[i] = gfs.u_new[i];
  }
}

__global__
 void error_check (struct GFS gfs)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  //int stride = blockDim.x * gridDim.x;
  const int n = gfs.n;
  const int i = index;
  const gpu_fp idx2 = gfs.idx2;

  if (index < n-1 && index > 0)
  {
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

  int blockSize =256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  const double t_start = get_time();
   
  
  printf("numBlocks = %d, blockSize = %d, totalThreads=%d, N=%d\n", numBlocks, blockSize, numBlocks * blockSize, N);
  
  int its = 0;
  for (;;)
  {
    for (int j = 0; j < 1000; j++)
    {
      gauss_seidel <<<numBlocks, blockSize>>> (gfs_managed);
      copy <<<numBlocks, blockSize>>> (gfs_managed);
    }
    error_check <<<numBlocks, blockSize>>> (gfs_managed);
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
