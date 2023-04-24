#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>  // needed for timing

// Solving a 2-D Poisson problem using Jacobi iterations.
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
  gpu_fp dy2;
  gpu_fp idx2;
  gpu_fp idy2;
  gpu_fp *u;
  gpu_fp *u_new;
  gpu_fp *src;
  gpu_fp *error;
};

static double l2_error(struct GFS gfs);

__global__
void jacobi(struct GFS gfs)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;//index_x
  int iy = blockIdx.y * blockDim.y + threadIdx.y;//index_y
  int stride_x = blockDim.x * gridDim.x;//
  int stride_y = blockDim.y * gridDim.y;//
  const int n = gfs.n;
  const gpu_fp dx2 = gfs.dx2;
  const gpu_fp dy2 = gfs.dy2;
  const gpu_fp idx2 = gfs.idx2;
  const gpu_fp idy2 = gfs.idy2;
  const int nx = gfs.nx//
  const int ny = gfs.ny//	 

 
  const int i = iy * n + ix; // TODO: Is this valid?
	const int ip1 = i + 1;
	const int im1 = i - 1;
	const int jp1 = i + n; 
	const int jm1 = i - n;
  if (i < n-1 && i > 0)//this accounts for ghost points and index likely needs to be changed to ix
  {
    gfs.u_new[i] =  (0.5 * idx2 * idy2) * (dy2 * (gfs.u[ip1] + gfs.u[im1]) + dx2 * (gfs.u[jp1] + gfs.u[jm1]) - dx2 * dy2 * gfs.src[i]);
  }

  for(int j = iy; j < ny - 1; j += stride_y)
  {
	  if(j == 0) continue;
		for(int i = ix; i < nx - 1; i += stride_x)
    {
		  if(i == 0) continue;
		  gfs.u_new[i + j * nx] = (dy2 * (gfs.u[i + 1 + j * nx] + gfs.u[i - 1 + j * nx]) + dx2 * (gfs.u[i + (j + 1) * nx] + gfs.u[i + (j - 1) * nx] - gfs.src[i + j * nx] * dx2 * dy2)) * (.5 * idx2 * idy2);
		  gfs.u[i + j * nx] = gfs.u_new[i + j * nx];
    }
	}
	//we also have to make sure we have the same numebr of threads as points in the x direction, and same for the y
	//4/18 lecture from 11:30-24:44

   __syncthreads(); //AV: is this enough for synchronization?
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
  int ix = blockIdx.x * blockDim.x + threadIdx.x;//index_x
  int iy = blockIdx.y * blockDim.y + threadIdx.y;//index_y
  const int i = ix + n * iy
	const int ip1 = i + 1;
	const int im1 = i - 1;
	const int jp1 = i + n; 
	const int jm1 = i - n;
  const gpu_fp idx2 = gfs.idx2;
  const gpu_fp idy2 = gfs.idy2

  if (i < n-1 && i > 0)
  {
    gfs.error[i] = (gfs.u[ip1] + gfs.u[im1] - 2 * gfs.u[i]) * idx2 + (gfs.u[jp1] + gfs.u[jm1] - 2 * gfs.u[i]) * idy2 - gfs.src[i];
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
  const int N = 1<<8;

  struct GFS gfs_managed;

 // const gpu_fp t_final = atof(argv[1]);

  CUDA_CHECK(cudaMallocManaged(&(gfs_managed.u), N*N*sizeof(gpu_fp)));
  CUDA_CHECK(cudaMallocManaged(&(gfs_managed.u_new), N*N*sizeof(gpu_fp)));
  CUDA_CHECK(cudaMallocManaged(&(gfs_managed.src), N*N*sizeof(gpu_fp)));
  CUDA_CHECK(cudaMallocManaged(&(gfs_managed.error), N*N*sizeof(gpu_fp)));

	//these are all used in the finite difference approxiamtion to solve the differential equation
  const gpu_fp dx = 1.0 / (N-1);//spacing between grid points in the x direction
  const gpu_fp dy = 1.0 / (N-1);//spacing in the y direction
  gfs_managed.n = N * N;//sets the grid points to N*N
  gfs_managed.dx2 = dx * dx;
  gfs_managed.idx2 = 1.0 / dx / dx;//sets the inverse of the squared spacing between grid points in the x-direction, which is also used in the finite difference approximations.
  gfs_managed.dy2 = dy * dy;
  gfs_managed.idy2 = 1.0 / dy / dy;

  //initialize x and y arrays on the host TODO
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++) {
			const gpu_fp x = i * dx;
			const gpu_fp y = j * dy;
			const int idx = j * N + i;
			gfs_managed.src[idx] = exp(-(x - 0.5) * (x - 0.5) * 50 - (y - 0.5) * (y - 0.5) * 50);
			gfs_managed.u[idx] = 0;
			gfs_managed.u_new[idx] = 0;
			gfs_managed.error[idx] = 0;
		}
	}

  dim3 blockSize(256, 256);
  dim3 numBlocks((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blocksize.y);

  const double t_start = get_time();
   
  
  // printf("numBlocks = %d, blockSize = %d, totalThreads=%d, N=%d\n", numBlocks, blockSize, numBlocks * blockSize, N);
  
  int its = 0;
  for (;;)
  {
    for (int j = 0; j < 100; j++) //j was originally 1000, setting to 100 for testing
    {
      jacobi <<<numBlocks, blockSize>>> (gfs_managed);
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


   
  printf("%d iterations of grid size %d ^2 took %fs\n",its, N,  t_end - t_start);
	for (int j = 0; j < N; j++)
	{
		for (int i = 0; i < N; i++)
		{
			const double x = i * dx;
			const double y = j * dy;
			const double idx = j * N + i
			printf("%20.16e %20.16e %20.16e\n",x, y, gfs_managed.u[idx]);
		}
	}

  // Free memory
  CUDA_CHECK(cudaFree(gfs_managed.u));
  CUDA_CHECK(cudaFree(gfs_managed.u_new));
  CUDA_CHECK(cudaFree(gfs_managed.src));
  CUDA_CHECK(cudaFree(gfs_managed.error));
  CUDA_CHECK(cudaDeviceReset());
  
  return 0;
}
