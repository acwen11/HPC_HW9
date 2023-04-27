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
	int nx;
	int ny;
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
  // int ix = blockIdx.x * blockDim.x + threadIdx.x;
  // int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int ix = threadIdx.x;
  int iy = threadIdx.y;
  // int stride_x = blockDim.x * gridDim.x;
  // int stride_y = blockDim.y * gridDim.y;
  const gpu_fp dx2 = gfs.dx2;
  const gpu_fp dy2 = gfs.dy2;
  const gpu_fp idx2 = gfs.idx2;
  const gpu_fp idy2 = gfs.idy2;
  const int nx = gfs.nx;
  const int ny = gfs.ny; 
 
  const int i = iy * nx + ix; 
	const int ip1 = i + 1;
	const int im1 = i - 1;
	const int jp1 = i + nx; 
	const int jm1 = i - nx;

  if ((ix < nx - 1 && ix > 0) && (iy < ny - 1 && iy > 0)) // 2D Boundaries
  {
    gfs.u_new[i] =  (0.5 * idx2 * idy2) * (dy2 * (gfs.u[ip1] + gfs.u[im1]) + dx2 * (gfs.u[jp1] + gfs.u[jm1]) - dx2 * dy2 * gfs.src[i]);
    // gfs.u_new[i] =  (0.5) * (idx2 * (gfs.u[ip1] + gfs.u[im1]) + idy2 * (gfs.u[jp1] + gfs.u[jm1]) - gfs.src[i]);
  }
}	


__global__
 void copy (struct GFS gfs)
{
  // int ix = blockIdx.x * blockDim.x + threadIdx.x;
  // int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int ix = threadIdx.x;
  int iy = threadIdx.y;
  const int nx = gfs.nx;
  const int ny = gfs.ny; 
 
  const int i = iy * nx + ix; 

  if ((ix < nx - 1 && ix > 0) && (iy < ny - 1 && iy > 0)) // 2D Boundaries
  {
    gfs.u[i] = gfs.u_new[i];
  }
}

__global__
 void error_check (struct GFS gfs)
{
  const int n = gfs.n;
  const int nx = gfs.nx;
  const int ny = gfs.ny; 
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  const int i = ix + nx * iy;

	const int ip1 = i + 1;
	const int im1 = i - 1;
	const int jp1 = i + nx; 
	const int jm1 = i - nx;
  const gpu_fp idx2 = gfs.idx2;
  const gpu_fp idy2 = gfs.idy2;
	// gfs.error[i] = (gfs.u[ip1] + gfs.u[im1] - 2 * gfs.u[i]) * idx2 + (gfs.u[jp1] + gfs.u[jm1] - 2 * gfs.u[i]) * idy2 - gfs.src[i];
	// gfs.error[i] *= gfs.error[i] /n;

  if ((ix < nx - 1 && ix > 0) && (iy < ny - 1 && iy > 0)) // 2D Boundaries
  {
    gfs.error[i] = (gfs.u[ip1] + gfs.u[im1] - 2 * gfs.u[i]) * idx2 + (gfs.u[jp1] + gfs.u[jm1] - 2 * gfs.u[i]) * idy2 - gfs.src[i];
    gfs.error[i] *= gfs.error[i] / n;
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
  // const int N = 1<<8;
  const int N = 16;

  struct GFS gfs_managed;

  // const gpu_fp t_final = atof(argv[1]);

  CUDA_CHECK(cudaMallocManaged(&(gfs_managed.u), N*N*sizeof(gpu_fp)));
  CUDA_CHECK(cudaMallocManaged(&(gfs_managed.u_new), N*N*sizeof(gpu_fp)));
  CUDA_CHECK(cudaMallocManaged(&(gfs_managed.src), N*N*sizeof(gpu_fp)));
  CUDA_CHECK(cudaMallocManaged(&(gfs_managed.error), N*N*sizeof(gpu_fp)));

	//these are all used in the finite difference approxiamtion to solve the differential equation
  const gpu_fp dx = 1.0 / (N-1); //spacing between grid points in the x direction
  const gpu_fp dy = 1.0 / (N-1); //spacing in the y direction
  gfs_managed.n = N * N; //sets the grid points to N*N
	gfs_managed.nx = N;
	gfs_managed.ny = N;
  gfs_managed.dx2 = dx * dx;
  gfs_managed.idx2 = 1.0 / dx / dx; //sets the inverse of the squared spacing between grid points in the x-direction, which is also used in the finite difference approximations.
  gfs_managed.dy2 = dy * dy;
  gfs_managed.idy2 = 1.0 / dy / dy;

  // Working: initialize x and y arrays on the host 
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

  dim3 blockSize(16, 16);
  dim3 numBlocks((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
	// int numBlocks = 1;

  const double t_start = get_time();
   
  
  // printf("numBlocks = %d, blockSize = %d, totalThreads=%d, N=%d\n", numBlocks, blockSize, numBlocks * blockSize, N);
  
  int its = 0;
  for (;;)
  {
    for (int j = 0; j < 1000; j++) 
    {
      jacobi <<<numBlocks, blockSize>>> (gfs_managed);
			// Debug output ////////////////
			// for (int i = 0; i < N*N; i++)
			// {
			// 	printf("Unew set to be %20.16e\n", gfs_managed.u_new[i]); 
			// 	printf("at idx %d using src of  %20.16e\n", i, gfs_managed.src[i]); 
			// }

			for (int jj = 1; jj < N-1; jj++) {
				for (int ii = 1; ii < N-1; ii++) {
					int i = jj * N + ii;
					int ip1 = i + 1;
					int im1 = i - 1;	
					int jp1 = i + N;
					int jm1 = i - N;
					double manual_calc =  (0.5 * gfs_managed.idx2 * gfs_managed.idy2) * (gfs_managed.dy2 * (gfs_managed.u[ip1] + gfs_managed.u[im1]) + gfs_managed.dx2 * (gfs_managed.u[jp1] + gfs_managed.u[jm1]) - gfs_managed.dx2 * gfs_managed.dy2 * gfs_managed.src[i]);
					printf("Unew - manual = %20.16e - %20.16e = %20.16e at idx %d\n", gfs_managed.u_new[i], manual_calc, gfs_managed.u_new[i] - manual_calc, i); 
				}
			}

			///////////////////////////////
      copy <<<numBlocks, blockSize>>> (gfs_managed);
			// Debug output ////////////////
			for (int i = 0; i < N*N; i++)
			{
				printf("Unew - U = %20.16e - %20.16e = %20.16e at idx %d\n", gfs_managed.u_new[i], gfs_managed.u[i], gfs_managed.u_new[i] - gfs_managed.u[i], i); 
			}
			///////////////////////////////
    }
    error_check <<<numBlocks, blockSize>>> (gfs_managed);
		// // Debug output ////////////////
		// for (int i = 0; i < N*N; i++)
		// {
		// 	printf("Error set to be %20.16e\n", gfs_managed.error[i]); 
		// }
		// ///////////////////////////////
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
			const int idx = j * N + i;
			// printf("%20.16e %20.16e %20.16e\n", x, y, gfs_managed.u[idx]); TODO:reenable when ready
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
