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


typedef double gpu_fp;
struct GF
{
  int n;
  gpu_fp *u_old;
  gpu_fp *u;
  gpu_fp *k1;
  gpu_fp *k2;
  gpu_fp *k3;
  gpu_fp *k4;
  gpu_fp dx;
  gpu_fp dt;
};

// Kernel function to perform step 1 of RK4
__global__
void RK4_step1(struct GF gf)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  const int n = gf.n;
  
  for (int i = index; i < n; i += stride)
    gf.u_old[i] = gf.u[i];

  __syncthreads(); 
  for (int i = index; i < n; i += stride)
  {
    int ip1 = i % (n-2) + 1;
    int im1 = (i + n -4)  % (n-2) + 1;
    gf.k1[i] = 0.5 * (gf.u[ip1] - gf.u[im1]) / gf.dx;
    gf.u[i] = gf.u_old[i] + .5 * gf.dt * gf.k1[i];
  }
}


// Kernel function to perform step 2 of RK4
__global__
void RK4_step2(struct GF gf)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  const int n = gf.n;
  
  for (int i = index; i < n; i += stride)
  {
    int ip1 = i % (n-2) + 1;
    int im1 = (i + n -4)  % (n-2) + 1;
    gf.k2[i] = 0.5 * (gf.u[ip1] - gf.u[im1]) / gf.dx;
    gf.u[i] = gf.u_old[i] + .5 * gf.dt * gf.k2[i];
  }
}

// Kernel function to perform step 3 of RK4
__global__
void RK4_step3(struct GF gf)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  const int n = gf.n;
  
  for (int i = index; i < n; i += stride)
  {
    int ip1 = i % (n-2) + 1;
    int im1 = (i + n -4)  % (n-2) + 1;
    gf.k3[i] = 0.5 * (gf.u[ip1] - gf.u[im1]) / gf.dx;
    gf.u[i] = gf.u_old[i] +  gf.dt * gf.k3[i];
  }
}


// Kernel function to perform step 4 of RK4
__global__
void RK4_step4(struct GF gf)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  const int n = gf.n;
  
  for (int i = index; i < n; i += stride)
  {
    int ip1 = i % (n-2) + 1;
    int im1 = (i + n -4)  % (n-2) + 1;
    gf.k4[i] = 0.5 * (gf.u[ip1] - gf.u[im1]) / gf.dx;
    gf.u[i] = gf.u_old[i] + 1./6. * gf.dt * (gf.k1[i] + 2 * gf.k2[i] + 2 * gf.k3[i] + gf.k4[i]);
  }
}


int main(void)
{
  int N = 1<<20;

  const double pi = 3.1415926535897932385;
  struct GF gf_host;
  struct GF gf_dev;
  CUDA_ERROR(cudaMalloc(&gf_dev.u, N*sizeof(gpu_fp)));
  CUDA_ERROR(cudaMalloc(&gf_dev.u_old, N*sizeof(gpu_fp)));
  CUDA_ERROR(cudaMalloc(&gf_dev.k1, N*sizeof(gpu_fp)));
  CUDA_ERROR(cudaMalloc(&gf_dev.k2, N*sizeof(gpu_fp)));
  CUDA_ERROR(cudaMalloc(&gf_dev.k3, N*sizeof(gpu_fp)));
  CUDA_ERROR(cudaMalloc(&gf_dev.k4, N*sizeof(gpu_fp)));

  gf_host.u = (gpu_fp *) malloc(N*sizeof(gpu_fp));
  gf_host.u_old = (gpu_fp *) malloc(N*sizeof(gpu_fp));
  gf_host.k1 = (gpu_fp *) malloc(N*sizeof(gpu_fp));
  gf_host.k2 = (gpu_fp *) malloc(N*sizeof(gpu_fp));
  gf_host.k3 = (gpu_fp *) malloc(N*sizeof(gpu_fp));
  gf_host.k4 = (gpu_fp *) malloc(N*sizeof(gpu_fp));

  const double dx = 1.0 / (N-2);
  const double dt = dx / 2;
  gf_dev.n = N;
  gf_dev.dx = dx;
  gf_dev.dt = dt;

  gf_host.n = N;
  gf_host.dx = dx;
  gf_host.dt = dt;
  
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    const double x = -dx + i * dx;
    const double y = sin(x * pi);
    const double y2 = y * y;
    const double y4 = y2 * y2;
    const double y6 = y2 * y4;

    gf_host.u[i] = y6;
  }

  CUDA_ERROR(cudaMemcpy(gf_dev.u, gf_host.u, N*sizeof(gpu_fp), cudaMemcpyHostToDevice));
  int blockSize = 128;
  int numBlocks = (N + blockSize - 1) / blockSize;
  // Run kernel on 1M elements on the GPU
   
  for (int i = 0; i < 0.1  / dt ; i ++)
  {
    RK4_step1<<<numBlocks, blockSize>>> (gf_dev);
    RK4_step2<<<numBlocks, blockSize>>> (gf_dev);
    RK4_step3<<<numBlocks, blockSize>>> (gf_dev);
    RK4_step4<<<numBlocks, blockSize>>> (gf_dev);
  }


  // Wait for GPU to finish before accessing on host
  CUDA_ERROR(cudaDeviceSynchronize());
  CUDA_ERROR(cudaMemcpy(gf_host.u, gf_dev.u,  N*sizeof(gpu_fp), cudaMemcpyDeviceToHost));

  for (int i = 0; i < N; i++)
  {
    const gpu_fp x = -dx + i * dx;
    printf("%f %f\n", x, gf_host.u[i]);
  }

  // Free memory
  CUDA_ERROR(cudaFree(gf_dev.u));
  CUDA_ERROR(cudaFree(gf_dev.u_old));
  CUDA_ERROR(cudaFree(gf_dev.k1));
  CUDA_ERROR(cudaFree(gf_dev.k2));
  CUDA_ERROR(cudaFree(gf_dev.k3));
  CUDA_ERROR(cudaFree(gf_dev.k4));
  
  free(gf_host.u);
  free(gf_host.u_old);
  free(gf_host.k1);
  free(gf_host.k2);
  free(gf_host.k3);
  free(gf_host.k4);

  return 0;
}
