#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>  // needed for timing

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


#define CUDA_ERROR(t_)                                            \
  do {                                                            \
       const long int  _err = t_;                                 \
       if (_err != cudaSuccess  )                                 \
       {                                                          \
         fprintf(stderr, "cuda function call %s failed\n", #t_);  \
         exit(-1);                                                \
       }                                                          \
    } while(0)

static void test_function(const gpu_fp x, const gpu_fp t,  gpu_fp *U, gpu_fp *K);
static gpu_fp l_inf_error(const gpu_fp t, struct GFS *gfs);

struct GF
{
  gpu_fp *old;
  gpu_fp *val;
  gpu_fp *k[5];
};

struct GFS
{
  int nvars;
  int n;
  gpu_fp dx;
  gpu_fp dt;
  struct GF var[2];
};


// Wave equation RHS. Periodic boundary conditions are assumed.
// Only works with 1 ghost zone.
__device__
 void dot(struct GFS gfs, int k)
{
  // gfs : structure containing pointers to grid function
  // k : step of RK4 (i.e., fill in k1, k2, k3, or k4 of RK4)

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  const int n = gfs.n;
  const gpu_fp dx2 = gfs.dx * gfs.dx;
  for (int i = index; i < n; i += stride)
  {
    int ip1 = i % (n-2) + 1;
    int im1 = (i + n -4)  % (n-2) + 1;
    gfs.var[0].k[k][i] = gfs.var[1].val[i];
    gfs.var[1].k[k][i] = (gfs.var[0].val[ip1] + gfs.var[0].val[im1] - 2.0 * gfs.var[0].val[i]) / dx2;
  }
}

__device__
 void intermediate_update_vars (struct GFS gfs, int k, gpu_fp factor)
{
  // gfs : structure containing pointers to grid function
  // k : step of RK4 (i.e., fill in k1, k2, k3, or k4 of RK4)
  // factor : fraction of dt used in the update. 

   int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  const int n = gfs.n;
  
  for (int v = 0; v < gfs.nvars; v++)
  {
    for (int i = index; i < n; i += stride)
    {
      gfs.var[v].val[i] = gfs.var[v].old[i] + factor * gfs.dt * gfs.var[v].k[k][i];
    }
  }
}

__device__
 void final_update_vars (struct GFS gfs)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  const int n = gfs.n;

  for (int v = 0; v < gfs.nvars; v++)
  {
    for (int i = index; i < n; i += stride)
    {
      gfs.var[v].val[i] = gfs.var[v].old[i] +
                 gfs.dt *
                   ( 
                            gfs.var[v].k[1][i] 
                      + 2 * gfs.var[v].k[2][i]
                      + 2 * gfs.var[v].k[3][i]
                      +     gfs.var[v].k[4][i]
                    ) / 6.0;
    }
  }
}

__device__
 void cycle_timelevels (struct GFS gfs)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  const int n = gfs.n;
  const gpu_fp dx2 = gfs.dx * gfs.dx;
  
  for (int v =0; v < gfs.nvars; v++)
  {
    for (int i = index; i < n; i += stride)
    { 
      gfs.var[v].old[i] = gfs.var[v].val[i];
    }
  }
}


// Kernel function to perform step 1 of RK4
__global__
void RK4_step1(struct GFS gfs)
{
  cycle_timelevels(gfs);
  dot(gfs, 1);
  intermediate_update_vars(gfs, 1, 0.5);
}


// Kernel function to perform step 2 of RK4
__global__
void RK4_step2(struct GFS gfs)
{
  dot(gfs, 2);
  intermediate_update_vars(gfs, 2, 0.5);
}

// Kernel function to perform step 3 of RK4
__global__
void RK4_step3(struct GFS gfs)
{
  dot(gfs, 3);
  intermediate_update_vars(gfs, 3, 1.0);
}

// Kernel function to perform step 4 of RK4
__global__
void RK4_step4(struct GFS gfs)
{
  dot(gfs, 4);
  final_update_vars(gfs);
}

static void test_function(const gpu_fp x, const gpu_fp t,  gpu_fp *U, gpu_fp *K)
{
  const gpu_fp pi = 3.1415926535897932385;
  const gpu_fp s = sin((x-t) * pi);
  const gpu_fp c = cos((x-t) * pi);
  const gpu_fp s2 = s * s;
  const gpu_fp s4 = s2 * s2;
  const gpu_fp s5 = s4 * s;
  const gpu_fp s6 = s2 * s4;

  *U = s6;
  *K = -6 * pi * s5 * c;
}

static gpu_fp l_inf_error(const gpu_fp t, struct GFS *gfs)
{
  gpu_fp error = 0;
  for (int i = 0; i < gfs->n; i++) {
    const gpu_fp x = -gfs->dx + i * gfs->dx;
    gpu_fp U;
    gpu_fp K;
    test_function(x, t, &U, &K);
    const gpu_fp lerror = fabs(U - gfs->var[0].val[i]);
    if (lerror > error)
    {
      error = lerror;
    }
  }
  return error;
}

int main(int argc, char **argv)
{
  const int N = 2<<10;

  const int nvars = 2;
  struct GFS gfs_host;
  struct GFS gfs_dev;

 // const gpu_fp t_final = atof(argv[1]);
  gfs_host.nvars = nvars;
  gfs_dev.nvars = nvars;

  for (int i = 0; i < nvars; i++)
  {
    CUDA_ERROR(cudaMalloc(&(gfs_dev.var[i].val), N*sizeof(gpu_fp)));
    CUDA_ERROR(cudaMalloc(&(gfs_dev.var[i].old), N*sizeof(gpu_fp)));
    CUDA_ERROR(cudaMalloc(&(gfs_dev.var[i].k[1]), N*sizeof(gpu_fp)));
    CUDA_ERROR(cudaMalloc(&(gfs_dev.var[i].k[2]), N*sizeof(gpu_fp)));
    CUDA_ERROR(cudaMalloc(&(gfs_dev.var[i].k[3]), N*sizeof(gpu_fp)));
    CUDA_ERROR(cudaMalloc(&(gfs_dev.var[i].k[4]), N*sizeof(gpu_fp)));

    gfs_host.var[i].val = (gpu_fp *) malloc(N*sizeof(gpu_fp));
    gfs_host.var[i].old = (gpu_fp *) malloc(N*sizeof(gpu_fp));
    gfs_host.var[i].k[1] = (gpu_fp *) malloc(N*sizeof(gpu_fp));
    gfs_host.var[i].k[2] = (gpu_fp *) malloc(N*sizeof(gpu_fp));
    gfs_host.var[i].k[3] = (gpu_fp *) malloc(N*sizeof(gpu_fp));
    gfs_host.var[i].k[4] = (gpu_fp *) malloc(N*sizeof(gpu_fp));
  }

  const gpu_fp dx = 1.0 / (N-2);
  const gpu_fp dt = dx / 2;
  gfs_dev.n = N;
  gfs_dev.dx = dx;
  gfs_dev.dt = dt;

  gfs_host.n = N;
  gfs_host.dx = dx;
  gfs_host.dt = dt;
  
  gpu_fp t = 0;
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    const gpu_fp x = -dx + i * dx;
    test_function(x, t, gfs_host.var[0].val + i,  gfs_host.var[1].val + i);
  }

  for (int i = 0; i < nvars; i++)
  {
    CUDA_ERROR(cudaMemcpy(gfs_dev.var[i].val, gfs_host.var[i].val, N*sizeof(gpu_fp), cudaMemcpyHostToDevice));
  }

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  const double t_start = get_time();
   
  int its = 0;
  for (; t < 1.0 ;)
  {
    RK4_step1<<<numBlocks, blockSize>>> (gfs_dev);
    RK4_step2<<<numBlocks, blockSize>>> (gfs_dev);
    RK4_step3<<<numBlocks, blockSize>>> (gfs_dev);
    RK4_step4<<<numBlocks, blockSize>>> (gfs_dev);
    t += gfs_host.dt;
    its ++;
  }

  const double t_end = get_time();

  // Wait for GPU to finish before accessing on host
  CUDA_ERROR(cudaDeviceSynchronize());


   
  for (int i = 0; i < nvars; i++)
  {
    CUDA_ERROR(cudaMemcpy(gfs_host.var[i].val, gfs_dev.var[i].val,  N*sizeof(gpu_fp), cudaMemcpyDeviceToHost));
  }

  const gpu_fp error = l_inf_error(t, &gfs_host);

  printf("%d iterations of grid size %d took %fs with error %20.16e\n",its, N,  t_end - t_start, (double) error);

//  for (int i = 0; i < N; i++)
//  {
//    const gpu_fp x = -dx + i * dx;
//    printf("%20.16e %20.16e\n", x, gfs_host.var[0].val[i]);
//  }

  // Free memory
  for (int i = 0; i < nvars; i++)
  {
    CUDA_ERROR(cudaFree(gfs_dev.var[i].val));
    CUDA_ERROR(cudaFree(gfs_dev.var[i].old));
    CUDA_ERROR(cudaFree(gfs_dev.var[i].k[1]));
    CUDA_ERROR(cudaFree(gfs_dev.var[i].k[2]));
    CUDA_ERROR(cudaFree(gfs_dev.var[i].k[3]));
    CUDA_ERROR(cudaFree(gfs_dev.var[i].k[4]));

    free(gfs_host.var[i].val);
    free(gfs_host.var[i].old);
    free(gfs_host.var[i].k[1]);
    free(gfs_host.var[i].k[2]);
    free(gfs_host.var[i].k[3]);
    free(gfs_host.var[i].k[4]);
  }
  
  return 0;
}
