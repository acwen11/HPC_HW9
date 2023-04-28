#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h> // needed for timing

#define TOL 1.0e-9

/*  A simple test code for a 2D Gauss-Seidel solve of
 *  U_xx + U_yy = exp(-((x-0.5)**2 + (y-0.5)**2) * 50)
 *  U(0,y) = U(1,y) = U(x,0) = U(x,1) = 0
 *
 *  Input: number of cells per direction
 */

void output_gf(const int nx, const int ny, const double x0, const double y0,
               const double dx, const double dy, const double *U);
void initialize(const int nx, const int ny, const double x0, const double y0,
                const double dx, const double dy, double *src, double *U);
void gauss_seidel(const int nx, const int ny, const double x0, const double y0,
                  const double dx, const double dy, const double *src,
                  double *U, const double omega);
double get_error(const int nx, const int ny, const double x0, const double y0,
                 const double dx, const double dy, const double *src,
                 const double *U);
double get_time(void);

int main(int argc, char **argv)
{

  if (argc != 2)
  {
    fprintf(stderr,
            "Usage: %s NXY\nNXY is the number of points per direction (total "
            "is NXY * NXY\n",
            argv[0]);
    exit(-1);
  }

  const int pnxy = atoi(argv[1]);
  if (pnxy < 3)
  {
    fprintf(stderr, "NXY must be a postive integer >= 3. Found %s instead\n ",
            argv[1]);
    exit(-1);
  }

  /* The spatial grid maps to [0, 1]x[0,1].
   */
  const double nx = pnxy;
  const double ny = pnxy;
  const double x0 = 0;
  const double y0 = 0;
  const double xmax = 1;
  const double ymax = 1;
  const double dx = (xmax - x0) / (nx - 1);
  const double dy = (ymax - y0) / (ny - 1);
  const double omega = 1; /* SOL parameters */


  double *source = calloc(nx * ny, sizeof(double));
  double *solution = calloc(nx * ny, sizeof(double));

  initialize(nx, ny, x0, y0, dx, dy, source, solution);

  double error = 1;

  int total_iterations = 0;

  const double t_start = get_time();
  while (error > TOL)
  {
    for (int its = 0; its < 100; its++)
    {
      gauss_seidel(nx, ny, x0, y0, dx, dy, source, solution, omega);
      total_iterations++;
    }

    error = get_error(nx, ny, x0, y0, dx, dy, source, solution);
    printf("%d %e\n", total_iterations, error);
  }
  const double t_end = get_time();

  printf("final error %e, total iterations %d, total time %.2f s\n", error,
         total_iterations, t_end - t_start);
  output_gf(nx, ny, x0, y0, dx, dy, solution);

  return 0;
}

/*
 * output_gf
 * Generate an ascii file containing the value of the solution vector
 * at each gridpoint. This can be used for visualization.
 *
 * nx (int, input): the number of gridpoints in the x direction. 
 * ny (int, input): the number of gridpoints in the y direction. 
 * x0 (double, input): x coordinate of the grid origin.
 * y0 (double, input): y coordinate of the grid origin.
 * dx (double, input): grid spacing in x direction.
 * dy (double, input): grid spacing in y direction.
 * U (*double, input): point to 2D data to output
 *
 * The output filename will be of the form 00..X.asc', where the
 * number X is incremented each time the function is called.
 *
 * Returns: Nothing
 */
void output_gf(const int nx, const int ny, const double x0, const double y0,
               const double dx, const double dy, const double *U)
{
  static int counter = 0;
  char name_buff[1024];
  snprintf(name_buff, 1024, "%07d.asc", counter);
  FILE *ofile = fopen(name_buff, "w");
  for (int j = 0; j < ny; ++j)
  {
    for (int i = 0; i < nx; ++i)
    {
      const int ij = i + j * nx;
      fprintf(ofile, "%20.16e %20.16e %20.16e\n", x0 + i * dx, y0 + j * dy,
              U[ij]);
    }
  //  fprintf(ofile, "\n");
  }
  ++counter;

  fclose(ofile);
  ofile = NULL;
}

// get_time will return a double containing the current time in
// seconds.
double get_time(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);

  return (tv.tv_sec) + 1.0e-6 * tv.tv_usec;
}

/*
 * initialize
 * Set up both the RHS gridfunction and initialize the solution
 * gridfunction.
 *
 * nx (int, input): the number of gridpoints in the x direction. 
 * ny (int, input): the number of gridpoints in the y direction. 
 * x0 (double, input): x coordinate of the grid origin.
 * y0 (double, input): y coordinate of the grid origin.
 * dx (double, input): grid spacing in x direction.
 * dy (double, input): grid spacing in y direction.
 * src (*double, output): pointer to the src array.
 * U (*double, output): pointer to the solution array.
 *
 *
 * Returns: Nothing
 */
void initialize(const int nx, const int ny, const double x0, const double y0,
                const double dx, const double dy, double *src, double *U)
{
  for (int j = 0; j < ny; j++)
  {
    const double y = y0 + j * dy - .5;
    for (int i = 0; i < nx; i++)
    {
      const int ij_index = i + j * nx;
      const double x = x0 + i * dx - .5;
      U[ij_index] = 0.0;
      src[ij_index] = exp(-(x * x + y * y) * 50);
    }
  }
}

/*
 * gauss_seidel
 * Perform 1 iteration of Gauss-Seidel relaxation
 * to solve D^2 U  - SRC = 0,
 * where R is the residual, SRC is the SRC gridfunction and D^2 is
 * the finite-difference equivalent to the Laplacian of U.
 *
 * nx (int, input): the number of gridpoints in the x direction. 
 * ny (int, input): the number of gridpoints in the y direction. 
 * x0 (double, input): x coordinate of the grid origin.
 * y0 (double, input): y coordinate of the grid origin.
 * dx (double, input): grid spacing in x direction.
 * dy (double, input): grid spacing in y direction.
 * src (*double, input): pointer to the src array.
 * U (*double, input/output): pointer to the solution array.
 * omega (double, input): SOL parameters should be in [0,1)
 *
 *
 * Returns: Nothing
 */
void gauss_seidel(const int nx, const int ny, const double x0, const double y0,
                  const double dx, const double dy, const double *src,
                  double *U, const double omega)
{
  const double dx2 = dx * dx;
  const double dy2 = dy * dy;

  const double idenom = 1.0 / (2 * (dx2 + dy2));

//uncomment one! of these to do proper parallelization
//collapse(2) is only for 2 loops, if there are 3, change to collapse(3)
//#pragma omp parallel for collapse(2) 
#pragma omp parallel for  
  for (int j = 1; j < ny - 1; j++)
  {
//#pragma omp parallel for
    for (int i = 1; i < nx - 1; i++)
    {
      const int ij = i + j * nx;
      const int ip1j = i + 1 + j * nx;
      const int im1j = i - 1 + j * nx;
      const int ijp1 = i + (j + 1) * nx;
      const int ijm1 = i + (j - 1) * nx;

      const double U_current = U[ij];
      const double U_new = (dy2 * (U[ip1j] + U[im1j]) +
                            dx2 * (U[ijp1] + U[ijm1]) - src[ij] * (dx2 * dy2)) *
                           idenom;
      U[ij] = (1 - omega) * U_current + omega * U_new;
    }
  }
}


/*
 * get_error
 * Calculate the L2 norm of the residual
 * D^2 U  - SRC = R,
 * where R is the residual, SRC is the SRC gridfunction and D^2 is
 * the finite-difference equivalent to the Laplacian of U.
 *
 * nx (int, input): the number of gridpoints in the x direction. 
 * ny (int, input): the number of gridpoints in the y direction. 
 * x0 (double, input): x coordinate of the grid origin.
 * y0 (double, input): y coordinate of the grid origin.
 * dx (double, input): grid spacing in x direction.
 * dy (double, input): grid spacing in y direction.
 * src (*double, input): pointer to source gridfunction.
 * U (*double, input): pointer to solution gridfunction
 *
 * Returns: (double) L2 norm of the error.
 */
double get_error(const int nx, const int ny, const double x0, const double y0,
                 const double dx, const double dy, const double *src,
                 const double *U)
{
  const double idx2 = 1 / (dx * dx);
  const double idy2 = 1 / (dy * dy);

  double error = 0;

//uncomment one! of these to do proper parallelization
//collapse(2) is only for 2 loops, if there are 3, change to collapse(3)
//#pragma omp parallel for collapse(2) 
#pragma omp parallel for reduction (+:error)
  for (int j = 1; j < ny - 1; j++)
  {
//#pragma omp parallel for reduction (+:error)
    for (int i = 1; i < nx - 1; i++)
    {
      const int ij = i + j * nx;
      const int ip1j = i + 1 + j * nx;
      const int im1j = i - 1 + j * nx;
      const int ijp1 = i + (j + 1) * nx;
      const int ijm1 = i + (j - 1) * nx;

      const double U_xx = (U[ip1j] + U[im1j] - 2 * U[ij]) * idx2;
      const double U_yy = (U[ijp1] + U[ijm1] - 2 * U[ij]) * idy2;
      const double lerror = U_xx + U_yy - src[ij];
      error += lerror * lerror;
    }
  }
  return sqrt(error / (nx + ny - 4));
}
