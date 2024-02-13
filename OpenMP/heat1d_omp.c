#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

#define OUT 1

// Function definitions
double boundary_condition(double x, double time);
double initial_condition(double x, double time);
double source(double x, double time);
void runSolver(int n, int num_threads);

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <N> <num_threads>\n", argv[0]);
        return 1;
    }

    int N = strtol(argv[1], NULL, 10); // Number of nodes
    int num_threads = strtol(argv[2], NULL, 10); // Number of OpenMP threads
    N = N * num_threads;
    double wtime = omp_get_wtime(); // Start timing
    runSolver(N, num_threads);
    wtime = omp_get_wtime() - wtime; // End timing

    printf("Execution time: %f seconds\n", wtime);
    return 0;
}

void runSolver(int n, int num_threads) {
    omp_set_num_threads(num_threads); // Set number of threads for parallel regions
  // CFL Condition is fixed
  double cfl = 0.5; 
  // Domain boundaries are fixed
  double x_min=0.0, x_max=1.0;
  // Diffusion coefficient is fixed
  double k   = 0.002;
  // Start time and end time are fixed
  double tstart = 0.0, tend = 10.0;  
  // Storage for node coordinates, solution field and next time level values
  double *x, *q, *qn;
  // Set the x coordinates of the n nodes padded with +2 ghost nodes. 
  x  = ( double*)malloc((n)*sizeof(double));
  q  = ( double*)malloc((n)*sizeof(double));
  qn = ( double*)malloc((n)*sizeof(double));
  // Write solution field to text file if size==1 only
  FILE *qfile, *xfile;
  double dx = ( x_max - x_min ) / ( double ) (n - 1 );
  double dt  = cfl*dx*dx/k; 
  int Nsteps = ceil(( tend - tstart )/dt);
  dt =  ( tend - tstart )/(( double )(Nsteps));     
  double time, time_new, wtime;  
  // find the coordinates for uniform spacing 
  for ( int i = 0; i < n; i++ ){
    // COMPLETE THIS PART
    x[i] = (double)(x_min + (dx*i));
  }
  // Set the values of q at the initial time.
  time = tstart; q[0] = 0.0; q[n+1] = 0.0;
  for (int i = 0; i < n; i++ ){
    q[i] = initial_condition(x[i],time);
  }
  // In single processor mode
  if (OUT==1){
    // write out the x coordinates for display.
    xfile = fopen ( "x_data.txt", "w" );
    for (int i = 0; i<n; i++ ){
      fprintf ( xfile, "  %f", x[i] );
    }
    fprintf ( xfile, "\n" );
    fclose ( xfile );
    // write out the initial solution for display.
    qfile = fopen ( "q_data.txt", "w" );
    for ( int i = 0; i < n; i++ ){
      fprintf ( qfile, "  %f", q[i] );
    }
    fprintf ( qfile, "\n" );
  }

    // Main computation loop
    for (int step = 1; step <= Nsteps; step++) {
        double time_new = time + step*dt; 
        #pragma omp parallel for
        // compute q values at all nodes except boundaries
        for (int i = 1; i < n-1; i++) {
            qn[i] = q[i] + dt*((k/(dx*dx)*(q[i-1]-(2*q[i])+q[i+1])) + source(x[i],time));
        }
        // update boundaries
        qn[0] = boundary_condition(x[0], time_new);
        qn[n-1] = boundary_condition(x[n-1], time_new);
        time = time_new;
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            q[i] = qn[i];
        }
        if (OUT==1){
            for ( int i = 0; i < n; i++ ){
                fprintf ( qfile, "  %f", q[i] );}
      fprintf ( qfile, "\n" );
    }
    }

    // Output results
  if(OUT==1)
    fclose ( qfile ); 

}
double boundary_condition ( double x, double time ){
  double value;

  // Left condition:
  if ( x < 0.5 ){
    value = 100.0 + 10.0 * sin ( time );
  }else{
    value = 75.0;
  }
  return value;
}
/*-----------------------------------------------------------*/
double initial_condition ( double x, double time ){
  double value;
  value = 95.0;

  return value;
}
/*-----------------------------------------------------------*/
double source ( double x, double time ){
  double value;

  value = 0.0;

  return value;
}
