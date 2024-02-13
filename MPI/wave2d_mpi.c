# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <time.h>
# include <string.h>
# include "mpi.h"

#define BUFSIZE 512


// Function definitions
/******************************************************************************/
int main ( int argc, char *argv[] );
double exactSoln( double c, double x, double y, double t );
void applyBC(double *data,  double *x, double *y, double c, double time, int nx, int ny, int rank, int size);
void solverPlot(char *fileName, double *x, double *y, int nx, int ny, double *data); 
double readInputFile(char *fileName, char* tag); 

// Solver Info
/******************************************************************************
  Purpose:
    wave2d solves the wave equation in parallel using MPI.
  Discussion:
    Discretize the equation for u(x,t):
      d^2 u/dt^2  =  c^2 * (d^2 u/dx^2 + d^2 u/dy^2)  
      for 0 < x < 1, 0 < y < 1, t>0
    with boundary conditions and Initial conditions obtained from the exact solutions:
      u(x,y, t) = sin ( 2 * pi * ( x - c * t ) )
   Usage: serial -> ./wave input.dat  parallel> mpirun -np 4 ./wave input.dat 
   Compile: serial -> gcc -o wave serial.c -lm //  parallel -> mpicc -o wave wave2d_mpi.c -lm
******************************************************************************/

int main ( int argc, char *argv[] ){
  
  // Read input file for solution parameters
  double tstart = readInputFile(argv[1], "TSART"); // Start time
  double tend   = readInputFile(argv[1], "TEND");  // End time
  double dt     = readInputFile(argv[1], "DT");    // Time step size

  // Global node number in x and y
  int NX        = (int) readInputFile(argv[1], "NX"); // Global node numbers in x direction
  int NY        = (int) readInputFile(argv[1], "NY"); // Global node numbers in y direction

  double xmax = readInputFile(argv[1], "XMAX"); // domain boundaries
  double xmin = readInputFile(argv[1], "XMIN"); // domain boundaries
  double ymax = readInputFile(argv[1], "YMAX"); // domain boundaries
  double ymin = readInputFile(argv[1], "YMIN"); // domain boundaries
  double c = readInputFile(argv[1], "CONSTANT_C");

  double *qn, *q0, *q1;               // Solution field at t+dt, t and t-dt
  static int frame=0; 

  // Initialize MPI, get size and rank, record time
  int rank, size;
  MPI_Init ( &argc, &argv );
  MPI_Comm_rank ( MPI_COMM_WORLD, &rank );
  MPI_Comm_size ( MPI_COMM_WORLD, &size );
  MPI_Status status;
  double wtime = MPI_Wtime();

  // DOMAIN DECOMPOSITION
  // For serial implementation nx = NX and ny = NY; 
  // For MPI implementation nx = NX and ny = NY/rank
  int nx = NX;      // local number of nodes in x direction
  int ny = NY/size;      // local number of nodes in y direction
  // ALLOCATE MEMORY for COORDINATES (x, y) and compute them
    // Correct memory allocation since local number of nodes may be rounded to an integer.
  double *x = ( double * ) malloc ( nx*ny*sizeof ( double ) );
  double *y = ( double * ) malloc ( nx*ny*sizeof ( double ) );
  // find uniform spacing in x and y directions
  // Correct uniform y spacing for the local number of nodes as it may be rounded to an integer.
  double hx = (xmax - xmin)/(NX-1.0); 
  double hy = (ymax - ymin)/((ny*size)-1.0);
  // Compute coordinates of the nodes
  // Compute y coordinates according to ranks
  for(int j=0; j < ny; ++j){ 
    for(int i=0; i < nx;++i){
      // Every processors vertical size is (rank*hy*(ny-1)). This is the MPI correction for vertical position.
      double xn = xmin + i*hx; 
      double yn = ymin + (j*hy) + (rank*hy*ny);
      //Every processor contains nx*ny nodes. Multiply this value with rank to calculate the global coordinate.
      x[i+j*nx] = xn; 
      y[i+j*nx] = yn;

    }
  }


  // ALLOCATE MEMORY for SOLUTION and its HISTORY
  // Create ghost nodes at the lower and upper y axes for MPI communications.
  // Solution at time (t+dt)
  qn = ( double * ) malloc ( ((nx*ny)+(2*nx)) * sizeof ( double ) );
  // Solution at time (t)
  q0 = ( double * ) malloc ( ((nx*ny)+(2*nx)) * sizeof ( double ) );
  // Solution at time t-dt
  q1 = ( double * ) malloc ( ((nx*ny)+(2*nx)) * sizeof ( double ) );

  // USE EXACT SOLUTION TO FILL HISTORY
    for(int j=0; j < ny; ++j){ 
      for(int i=0; i < nx;++i){
        const double xn = x[i+j*nx]; 
        const double yn = y[i+j*nx]; 
        // Exact solutions at history tstart and tstart+dt
        q0[i+ (j*nx) + nx] = exactSoln(c, xn, yn, tstart + dt);  
        q1[i+ (j*nx) + nx] = exactSoln(c, xn, yn, tstart);
    }
  }

 
  // Write the initial solution 
    {
    char fname[BUFSIZ];
    sprintf(fname, "test_%04d_core%d_.csv", frame++,rank);
    solverPlot(fname, x, y, nx, ny, q1);
    }
// RUN SOLVER 
  int Noutput = 10000; 
  int Nsteps=(tend - tstart)/dt;     // Assume  dt divides (tend- tstart)
  double alphax2 = pow((c*dt/hx),2); 
  double alphay2 = pow((c*dt/hy),2);
  
  // We already have 2 steps computed with exact solution
  double time = dt; 
  // for every time step
  for(int tstep = 2; tstep<=Nsteps+1; ++tstep){
    // increase  time
    time = tstart + tstep*dt; 
    
    // Apply Boundary Conditions i.e. at i, j = 0, i,j = nx-1, ny-1
    applyBC(q0, x, y, c, time, nx, ny, rank, size); 
    int downcomms = 0;
    int uppercomms = 1;
    //Set up Communications
    if (rank > 0) {
        // Send the bottom row of the current domain to the upper ghost row of the previous rank
        MPI_Send(&q0[nx], nx, MPI_DOUBLE, rank - 1, downcomms, MPI_COMM_WORLD);

        // Receive the upper row of the previous domain at the lower ghost row of the working rank
        MPI_Recv(&q0[0], nx, MPI_DOUBLE, rank - 1, uppercomms, MPI_COMM_WORLD, &status);
    }

    if (rank < size - 1) {
        // Send the upper row of the current domain to the lower ghost row of the next rank
        MPI_Send(&q0[nx + (nx * (ny - 1))], nx, MPI_DOUBLE, rank + 1, uppercomms, MPI_COMM_WORLD);

        // Receive the bottom row of the next domain at the upper ghost row of the working rank
        MPI_Recv(&q0[nx + (nx * ny)], nx, MPI_DOUBLE, rank + 1, downcomms, MPI_COMM_WORLD, &status);
    }

    // Update solution using second order central differencing in time and space

    for(int j=0; j < ny; ++j){
      for(int i=1; i < nx-1;++i){// exclude left and right boundaries
        const int n0   = i + j*nx + nx; 
        const int nim1 = i - 1 + j*nx + nx; // node i-1,j
        const int nip1 = i + 1 + j*nx + nx; // node i+1,j
        const int njm1 = i + (j-1)*nx + nx; // node i, j-1
        const int njp1 = i + (j+1)*nx + nx; // node i, j+1
        // update solution
        //Check if lower boundary
        if(rank == 0){
          if (n0 >= 2*nx){
            qn[n0] = 2.0*q0[n0] - q1[n0] + alphax2*(q0[nip1]- 2.0*q0[n0] + q0[nim1])
                                        + alphay2*(q0[njp1] -2.0*q0[n0] + q0[njm1]); 
          }
        }
        //Check if upper boundary
        else if(rank == size-1){
          if (n0 < (nx + nx*(ny-1))){
            qn[n0] = 2.0*q0[n0] - q1[n0] + alphax2*(q0[nip1]- 2.0*q0[n0] + q0[nim1])
                                        + alphay2*(q0[njp1] -2.0*q0[n0] + q0[njm1]); 
          }
        }
        else{ 
            qn[n0] = 2.0*q0[n0] - q1[n0] + alphax2*(q0[nip1]- 2.0*q0[n0] + q0[nim1])
                                        + alphay2*(q0[njp1] -2.0*q0[n0] + q0[njm1]);
        } 
      }
    }

    // Update history q1 = q0; q0 = qn, except the boundaries
    for(int j=0; j < ny; ++j){ 
      for(int i=1; i < nx-1;++i){
        int n0   = i + j*nx + nx;
        //Check if lower boundary
        if(rank == 0){
          if (n0 >= 2*nx){
            q1[n0] = q0[n0]; 
            q0[n0] = qn[n0];
          }
        }
        //Check if upper boundary
        else if(rank == size-1){
          if (n0 < (nx + nx*(ny-1))){
            q1[n0] = q0[n0]; 
            q0[n0] = qn[n0];
          }
        }
        else{  
          q1[n0] = q0[n0]; 
          q0[n0] = qn[n0];
        }
        }
      }        
    // Dampout a csv file for postprocessing
    if(tstep%Noutput == 0){ 
    char fname[BUFSIZ];
    sprintf(fname, "test_%04d_core%d_.csv", frame++,rank);
    solverPlot(fname, x, y, nx, ny, q1);
    }
  }
  // Compute Linf norm of error at tend
    double linf = 0.0; 
    for(int j=0; j < ny; ++j){ 
      for(int i=0; i < nx;++i){
         double xn = x[i+ j*nx]; 
         double yn = y[i+ j*nx]; 
         // solution and the exact one
         double qn = q0[i + j*nx + nx]; 
         double qe = exactSoln(c, xn, yn, time);  
         linf  = fabs(qn-qe)>linf ? fabs(qn -qe):linf;
      }
    }
  // Max_Reduce the Infinity Error
  double global_linf = 0.0;
  MPI_Reduce(&linf, &global_linf, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  // Record the final time.
  wtime = MPI_Wtime( )-wtime;
  double global_time = 0.0; 
  MPI_Reduce( &wtime, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  //Print Elapsed Time and Infinity Error
  if(rank==0){
    printf ( "  Wall clock elapsed seconds = %f for %d number of processors.\n", global_time,size );
    printf("    Infinity norm of the error: %.4e %.8e \n", global_linf, time);
  }
  free(x);
  free(y);
  free(qn);
  free(q0);
  free(q1);

  //Terminate MPI   
  MPI_Finalize ( );   
  return 0;
}

/***************************************************************************************/
double exactSoln( double c, double x, double y, double t){
  const double pi = 3.141592653589793; 
  double value = sin( 2.0*pi*( x - c*t));
  return value;
}

/***************************************************************************************/
void applyBC(double *data,  double *x, double *y, double c, double time, int nx, int ny, int rank, int size){

  // Apply Boundary Conditions
  double xn, yn; 
  //Modfiy Boundary Condition loops according to domain decomposition and ranks.
  for(int j=0; j<ny;++j){ // left right boundaries i.e. i=0 and i=nx-1
    xn = x[0 + j*nx]; 
    yn = y[0 + j*nx];    
    data[j*nx + nx] = exactSoln(c, xn, yn, time); 

    xn = x[nx-1 + j*nx]; 
    yn = y[nx-1 + j*nx];    
    data[nx-1 + j*nx + nx] = exactSoln(c, xn, yn, time); 
  }

  for(int i=0; i< nx; ++i){ // top and  bottom boundaries i.e. j=0 and j=ny-1
    xn = x[i]; 
    yn = y[i];
    //Check if rank == 0 since it contains vertical down boundary.
    if(rank==0){
      data[i + nx] = exactSoln(c, xn, yn, time);
    }
    xn = x[i+ (ny-1)*nx]; 
    yn = y[i+ (ny-1)*nx];
    //Check if rank == size -1 since it contains vertical up boundary.
    if(rank==(size-1)){     
      data[i + nx + (ny-1)*nx] = exactSoln(c, xn, yn, time);
    }
  }
}

/* ************************************************************************** */
void solverPlot(char *fileName, double *x, double *y, int nx, int ny, double *Q){
    FILE *fp = fopen(fileName, "w");
    if (fp == NULL) {
        printf("Error opening file\n");
        return;
    }

    fprintf(fp, "X,Y,Z,Q \n");
    for(int j=0; j < ny; ++j){ 
      for(int i=0; i < nx;++i){
        const double xn = x[i+j*nx]; 
        const double yn = y[i+j*nx];
        fprintf(fp, "%.8f, %.8f,%.8f,%.8f\n", xn, yn, 0.0, Q[i + j*nx + nx]);
      }
    }
}


/* ************************************************************************** */
double readInputFile(char *fileName, char* tag){
  FILE *fp = fopen(fileName, "r");
  if (fp == NULL) {
    printf("Error opening the input file\n");
    return -1;
  }

  int sk = 0; 
  double result; 
  char buffer[BUFSIZE];
  char fileTag[BUFSIZE]; 
  while(fgets(buffer, BUFSIZE, fp) != NULL){
    sscanf(buffer, "%s", fileTag);
    if(strstr(fileTag, tag)){
      fgets(buffer, BUFSIZE, fp);
      sscanf(buffer, "%lf", &result); 
      return result;
    }
    sk++;
  }

  if(sk==0){
    printf("could not find the tag: %s in the file %s\n", tag, fileName);
  }
}
