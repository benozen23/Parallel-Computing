%%cu
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda.h"

int writeMandelbrot(const char *fileName, int width, int height, float *img, int minI, int maxI);

#define MXITER 1000

/*******************************************************************************/
// Define a complex number
typedef struct {
  double x;
  double y;
}complex_t;

/*******************************************************************************/
// perform Mandelbrot iteration on a grid of numbers in the complex plane
// record the  iteration counts in the count array
__global__ void mandelbrot(int Nre, int Nim, complex_t cmin, complex_t dc, float *count){
  int n = blockIdx.y * blockDim.y + threadIdx.y;//compute imaginary cuda point
  int m = blockIdx.x * blockDim.x + threadIdx.x;//compute real cuda point
  if (m < Nre && n < Nim){//check if boundary is reached.
    complex_t c;
    c.x = cmin.x + dc.x*m;
    c.y = cmin.y + dc.y*n;
    int iter;
    complex_t z = c;
    //Testpoint function
    for(iter=0; iter<MXITER; iter++){
      // real part of z^2 + c
      double tmp = (z.x*z.x) - (z.y*z.y) + c.x;
      // update with imaginary part of z^2 + c
      z.y = z.x*z.y*2. + c.y;
      // update real part
      z.x = tmp;
      // check bound
      if((z.x*z.x+z.y*z.y)>4.0){
        count[m+n*Nre] = iter;
        break;
      }
    }
    count[m+n*Nre] = iter;
  }
}

#/*******************************************************************************/
int main(int argc, char **argv){

  // to create a 4096x4096 pixel image
  // usage: ./mandelbrot 4096 4096

  int Nre = (argc==3) ? atoi(argv[1]): 8192;
  int Nim = (argc==3) ? atoi(argv[2]): 8192;
  int bsize = 32;
  int gsize = Nre/bsize;

  dim3 block_dim(bsize, bsize,1);
  dim3 grid_dim(gsize, gsize,1);
  // storage for the iteration counts
  //cpu
  float *count;
  count = (float*) malloc(Nre*Nim*sizeof(float));
  //gpu
  float *cuda_count;
  cudaMalloc((void**)&cuda_count, Nre * Nim * sizeof(float));

  // Parameters for a bounding box for "c" that generates an interesting image
  // const float centRe = -.759856, centIm= .125547;
  // const float diam  = 0.151579;
  //Default Settings
  //  const float centRe = -0.5, centIm= 0;
  //const float diam  = 3.0;
  const float centRe = -0.759856, centIm= 0.125547;
  const float diam  = 0.151579;

  complex_t cmin;
  complex_t cmax;
  complex_t dc;

  cmin.x = centRe - 0.5*diam;
  cmax.x = centRe + 0.5*diam;
  cmin.y = centIm - 0.5*diam;
  cmax.y = centIm + 0.5*diam;

  //set step sizes
  dc.x = (cmax.x-cmin.x)/(Nre-1);
  dc.y = (cmax.y-cmin.y)/(Nim-1);

  //start Cuda time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);

  // compute mandelbrot set
  mandelbrot<<<grid_dim, block_dim>>>(Nre, Nim, cmin, dc, cuda_count);

  // copy from the GPU back to the host header
  cudaMemcpy(count, cuda_count, Nre * Nim * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(cuda_count);

  //Finalize Cuda time
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);

  float time = 0;
  cudaEventElapsedTime(&time, start, stop);
  // print elapsed time
  printf("elapsed = %f\n", (double)(time/1000));

  // output mandelbrot to ppm format image
  printf("Printing mandelbrot.ppm...");
  writeMandelbrot("mandelbrot.ppm", Nre, Nim, count, 0, 80);
  printf("done.\n");

  free(count);

  exit(0);
  return 0;
}


/* Output data as PPM file */
void saveppm(const char *filename, unsigned char *img, int width, int height){

  #/* FILE pointer */
  FILE *f;

  #/* Open file for writing */
  f = fopen(filename, "wb");

  #/* PPM header info, including the size of the image */
  fprintf(f, "P6 %d %d %d\n", width, height, 255);

  #/* Write the image data to the file - remember 3 byte per pixel */
  fwrite(img, 3, width*height, f);

  #/* Make sure you close the file */
  fclose(f);
}



int writeMandelbrot(const char *fileName, int width, int height, float *img, int minI, int maxI){

  int n, m;
  unsigned char *rgb   = (unsigned char*) calloc(3*width*height, sizeof(unsigned char));

  for(n=0;n<height;++n){
    for(m=0;m<width;++m){
      int id = m+n*width;
      int I = (int) (768*sqrt((double)(img[id]-minI)/(maxI-minI)));

      // change this to change palette
      if(I<256)      rgb[3*id+2] = 255-I;
      else if(I<512) rgb[3*id+1] = 511-I;
      else if(I<768) rgb[3*id+0] = 767-I;
      else if(I<1024) rgb[3*id+0] = 1023-I;
      else if(I<1536) rgb[3*id+1] = 1535-I;
      else if(I<2048) rgb[3*id+2] = 2047-I;

    }
  }

  saveppm(fileName, rgb, width, height);

  free(rgb);
}