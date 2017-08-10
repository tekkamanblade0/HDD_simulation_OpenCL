#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <assert.h>
#include <iomanip>

using namespace std;
#include "clFFT.h"
#include "Parameters.h"

#define   d    delta_x

int FFT_3D_OpenCL(cl_mem buffersIn[], //cl_mem buffersOut[], 
				const char* direction, int sizex, int sizey, int sizez,
				cl_context &ctx, cl_command_queue &queue)
{
  int i;

  /* OpenCL variables. */
  cl_int err;
//   cl_platform_id platform = 0;
//   cl_device_id device = 0;
//   cl_context ctx = 0;
//   cl_command_queue queue = 0;

  /* Input and Output  buffer. */
//   cl_mem buffersIn[2]  = {0, 0};
//   cl_mem buffersOut[2] = {0, 0};

  /* Temporary buffer. */
  cl_mem tmpBuffer = 0;

  /* Size of temp buffer. */
  size_t tmpBufferSize = 0;
  int status = 0;
  int ret = 0;

  /* Total size of FFT. */
  size_t N = sizex*sizey*sizez;

  /* FFT library realted declarations. */
  clfftPlanHandle planHandle;
  clfftDim dim = CLFFT_3D;
  size_t clLengths[3] = {sizex, sizey, sizez};

  /* Setup OpenCL environment. */
//   err = clGetPlatformIDs(1, &platform, NULL);
//   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

//   /* Create an OpenCL context. */
//   ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

//   /* Create a command queue. */
//   queue = clCreateCommandQueue(ctx, device, 0, &err);

  /* Setup clFFT. */
  clfftSetupData fftSetup;
  err = clfftInitSetupData(&fftSetup);
  err = clfftSetup(&fftSetup);

  /* Create a default plan for a complex FFT. */
  err = clfftCreateDefaultPlan(&planHandle, ctx, dim, clLengths);

  /* Set plan parameters. */
  err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
  err = clfftSetLayout(planHandle, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR);
  //err = clfftSetResultLocation(planHandle, CLFFT_OUTOFPLACE);
  err = clfftSetResultLocation(planHandle, CLFFT_INPLACE);

  /* Bake the plan. */
  err = clfftBakePlan(planHandle, 1, &queue, NULL, NULL);

  /* Real and Imaginary arrays. */
//   cl_float* inReal  = (cl_float*) malloc (N * sizeof (cl_float));
//   cl_float* inImag  = (cl_float*) malloc (N * sizeof (cl_float));
//   cl_float* outReal = (cl_float*) malloc (N * sizeof (cl_float));
//   cl_float* outImag = (cl_float*) malloc (N * sizeof (cl_float));

  /* Initialization of inReal, inImag, outReal and outImag. */
//   for(i=0; i<N; i++)
//   {
//     inReal[i]  = tab[0][i];
//     inImag[i]  = 0.0f;
//     outReal[i] = 0.0f;
//     outImag[i] = 0.0f;
//   }

  /* Create temporary buffer. */
  status = clfftGetTmpBufSize(planHandle, &tmpBufferSize);

  if ((status == 0) && (tmpBufferSize > 0)) {
    tmpBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE, tmpBufferSize, 0, &err);
    if (err != CL_SUCCESS)
      printf("Error with tmpBuffer clCreateBuffer\n");
  }

  /* Prepare OpenCL memory objects : create buffer for input. */
//   buffersIn[0] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
//       N * sizeof(cl_float), inReal, &err);
//   if (err != CL_SUCCESS)
//     printf("Error with buffersIn[0] clCreateBuffer\n");

  /* Enqueue write tab array into buffersIn[0]. */
//   err = clEnqueueWriteBuffer(queue, buffersIn[0], CL_TRUE, 0, N *
//       sizeof(float),
//       inReal, 0, NULL, NULL);
//   if (err != CL_SUCCESS)
//     printf("Error with buffersIn[0] clEnqueueWriteBuffer\n");

  /* Prepare OpenCL memory objects : create buffer for input. */
//   buffersIn[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
//       N * sizeof(cl_float), inImag, &err);
//   if (err != CL_SUCCESS)
//     printf("Error with buffersIn[1] clCreateBuffer\n");

  /* Enqueue write tab array into buffersIn[1]. */
//   err = clEnqueueWriteBuffer(queue, buffersIn[1], CL_TRUE, 0, N * sizeof(float),
//       inImag, 0, NULL, NULL);
//   if (err != CL_SUCCESS)
//     printf("Error with buffersIn[1] clEnqueueWriteBuffer\n");

  /* Prepare OpenCL memory objects : create buffer for output. */
//   buffersOut[0] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N *
//       sizeof(cl_float), outReal, &err);
//   if (err != CL_SUCCESS)
//     printf("Error with buffersOut[0] clCreateBuffer\n");

//   buffersOut[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N *
//       sizeof(cl_float), outImag, &err);
//   if (err != CL_SUCCESS)
//     printf("Error with buffersOut[1] clCreateBuffer\n");

  /* Execute Forward or Backward FFT. */
  if(strcmp(direction,"forward") == 0)
  {
    /* Execute the plan. */
    err = clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &queue, 0, NULL, NULL,
        buffersIn, NULL, tmpBuffer);
  }
  else if(strcmp(direction,"backward") == 0)
  {
    /* Execute the plan. */
    err = clfftEnqueueTransform(planHandle, CLFFT_BACKWARD, 1, &queue, 0, NULL, NULL,
        buffersIn, NULL, tmpBuffer);
  }

  /* Wait for calculations to be finished. */
  err = clFinish(queue);

  /* Fetch results of calculations : Real and Imaginary. */
//   err = clEnqueueReadBuffer(queue, buffersOut[0], CL_TRUE, 0, N * sizeof(float), tab[0],
//       0, NULL, NULL);
//   err = clEnqueueReadBuffer(queue, buffersOut[1], CL_TRUE, 0, N * sizeof(float), tab[1],
//       0, NULL, NULL);

  /* Release OpenCL memory objects. */
//   clReleaseMemObject(buffersIn[0]);
//   clReleaseMemObject(buffersIn[1]);
//   clReleaseMemObject(buffersOut[0]);
//   clReleaseMemObject(buffersOut[1]);
  clReleaseMemObject(tmpBuffer);

  /* Release the plan. */
  err = clfftDestroyPlan(&planHandle);

  /* Release clFFT library. */
  clfftTeardown();

  /* Release OpenCL working objects. */
//   clReleaseCommandQueue(queue);
//   clReleaseContext(ctx);

  return ret;
}


static bool G_matrix(int lx_zero_pad, int ly_zero_pad, int lz_zero_pad, cl_command_queue &queue)
{
	int idx;
	double x, y, z;
	double xmd, xpd, ymd, ypd, zmd, zpd; // xmd: x minus dx/2; xpd: x plus dx/2
	cl_int err=0;

	Gxx_1d_real  = (double*) malloc(lx_zero_pad*ly_zero_pad*lz_zero_pad*sizeof(double));
	Gxy_1d_real  = (double*) malloc(lx_zero_pad*ly_zero_pad*lz_zero_pad*sizeof(double));
	Gxz_1d_real  = (double*) malloc(lx_zero_pad*ly_zero_pad*lz_zero_pad*sizeof(double));
	Gyx_1d_real  = (double*) malloc(lx_zero_pad*ly_zero_pad*lz_zero_pad*sizeof(double));
	Gyy_1d_real  = (double*) malloc(lx_zero_pad*ly_zero_pad*lz_zero_pad*sizeof(double));
	Gyz_1d_real  = (double*) malloc(lx_zero_pad*ly_zero_pad*lz_zero_pad*sizeof(double));
	Gzx_1d_real  = (double*) malloc(lx_zero_pad*ly_zero_pad*lz_zero_pad*sizeof(double));
	Gzy_1d_real  = (double*) malloc(lx_zero_pad*ly_zero_pad*lz_zero_pad*sizeof(double));
	Gzz_1d_real  = (double*) malloc(lx_zero_pad*ly_zero_pad*lz_zero_pad*sizeof(double));

	Gxx_1d_imag  = (double*) malloc(lx_zero_pad*ly_zero_pad*lz_zero_pad*sizeof(double));
	Gxy_1d_imag  = (double*) malloc(lx_zero_pad*ly_zero_pad*lz_zero_pad*sizeof(double));
	Gxz_1d_imag  = (double*) malloc(lx_zero_pad*ly_zero_pad*lz_zero_pad*sizeof(double));
	Gyx_1d_imag  = (double*) malloc(lx_zero_pad*ly_zero_pad*lz_zero_pad*sizeof(double));
	Gyy_1d_imag  = (double*) malloc(lx_zero_pad*ly_zero_pad*lz_zero_pad*sizeof(double));
	Gyz_1d_imag  = (double*) malloc(lx_zero_pad*ly_zero_pad*lz_zero_pad*sizeof(double));
	Gzx_1d_imag  = (double*) malloc(lx_zero_pad*ly_zero_pad*lz_zero_pad*sizeof(double));
	Gzy_1d_imag  = (double*) malloc(lx_zero_pad*ly_zero_pad*lz_zero_pad*sizeof(double));
	Gzz_1d_imag  = (double*) malloc(lx_zero_pad*ly_zero_pad*lz_zero_pad*sizeof(double));

	
	for (int k = 0; k < 2*Nz-1; k++){
		for (int j = 0; j < 2*Ny-1; j++){
			for (int i = 0; i < 2*Nx-1; i++){
				x = ((i + 1) - Nx) * d;
				y = ((j + 1) - Ny) * d;
				z = ((k + 1) - Nz) * d;

				xmd = pow(x-d/2, 2);
				xpd = pow(x+d/2, 2);
				ymd = pow(y-d/2, 2);
				ypd = pow(y+d/2, 2);
				zmd = pow(z-d/2, 2);
				zpd = pow(z+d/2, 2);
				
				idx = i + j*(lx_zero_pad) + k*(lx_zero_pad)*(ly_zero_pad);				
				Gxx_1d_real[idx] =  ((atan( (y-d/2) * (z-d/2)/((x-d/2) * pow(xmd + ymd + zmd, 0.5)) ) + 
									   atan( (y+d/2) * (z+d/2)/((x-d/2) * pow(xmd + ypd + zpd, 0.5)) )) - 
					                  (atan( (y-d/2) * (z+d/2)/((x-d/2) * pow(xmd + ymd + zpd, 0.5)) ) + 
									   atan( (y+d/2) * (z-d/2)/((x-d/2) * pow(xmd + ypd + zmd, 0.5)) ))) - 
							         ((atan( (y-d/2) * (z-d/2)/((x+d/2) * pow(xpd + ymd + zmd, 0.5)) ) + 
					                   atan( (y+d/2) * (z+d/2)/((x+d/2) * pow(xpd + ypd + zpd, 0.5)) )) - 
					                  (atan( (y-d/2) * (z+d/2)/((x+d/2) * pow(xpd + ymd + zpd, 0.5)) ) + 
					                   atan( (y+d/2) * (z-d/2)/((x+d/2) * pow(xpd + ypd + zmd, 0.5)) )));
				Gxy_1d_real[idx] =  ( log( 4 * (-(z-d/2) + pow(xmd + ymd + zmd, 0.5) ) * (-(z+d/2) + pow(xpd + ymd + zpd, 0.5) )) - 
					                   log( 4 * (-(z-d/2) + pow(xpd + ymd + zmd, 0.5) ) * (-(z+d/2) + pow(xmd + ymd + zpd, 0.5) ))) - 
					                 ( log( 4 * (-(z-d/2) + pow(xmd + ypd + zmd, 0.5) ) * (-(z+d/2) + pow(xpd + ypd + zpd, 0.5) )) - 
					                   log( 4 * (-(z-d/2) + pow(xpd + ypd + zmd, 0.5) ) * (-(z+d/2) + pow(xmd + ypd + zpd, 0.5) )));
				
				Gxz_1d_real[idx] = ( log( 4 * (-(y-d/2) + pow(xmd + ymd + zmd, 0.5) ) * (-(y+d/2) + pow(xpd + ypd + zmd, 0.5) )) - 
					                  log( 4 * (-(y-d/2) + pow(xpd + ymd + zmd, 0.5) ) * (-(y+d/2) + pow(xmd + ypd + zmd, 0.5) ))) - 
					                ( log( 4 * (-(y-d/2) + pow(xmd + ymd + zpd, 0.5) ) * (-(y+d/2) + pow(xpd + ypd + zpd, 0.5) )) - 
					                  log( 4 * (-(y-d/2) + pow(xpd + ymd + zpd, 0.5) ) * (-(y+d/2) + pow(xmd + ypd + zpd, 0.5) )));
				Gyx_1d_real[idx] = ( log( 4 * (-(z-d/2) + pow(xmd + ymd + zmd, 0.5) ) * (-(z+d/2) + pow(xmd + ypd + zpd, 0.5) )) - 
					                  log( 4 * (-(z-d/2) + pow(xmd + ypd + zmd, 0.5) ) * (-(z+d/2) + pow(xmd + ymd + zpd, 0.5) ))) - 
					                ( log( 4 * (-(z-d/2) + pow(xpd + ymd + zmd, 0.5) ) * (-(z+d/2) + pow(xpd + ypd + zpd, 0.5) )) - 
					                  log( 4 * (-(z-d/2) + pow(xpd + ypd + zmd, 0.5) ) * (-(z+d/2) + pow(xpd + ymd + zpd, 0.5) )));
				Gyy_1d_real[idx] = ((atan( (x-d/2) * (z-d/2)/((y-d/2) * pow(xmd + ymd + zmd, 0.5)) ) + 
					                  atan( (x+d/2) * (z+d/2)/((y-d/2) * pow(xpd + ymd + zpd, 0.5)) )) - 
					                 (atan( (x-d/2) * (z+d/2)/((y-d/2) * pow(xmd + ymd + zpd, 0.5)) ) +
					                  atan( (x+d/2) * (z-d/2)/((y-d/2) * pow(xpd + ymd + zmd, 0.5)) ))) - 
					                ((atan( (x-d/2) * (z-d/2)/((y+d/2) * pow(xmd + ypd + zmd, 0.5)) ) + 
					                  atan( (x+d/2) * (z+d/2)/((y+d/2) * pow(xpd + ypd + zpd, 0.5)) )) - 
					                 (atan( (x-d/2) * (z+d/2)/((y+d/2) * pow(xmd + ypd + zpd, 0.5)) ) + 
					                  atan( (x+d/2) * (z-d/2)/((y+d/2) * pow(xpd + ypd + zmd, 0.5)) )));
				Gyz_1d_real[idx] = ( log( 4 * (-(x-d/2) + pow(xmd + ymd + zmd, 0.5) ) * (-(x+d/2) + pow(xpd + ypd + zmd, 0.5) )) - 
					                  log( 4 * (-(x-d/2) + pow(xmd + ypd + zmd, 0.5) ) * (-(x+d/2) + pow(xpd + ymd + zmd, 0.5) ))) - 
					                ( log( 4 * (-(x-d/2) + pow(xmd + ymd + zpd, 0.5) ) * (-(x+d/2) + pow(xpd + ypd + zpd, 0.5) )) - 
					                  log( 4 * (-(x-d/2) + pow(xmd + ypd + zpd, 0.5) ) * (-(x+d/2) + pow(xpd + ymd + zpd, 0.5) )));
				Gzx_1d_real[idx] = ( log( 4 * (-(y-d/2) + pow(xmd + ymd + zmd, 0.5) ) * (-(y+d/2) + pow(xmd + ypd + zpd, 0.5) )) - 
					                  log( 4 * (-(y-d/2) + pow(xmd + ymd + zpd, 0.5) ) * (-(y+d/2) + pow(xmd + ypd + zmd, 0.5) ))) - 
					                ( log( 4 * (-(y-d/2) + pow(xpd + ymd + zmd, 0.5) ) * (-(y+d/2) + pow(xpd + ypd + zpd, 0.5) )) - 
					                  log( 4 * (-(y-d/2) + pow(xpd + ymd + zpd, 0.5) ) * (-(y+d/2) + pow(xpd + ypd + zmd, 0.5) )));
				Gzy_1d_real[idx] = ( log( 4 * (-(x-d/2) + pow(xmd + ymd + zmd, 0.5) ) * (-(x+d/2) + pow(xpd + ymd + zpd, 0.5) )) - 
					                  log( 4 * (-(x-d/2) + pow(xmd + ymd + zpd, 0.5) ) * (-(x+d/2) + pow(xpd + ymd + zmd, 0.5) ))) - 
					                ( log( 4 * (-(x-d/2) + pow(xmd + ypd + zmd, 0.5) ) * (-(x+d/2) + pow(xpd + ypd + zpd, 0.5) )) - 
					                  log( 4 * (-(x-d/2) + pow(xmd + ypd + zpd, 0.5) ) * (-(x+d/2) + pow(xpd + ypd + zmd, 0.5) )));
				Gzz_1d_real[idx] = ((atan( (x-d/2) * (y-d/2)/((z-d/2) * pow(xmd + ymd + zmd, 0.5)) ) + 
					                  atan( (x+d/2) * (y+d/2)/((z-d/2) * pow(xpd + ypd + zmd, 0.5)) )) - 
					                 (atan( (x-d/2) * (y+d/2)/((z-d/2) * pow(xmd + ypd + zmd, 0.5)) ) + 
					                  atan( (x+d/2) * (y-d/2)/((z-d/2) * pow(xpd + ymd + zmd, 0.5)) ))) - 
					                ((atan( (x-d/2) * (y-d/2)/((z+d/2) * pow(xmd + ymd + zpd, 0.5)) ) + 
					                  atan( (x+d/2) * (y+d/2)/((z+d/2) * pow(xpd + ypd + zpd, 0.5)) )) - 
					                 (atan( (x-d/2) * (y+d/2)/((z+d/2) * pow(xmd + ypd + zpd, 0.5)) ) + 
					                  atan( (x+d/2) * (y-d/2)/((z+d/2) * pow(xpd + ymd + zpd, 0.5)) )));

			 	Gxx_1d_imag[idx] = 0.;
				Gxy_1d_imag[idx] = 0.;
				Gxz_1d_imag[idx] = 0.;
				Gyx_1d_imag[idx] = 0.;
				Gyy_1d_imag[idx] = 0.;
				Gyz_1d_imag[idx] = 0.;
				Gzx_1d_imag[idx] = 0.;
				Gzy_1d_imag[idx] = 0.;
				Gzz_1d_imag[idx] = 0.;
			}
		}
	}

	err = clEnqueueWriteBuffer(queue, dev_Gxx_cufft[0], CL_TRUE, 0, sizeof(cl_double)*(lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad), Gxx_1d_real, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, dev_Gxy_cufft[0], CL_TRUE, 0, sizeof(cl_double)*(lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad), Gxy_1d_real, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, dev_Gxz_cufft[0], CL_TRUE, 0, sizeof(cl_double)*(lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad), Gxz_1d_real, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, dev_Gyx_cufft[0], CL_TRUE, 0, sizeof(cl_double)*(lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad), Gyx_1d_real, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, dev_Gyy_cufft[0], CL_TRUE, 0, sizeof(cl_double)*(lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad), Gyy_1d_real, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, dev_Gyz_cufft[0], CL_TRUE, 0, sizeof(cl_double)*(lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad), Gyz_1d_real, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, dev_Gzx_cufft[0], CL_TRUE, 0, sizeof(cl_double)*(lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad), Gzx_1d_real, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, dev_Gzy_cufft[0], CL_TRUE, 0, sizeof(cl_double)*(lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad), Gzy_1d_real, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, dev_Gzz_cufft[0], CL_TRUE, 0, sizeof(cl_double)*(lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad), Gzz_1d_real, 0, NULL, NULL);

	err |= clEnqueueWriteBuffer(queue, dev_Gxx_cufft[1], CL_TRUE, 0, sizeof(cl_double)*(lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad), Gxx_1d_imag, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, dev_Gxy_cufft[1], CL_TRUE, 0, sizeof(cl_double)*(lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad), Gxy_1d_imag, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, dev_Gxz_cufft[1], CL_TRUE, 0, sizeof(cl_double)*(lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad), Gxz_1d_imag, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, dev_Gyx_cufft[1], CL_TRUE, 0, sizeof(cl_double)*(lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad), Gyx_1d_imag, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, dev_Gyy_cufft[1], CL_TRUE, 0, sizeof(cl_double)*(lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad), Gyy_1d_imag, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, dev_Gyz_cufft[1], CL_TRUE, 0, sizeof(cl_double)*(lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad), Gyz_1d_imag, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, dev_Gzx_cufft[1], CL_TRUE, 0, sizeof(cl_double)*(lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad), Gzx_1d_imag, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, dev_Gzy_cufft[1], CL_TRUE, 0, sizeof(cl_double)*(lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad), Gzy_1d_imag, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, dev_Gzz_cufft[1], CL_TRUE, 0, sizeof(cl_double)*(lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad), Gzz_1d_imag, 0, NULL, NULL);
 
    if (err != CL_SUCCESS)
    {
        cout<<"Error: Failed to write to source array (G-matrix)!"<<endl;
        exit(1);
    }

	///// Watch /////
	/*FILE *p1;
	fopen_s(&p1, "p1.dat", "w");
	for (int k = 0; k < lz_zero_pad; k++){
		for (int j = 0; j < ly_zero_pad; j++){
			for (int i = 0; i < lx_zero_pad; i++){
				fprintf(p1, " %15.5f", real(Gzz_1d_cmplx[i+j*lx_zero_pad+k*lx_zero_pad*ly_zero_pad]));
			}
			fprintf(p1, "\n");
		}
		fprintf(p1, "\n \n");
	}
	fclose(p1);*/
	///// Watch /////

	if (Gxx_1d_real)   free (Gxx_1d_real);
	if (Gxy_1d_real)   free (Gxy_1d_real);
	if (Gxz_1d_real)   free (Gxz_1d_real);
	if (Gyx_1d_real)   free (Gyx_1d_real);
	if (Gyy_1d_real)   free (Gyy_1d_real);
	if (Gyz_1d_real)   free (Gyz_1d_real);
	if (Gzx_1d_real)   free (Gzx_1d_real);
	if (Gzy_1d_real)   free (Gzy_1d_real);
	if (Gzz_1d_real)   free (Gzz_1d_real);

	if (Gxx_1d_imag)   free (Gxx_1d_imag);
	if (Gxy_1d_imag)   free (Gxy_1d_imag);
	if (Gxz_1d_imag)   free (Gxz_1d_imag);
	if (Gyx_1d_imag)   free (Gyx_1d_imag);
	if (Gyy_1d_imag)   free (Gyy_1d_imag);
	if (Gyz_1d_imag)   free (Gyz_1d_imag);
	if (Gzx_1d_imag)   free (Gzx_1d_imag);
	if (Gzy_1d_imag)   free (Gzy_1d_imag);
	if (Gzz_1d_imag)   free (Gzz_1d_imag);
	return true;
}



int G_tensor(int lx_zero_pad, int ly_zero_pad, int lz_zero_pad, 
			cl_context &ctx, cl_command_queue &queue)
{
	// Demag Tensor
	if (!G_matrix(lx_zero_pad, ly_zero_pad, lz_zero_pad, queue)) 
	{ 
		cout<<"G_matrix() fails!"<<endl; 
		exit(1); 
	}

	//---------- Set to Device ----------//


	/////// Watch /////
	// {		cl_event event = 0;
	// 		double *out=new double[(lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad)];
	// 		cl_int err = clEnqueueReadBuffer(queue, dev_Gyx_cufft[0], CL_TRUE, 0, sizeof(cl_double) * (lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad), out, 0, NULL, &event);
	// 		if (err != CL_SUCCESS)
	// 		{
	// 			printf("Error: Failed to read output array! %d\n", err);
	// 			exit(1);
	// 		}
							
	// 		ofstream wfile_o;
	// 		wfile_o.open("dev_Gyx_cufft.dat");
	// 		for (int k = 0; k < (lz_zero_pad); k++){
	// 			for (int j = 0; j < (ly_zero_pad); j++){
	// 				for (int i = 0; i < (lx_zero_pad); i++){
	// 								//fprintf(watch, " %lf", out[i+j*(Nx)+k*(Nx)*(Ny)]);
	// 					wfile_o<<setw(10)<<out[i+j*(lx_zero_pad)+k*(lx_zero_pad)*(ly_zero_pad)]<<" ";
	// 				}
	// 							//fprintf(watch, "\n");
	// 				wfile_o<<endl;
	// 			}
	// 						//fprintf(watch, "\n \n");
	// 		wfile_o<<endl<<endl;
	// 		}
	// 					//fclose(watch);
	// 		wfile_o.close();
	// 		delete []out;
	// }

    

	
	//---------- CUFFT ----------//
	FFT_3D_OpenCL(dev_Gxx_cufft,    //cl_mem buffersOut[], 
				"forward", lx_zero_pad, ly_zero_pad, lz_zero_pad, ctx, queue);
	FFT_3D_OpenCL(dev_Gxy_cufft,     //cl_mem buffersOut[], 
				"forward", lx_zero_pad, ly_zero_pad, lz_zero_pad, ctx, queue);
	FFT_3D_OpenCL(dev_Gxz_cufft,     //cl_mem buffersOut[], 
				"forward", lx_zero_pad, ly_zero_pad, lz_zero_pad, ctx, queue);

	FFT_3D_OpenCL(dev_Gyx_cufft,     //cl_mem buffersOut[], 
				"forward", lx_zero_pad, ly_zero_pad, lz_zero_pad, ctx, queue);
	FFT_3D_OpenCL(dev_Gyy_cufft,     //cl_mem buffersOut[], 
				"forward", lx_zero_pad, ly_zero_pad, lz_zero_pad, ctx, queue);
	FFT_3D_OpenCL(dev_Gyz_cufft,     //cl_mem buffersOut[], 
				"forward", lx_zero_pad, ly_zero_pad, lz_zero_pad, ctx, queue);

	FFT_3D_OpenCL(dev_Gzx_cufft,     //cl_mem buffersOut[], 
				"forward", lx_zero_pad, ly_zero_pad, lz_zero_pad, ctx, queue);
	FFT_3D_OpenCL(dev_Gzy_cufft,     //cl_mem buffersOut[], 
				"forward", lx_zero_pad, ly_zero_pad, lz_zero_pad, ctx, queue);
	FFT_3D_OpenCL(dev_Gzz_cufft,      //cl_mem buffersOut[], 
				"forward", lx_zero_pad, ly_zero_pad, lz_zero_pad, ctx, queue);

	// {		cl_event event = 0;
	// 		double *out=new double[(lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad)];
	// 		cl_int err = clEnqueueReadBuffer(queue, dev_Gyx_cufft[0], CL_TRUE, 0, sizeof(cl_double) * (lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad), out, 0, NULL, &event);
	// 		if (err != CL_SUCCESS)
	// 		{
	// 			printf("Error: Failed to read output array! %d\n", err);
	// 			exit(1);
	// 		}
							
	// 		ofstream wfile_o;
	// 		wfile_o.open("dev_Gyx_cufft_f.dat");
	// 		for (int k = 0; k < (lz_zero_pad); k++){
	// 			for (int j = 0; j < (ly_zero_pad); j++){
	// 				for (int i = 0; i < (lx_zero_pad); i++){
	// 								//fprintf(watch, " %lf", out[i+j*(Nx)+k*(Nx)*(Ny)]);
	// 					wfile_o<<setw(10)<<out[i+j*(lx_zero_pad)+k*(lx_zero_pad)*(ly_zero_pad)]<<" ";
	// 				}
	// 							//fprintf(watch, "\n");
	// 				wfile_o<<endl;
	// 			}
	// 						//fprintf(watch, "\n \n");
	// 		wfile_o<<endl<<endl;
	// 		}
	// 					//fclose(watch);
	// 		wfile_o.close();
	// 		delete []out;
	// }

	/////// Watch /////

	return 1;

}


int Hms(int lx_zero_pad, int ly_zero_pad, int lz_zero_pad,
		cl_context &ctx, cl_command_queue &queue,
		cl_kernel &ker_M_ini, cl_kernel &ker_fft_M_G, cl_kernel &ker_fft_hd)
{
	double x, y, z, xx, yy, zz;
	cl_int err;
	cl_event event = 0;	
	int ret = 0;

	//---------- CUFFT ----------//
	err = clSetKernelArg(ker_M_ini, 0, sizeof(cl_mem),  &(dev_Mx_cufft[0]));
    err |= clSetKernelArg(ker_M_ini, 1, sizeof(cl_mem),  &(dev_Mx_cufft[1]));
    err |= clSetKernelArg(ker_M_ini, 2, sizeof(cl_mem),  &(dev_My_cufft[0]));
    err |= clSetKernelArg(ker_M_ini, 3, sizeof(cl_mem),  &(dev_My_cufft[1]));
    err |= clSetKernelArg(ker_M_ini, 4, sizeof(cl_mem),  &(dev_Mz_cufft[0]));
    err |= clSetKernelArg(ker_M_ini, 5, sizeof(cl_mem),  &(dev_Mz_cufft[1]));
    err |= clSetKernelArg(ker_M_ini, 6, sizeof(cl_mem),  &dev_Mx);
    err |= clSetKernelArg(ker_M_ini, 7, sizeof(cl_mem),  &dev_My);
    err |= clSetKernelArg(ker_M_ini, 8, sizeof(cl_mem),  &dev_Mz);
    err |= clSetKernelArg(ker_M_ini, 9, sizeof(cl_int),  &lx_zero_pad);
    err |= clSetKernelArg(ker_M_ini, 10, sizeof(cl_int),  &ly_zero_pad);
    err |= clSetKernelArg(ker_M_ini, 11, sizeof(cl_int),  &lz_zero_pad);

    if (err != CL_SUCCESS)
    {
        cout<<"Error: Failed to set ker_M_ini arguments!"<<err<<endl;
		//ret = 1;
		return 2;
        //exit(1);
    } 
    err = clEnqueueNDRangeKernel(queue, ker_M_ini, 3, NULL, numWorkItems, NULL, 0, NULL, &event);
    if (err)
    {
         cout<<"Error: Failed to execute kernel ker_M_ini!"<<endl;
		 return 2;
         //exit(1);
    }
    err = clWaitForEvents(1, &event);
	if (err)
    {
         cout<<"Error: Wait For ker_M_ini!"<<endl;
		 ret = 1;
         //exit(1);
    }

//////Watch//////////////
	// 	{		cl_event event = 0;
	// 		double *out=new double[(lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad)];
	// 		cl_int err = clEnqueueReadBuffer(queue, dev_Mz_cufft[0], CL_TRUE, 0, sizeof(cl_double) * (lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad), out, 0, NULL, &event);
	// 		if (err != CL_SUCCESS)
	// 		{
	// 			printf("Error: Failed to read output array! %d\n", err);
	// 			exit(1);
	// 		}
	// 		err = clWaitForEvents(1, &event);			
	// 		ofstream wfile_o;
	// 		wfile_o.open("dev_Mz_cufft.dat");
	// 		for (int k = 0; k < (lz_zero_pad); k++){
	// 			for (int j = 0; j < (ly_zero_pad); j++){
	// 				for (int i = 0; i < (lx_zero_pad); i++){
	// 								//fprintf(watch, " %lf", out[i+j*(Nx)+k*(Nx)*(Ny)]);
	// 					wfile_o<<setw(10)<<out[i+j*(lx_zero_pad)+k*(lx_zero_pad)*(ly_zero_pad)]<<" ";
	// 				}
	// 							//fprintf(watch, "\n");
	// 				wfile_o<<endl;
	// 			}
	// 						//fprintf(watch, "\n \n");
	// 		wfile_o<<endl<<endl;
	// 		}
	// 					//fclose(watch);
	// 		wfile_o.close();
	// 		delete []out;
	// }

	FFT_3D_OpenCL(dev_Mx_cufft,    //cl_mem buffersOut[], 
				"forward", lx_zero_pad, ly_zero_pad, lz_zero_pad, ctx, queue);
	FFT_3D_OpenCL(dev_My_cufft,     //cl_mem buffersOut[], 
				"forward", lx_zero_pad, ly_zero_pad, lz_zero_pad, ctx, queue);
	FFT_3D_OpenCL(dev_Mz_cufft,     //cl_mem buffersOut[], 
				"forward", lx_zero_pad, ly_zero_pad, lz_zero_pad, ctx, queue);


	////////Watch//////////////
	// {		cl_event event = 0;
	// 		double *out=new double[(lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad)];
	// 		cl_int err = clEnqueueReadBuffer(queue, dev_Mz_cufft[0], CL_TRUE, 0, sizeof(cl_double) * (lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad), out, 0, NULL, &event);
	// 		if (err != CL_SUCCESS)
	// 		{
	// 			printf("Error: Failed to read output array! %d\n", err);
	// 			exit(1);
	// 		}
	// 		err = clWaitForEvents(1, &event);			
	// 		ofstream wfile_o;
	// 		wfile_o.open("dev_Mz_cufft_f.dat");
	// 		for (int k = 0; k < (lz_zero_pad); k++){
	// 			for (int j = 0; j < (ly_zero_pad); j++){
	// 				for (int i = 0; i < (lx_zero_pad); i++){
	// 								//fprintf(watch, " %lf", out[i+j*(Nx)+k*(Nx)*(Ny)]);
	// 					wfile_o<<setw(10)<<out[i+j*(lx_zero_pad)+k*(lx_zero_pad)*(ly_zero_pad)]<<" ";
	// 				}
	// 							//fprintf(watch, "\n");
	// 				wfile_o<<endl;
	// 			}
	// 						//fprintf(watch, "\n \n");
	// 		wfile_o<<endl<<endl;
	// 		}
	// 					//fclose(watch);
	// 		wfile_o.close();
	// 		delete []out;
	// }


	err = clSetKernelArg(ker_fft_M_G, 0, sizeof(cl_mem),  &(dev_Gxx_cufft[0]));
    err |= clSetKernelArg(ker_fft_M_G, 1, sizeof(cl_mem),  &(dev_Gxy_cufft[0]));
    err |= clSetKernelArg(ker_fft_M_G, 2, sizeof(cl_mem),  &(dev_Gxz_cufft[0]));
	err |= clSetKernelArg(ker_fft_M_G, 3, sizeof(cl_mem),  &(dev_Gyx_cufft[0]));
    err |= clSetKernelArg(ker_fft_M_G, 4, sizeof(cl_mem),  &(dev_Gyy_cufft[0]));
    err |= clSetKernelArg(ker_fft_M_G, 5, sizeof(cl_mem),  &(dev_Gyz_cufft[0]));
	err |= clSetKernelArg(ker_fft_M_G, 6, sizeof(cl_mem),  &(dev_Gzx_cufft[0]));
	err |= clSetKernelArg(ker_fft_M_G, 7, sizeof(cl_mem),  &(dev_Gzy_cufft[0]));
    err |= clSetKernelArg(ker_fft_M_G, 8, sizeof(cl_mem),  &(dev_Gzz_cufft[0]));
	err |= clSetKernelArg(ker_fft_M_G, 9, sizeof(cl_mem),  &(dev_Mx_cufft[0]));
	err |= clSetKernelArg(ker_fft_M_G, 10, sizeof(cl_mem),  &(dev_My_cufft[0]));
    err |= clSetKernelArg(ker_fft_M_G, 11, sizeof(cl_mem),  &(dev_Mz_cufft[0]));
	err |= clSetKernelArg(ker_fft_M_G, 12, sizeof(cl_mem),  &(dev_Hd_x_cufft[0]));
	err |= clSetKernelArg(ker_fft_M_G, 13, sizeof(cl_mem),  &(dev_Hd_y_cufft[0]));
    err |= clSetKernelArg(ker_fft_M_G, 14, sizeof(cl_mem),  &(dev_Hd_z_cufft[0]));

	err |= clSetKernelArg(ker_fft_M_G, 15, sizeof(cl_mem),  &(dev_Gxx_cufft[1]));
    err |= clSetKernelArg(ker_fft_M_G, 16, sizeof(cl_mem),  &(dev_Gxy_cufft[1]));
    err |= clSetKernelArg(ker_fft_M_G, 17, sizeof(cl_mem),  &(dev_Gxz_cufft[1]));
	err |= clSetKernelArg(ker_fft_M_G, 18, sizeof(cl_mem),  &(dev_Gyx_cufft[1]));
    err |= clSetKernelArg(ker_fft_M_G, 19, sizeof(cl_mem),  &(dev_Gyy_cufft[1]));
    err |= clSetKernelArg(ker_fft_M_G, 20, sizeof(cl_mem),  &(dev_Gyz_cufft[1]));
	err |= clSetKernelArg(ker_fft_M_G, 21, sizeof(cl_mem),  &(dev_Gzx_cufft[1]));
	err |= clSetKernelArg(ker_fft_M_G, 22, sizeof(cl_mem),  &(dev_Gzy_cufft[1]));
    err |= clSetKernelArg(ker_fft_M_G, 23, sizeof(cl_mem),  &(dev_Gzz_cufft[1]));
	err |= clSetKernelArg(ker_fft_M_G, 24, sizeof(cl_mem),  &(dev_Mx_cufft[1]));
	err |= clSetKernelArg(ker_fft_M_G, 25, sizeof(cl_mem),  &(dev_My_cufft[1]));
    err |= clSetKernelArg(ker_fft_M_G, 26, sizeof(cl_mem),  &(dev_Mz_cufft[1]));
	err |= clSetKernelArg(ker_fft_M_G, 27, sizeof(cl_mem),  &(dev_Hd_x_cufft[1]));
	err |= clSetKernelArg(ker_fft_M_G, 28, sizeof(cl_mem),  &(dev_Hd_y_cufft[1]));
    err |= clSetKernelArg(ker_fft_M_G, 29, sizeof(cl_mem),  &(dev_Hd_z_cufft[1]));

    err |= clSetKernelArg(ker_fft_M_G, 30, sizeof(cl_int),  &lx_zero_pad);
    err |= clSetKernelArg(ker_fft_M_G, 31, sizeof(cl_int),  &ly_zero_pad);
    err |= clSetKernelArg(ker_fft_M_G, 32, sizeof(cl_int),  &lz_zero_pad);

    if (err != CL_SUCCESS)
    {
        cout<<"Error: Failed to set ker_fft_M_G arguments!"<<err<<endl;
        return 2;
		//exit(1);
    } 
	size_t numWorkItems_ext[3] = {(size_t)lx_zero_pad, (size_t)ly_zero_pad, (size_t)lz_zero_pad};
    err = clEnqueueNDRangeKernel(queue, ker_fft_M_G, 3, NULL, numWorkItems_ext, NULL, 0, NULL, &event);
    if (err)
    {
         cout<<"Error: Failed to execute kernel ker_fft_M_G!"<<endl;
		 return 2;
         //exit(1);
    }
    err = clWaitForEvents(1, &event);
	if (err)
    {
         cout<<"Error: Wait For ker_fft_M_G!"<<endl;
		 ret = 1;
         //exit(1);
    }


	////////Watch//////////////
	// {		cl_event event = 0;
	// 		double *out=new double[(lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad)];
	// 		cl_int err = clEnqueueReadBuffer(queue, dev_Gxz_cufft[0], CL_TRUE, 0, sizeof(cl_double) * (lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad), out, 0, NULL, &event);
	// 		if (err != CL_SUCCESS)
	// 		{
	// 			printf("Error: Failed to read output array! %d\n", err);
	// 			exit(1);
	// 		}
	// 		err = clWaitForEvents(1, &event);			
	// 		ofstream wfile_o;
	// 		wfile_o.open("dev_Gxz_cufft.dat");
	// 		for (int k = 0; k < (lz_zero_pad); k++){
	// 			for (int j = 0; j < (ly_zero_pad); j++){
	// 				for (int i = 0; i < (lx_zero_pad); i++){
	// 								//fprintf(watch, " %lf", out[i+j*(Nx)+k*(Nx)*(Ny)]);
	// 					wfile_o<<setw(10)<<out[i+j*(lx_zero_pad)+k*(lx_zero_pad)*(ly_zero_pad)]<<" ";
	// 				}
	// 							//fprintf(watch, "\n");
	// 				wfile_o<<endl;
	// 			}
	// 						//fprintf(watch, "\n \n");
	// 		wfile_o<<endl<<endl;
	// 		}
	// 					//fclose(watch);
	// 		wfile_o.close();
	// 		delete []out;
	// }

	// {		cl_event event = 0;
	// 		double *out=new double[(lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad)];
	// 		cl_int err = clEnqueueReadBuffer(queue, dev_Hd_z_cufft[0], CL_TRUE, 0, sizeof(cl_double) * (lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad), out, 0, NULL, &event);
	// 		if (err != CL_SUCCESS)
	// 		{
	// 			printf("Error: Failed to read output array! %d\n", err);
	// 			exit(1);
	// 		}
	// 		err = clWaitForEvents(1, &event);			
	// 		ofstream wfile_o;
	// 		wfile_o.open("dev_Hd_z_cufft.dat");
	// 		for (int k = 0; k < (lz_zero_pad); k++){
	// 			for (int j = 0; j < (ly_zero_pad); j++){
	// 				for (int i = 0; i < (lx_zero_pad); i++){
	// 								//fprintf(watch, " %lf", out[i+j*(Nx)+k*(Nx)*(Ny)]);
	// 					wfile_o<<setw(10)<<out[i+j*(lx_zero_pad)+k*(lx_zero_pad)*(ly_zero_pad)]<<" ";
	// 				}
	// 							//fprintf(watch, "\n");
	// 				wfile_o<<endl;
	// 			}
	// 						//fprintf(watch, "\n \n");
	// 		wfile_o<<endl<<endl;
	// 		}
	// 					//fclose(watch);
	// 		wfile_o.close();
	// 		delete []out;
	// }
	// Kernel_CUFFT_M_times_G<<<grids2, blocks2>>>(dev_Gxx_cufft,  dev_Gxy_cufft,  dev_Gxz_cufft,
	// 										    dev_Gyx_cufft,  dev_Gyy_cufft,  dev_Gyz_cufft,
	// 										    dev_Gzx_cufft,  dev_Gzy_cufft,  dev_Gzz_cufft,
	// 										    dev_Mx_cufft,   dev_My_cufft,   dev_Mz_cufft,
	// 										    dev_Hd_x_cufft, dev_Hd_y_cufft, dev_Hd_z_cufft,
	// 										    lx_zero_pad, ly_zero_pad, lz_zero_pad);

	//////---------- CUIFFT ----------//
	FFT_3D_OpenCL(dev_Hd_x_cufft,    //cl_mem buffersOut[], 
				"backward", lx_zero_pad, ly_zero_pad, lz_zero_pad, ctx, queue);
	FFT_3D_OpenCL(dev_Hd_y_cufft,     //cl_mem buffersOut[], 
				"backward", lx_zero_pad, ly_zero_pad, lz_zero_pad, ctx, queue);
	FFT_3D_OpenCL(dev_Hd_z_cufft,     //cl_mem buffersOut[], 
				"backward", lx_zero_pad, ly_zero_pad, lz_zero_pad, ctx, queue);


	// {		cl_event event = 0;
	// 		double *out=new double[(lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad)];
	// 		cl_int err = clEnqueueReadBuffer(queue, dev_Hd_x_cufft[0], CL_TRUE, 0, sizeof(cl_double) * (lx_zero_pad)*(ly_zero_pad)*(lz_zero_pad), out, 0, NULL, &event);
	// 		if (err != CL_SUCCESS)
	// 		{
	// 			printf("Error: Failed to read output array! %d\n", err);
	// 			exit(1);
	// 		}
	// 		err = clWaitForEvents(1, &event);			
	// 		ofstream wfile_o;
	// 		wfile_o.open("dev_Hd_x_cufft.dat");
	// 		for (int k = 0; k < (lz_zero_pad); k++){
	// 			for (int j = 0; j < (ly_zero_pad); j++){
	// 				for (int i = 0; i < (lx_zero_pad); i++){
	// 								//fprintf(watch, " %lf", out[i+j*(Nx)+k*(Nx)*(Ny)]);
	// 					wfile_o<<setw(10)<<out[i+j*(lx_zero_pad)+k*(lx_zero_pad)*(ly_zero_pad)]<<" ";
	// 				}
	// 							//fprintf(watch, "\n");
	// 				wfile_o<<endl;
	// 			}
	// 						//fprintf(watch, "\n \n");
	// 		wfile_o<<endl<<endl;
	// 		}
	// 					//fclose(watch);
	// 		wfile_o.close();
	// 		delete []out;
	// }



	// Shift the Hms field
	//size_t numWorkItems_ext[3] = {(size_t)lx_zero_pad, (size_t)ly_zero_pad, (size_t)lz_zero_pad};
	err = clSetKernelArg(ker_fft_hd, 0, sizeof(cl_mem),  &(dev_Hd_x_cufft[0]));
    err |= clSetKernelArg(ker_fft_hd, 1, sizeof(cl_mem),  &(dev_Hd_y_cufft[0]));
    err |= clSetKernelArg(ker_fft_hd, 2, sizeof(cl_mem),  &(dev_Hd_z_cufft[0]));
	err |= clSetKernelArg(ker_fft_hd, 3, sizeof(cl_mem),  &(dev_Hd_x));
    err |= clSetKernelArg(ker_fft_hd, 4, sizeof(cl_mem),  &(dev_Hd_y));
    err |= clSetKernelArg(ker_fft_hd, 5, sizeof(cl_mem),  &(dev_Hd_z));
	
    err |= clSetKernelArg(ker_fft_hd, 6, sizeof(cl_int),  &lx_zero_pad);
    err |= clSetKernelArg(ker_fft_hd, 7, sizeof(cl_int),  &ly_zero_pad);
    err |= clSetKernelArg(ker_fft_hd, 8, sizeof(cl_int),  &lz_zero_pad);

    if (err != CL_SUCCESS)
    {
        cout<<"Error: Failed to set ker_fft_hd arguments!"<<err<<endl;
        //exit(1);
		return 2;
    } 
    err = clEnqueueNDRangeKernel(queue, ker_fft_hd, 3, NULL, numWorkItems, NULL, 0, NULL, &event);
    if (err)
    {
         cout<<"Error: Failed to execute kernel ker_fft_hd!"<<endl;
         //exit(1);
		 return 2;
    }
    err = clWaitForEvents(1, &event);
	if (err)
    {
         cout<<"Error: Wait For ker_fft_hd!"<<endl;
		 ret = 1;
         //exit(1);
    }
///////watch///////
	// {		cl_event event = 0;
	// 		double *out=new double[Nx * Ny * Nz];
	// 		cl_int err = clEnqueueReadBuffer(queue, dev_Hd_z, CL_TRUE, 0, sizeof(cl_double) * Nx * Ny * Nz, out, 0, NULL, &event);
	// 		if (err != CL_SUCCESS)
	// 		{
	// 			printf("Error: Failed to read output array! %d\n", err);
	// 			exit(1);
	// 		}
	// 		err = clWaitForEvents(1, &event);			
	// 		ofstream wfile_o;
	// 		wfile_o.open("dev_Hd_z.dat");
	// 		for (int k = 0; k < Nz; k++){
	// 			for (int j = 0; j < Ny; j++){
	// 				for (int i = 0; i < Nx; i++){
	// 								//fprintf(watch, " %lf", out[i+j*(Nx)+k*(Nx)*(Ny)]);
	// 					wfile_o<<setw(10)<<out[i+j*Nx+k*Nx*Ny]<<" ";
	// 				}
	// 							//fprintf(watch, "\n");
	// 				wfile_o<<endl;
	// 			}
	// 						//fprintf(watch, "\n \n");
	// 		wfile_o<<endl<<endl;
	// 		}
	// 					//fclose(watch);
	// 		wfile_o.close();
	// 		delete []out;
	// }

	// int idx, idxx = 0;
	// double H_temp;
	// for (int k = Nz-1; k < 2*Nz-1; k++){
	// 	for (int j = Ny-1; j < 2*Ny-1; j++){
	// 		for (int i = Nx-1; i < 2*Nx-1; i++){
	// 			idx = i + j*lx_zero_pad + k*lx_zero_pad*ly_zero_pad;
	// 			Hd_x_1d_shift[idxx] = Hd_x_1d[idx];
	// 			Hd_y_1d_shift[idxx] = Hd_y_1d[idx];
	// 			Hd_z_1d_shift[idxx] = Hd_z_1d[idx];
	// 			idxx++;		
	// 		}
	// 	}
	// }


// 	return 1;
// ERROR:


	return ret;
}



#undef d