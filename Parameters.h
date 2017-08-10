//----Document created by Yipeng Jiao (jiaox058@umn.edu) on 01/10/2015-----

//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#ifndef __PARAMETER__
#define __PARAMETER__

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>

#include "Parameters_cl.clh"

using namespace std;

// Set GPU
//#define  ID                1

extern size_t numWorkItems[];
extern size_t *numWorkItems_local;
//size_t numWorkItems_sum = numWorkItems[0] * numWorkItems[1] * numWorkItems[2];

extern double offset;
unsigned long long int extern iseed;
extern double kb, delta_t, h, delta_x, v; // head velocity in (cm)

extern double FWHMx, FWHMy, deltaTemp;
extern double extHstat_x, extHstat_y, delta_HP;
extern double extTstat_x, extTstat_y, delta_TP;
// Variables
const double extern Aex, Ku, Ms, Ms_soft; 
long extern idum0;



// Array for device
extern cl_mem dev_Aex1, dev_Aex2, dev_Ku, dev_Ms, dev_alpha, dev_gamma;
extern cl_mem dev_Aex1_temp, dev_Aex2_temp, dev_Ku_temp, dev_Ms_temp, dev_alpha_temp;

extern cl_mem theta, phi;

extern cl_mem dev_GasArray, dev_streams;

extern cl_mem dev_theta, dev_phi;

extern cl_mem dev_Hth_x, dev_Hth_y, dev_Hth_z;
extern cl_mem dev_Hk_x , dev_Hk_y , dev_Hk_z ;
extern cl_mem dev_Ha_x , dev_Ha_y , dev_Ha_z ; 
extern cl_mem dev_Hd_x , dev_Hd_y , dev_Hd_z ;

extern cl_mem dev_Happl_x, dev_Happl_y, dev_Happl_z;
extern cl_mem dev_D, dev_T;

extern cl_mem dev_a_theta, dev_b_theta, dev_c_theta, dev_d_theta,
			  dev_a_phi, dev_b_phi, dev_c_phi, dev_d_phi;



extern cl_mem dev_d_theta_d_t, dev_d_phi_d_t;
extern cl_mem dev_Mx, dev_My, dev_Mz;
extern cl_mem dev_M_temp_x, dev_M_temp_y, dev_M_temp_z;

extern cl_mem   dev_indicator1, dev_indicator1_temp, 
			  dev_indicator2, dev_indicator2_temp,  // Voronoi-cell inner point indicator 
              dev_indicator3;

extern cl_mem dev_std_Aex, dev_std_Ku;

extern cl_mem dev_watch2;

extern cl_mem dev_extT, dev_extH_x, dev_extH_y, dev_extH_z;

// Array for host
extern double host_Aex[Nx*Ny*Nz], host_Aratio[Nx*Ny*Nz], host_Aex1[Nx*Ny*Nz], host_Aex2[Nx*Ny*Nz],
              host_Ku[Nx*Ny*Nz], host_Ms[Nx*Ny*Nz], host_alpha[Nx*Ny*Nz], host_gamma[Nx*Ny*Nz];

// extern int     indicator1[Nx*Ny*Nz], indicator2[Nx*Ny*Nz], indicator3[Nx*Ny*Nz];  // indicator1 = is_in_polygon; indicator2 = which_polygon
// extern double  indicator4[Nx*Ny*Nz], indicator5[Nx*Ny*Nz];

extern double  T[Nx*Ny*Nz], D[Nx*Ny*Nz], Happl_x[Nx*Ny*Nz], Happl_y[Nx*Ny*Nz], Happl_z[Nx*Ny*Nz];
extern double  extT[EXTTsize_x][EXTTsize_y],
			   extH_x[EXTHsize_x][EXTHsize_y], extH_y[EXTHsize_x][EXTHsize_y], extH_z[EXTHsize_x][EXTHsize_y];
extern double  extH_s[30000];

extern double  temp_Happl_x[Nx*Ny*Nz], temp_Happl_y[Nx*Ny*Nz], temp_Happl_z[Nx*Ny*Nz];
extern double  T_Cr, Happl;

extern double  Mx[Nx*Ny*Nz], My[Nx*Ny*Nz], Mz[Nx*Ny*Nz], 
			   watch1[(Nx+2)*(Ny+2)*(Nz+2)], watch2[Nx*Ny*Nz], watch3[Nx*Ny*Nz], watch4[Nx*Ny*Nz], watch5[Nx*Ny*Nz];

extern float   Mx_float[Nx*Ny*Nz], My_float[Nx*Ny*Nz], Mz_float[Nx*Ny*Nz];
extern double  Mx_t_bar[Nx*Ny*Nz], My_t_bar[Nx*Ny*Nz], Mz_t_bar[Nx*Ny*Nz];
extern int     watch1_int[(Nx+2)*(Ny+2)*(Nz+2)], watch2_int[Nx*Ny*Nz], watch3_int[Nx*Ny*Nz];


extern double  M_bar[TOTAL_TIME], Mx_bar1[TOTAL_TIME], My_bar1[TOTAL_TIME], Mz_bar1[TOTAL_TIME], 
                                  Mx_bar2[TOTAL_TIME], My_bar2[TOTAL_TIME], Mz_bar2[TOTAL_TIME],
								  Mx_bar3[TOTAL_TIME], My_bar3[TOTAL_TIME], Mz_bar3[TOTAL_TIME];

extern double  std_Ku[Nx*Ny*Nz],std_Aex[Nx*Ny*Nz];

extern double  gasarray[DEG_FREEDOM];


extern double x[], y[];
extern double a, b, xbar, ybar, xsqr_sum, xy_sum, std_torq, std_Keff, Keff,
			  sin_2theta_bar, torque_pp_bar, sin_2theta_bar_temp, torque_pp_bar_temp,	
			  My_bar_t, Mz_bar_t, M_bar_t, Ku_bar_t, torque_pp_bar_t, sin_2theta_bar_t;

// Head moving variable; head_dist: head's moving distance; NextBit: used to flip the applied field.
extern int HeadDist;
//extern long int  NextBit = 957657373; //45341;    
extern int  NextBit, temp_NextBit;
extern int  rise_time, temp_time, BL_t;
extern int	ID;
extern int	extTP, extHP, extHS;

extern ifstream rfile1, rfile2, rfile_kernel;
extern ofstream wfile100;



// FFT on Hms
// Arrays
/*static float   *Mx_1d  ,  *My_1d  ,  *Mz_1d  ,
*Gxx_1d ,  *Gxy_1d ,  *Gxz_1d ,
*Gyx_1d ,  *Gyy_1d ,  *Gyz_1d ,
*Gzx_1d ,  *Gzy_1d ,  *Gzz_1d ;*/


extern double  *Hd_x_1d_cmplx, *Hd_y_1d_cmplx, *Hd_z_1d_cmplx,
					   *Mx_1d_cmplx, *My_1d_cmplx, *Mz_1d_cmplx,
                       *Gxx_1d_real, *Gxy_1d_real, *Gxz_1d_real, 
                       *Gyx_1d_real, *Gyy_1d_real, *Gyz_1d_real, 
                       *Gzx_1d_real, *Gzy_1d_real, *Gzz_1d_real,

                       *Gxx_1d_imag, *Gxy_1d_imag, *Gxz_1d_imag, 
                       *Gyx_1d_imag, *Gyy_1d_imag, *Gyz_1d_imag, 
                       *Gzx_1d_imag, *Gzy_1d_imag, *Gzz_1d_imag;

//static double *Hd_x_1d, *Hd_y_1d, *Hd_z_1d;
//static double *Hd_x_1d_shift, *Hd_y_1d_shift, *Hd_z_1d_shift;


extern cl_mem //dev_Gxx[2], dev_Gxy[2], dev_Gxz[2],
	   //dev_Gyx[2], dev_Gyy[2], dev_Gyz[2],
	   //dev_Gzx[2], dev_Gzy[2], dev_Gzz[2],
	   dev_Gxx_cufft[2], dev_Gxy_cufft[2], dev_Gxz_cufft[2],
       dev_Gyx_cufft[2], dev_Gyy_cufft[2], dev_Gyz_cufft[2],
       dev_Gzx_cufft[2], dev_Gzy_cufft[2], dev_Gzz_cufft[2],
       dev_Mx_cufft[2],  dev_My_cufft[2],  dev_Mz_cufft[2],
       dev_Hd_x_cufft[2],  dev_Hd_y_cufft[2],  dev_Hd_z_cufft[2];

//static cufftHandle plan;
//static FILE *rfile2, *wfile100;

//watch
extern int watchcount;


#endif  //__PARAMETER__