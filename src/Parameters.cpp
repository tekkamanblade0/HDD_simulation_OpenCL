//----Document created by Yipeng Jiao (jiaox058@umn.edu) on 01/10/2015-----

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include "Parameters.h"

size_t numWorkItems[3] = {Nx, Ny, Nz};
size_t *numWorkItems_local = NULL;
//size_t numWorkItems_sum = numWorkItems[0] * numWorkItems[1] * numWorkItems[2];

double offset = 5;
unsigned long long int iseed = 114;
double kb = 1.38e-16,
			 delta_t = 2.0e-14,
			 h = delta_t,
			 delta_x = 1.50e-7,
			 v = 2000; // head velocity in (cm)

double FWHMx = 50, FWHMy = 50, deltaTemp = 0.0;
double extHstat_x = 0, extHstat_y = 0, delta_HP = 1.00e-7;
double extTstat_x = 0, extTstat_y = 0, delta_TP = 1.50e-7;
// Variables
const double Aex = 1.00e-6, Ku = 4.14e7, Ms = 925.07, Ms_soft = 1400; 
long idum0;



// Array for device
cl_mem dev_Aex1 = NULL, dev_Aex2 = NULL, dev_Ku = NULL, dev_Ms = NULL, dev_alpha = NULL, dev_gamma = NULL;
cl_mem dev_Aex1_temp = NULL, dev_Aex2_temp = NULL, dev_Ku_temp = NULL, dev_Ms_temp = NULL, dev_alpha_temp = NULL;

cl_mem theta = 0, phi = 0;

cl_mem dev_GasArray = NULL, dev_streams = NULL;

cl_mem dev_theta = NULL, dev_phi = NULL;

cl_mem dev_Hth_x = NULL, dev_Hth_y = NULL, dev_Hth_z = NULL;
cl_mem dev_Hk_x  = NULL, dev_Hk_y  = NULL, dev_Hk_z  = NULL;
cl_mem dev_Ha_x  = NULL, dev_Ha_y  = NULL, dev_Ha_z  = NULL; 
cl_mem dev_Hd_x  = NULL, dev_Hd_y  = NULL, dev_Hd_z  = NULL;

cl_mem dev_Happl_x = NULL, dev_Happl_y = NULL, dev_Happl_z = NULL;
cl_mem dev_D = NULL, dev_T = NULL;

cl_mem dev_a_theta = NULL, dev_b_theta = NULL, dev_c_theta = NULL, dev_d_theta = NULL,
			  dev_a_phi = NULL, dev_b_phi = NULL, dev_c_phi = NULL, dev_d_phi = NULL;



cl_mem dev_d_theta_d_t = NULL, dev_d_phi_d_t = NULL;
cl_mem dev_Mx = NULL, dev_My = NULL, dev_Mz = NULL;
cl_mem dev_M_temp_x = NULL, dev_M_temp_y = NULL, dev_M_temp_z = NULL;

cl_mem   dev_indicator1 = NULL, dev_indicator1_temp = NULL, 
			  dev_indicator2 = NULL, dev_indicator2_temp = NULL,  // Voronoi-cell inner point indicator 
              dev_indicator3 = NULL;

cl_mem dev_std_Aex = NULL, dev_std_Ku = NULL;

cl_mem dev_watch2 = NULL;

cl_mem dev_extT = NULL, dev_extH_x = NULL, dev_extH_y = NULL, dev_extH_z = NULL;

// Array for host
double host_Aex[Nx*Ny*Nz], host_Aratio[Nx*Ny*Nz], host_Aex1[Nx*Ny*Nz], host_Aex2[Nx*Ny*Nz],
              host_Ku[Nx*Ny*Nz], host_Ms[Nx*Ny*Nz], host_alpha[Nx*Ny*Nz], host_gamma[Nx*Ny*Nz];

// extern int     indicator1[Nx*Ny*Nz], indicator2[Nx*Ny*Nz], indicator3[Nx*Ny*Nz];  // indicator1 = is_in_polygon; indicator2 = which_polygon
// extern double  indicator4[Nx*Ny*Nz], indicator5[Nx*Ny*Nz];

double  T[Nx*Ny*Nz], D[Nx*Ny*Nz], Happl_x[Nx*Ny*Nz], Happl_y[Nx*Ny*Nz], Happl_z[Nx*Ny*Nz];
double  extT[EXTTsize_x][EXTTsize_y],
			   extH_x[EXTHsize_x][EXTHsize_y], extH_y[EXTHsize_x][EXTHsize_y], extH_z[EXTHsize_x][EXTHsize_y];
double  extH_s[30000];

double  temp_Happl_x[Nx*Ny*Nz], temp_Happl_y[Nx*Ny*Nz], temp_Happl_z[Nx*Ny*Nz];
double  T_Cr, Happl = +10000;

double  Mx[Nx*Ny*Nz], My[Nx*Ny*Nz], Mz[Nx*Ny*Nz], 
			   watch1[(Nx+2)*(Ny+2)*(Nz+2)], watch2[Nx*Ny*Nz], watch3[Nx*Ny*Nz], watch4[Nx*Ny*Nz], watch5[Nx*Ny*Nz];

float   Mx_float[Nx*Ny*Nz], My_float[Nx*Ny*Nz], Mz_float[Nx*Ny*Nz];
double  Mx_t_bar[Nx*Ny*Nz], My_t_bar[Nx*Ny*Nz], Mz_t_bar[Nx*Ny*Nz];
int     watch1_int[(Nx+2)*(Ny+2)*(Nz+2)], watch2_int[Nx*Ny*Nz], watch3_int[Nx*Ny*Nz];


double  M_bar[TOTAL_TIME], Mx_bar1[TOTAL_TIME], My_bar1[TOTAL_TIME], Mz_bar1[TOTAL_TIME], 
                                  Mx_bar2[TOTAL_TIME], My_bar2[TOTAL_TIME], Mz_bar2[TOTAL_TIME],
								  Mx_bar3[TOTAL_TIME], My_bar3[TOTAL_TIME], Mz_bar3[TOTAL_TIME];

double  std_Ku[Nx*Ny*Nz],std_Aex[Nx*Ny*Nz];

double  gasarray[DEG_FREEDOM];


double x[(TOTAL_TIME>>1)+1], y[(TOTAL_TIME>>1)+1];
double a, b, xbar, ybar, xsqr_sum, xy_sum, std_torq, std_Keff, Keff,
			  sin_2theta_bar, torque_pp_bar, sin_2theta_bar_temp, torque_pp_bar_temp,	
			  My_bar_t, Mz_bar_t, M_bar_t, Ku_bar_t, torque_pp_bar_t, sin_2theta_bar_t;

// Head moving variable; head_dist: head's moving distance; NextBit: used to flip the applied field.
int HeadDist = -1;
//extern long int  NextBit = 957657373; //45341;    
int  NextBit = 0, temp_NextBit = 0;
int  rise_time =0, temp_time, BL_t = 22500;
int	ID = 0;
int	extTP = 0, extHP = 0, extHS = 0;

ifstream rfile1, rfile2, rfile_kernel;
ofstream wfile100;



// FFT on Hms
// Arrays
/*static float   *Mx_1d   = NULL,  *My_1d   = NULL,  *Mz_1d   = NULL,
*Gxx_1d  = NULL,  *Gxy_1d  = NULL,  *Gxz_1d  = NULL,
*Gyx_1d  = NULL,  *Gyy_1d  = NULL,  *Gyz_1d  = NULL,
*Gzx_1d  = NULL,  *Gzy_1d  = NULL,  *Gzz_1d  = NULL;*/


double  *Hd_x_1d_cmplx = NULL, *Hd_y_1d_cmplx = NULL, *Hd_z_1d_cmplx = NULL,
					   *Mx_1d_cmplx   = NULL, *My_1d_cmplx   = NULL, *Mz_1d_cmplx   = NULL,
                       *Gxx_1d_real  = NULL, *Gxy_1d_real  = NULL, *Gxz_1d_real  = NULL, 
                       *Gyx_1d_real  = NULL, *Gyy_1d_real  = NULL, *Gyz_1d_real  = NULL, 
                       *Gzx_1d_real  = NULL, *Gzy_1d_real  = NULL, *Gzz_1d_real  = NULL,

                       *Gxx_1d_imag  = NULL, *Gxy_1d_imag  = NULL, *Gxz_1d_imag  = NULL, 
                       *Gyx_1d_imag  = NULL, *Gyy_1d_imag  = NULL, *Gyz_1d_imag  = NULL, 
                       *Gzx_1d_imag  = NULL, *Gzy_1d_imag  = NULL, *Gzz_1d_imag  = NULL;

//static double *Hd_x_1d = NULL, *Hd_y_1d = NULL, *Hd_z_1d = NULL;
//static double *Hd_x_1d_shift = NULL, *Hd_y_1d_shift = NULL, *Hd_z_1d_shift = NULL;


cl_mem //dev_Gxx[2], dev_Gxy[2], dev_Gxz[2],
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
int watchcount = 0;