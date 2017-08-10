//----Document created by Yipeng Jiao (jiaox058@umn.edu) on 01/11/2015-----
//Gussian distribution generator is based on parts of clProbDist project 
//(https://github.com/umontreal-simul/clProbDist)

#define CLRNG_SINGLE_PRECISION                                   
#include <clRNG/mrg31k3p.clh>  
#define CL_TRUE     true
#define CL_FALSE    false
#ifndef NULL
#define NULL ((void*)0)
#endif

#include "Parameters_cl.clh"

#define GetElement(pData, i, j, k, pitch_x, pitch_y)             pData[(k)*(pitch_x)*(pitch_y) + (j)*(pitch_x) + (i)]
#define GetElement_Int(pData, i, j, k, pitch_x, pitch_y)         pData[(k)*(pitch_x)*(pitch_y) + (j)*(pitch_x) + (i)]
#define SetElement(pData, i, j, k, pitch_x, pitch_y, value)      pData[(k)*(pitch_x)*(pitch_y) + (j)*(pitch_x) + (i)] = value
#define SetElement_Int(pData, i, j, k, pitch_x, pitch_y, value)  pData[(k)*(pitch_x)*(pitch_y) + (j)*(pitch_x) + (i)] = value



// double GetElement(__global double* pData, int i, int j, int k, int pitch_x, int pitch_y)
// {
// 	return pData[k*pitch_x*pitch_y + j*pitch_x + i];
// }
// int GetElement_Int(__global int* pData, int i, int j, int k, int pitch_x, int pitch_y)
// {
// 	return pData[k*pitch_x*pitch_y + j*pitch_x + i];
// }
// /*cufftComplex GetElement_Complex(cufftComplex* pData, int i, int j, int k, int pitch_x, int pitch_y)
// {
// 	return pData[k*pitch_x*pitch_y + j*pitch_x + i];
// }*/
// double SetElement(__global double* pData, int i, int j, int k, int pitch_x, int pitch_y, double value)
// {
// 	pData[k*pitch_x*pitch_y + j*pitch_x + i] = value;
// }
// int SetElement_Int(__global int* pData, int i, int j, int k, int pitch_x, int pitch_y, int value)
// {
// 	pData[k*pitch_x*pitch_y + j*pitch_x + i] = value;
// }

double dtheta_dt(int i, int j, int k, 
				__global double* dev_theta,	__global double* dev_phi, 
				__global double* dev_x_theta, __global double* dev_x_phi,
				__global double* dev_Ha_x,	__global double* dev_Ha_y, __global double* dev_Ha_z,
				__global double* dev_Hth_x,	__global double* dev_Hth_y, __global double* dev_Hth_z,
				__global double* dev_Hd_x,	__global double* dev_Hd_y, __global double* dev_Hd_z,
				__global double* dev_Happl_x, __global double* dev_Happl_y, __global double* dev_Happl_z,	
				double hh, __global double* dev_Ku,
				__global double* dev_Ms, __global double* dev_alpha, __global double* dev_gamma)
{
	double theta_tmp = GetElement(dev_theta, i, j, k, Nx, Ny);
	double phi_tmp   = GetElement(dev_phi,   i, j, k, Nx, Ny);
	double Ms = GetElement(dev_Ms, i, j, k, Nx, Ny), 
		   Ku = GetElement(dev_Ku, i, j, k, Nx, Ny), 
		   alpha = GetElement(dev_alpha, i, j, k, Nx, Ny),
		   gamma = GetElement(dev_gamma, i, j, k, Nx, Ny);

	if (fabs(theta_tmp) > 2 * PI)
		theta_tmp = fmod(fabs(theta_tmp), 2 * PI);
	else if (fabs(theta_tmp) > PI && fabs(theta_tmp) <= 2 * PI)
		theta_tmp = 2 * PI - fabs(theta_tmp);
	else
		theta_tmp = fabs(theta_tmp);

	phi_tmp = phi_tmp - (int)((phi_tmp/2/PI)) * 2 * PI;   


	double theta_add_dtheta = theta_tmp + hh * GetElement(dev_x_theta, i, j, k, Nx, Ny);
	double phi_add_dphi     = phi_tmp   + hh * GetElement(dev_x_phi  , i, j, k, Nx, Ny);

	return  gamma * (alpha * cos(theta_add_dtheta) * cos(phi_add_dphi) - sin(phi_add_dphi))*
		    (GetElement(dev_Ha_x, i, j, k, Nx, Ny) + GetElement(dev_Hth_x, i, j, k, Nx, Ny) + GetElement(dev_Hd_x, i, j, k, Nx, Ny) + GetElement(dev_Happl_x, i, j, k, Nx, Ny)) + 
		    gamma * (alpha * cos(theta_add_dtheta) * sin(phi_add_dphi) + cos(phi_add_dphi))*
		    (GetElement(dev_Ha_y, i, j, k, Nx, Ny) + GetElement(dev_Hth_y, i, j, k, Nx, Ny) + GetElement(dev_Hd_y, i, j, k, Nx, Ny) + GetElement(dev_Happl_y, i, j, k, Nx, Ny)) -
	   	    gamma * alpha * sin(theta_add_dtheta) * (2*Ku/Ms*cos(theta_add_dtheta) + GetElement(dev_Ha_z, i, j, k, Nx, Ny) + GetElement(dev_Hth_z, i, j, k, Nx, Ny) + GetElement(dev_Hd_z, i, j, k, Nx, Ny) + GetElement(dev_Happl_z, i, j, k, Nx, Ny));   

} 

double dphi_dt(int i, int j, int k, 
			   __global double* dev_theta, __global double* dev_phi, 
			   __global double* dev_x_theta, __global double* dev_x_phi,
			   __global double* dev_Ha_x, __global double* dev_Ha_y, __global double* dev_Ha_z,
			   __global double* dev_Hth_x, __global double* dev_Hth_y, __global double* dev_Hth_z,
			   __global double* dev_Hd_x, __global double* dev_Hd_y, __global double* dev_Hd_z,
			   __global double* dev_Happl_x, __global double* dev_Happl_y, __global double* dev_Happl_z,
			   double hh, __global double* dev_Ku,
			   __global double* dev_Ms, __global double* dev_alpha, __global double* dev_gamma)
{
	double theta_tmp = GetElement(dev_theta, i, j, k, Nx, Ny);
	double phi_tmp   = GetElement(dev_phi,   i, j, k, Nx, Ny);
	double Ms = GetElement(dev_Ms, i, j, k, Nx, Ny), 
		   Ku = GetElement(dev_Ku, i, j, k, Nx, Ny), 
		   alpha = GetElement(dev_alpha, i, j, k, Nx, Ny),
		   gamma = GetElement(dev_gamma, i, j, k, Nx, Ny);

	if (fabs(theta_tmp) > 2 * PI)
		theta_tmp = fmod(fabs(theta_tmp), 2 * PI);
	else if (fabs(theta_tmp) > PI && fabs(theta_tmp) <= 2 * PI)
		theta_tmp = 2 * PI - fabs(theta_tmp);
	else
		theta_tmp = fabs(theta_tmp);

	phi_tmp = phi_tmp - (int)((phi_tmp/2/PI)) * 2 * PI;   


	double theta_add_dtheta = theta_tmp + hh * GetElement(dev_x_theta, i, j, k, Nx, Ny);
	double phi_add_dphi     = phi_tmp   + hh * GetElement(dev_x_phi  , i, j, k, Nx, Ny);


	return  gamma * (-alpha*sin(phi_add_dphi)/sin(theta_add_dtheta) - cos(theta_add_dtheta)/sin(theta_add_dtheta)*cos(phi_add_dphi)) * 
		            (GetElement(dev_Ha_x, i, j, k, Nx, Ny) + GetElement(dev_Hth_x, i, j, k, Nx, Ny) + GetElement(dev_Hd_x, i, j, k, Nx, Ny) + GetElement(dev_Happl_x, i, j, k, Nx, Ny)) + 
  		    gamma * ( alpha*cos(phi_add_dphi)/sin(theta_add_dtheta) - cos(theta_add_dtheta)/sin(theta_add_dtheta)*sin(phi_add_dphi)) *
	   	            (GetElement(dev_Ha_y, i, j, k, Nx, Ny) + GetElement(dev_Hth_y, i, j, k, Nx, Ny) + GetElement(dev_Hd_y, i, j, k, Nx, Ny) + GetElement(dev_Happl_y, i, j, k, Nx, Ny)) +
 		    gamma * (2*Ku/Ms*cos(theta_add_dtheta) + GetElement(dev_Ha_z, i, j, k, Nx, Ny) + GetElement(dev_Hth_z, i, j, k, Nx, Ny) + GetElement(dev_Hd_z, i, j, k, Nx, Ny) + GetElement(dev_Happl_z, i, j, k, Nx, Ny));  
}

__kernel void Kernel_Initialization(__global double* dev_theta, __global double* dev_phi,
									__global double* dev_a_theta, __global double* dev_b_theta, 
									__global double* dev_c_theta, __global double* dev_d_theta,
									__global double* dev_a_phi, __global double* dev_b_phi, 
									__global double* dev_c_phi, __global double* dev_d_phi,
									__global double* dev_Ha_x, __global double* dev_Ha_y, __global double* dev_Ha_z, 
									__global double* dev_Hth_x, __global double* dev_Hth_y, __global double* dev_Hth_z,
									__global double* dev_Hk_x, __global double* dev_Hk_y, __global double* dev_Hk_z,
									__global double* dev_Hd_x, __global double* dev_Hd_y, __global double* dev_Hd_z,
									__global double* dev_d_theta_d_t, __global double* dev_d_phi_d_t,
									__global int* dev_indicator1, __global int* dev_indicator3)
{
	/*int x = blockIdx.x * BLOCK_SIZE_X + threadIdx.x,
	    y = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y,
		z = blockIdx.z * BLOCK_SIZE_Z + threadIdx.z; */
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	//printf("%d ", dev_indicator3[x * Ny * Nz + y * Nz + z]);
	//printf("%f ", dev_theta[x * Ny * Nz + y * Nz + z]);

	/*if (GetElement_Int(dev_indicator1, x, y, z, Nx, Ny) == 3 && GetElement_Int(dev_indicator3, x, y, z, Nx, Ny) == 1)
	{SetElement(dev_theta,       x, y, z, Nx, Ny, Ini_THETA_Up);}
	if (GetElement_Int(dev_indicator1, x, y, z, Nx, Ny) == 3 && GetElement_Int(dev_indicator3, x, y, z, Nx, Ny) == 2)
	{SetElement(dev_theta,       x, y, z, Nx, Ny, Ini_THETA_Down);}
	if (GetElement_Int(dev_indicator1, x, y, z, Nx, Ny) == 3 && GetElement_Int(dev_indicator3, x, y, z, Nx, Ny) == 0)
	{SetElement(dev_theta,       x, y, z, Nx, Ny, Ini_THETA_Down);}*/

	if (GetElement_Int(dev_indicator3, x, y, z, Nx, Ny) == 1)
	{SetElement(dev_theta,       x, y, z, Nx, Ny, Ini_THETA_Up);}
	if (GetElement_Int(dev_indicator3, x, y, z, Nx, Ny) == 2)
	{SetElement(dev_theta,       x, y, z, Nx, Ny, Ini_THETA_Down);}
	if (GetElement_Int(dev_indicator3, x, y, z, Nx, Ny) == 0)
	{SetElement(dev_theta,       x, y, z, Nx, Ny, PI/2+0.01);}

	SetElement(dev_phi,         x, y, z, Nx, Ny, 0.);
	SetElement(dev_a_theta,     x, y, z, Nx, Ny, 0.);
	SetElement(dev_b_theta,     x, y, z, Nx, Ny, 0.);
	SetElement(dev_c_theta,     x, y, z, Nx, Ny, 0.);
	SetElement(dev_d_theta,     x, y, z, Nx, Ny, 0.);
	SetElement(dev_a_phi,       x, y, z, Nx, Ny, 0.);
	SetElement(dev_b_phi,       x, y, z, Nx, Ny, 0.);
	SetElement(dev_c_phi,       x, y, z, Nx, Ny, 0.);
	SetElement(dev_d_phi,       x, y, z, Nx, Ny, 0.);
	SetElement(dev_Ha_x,        x, y, z, Nx, Ny, 0.);
	SetElement(dev_Ha_y,        x, y, z, Nx, Ny, 0.);
	SetElement(dev_Ha_z,        x, y, z, Nx, Ny, 0.);
	SetElement(dev_Hth_x,       x, y, z, Nx, Ny, 0.);
	SetElement(dev_Hth_y,       x, y, z, Nx, Ny, 0.);
	SetElement(dev_Hth_z,       x, y, z, Nx, Ny, 0.);
	SetElement(dev_Hk_x,        x, y, z, Nx, Ny, 0.);
	SetElement(dev_Hk_y,        x, y, z, Nx, Ny, 0.);
	SetElement(dev_Hk_z,        x, y, z, Nx, Ny, 0.);
	SetElement(dev_Hd_x,        x, y, z, Nx, Ny, 0.);
	SetElement(dev_Hd_y,        x, y, z, Nx, Ny, 0.);
	SetElement(dev_Hd_z,        x, y, z, Nx, Ny, 0.);
	SetElement(dev_d_theta_d_t, x, y, z, Nx, Ny, 0.);
	SetElement(dev_d_phi_d_t,   x, y, z, Nx, Ny, 0.);

	//printf("%f ", dev_theta[x + y * Nx + z * Nx * Ny]);
}

__kernel void Kernel_dev_indicator_with_apron(__global int* dev_indicator1, __global int* dev_indicator1_temp, 
												__global int* dev_indicator2, __global int* dev_indicator2_temp)
{
	// int x = blockIdx.x * BLOCK_SIZE_X + threadIdx.x,
	//     y = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y,
	// 	   z = blockIdx.z * BLOCK_SIZE_Z + threadIdx.z;
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	int temp1, temp2;
	
	temp1 = GetElement_Int(dev_indicator1, x, y, z, Nx, Ny);
	temp2 = GetElement_Int(dev_indicator2, x, y, z, Nx, Ny);
	//__syncthreads();

	// Inner cube
	SetElement_Int(dev_indicator1_temp, x+1, y+1, z+1, Nx+2, Ny+2, temp1);
	SetElement_Int(dev_indicator2_temp, x+1, y+1, z+1, Nx+2, Ny+2, temp2);
	//__syncthreads();

	// up & down faces
	if (z == Nz-1) {
		SetElement_Int(dev_indicator1_temp, x+1, y+1, 0, Nx+2, Ny+2, 0);
		SetElement_Int(dev_indicator2_temp, x+1, y+1, 0, Nx+2, Ny+2, 0);
	}
	if (z == 0) {
		SetElement_Int(dev_indicator1_temp, x+1, y+1, Nz+1, Nx+2, Ny+2, 0);
		SetElement_Int(dev_indicator2_temp, x+1, y+1, Nz+1, Nx+2, Ny+2, 0);
	}
	//__syncthreads();

	// left and right faces
	if (x == Nx-1) {
		SetElement_Int(dev_indicator1_temp, 0, y+1, z+1, Nx+2, Ny+2, temp1);
		SetElement_Int(dev_indicator2_temp, 0, y+1, z+1, Nx+2, Ny+2, temp2);
	}
	if (x == 0) {
		SetElement_Int(dev_indicator1_temp, Nx+1, y+1, z+1, Nx+2, Ny+2, temp1);
		SetElement_Int(dev_indicator2_temp, Nx+1, y+1, z+1, Nx+2, Ny+2, temp2);
	}
	//__syncthreads();

	// front and back faces
	if (y == Ny-1) {
		SetElement_Int(dev_indicator1_temp, x+1, 0, z+1, Nx+2, Ny+2, temp1);
		SetElement_Int(dev_indicator2_temp, x+1, 0, z+1, Nx+2, Ny+2, temp2);
	}
	if (y == 0) {
		SetElement_Int(dev_indicator1_temp, x+1, Ny+1, z+1, Nx+2, Ny+2, temp1);
		SetElement_Int(dev_indicator2_temp, x+1, Ny+1, z+1, Nx+2, Ny+2, temp2);
	}
	//__syncthreads();

}

__kernel void Kernel_d_theta_phi_d_t(__global double* dev_d_theta_d_t, __global double* dev_d_phi_d_t,
									 __global double* dev_theta,       __global double* dev_phi, 
									 __global double* dev_a_theta,     __global double* dev_a_phi,
									 __global double* dev_Ha_x,        __global double* dev_Ha_y,    __global double* dev_Ha_z,
									 __global double* dev_Hth_x,       __global double* dev_Hth_y,   __global double* dev_Hth_z,
									 __global double* dev_Hd_x,        __global double* dev_Hd_y,    __global double* dev_Hd_z,
									 __global double* dev_Happl_x,     __global double* dev_Happl_y, __global double* dev_Happl_z,
									 __global double* dev_Ku,          __global double* dev_Ms,      
								     __global double* dev_alpha,  	   __global double* dev_gamma)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	SetElement(dev_d_theta_d_t, x, y, z, Nx, Ny, dtheta_dt(x, y, z, 
		                                                   dev_theta, dev_phi, 
													 	   dev_a_theta, dev_a_phi,
														   dev_Ha_x, dev_Ha_y, dev_Ha_z,
									                       dev_Hth_x, dev_Hth_y, dev_Hth_z,
														   dev_Hd_x, dev_Hd_y, dev_Hd_z,
														   dev_Happl_x, dev_Happl_y, dev_Happl_z,
														   ZERO, dev_Ku, dev_Ms, dev_alpha, dev_gamma));
	SetElement(dev_d_phi_d_t,   x, y, z, Nx, Ny, dphi_dt  (x, y, z, 
		                                                   dev_theta, dev_phi, 
														   dev_a_theta, dev_a_phi,
														   dev_Ha_x, dev_Ha_y, dev_Ha_z,
									                       dev_Hth_x, dev_Hth_y, dev_Hth_z,
														   dev_Hd_x, dev_Hd_y, dev_Hd_z,
														   dev_Happl_x, dev_Happl_y, dev_Happl_z,
														   ZERO, dev_Ku, dev_Ms, dev_alpha, dev_gamma));

	//__syncthreads();
}

__kernel void Kernel_M_A_with_apron(__global double* dev_M_temp_x, __global double* dev_M_temp_y, __global double* dev_M_temp_z, 
									__global double* dev_Ms_temp, __global double* dev_Aex1_temp, __global double* dev_Aex2_temp,
									__global double* dev_theta,    __global double* dev_x_theta,  __global double* dev_phi, __global double* dev_x_phi,
									__global double* dev_Ms, __global double* dev_Aex1, __global double* dev_Aex2, const double hh)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	double temp_x, temp_y, temp_z;
	double Ms_temp = GetElement(dev_Ms, x, y, z, Nx, Ny), Aex1_temp = GetElement(dev_Aex1, x, y, z, Nx, Ny),
	Aex2_temp = GetElement(dev_Aex2, x, y, z, Nx, Ny);
	
	temp_x = Ms_temp * sin(GetElement(dev_theta, x, y, z, Nx, Ny) + hh * GetElement(dev_x_theta, x, y, z, Nx, Ny)) * 
		               cos(GetElement(dev_phi,   x, y, z, Nx, Ny) + hh * GetElement(dev_x_phi,   x, y, z, Nx, Ny));
	temp_y = Ms_temp * sin(GetElement(dev_theta, x, y, z, Nx, Ny) + hh * GetElement(dev_x_theta, x, y, z, Nx, Ny)) * 
		               sin(GetElement(dev_phi,   x, y, z, Nx, Ny) + hh * GetElement(dev_x_phi,   x, y, z, Nx, Ny));
	temp_z = Ms_temp * cos(GetElement(dev_theta, x, y, z, Nx, Ny) + hh * GetElement(dev_x_theta, x, y, z, Nx, Ny));

	// //__syncthreads();
	

	// // inner cube of M_temp is given values
	SetElement(dev_M_temp_x, x+1, y+1, z+1, Nx+2, Ny+2, temp_x);
	SetElement(dev_M_temp_y, x+1, y+1, z+1, Nx+2, Ny+2, temp_y);
	SetElement(dev_M_temp_z, x+1, y+1, z+1, Nx+2, Ny+2, temp_z);
	SetElement(dev_Ms_temp,  x+1, y+1, z+1, Nx+2, Ny+2, Ms_temp);
	SetElement(dev_Aex1_temp, x+1, y+1, z+1, Nx+2, Ny+2, Aex1_temp);
	SetElement(dev_Aex2_temp, x+1, y+1, z+1, Nx+2, Ny+2, Aex2_temp);
	
	//__syncthreads();
	// up & down faces are given values
	if (z == Nz-1) {
	//	temp_x = GetElement(dev_M_temp_x, x+1, y+1, z+1, Nx+2, Ny+2);
		SetElement(dev_M_temp_x, x+1, y+1, 0, Nx+2, Ny+2, 0.);
	//	temp_y = GetElement(dev_M_temp_y, x+1, y+1, z+1, Nx+2, Ny+2);
		SetElement(dev_M_temp_y, x+1, y+1, 0, Nx+2, Ny+2, 0.);
	//	temp_z = GetElement(dev_M_temp_z, x+1, y+1, z+1, Nx+2, Ny+2);
		SetElement(dev_M_temp_z, x+1, y+1, 0, Nx+2, Ny+2, 0.);
		SetElement(dev_Ms_temp, x+1, y+1, 0, Nx+2, Ny+2, 0.);
		SetElement(dev_Aex1_temp, x+1, y+1, 0, Nx+2, Ny+2, 0.);
		SetElement(dev_Aex2_temp, x+1, y+1, 0, Nx+2, Ny+2, 0.);
	}
	if (z == 0) {
	//	temp_x = GetElement(dev_M_temp_x, x+1, y+1, z+1, Nx+2, Ny+2);
		SetElement(dev_M_temp_x, x+1, y+1, Nz+1, Nx+2, Ny+2, 0.);
	//	temp_y = GetElement(dev_M_temp_y, x+1, y+1, z+1, Nx+2, Ny+2);
		SetElement(dev_M_temp_y, x+1, y+1, Nz+1, Nx+2, Ny+2, 0.);
	//	temp_z = GetElement(dev_M_temp_z, x+1, y+1, z+1, Nx+2, Ny+2);
		SetElement(dev_M_temp_z, x+1, y+1, Nz+1, Nx+2, Ny+2, 0.);
		SetElement(dev_Ms_temp, x+1, y+1, Nz+1, Nx+2, Ny+2, 0.);
		SetElement(dev_Aex1_temp, x+1, y+1, Nz+1, Nx+2, Ny+2, 0.);
		SetElement(dev_Aex2_temp, x+1, y+1, Nz+1, Nx+2, Ny+2, 0.);
	}
	//__syncthreads();
	// left and right faces are given values
	if (x == Nx-1) {
		/*temp_x = GetElement(dev_M_temp_x, x+1, y+1, z+1, Nx+2, Ny+2);
		SetElement(dev_M_temp_x, 0, y+1, z+1, Nx+2, Ny+2, temp_x);*/
		SetElement(dev_M_temp_x, 0, y+1, z+1, Nx+2, Ny+2, 0.);
		/*temp_y = GetElement(dev_M_temp_y, x+1, y+1, z+1, Nx+2, Ny+2);
		SetElement(dev_M_temp_y, 0, y+1, z+1, Nx+2, Ny+2, temp_y);*/
		SetElement(dev_M_temp_y, 0, y+1, z+1, Nx+2, Ny+2, 0.);
		/*temp_z = GetElement(dev_M_temp_z, x+1, y+1, z+1, Nx+2, Ny+2);
		SetElement(dev_M_temp_z, 0, y+1, z+1, Nx+2, Ny+2, temp_z);*/
		SetElement(dev_M_temp_z, 0, y+1, z+1, Nx+2, Ny+2, 0.);
		//SetElement(dev_Ms_temp, 0, y+1, z+1, Nx+2, Ny+2, GetElement(dev_Ms_temp, x+1, y+1, z+1, Nx+2, Ny+2));
		SetElement(dev_Ms_temp, 0, y+1, z+1, Nx+2, Ny+2, GetElement(dev_Ms, x, y, z, Nx, Ny));
        //SetElement(dev_Aex1_temp, 0, y+1, z+1, Nx+2, Ny+2, GetElement(dev_Aex1_temp, x+1, y+1, z+1, Nx+2, Ny+2));
        SetElement(dev_Aex1_temp, 0, y+1, z+1, Nx+2, Ny+2, GetElement(dev_Aex1, x, y, z, Nx, Ny));
		//SetElement(dev_Aex2_temp, 0, y+1, z+1, Nx+2, Ny+2, GetElement(dev_Aex2_temp, x+1, y+1, z+1, Nx+2, Ny+2));
		SetElement(dev_Aex2_temp, 0, y+1, z+1, Nx+2, Ny+2, GetElement(dev_Aex2, x, y, z, Nx, Ny));		
	}
	if (x == 0) {
		/*temp_x = GetElement(dev_M_temp_x, x+1, y+1, z+1, Nx+2, Ny+2);
		SetElement(dev_M_temp_x, Nx+1, y+1, z+1, Nx+2, Ny+2, temp_x);*/
		SetElement(dev_M_temp_x, Nx+1, y+1, z+1, Nx+2, Ny+2, 0.);
		/*temp_y = GetElement(dev_M_temp_y, x+1, y+1, z+1, Nx+2, Ny+2);
		SetElement(dev_M_temp_y, Nx+1, y+1, z+1, Nx+2, Ny+2, temp_y);*/
		SetElement(dev_M_temp_y, Nx+1, y+1, z+1, Nx+2, Ny+2, 0.);
		/*temp_z = GetElement(dev_M_temp_z, x+1, y+1, z+1, Nx+2, Ny+2);
		SetElement(dev_M_temp_z, Nx+1, y+1, z+1, Nx+2, Ny+2, temp_z);*/
		SetElement(dev_M_temp_z, Nx+1, y+1, z+1, Nx+2, Ny+2, 0.);
		//SetElement(dev_Ms_temp, Nx+1, y+1, z+1, Nx+2, Ny+2, GetElement(dev_Ms_temp, x+1, y+1, z+1, Nx+2, Ny+2));
		SetElement(dev_Ms_temp, Nx+1, y+1, z+1, Nx+2, Ny+2, GetElement(dev_Ms, x, y, z, Nx, Ny));
		//SetElement(dev_Aex1_temp, Nx+1, y+1, z+1, Nx+2, Ny+2, GetElement(dev_Aex1_temp, x+1, y+1, z+1, Nx+2, Ny+2));
		SetElement(dev_Aex1_temp, Nx+1, y+1, z+1, Nx+2, Ny+2, GetElement(dev_Aex1, x, y, z, Nx, Ny));
		//SetElement(dev_Aex2_temp, Nx+1, y+1, z+1, Nx+2, Ny+2, GetElement(dev_Aex2_temp, x+1, y+1, z+1, Nx+2, Ny+2));
		SetElement(dev_Aex2_temp, Nx+1, y+1, z+1, Nx+2, Ny+2, GetElement(dev_Aex2, x, y, z, Nx, Ny));
	}
	//__syncthreads();

	// front and back faces are given values
	if (y == Ny-1) {
		/*temp_x = GetElement(dev_M_temp_x, x+1, y+1, z+1, Nx+2, Ny+2);
		SetElement(dev_M_temp_x, x+1, 0, z+1, Nx+2, Ny+2, temp_x);*/
		SetElement(dev_M_temp_x, x+1, 0, z+1, Nx+2, Ny+2, 0.);
		/*temp_y = GetElement(dev_M_temp_y, x+1, y+1, z+1, Nx+2, Ny+2);
		SetElement(dev_M_temp_y, x+1, 0, z+1, Nx+2, Ny+2, temp_y);*/
		SetElement(dev_M_temp_y, x+1, 0, z+1, Nx+2, Ny+2, 0.);
		/*temp_z = GetElement(dev_M_temp_z, x+1, y+1, z+1, Nx+2, Ny+2);
		SetElement(dev_M_temp_z, x+1, 0, z+1, Nx+2, Ny+2, temp_z);*/
		SetElement(dev_M_temp_z, x+1, 0, z+1, Nx+2, Ny+2, 0.);
		//SetElement(dev_Ms_temp, x+1, 0, z+1, Nx+2, Ny+2, GetElement(dev_Ms_temp, x+1, y+1, z+1, Nx+2, Ny+2));
		SetElement(dev_Ms_temp, x+1, 0, z+1, Nx+2, Ny+2, GetElement(dev_Ms, x, y, z, Nx, Ny));
		//SetElement(dev_Aex1_temp, x+1, 0, z+1, Nx+2, Ny+2, GetElement(dev_Aex1_temp, x+1, y+1, z+1, Nx+2, Ny+2));
		SetElement(dev_Aex1_temp, x+1, 0, z+1, Nx+2, Ny+2, GetElement(dev_Aex1, x, y, z, Nx, Ny));
		//SetElement(dev_Aex2_temp, x+1, 0, z+1, Nx+2, Ny+2, GetElement(dev_Aex2_temp, x+1, y+1, z+1, Nx+2, Ny+2));
		SetElement(dev_Aex2_temp, x+1, 0, z+1, Nx+2, Ny+2, GetElement(dev_Aex2, x, y, z, Nx, Ny));
	}
	if (y == 0) {
		/*temp_x = GetElement(dev_M_temp_x, x+1, y+1, z+1, Nx+2, Ny+2);
		SetElement(dev_M_temp_x, x+1, Ny+1, z+1, Nx+2, Ny+2, temp_x);*/
		SetElement(dev_M_temp_x, x+1, Ny+1, z+1, Nx+2, Ny+2, 0.);
		/*temp_y = GetElement(dev_M_temp_y, x+1, y+1, z+1, Nx+2, Ny+2);
		SetElement(dev_M_temp_y, x+1, Ny+1, z+1, Nx+2, Ny+2, temp_y);*/
		SetElement(dev_M_temp_y, x+1, Ny+1, z+1, Nx+2, Ny+2, 0.);
		/*temp_z = GetElement(dev_M_temp_z, x+1, y+1, z+1, Nx+2, Ny+2);
		SetElement(dev_M_temp_z, x+1, Ny+1, z+1, Nx+2, Ny+2, temp_z);*/
		SetElement(dev_M_temp_z, x+1, Ny+1, z+1, Nx+2, Ny+2, 0.);
		//SetElement(dev_Ms_temp, x+1, Ny+1, z+1, Nx+2, Ny+2, GetElement(dev_Ms_temp, x+1, y+1, z+1, Nx+2, Ny+2));
		SetElement(dev_Ms_temp, x+1, Ny+1, z+1, Nx+2, Ny+2, GetElement(dev_Ms, x, y, z, Nx, Ny));
		//SetElement(dev_Aex1_temp, x+1, Ny+1, z+1, Nx+2, Ny+2, GetElement(dev_Aex1_temp, x+1, y+1, z+1, Nx+2, Ny+2));
		SetElement(dev_Aex1_temp, x+1, Ny+1, z+1, Nx+2, Ny+2, GetElement(dev_Aex1, x, y, z, Nx, Ny));
		//SetElement(dev_Aex2_temp, x+1, Ny+1, z+1, Nx+2, Ny+2, GetElement(dev_Aex2_temp, x+1, y+1, z+1, Nx+2, Ny+2));
		SetElement(dev_Aex2_temp, x+1, Ny+1, z+1, Nx+2, Ny+2, GetElement(dev_Aex2, x, y, z, Nx, Ny));
	}
	//__syncthreads();

	// /* lines */
	// /* points */
}

__kernel void Kernel_Ha_with_apron(__global double* dev_M_temp_x, __global double* dev_M_temp_y, __global double* dev_M_temp_z, 
								   __global double* dev_Ms_temp, __global double* dev_Aex1_temp, __global double* dev_Aex2_temp,
								   __global double* dev_Ha_x,     __global double* dev_Ha_y,     __global double* dev_Ha_z,
								   __global int* dev_indicator1_temp, __global int* dev_indicator2_temp, 
								   __global double* dev_Ms, __global double* dev_Aex1, __global double* dev_Aex2, const double delta_x)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	int i_tmp = x + 1, 
		j_tmp = y + 1, 
		k_tmp = z + 1;
	double temp_x, temp_y, temp_z;

	double Ms = GetElement(dev_Ms, x, y, z, Nx, Ny),      Aex1 = GetElement(dev_Aex1, x, y, z, Nx, Ny),  Aex2 = GetElement(dev_Aex2, x, y, z, Nx, Ny),
	       Ms_xp = GetElement(dev_Ms_temp, i_tmp+1, j_tmp, k_tmp, Nx+2, Ny+2), Aex_xp = GetElement(dev_Aex2_temp, i_tmp+1, j_tmp, k_tmp, Nx+2, Ny+2),
		   Ms_xm = GetElement(dev_Ms_temp, i_tmp-1, j_tmp, k_tmp, Nx+2, Ny+2), Aex_xm = GetElement(dev_Aex2_temp, i_tmp-1, j_tmp, k_tmp, Nx+2, Ny+2),
		   Ms_yp = GetElement(dev_Ms_temp, i_tmp, j_tmp+1, k_tmp, Nx+2, Ny+2), Aex_yp = GetElement(dev_Aex2_temp, i_tmp, j_tmp+1, k_tmp, Nx+2, Ny+2),
		   Ms_ym = GetElement(dev_Ms_temp, i_tmp, j_tmp-1, k_tmp, Nx+2, Ny+2), Aex_ym = GetElement(dev_Aex2_temp, i_tmp, j_tmp-1, k_tmp, Nx+2, Ny+2),
		   Ms_zp = GetElement(dev_Ms_temp, i_tmp, j_tmp, k_tmp+1, Nx+2, Ny+2), Aex_zp = GetElement(dev_Aex1_temp, i_tmp, j_tmp, k_tmp+1, Nx+2, Ny+2),
		   Ms_zm = GetElement(dev_Ms_temp, i_tmp, j_tmp, k_tmp-1, Nx+2, Ny+2), Aex_zm = GetElement(dev_Aex1_temp, i_tmp, j_tmp, k_tmp-1, Nx+2, Ny+2);

	double Ms_XP_sqr, Ms_XM_sqr, Ms_YP_sqr, Ms_YM_sqr, Ms_ZP_sqr, Ms_ZM_sqr, Ms_sqr;

		
	/* Write to Ha field. */
	double Ms_XP,  Ms_XM, 
		   Ms_YP,  Ms_YM,
		   Ms_ZP,  Ms_ZM,
	       Aex_XP, Aex_XM, 
		   Aex_YP, Aex_YM, 
		   Aex_ZP, Aex_ZM,
		   Aex_tmp, weight = 0.02, // Grain boundary exchange (x/y/z's plus/minus)
		   exch_Cr = 0.0;
	
	// Break the exchange when confronting grain boundaries
	Aex_tmp = 0.0;
	// FePt Layer
	if (GetElement_Int(dev_indicator1_temp, i_tmp, j_tmp, k_tmp, Nx+2, Ny+2) != 0) {
		if ((GetElement_Int(dev_indicator1_temp, i_tmp+1, j_tmp, k_tmp, Nx+2, Ny+2) == 0)) 
		{Aex_XP = Aex_tmp;
		 Ms_XP = Ms;}
		else if ((GetElement_Int(dev_indicator2_temp, i_tmp,   j_tmp, k_tmp, Nx+2, Ny+2) != 
			      GetElement_Int(dev_indicator2_temp, i_tmp+1, j_tmp, k_tmp, Nx+2, Ny+2)))
		{Aex_XP = weight*(Aex2+Aex_xp)/2;
		 Ms_XP  = pow(Ms*Ms_xp, 0.5);}
		else 
		{Aex_XP = (Aex2+Aex_xp)/2;
		 Ms_XP  = pow(Ms*Ms_xp, 0.5);}

		if ((GetElement_Int(dev_indicator1_temp, i_tmp-1, j_tmp, k_tmp, Nx+2, Ny+2) == 0)) 
		{Aex_XM = Aex_tmp;
		 Ms_XM = Ms;}
		else if ((GetElement_Int(dev_indicator2_temp, i_tmp,   j_tmp, k_tmp, Nx+2, Ny+2) != 
			      GetElement_Int(dev_indicator2_temp, i_tmp-1, j_tmp, k_tmp, Nx+2, Ny+2)))
		{Aex_XM = weight*(Aex2+Aex_xm)/2;
		 Ms_XM  = pow(Ms*Ms_xm, 0.5);}
		else 
		{Aex_XM = (Aex2+Aex_xm)/2;
		 Ms_XM  = pow(Ms*Ms_xm, 0.5);}
		
		if ((GetElement_Int(dev_indicator1_temp, i_tmp, j_tmp+1, k_tmp, Nx+2, Ny+2) == 0)) 
		{Aex_YP = Aex_tmp;
		 Ms_YP = Ms;}
		else if ((GetElement_Int(dev_indicator2_temp, i_tmp,   j_tmp, k_tmp, Nx+2, Ny+2) != 
			      GetElement_Int(dev_indicator2_temp, i_tmp, j_tmp+1, k_tmp, Nx+2, Ny+2)))
		{Aex_YP = weight*(Aex2+Aex_yp)/2;
		 Ms_YP  = pow(Ms*Ms_yp, 0.5);}
		else 
		{Aex_YP = (Aex2+Aex_yp)/2;
		 Ms_YP  = pow(Ms*Ms_yp, 0.5);}
		
		if ((GetElement_Int(dev_indicator1_temp, i_tmp, j_tmp-1, k_tmp, Nx+2, Ny+2) == 0)) 
		{Aex_YM = Aex_tmp;
		 Ms_YM = Ms;}
		else if ((GetElement_Int(dev_indicator2_temp, i_tmp,   j_tmp, k_tmp, Nx+2, Ny+2) != 
			      GetElement_Int(dev_indicator2_temp, i_tmp, j_tmp-1, k_tmp, Nx+2, Ny+2)))
		{Aex_YM = weight*(Aex2+Aex_ym)/2;
		 Ms_YM  = pow(Ms*Ms_ym, 0.5);}
		else 
		{Aex_YM = (Aex2+Aex_ym)/2;
		 Ms_YM  = pow(Ms*Ms_ym, 0.5);}

		Aex_ZP = Aex1;
		Aex_ZM = Aex1;
		Ms_ZP = Ms;
		Ms_ZM = Ms;
		
	}
	else 
	{
		Aex_XP = Aex_tmp; 
		Aex_XM = Aex_tmp; 
		Aex_YP = Aex_tmp; 
		Aex_YM = Aex_tmp; 
		Aex_ZP = Aex_tmp; 
		Aex_ZM = Aex_tmp;
		Ms_XP = Ms;
		Ms_XM = Ms;
		Ms_YP = Ms;
		Ms_YM = Ms;
		Ms_ZP = Ms;
		Ms_ZM = Ms;
	}

	//__syncthreads();
	//	Aex_xp = Aex; Aex_xm = Aex; Aex_yp = Aex; Aex_ym = Aex; Aex_zp = Aex; Aex_zm = Aex;
	
	Ms_XP_sqr = pow(Ms_XP, 2.0);
	Ms_XM_sqr = pow(Ms_XM, 2.0);
	Ms_YP_sqr = pow(Ms_YP, 2.0);
	Ms_YM_sqr = pow(Ms_YM, 2.0);
	Ms_ZP_sqr = pow(Ms_ZP, 2.0);
	Ms_ZM_sqr = pow(Ms_ZM, 2.0);
	
	temp_x = 2 / pow(delta_x, 2.0) * 
				(Aex_XP* GetElement(dev_M_temp_x, i_tmp+1, j_tmp,   k_tmp,   Nx+2, Ny+2)/ Ms_XP_sqr + 
				 Aex_XM* GetElement(dev_M_temp_x, i_tmp-1, j_tmp,   k_tmp,   Nx+2, Ny+2)/ Ms_XM_sqr + 
				 Aex_YP* GetElement(dev_M_temp_x, i_tmp,   j_tmp+1, k_tmp,   Nx+2, Ny+2)/ Ms_YP_sqr + 
				 Aex_YM* GetElement(dev_M_temp_x, i_tmp,   j_tmp-1, k_tmp,   Nx+2, Ny+2)/ Ms_YM_sqr + 
				 Aex_ZP* GetElement(dev_M_temp_x, i_tmp,   j_tmp,   k_tmp+1, Nx+2, Ny+2)/ Ms_ZP_sqr    + 
				 Aex_ZM* GetElement(dev_M_temp_x, i_tmp,   j_tmp,   k_tmp-1, Nx+2, Ny+2)/ Ms_ZM_sqr);      
	SetElement(dev_Ha_x, x, y, z, Nx, Ny, temp_x);

	temp_y = 2 / pow(delta_x, 2.0) * 
				(Aex_XP* GetElement(dev_M_temp_y, i_tmp+1, j_tmp,   k_tmp,   Nx+2, Ny+2)/ Ms_XP_sqr + 
				 Aex_XM* GetElement(dev_M_temp_y, i_tmp-1, j_tmp,   k_tmp,   Nx+2, Ny+2)/ Ms_XM_sqr + 
				 Aex_YP* GetElement(dev_M_temp_y, i_tmp,   j_tmp+1, k_tmp,   Nx+2, Ny+2)/ Ms_YP_sqr + 
				 Aex_YM* GetElement(dev_M_temp_y, i_tmp,   j_tmp-1, k_tmp,   Nx+2, Ny+2)/ Ms_YM_sqr + 
				 Aex_ZP* GetElement(dev_M_temp_y, i_tmp,   j_tmp,   k_tmp+1, Nx+2, Ny+2)/ Ms_ZP_sqr    + 
				 Aex_ZM* GetElement(dev_M_temp_y, i_tmp,   j_tmp,   k_tmp-1, Nx+2, Ny+2)/ Ms_ZM_sqr);      
	SetElement(dev_Ha_y, x, y, z, Nx, Ny, temp_y);

	temp_z = 2 / pow(delta_x, 2.0) * 
			    (Aex_XP* GetElement(dev_M_temp_z, i_tmp+1, j_tmp,   k_tmp,   Nx+2, Ny+2)/ Ms_XP_sqr + 
			     Aex_XM* GetElement(dev_M_temp_z, i_tmp-1, j_tmp,   k_tmp,   Nx+2, Ny+2)/ Ms_XM_sqr + 
			     Aex_YP* GetElement(dev_M_temp_z, i_tmp,   j_tmp+1, k_tmp,   Nx+2, Ny+2)/ Ms_YP_sqr + 
			     Aex_YM* GetElement(dev_M_temp_z, i_tmp,   j_tmp-1, k_tmp,   Nx+2, Ny+2)/ Ms_YM_sqr + 
			     Aex_ZP* GetElement(dev_M_temp_z, i_tmp,   j_tmp,   k_tmp+1, Nx+2, Ny+2)/ Ms_ZP_sqr    + 
			     Aex_ZM* GetElement(dev_M_temp_z, i_tmp,   j_tmp,   k_tmp-1, Nx+2, Ny+2)/ Ms_ZM_sqr);
	SetElement(dev_Ha_z, x, y, z, Nx, Ny, temp_z);

	//__syncthreads();
	
	/////// Watch /////
	//SetElement(dev_watch2, x, y, z, Nx, Ny, GetElement(dev_M_temp_z, i_tmp+1, j_tmp,   k_tmp,   Nx+2, Ny+2));
	//__syncthreads();

} 

__kernel void Kernel_a_theta_phi(__global double* dev_a_theta, __global double* dev_a_phi,
								        __global double* dev_d_theta_d_t, __global double* dev_d_phi_d_t)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	SetElement(dev_a_theta, x, y, z, Nx, Ny, GetElement(dev_d_theta_d_t, x, y, z, Nx, Ny));
	SetElement(dev_a_phi,   x, y, z, Nx, Ny, GetElement(dev_d_phi_d_t,   x, y, z, Nx, Ny));

	//__syncthreads();
}


// // __kernel void Kernel_b_theta_phi(double* dev_b_theta, double* dev_b_phi,
// // 	                                      double* dev_theta, double* dev_phi, 
// // 										  double* dev_a_theta, double* dev_a_phi,
// // 										  double* dev_Ha_x, double* dev_Ha_y, double* dev_Ha_z,
// // 									      double* dev_Hth_x, double* dev_Hth_y, double* dev_Hth_z,
// // 										  double* dev_Hd_x, double* dev_Hd_y, double* dev_Hd_z,
// // 										  double* dev_Happl_x, double* dev_Happl_y, double* dev_Happl_z,
// // 									      double* dev_Ku, double* dev_Ms, double* dev_alpha, double* dev_gamma, double h)
// // {
// // 	int x = get_global_id(0);
// // 	int y = get_global_id(1);
// // 	int z = get_global_id(2);

// // 	SetElement(dev_b_theta, x, y, z, Nx, Ny, dtheta_dt(x, y, z, 
// // 		                                               dev_theta, dev_phi, 
// // 													   dev_a_theta, dev_a_phi,
// // 													   dev_Ha_x, dev_Ha_y, dev_Ha_z,
// // 									                   dev_Hth_x, dev_Hth_y, dev_Hth_z,
// // 													   dev_Hd_x, dev_Hd_y, dev_Hd_z,
// // 													   dev_Happl_x, dev_Happl_y, dev_Happl_z,
// // 													   h/2, dev_Ku, dev_Ms, dev_alpha, dev_gamma));
// // 	SetElement(dev_b_phi,   x, y, z, Nx, Ny, dphi_dt  (x, y, z, 
// // 		                                               dev_theta, dev_phi, 
// // 													   dev_a_theta, dev_a_phi,
// // 													   dev_Ha_x, dev_Ha_y, dev_Ha_z,
// // 									                   dev_Hth_x, dev_Hth_y, dev_Hth_z,
// // 													   dev_Hd_x, dev_Hd_y, dev_Hd_z,
// // 													   dev_Happl_x, dev_Happl_y, dev_Happl_z,
// // 													   h/2, dev_Ku, dev_Ms, dev_alpha, dev_gamma));

// // 	//__syncthreads();
// // }

// // __kernel void Kernel_c_theta_phi(double* dev_c_theta, double* dev_c_phi,
// // 	                                      double* dev_theta, double* dev_phi, 
// // 										  double* dev_b_theta, double* dev_b_phi,
// // 										  double* dev_Ha_x, double* dev_Ha_y, double* dev_Ha_z,
// // 									      double* dev_Hth_x, double* dev_Hth_y, double* dev_Hth_z,
// // 										  double* dev_Hd_x, double* dev_Hd_y, double* dev_Hd_z,
// // 										  double* dev_Happl_x, double* dev_Happl_y, double* dev_Happl_z,
// // 									      double* dev_Ku, double* dev_Ms, double* dev_alpha, double* dev_gamma, double h)
// // {
// // 	int x = get_global_id(0);
// // 	int y = get_global_id(1);
// // 	int z = get_global_id(2);

// // 	SetElement(dev_c_theta, x, y, z, Nx, Ny, dtheta_dt(x, y, z, 
// // 		                                               dev_theta, dev_phi, 
// // 													   dev_b_theta, dev_b_phi,
// // 													   dev_Ha_x, dev_Ha_y, dev_Ha_z,
// // 									                   dev_Hth_x, dev_Hth_y, dev_Hth_z,
// // 													   dev_Hd_x, dev_Hd_y, dev_Hd_z,
// // 													   dev_Happl_x, dev_Happl_y, dev_Happl_z,
// // 													   h/2, dev_Ku, dev_Ms, dev_alpha, dev_gamma));
// // 	SetElement(dev_c_phi,   x, y, z, Nx, Ny, dphi_dt  (x, y, z, 
// // 		                                               dev_theta, dev_phi, 
// // 													   dev_b_theta, dev_b_phi,
// // 													   dev_Ha_x, dev_Ha_y, dev_Ha_z,
// // 									                   dev_Hth_x, dev_Hth_y, dev_Hth_z,
// // 													   dev_Hd_x, dev_Hd_y, dev_Hd_z,
// // 													   dev_Happl_x, dev_Happl_y, dev_Happl_z,
// // 													   h/2, dev_Ku, dev_Ms, dev_alpha, dev_gamma));

// // 	//__syncthreads();
// // }

// // __kernel void Kernel_d_theta_phi(double* dev_d_theta, double* dev_d_phi,
// //    									      double* dev_theta, double* dev_phi, 
// // 										  double* dev_c_theta, double* dev_c_phi,
// // 										  double* dev_Ha_x, double* dev_Ha_y, double* dev_Ha_z,
// // 									      double* dev_Hth_x, double* dev_Hth_y, double* dev_Hth_z,
// // 										  double* dev_Hd_x, double* dev_Hd_y, double* dev_Hd_z,
// // 										  double* dev_Happl_x, double* dev_Happl_y, double* dev_Happl_z,
// // 									      double* dev_Ku, double* dev_Ms, double* dev_alpha, double* dev_gamma, double h)
// // {
// // 	int x = get_global_id(0);
// // 	int y = get_global_id(1);
// // 	int z = get_global_id(2);

// // 	SetElement(dev_d_theta, x, y, z, Nx, Ny, dtheta_dt(x, y, z, 
// // 		                                               dev_theta, dev_phi, 
// // 													   dev_c_theta, dev_c_phi,
// // 													   dev_Ha_x, dev_Ha_y, dev_Ha_z,
// // 									                   dev_Hth_x, dev_Hth_y, dev_Hth_z,
// // 												       dev_Hd_x, dev_Hd_y, dev_Hd_z,
// // 													   dev_Happl_x, dev_Happl_y, dev_Happl_z,
// // 													   h, dev_Ku, dev_Ms, dev_alpha, dev_gamma));
// // 	SetElement(dev_d_phi,   x, y, z, Nx, Ny, dphi_dt  (x, y, z, 
// // 		                                               dev_theta, dev_phi, 
// // 													   dev_c_theta, dev_c_phi,
// // 													   dev_Ha_x, dev_Ha_y, dev_Ha_z,
// // 									                   dev_Hth_x, dev_Hth_y, dev_Hth_z,
// // 													   dev_Hd_x, dev_Hd_y, dev_Hd_z,
// // 													   dev_Happl_x, dev_Happl_y, dev_Happl_z,
// // 													   h, dev_Ku, dev_Ms, dev_alpha, dev_gamma));

// // 	//__syncthreads();
// // }

__kernel void Kernel_x_theta_phi(__global double* dev_x_theta, __global double* dev_x_phi,
	                             __global double* dev_theta, __global double* dev_phi, 
								 __global double* dev_y_theta, __global double* dev_y_phi,
								 __global double* dev_Ha_x, __global double* dev_Ha_y, __global double* dev_Ha_z,
								 __global double* dev_Hth_x, __global double* dev_Hth_y, __global double* dev_Hth_z,
								 __global double* dev_Hd_x, __global double* dev_Hd_y, __global double* dev_Hd_z,
								 __global double* dev_Happl_x, __global double* dev_Happl_y, __global double* dev_Happl_z,
								 __global double* dev_Ku, __global double* dev_Ms, 
								 __global double* dev_alpha, __global double* dev_gamma, 
										  const double h)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	SetElement(dev_x_theta, x, y, z, Nx, Ny, dtheta_dt(x, y, z, 
		                                               dev_theta, dev_phi, 
													   dev_y_theta, dev_y_phi,
													   dev_Ha_x, dev_Ha_y, dev_Ha_z,
									                   dev_Hth_x, dev_Hth_y, dev_Hth_z,
													   dev_Hd_x, dev_Hd_y, dev_Hd_z,
													   dev_Happl_x, dev_Happl_y, dev_Happl_z,
													   h, dev_Ku, dev_Ms, dev_alpha, dev_gamma));
	SetElement(dev_x_phi,   x, y, z, Nx, Ny, dphi_dt  (x, y, z, 
		                                               dev_theta, dev_phi, 
													   dev_y_theta, dev_y_phi,
													   dev_Ha_x, dev_Ha_y, dev_Ha_z,
									                   dev_Hth_x, dev_Hth_y, dev_Hth_z,
													   dev_Hd_x, dev_Hd_y, dev_Hd_z,
													   dev_Happl_x, dev_Happl_y, dev_Happl_z,
													   h, dev_Ku, dev_Ms, dev_alpha, dev_gamma));

	//__syncthreads();
}

__kernel void Kernel_time_increment(__global double* dev_theta, __global double* dev_phi,
									__global double* dev_a_theta, __global double* dev_b_theta, 
									__global double* dev_c_theta, __global double* dev_d_theta,   
									__global double* dev_a_phi, __global double* dev_b_phi, 
									__global double* dev_c_phi, __global double* dev_d_phi,   
									__global double* dev_Mx, __global double* dev_My, 
									__global double* dev_Mz, __global int* dev_indicator1,
											 const double h, __global double* dev_Ms)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	double temp;
	double Ms_temp = GetElement(dev_Ms, x, y, z, Nx, Ny);

	/* rescaling theta and then time increment */    				
	temp = GetElement(dev_theta, x, y, z, Nx, Ny);
	if (fabs(temp) > 2*PI)				
		SetElement(dev_theta, x, y, z, Nx, Ny, fmod(fabs(temp), 2*PI));
	else if (fabs(temp) > PI && fabs(temp) <= 2*PI)
		SetElement(dev_theta, x, y, z, Nx, Ny, 2*PI - fabs(temp));
	else
		SetElement(dev_theta, x, y, z, Nx, Ny, fabs(temp));
	
	temp = GetElement(dev_theta, x, y, z, Nx, Ny) + h/6*(GetElement(dev_a_theta, x, y, z, Nx, Ny)   + 
													     GetElement(dev_b_theta, x, y, z, Nx, Ny)*2 + 
													     GetElement(dev_c_theta, x, y, z, Nx, Ny)*2 +
													     GetElement(dev_d_theta, x, y, z, Nx, Ny));
	SetElement(dev_theta, x, y, z, Nx, Ny, temp);

	/* rescaling phi and then time increment */
	temp = GetElement(dev_phi, x, y, z, Nx, Ny);
	SetElement(dev_phi, x, y, z, Nx, Ny, temp - ((int)(temp/2/PI))*2*PI);
	
	temp = GetElement(dev_phi, x, y, z, Nx, Ny) + h/6*(GetElement(dev_a_phi, x, y, z, Nx, Ny)   + 
		                                               GetElement(dev_b_phi, x, y, z, Nx, Ny)*2 + 
					                                   GetElement(dev_c_phi, x, y, z, Nx, Ny)*2 +
					                                   GetElement(dev_d_phi, x, y, z, Nx, Ny));
	SetElement(dev_phi, x, y, z, Nx, Ny, temp);
    

	//---- rescaling theta and phi ---------------------//
	temp = GetElement(dev_theta, x, y, z, Nx, Ny);
	if (fabs(temp) > 2*PI)				
		SetElement(dev_theta, x, y, z, Nx, Ny, fmod(fabs(temp), 2*PI));
	else if (fabs(temp) > PI && fabs(temp) <= 2*PI)
		SetElement(dev_theta, x, y, z, Nx, Ny, 2*PI - fabs(temp));
	else
		SetElement(dev_theta, x, y, z, Nx, Ny, fabs(temp));

	temp = GetElement(dev_phi, x, y, z, Nx, Ny);
	SetElement(dev_phi, x, y, z, Nx, Ny, temp - (int)((temp/2/PI))*2*PI);
	//--------------------------------------------------//
	
	// Set magnetization on grain boundary to be zero
	if (GetElement_Int(dev_indicator1, x, y, z, Nx, Ny) == 0) {
		SetElement(dev_Mx, x, y, z, Nx, Ny, 0);
		SetElement(dev_My, x, y, z, Nx, Ny, 0);
		SetElement(dev_Mz, x, y, z, Nx, Ny, 0);
	}
	else {
		SetElement(dev_Mx, x, y, z, Nx, Ny, Ms_temp*sin(GetElement(dev_theta, x, y, z, Nx, Ny))*cos(GetElement(dev_phi, x, y, z, Nx, Ny)));
		SetElement(dev_My, x, y, z, Nx, Ny, Ms_temp*sin(GetElement(dev_theta, x, y, z, Nx, Ny))*sin(GetElement(dev_phi, x, y, z, Nx, Ny)));
		SetElement(dev_Mz, x, y, z, Nx, Ny, Ms_temp*cos(GetElement(dev_theta, x, y, z, Nx, Ny)));
	}

	//__syncthreads();
}

__kernel void Kernel_Hk_field(__global double* dev_theta, __global double* dev_phi,
							  __global double* dev_Hk_x,  __global double* dev_Hk_y, __global double* dev_Hk_z,
							  __global double* dev_Ku, __global double* dev_Ms)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	double temp_z;
	double Ku_temp = GetElement(dev_Ku, x, y, z, Nx, Ny), Ms_temp = GetElement(dev_Ms, x, y, z, Nx, Ny);

	temp_z = 2 * Ku_temp / Ms_temp * cos(GetElement(dev_theta, x, y, z, Nx, Ny));
	SetElement(dev_Hk_x, x, y, z, Nx, Ny, 0.);
	SetElement(dev_Hk_y, x, y, z, Nx, Ny, 0.);
	SetElement(dev_Hk_z, x, y, z, Nx, Ny, temp_z);

	//__syncthreads();
}

__kernel void Kernel_Hth_field(__global double* dev_GasArray, __global double* dev_Hth_x, 
								__global double* dev_Hth_y, __global double* dev_Hth_z, __global double* dev_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	int index = z * Nx * Ny + y * Nx + x;
	double temp_x, temp_y, temp_z;

	temp_x = dev_GasArray[index*3  ] * pow(GetElement(dev_D, x, y, z, Nx, Ny), 0.5);
	SetElement(dev_Hth_x, x, y, z, Nx, Ny, temp_x);

	temp_y = dev_GasArray[index*3+1] * pow(GetElement(dev_D, x, y, z, Nx, Ny), 0.5);
	SetElement(dev_Hth_y, x, y, z, Nx, Ny, temp_y);

	temp_z = dev_GasArray[index*3+2] * pow(GetElement(dev_D, x, y, z, Nx, Ny), 0.5);
	SetElement(dev_Hth_z, x, y, z, Nx, Ny, temp_z);

	//printf("%lf\t", temp_z);

	//__syncthreads();
}

 __kernel void Kernel_M_cufft_Initialization(//__global double* dev_Mx_cufft[],
											__global double* dev_Mx_cufft0, __global double* dev_Mx_cufft1,
											__global double* dev_My_cufft0, __global double* dev_My_cufft1,
											__global double* dev_Mz_cufft0, __global double* dev_Mz_cufft1,
											__global double* dev_Mx, __global double* dev_My, __global double* dev_Mz,
                                            const int lx_zero_pad, const int ly_zero_pad, const int lz_zero_pad)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	//int idxx = i + j * lx_zero_pad + k * lx_zero_pad*ly_zero_pad;
	//dev_Mx_cufft0[idxx];
	//////Mx/////////
	SetElement(dev_Mx_cufft0, x,      y,      z,      lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_Mx_cufft0, x + Nx, y,      z,      lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_Mx_cufft0, x,      y + Ny, z,      lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_Mx_cufft0, x + Nx, y + Ny, z,      lx_zero_pad, ly_zero_pad, 0);

	SetElement(dev_Mx_cufft0, x,      y,      z + Nz, lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_Mx_cufft0, x + Nx, y,      z + Nz, lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_Mx_cufft0, x,      y + Ny, z + Nz, lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_Mx_cufft0, x + Nx, y + Ny, z + Nz, lx_zero_pad, ly_zero_pad, 0);

	SetElement(dev_Mx_cufft1, x,      y,      z,      lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_Mx_cufft1, x + Nx, y,      z,      lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_Mx_cufft1, x,      y + Ny, z,      lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_Mx_cufft1, x + Nx, y + Ny, z,      lx_zero_pad, ly_zero_pad, 0);

	SetElement(dev_Mx_cufft1, x,      y,      z + Nz, lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_Mx_cufft1, x + Nx, y,      z + Nz, lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_Mx_cufft1, x,      y + Ny, z + Nz, lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_Mx_cufft1, x + Nx, y + Ny, z + Nz, lx_zero_pad, ly_zero_pad, 0);

	SetElement(dev_Mx_cufft0, x, y, z, lx_zero_pad, ly_zero_pad, GetElement(dev_Mx, x, y, z, Nx, Ny));

	///////My////////
	SetElement(dev_My_cufft0, x,      y,      z,      lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_My_cufft0, x + Nx, y,      z,      lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_My_cufft0, x,      y + Ny, z,      lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_My_cufft0, x + Nx, y + Ny, z,      lx_zero_pad, ly_zero_pad, 0);

	SetElement(dev_My_cufft0, x,      y,      z + Nz, lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_My_cufft0, x + Nx, y,      z + Nz, lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_My_cufft0, x,      y + Ny, z + Nz, lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_My_cufft0, x + Nx, y + Ny, z + Nz, lx_zero_pad, ly_zero_pad, 0);

	SetElement(dev_My_cufft1, x,      y,      z,      lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_My_cufft1, x + Nx, y,      z,      lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_My_cufft1, x,      y + Ny, z,      lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_My_cufft1, x + Nx, y + Ny, z,      lx_zero_pad, ly_zero_pad, 0);

	SetElement(dev_My_cufft1, x,      y,      z + Nz, lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_My_cufft1, x + Nx, y,      z + Nz, lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_My_cufft1, x,      y + Ny, z + Nz, lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_My_cufft1, x + Nx, y + Ny, z + Nz, lx_zero_pad, ly_zero_pad, 0);

	SetElement(dev_My_cufft0, x, y, z, lx_zero_pad, ly_zero_pad, GetElement(dev_My, x, y, z, Nx, Ny));
//////////Mz//////

	SetElement(dev_Mz_cufft0, x,      y,      z,      lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_Mz_cufft0, x + Nx, y,      z,      lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_Mz_cufft0, x,      y + Ny, z,      lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_Mz_cufft0, x + Nx, y + Ny, z,      lx_zero_pad, ly_zero_pad, 0);

	SetElement(dev_Mz_cufft0, x,      y,      z + Nz, lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_Mz_cufft0, x + Nx, y,      z + Nz, lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_Mz_cufft0, x,      y + Ny, z + Nz, lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_Mz_cufft0, x + Nx, y + Ny, z + Nz, lx_zero_pad, ly_zero_pad, 0);

	SetElement(dev_Mz_cufft1, x,      y,      z,      lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_Mz_cufft1, x + Nx, y,      z,      lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_Mz_cufft1, x,      y + Ny, z,      lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_Mz_cufft1, x + Nx, y + Ny, z,      lx_zero_pad, ly_zero_pad, 0);

	SetElement(dev_Mz_cufft1, x,      y,      z + Nz, lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_Mz_cufft1, x + Nx, y,      z + Nz, lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_Mz_cufft1, x,      y + Ny, z + Nz, lx_zero_pad, ly_zero_pad, 0);
	SetElement(dev_Mz_cufft1, x + Nx, y + Ny, z + Nz, lx_zero_pad, ly_zero_pad, 0);

	SetElement(dev_Mz_cufft0, x, y, z, lx_zero_pad, ly_zero_pad, GetElement(dev_Mz, x, y, z, Nx, Ny));

}	

cl_double2 ComplexMul(__global double* a1, __global double* a2, __global double* b1, __global double* b2, int i, int j, int k, int pitch_x, int pitch_y)
{
        cl_double2 c;
        c.x = a1[k*pitch_x*pitch_y + j*pitch_x + i] * b1[k*pitch_x*pitch_y + j*pitch_x + i] - a2[k*pitch_x*pitch_y + j*pitch_x + i] * b2[k*pitch_x*pitch_y + j*pitch_x + i];
        c.y = a1[k*pitch_x*pitch_y + j*pitch_x + i] * b2[k*pitch_x*pitch_y + j*pitch_x + i] + a2[k*pitch_x*pitch_y + j*pitch_x + i] * b1[k*pitch_x*pitch_y + j*pitch_x + i];
        return c;
}

__kernel void Kernel_CUFFT_M_times_G(__global double* dev_Gxx_cufft0, __global double* dev_Gxy_cufft0,   __global double* dev_Gxz_cufft0,
											  __global double* dev_Gyx_cufft0,  __global double* dev_Gyy_cufft0,  __global double* dev_Gyz_cufft0,
											  __global double* dev_Gzx_cufft0,  __global double* dev_Gzy_cufft0,  __global double* dev_Gzz_cufft0,
											  __global double* dev_Mx_cufft0,   __global double* dev_My_cufft0,   __global double* dev_Mz_cufft0,
											  __global double* dev_Hd_x_cufft0, __global double* dev_Hd_y_cufft0, __global double* dev_Hd_z_cufft0,

											  __global double* dev_Gxx_cufft1, __global double* dev_Gxy_cufft1,   __global double* dev_Gxz_cufft1,
											  __global double* dev_Gyx_cufft1,  __global double* dev_Gyy_cufft1,  __global double* dev_Gyz_cufft1,
											  __global double* dev_Gzx_cufft1,  __global double* dev_Gzy_cufft1,  __global double* dev_Gzz_cufft1,
											  __global double* dev_Mx_cufft1,   __global double* dev_My_cufft1,   __global double* dev_Mz_cufft1,
											  __global double* dev_Hd_x_cufft1, __global double* dev_Hd_y_cufft1, __global double* dev_Hd_z_cufft1,

											  const int lx_zero_pad, const int ly_zero_pad, const int lz_zero_pad)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	cl_double2 adder1, adder2, adder3,
                 Hd_x, Hd_y, Hd_z;

	adder1 = ComplexMul(dev_Gxx_cufft0, dev_Gxx_cufft1, dev_Mx_cufft0, dev_Mx_cufft1, x, y, z, lx_zero_pad, ly_zero_pad); 
	adder2 = ComplexMul(dev_Gxy_cufft0, dev_Gxy_cufft1, dev_My_cufft0, dev_My_cufft1, x, y, z, lx_zero_pad, ly_zero_pad);
	adder3 = ComplexMul(dev_Gxz_cufft0, dev_Gxz_cufft1, dev_Mz_cufft0, dev_Mz_cufft1, x, y, z, lx_zero_pad, ly_zero_pad);
	//__syncthreads();
	Hd_x.x = adder1.x + adder2.x + adder3.x;
	Hd_x.y = adder1.y + adder2.y + adder3.y;
	
	adder1 = ComplexMul(dev_Gyx_cufft0, dev_Gyx_cufft1, dev_Mx_cufft0, dev_Mx_cufft1, x, y, z, lx_zero_pad, ly_zero_pad); 
	adder2 = ComplexMul(dev_Gyy_cufft0, dev_Gyy_cufft1, dev_My_cufft0, dev_My_cufft1, x, y, z, lx_zero_pad, ly_zero_pad);
	adder3 = ComplexMul(dev_Gyz_cufft0, dev_Gyz_cufft1, dev_Mz_cufft0, dev_Mz_cufft1, x, y, z, lx_zero_pad, ly_zero_pad);
	//__syncthreads();
	Hd_y.x = adder1.x + adder2.x + adder3.x;
	Hd_y.y = adder1.y + adder2.y + adder3.y;
	
	adder1 = ComplexMul(dev_Gzx_cufft0, dev_Gzx_cufft1, dev_Mx_cufft0, dev_Mx_cufft1, x, y, z, lx_zero_pad, ly_zero_pad); 
	adder2 = ComplexMul(dev_Gzy_cufft0, dev_Gzy_cufft1, dev_My_cufft0, dev_My_cufft1, x, y, z, lx_zero_pad, ly_zero_pad);
	adder3 = ComplexMul(dev_Gzz_cufft0, dev_Gzz_cufft1, dev_Mz_cufft0, dev_Mz_cufft1, x, y, z, lx_zero_pad, ly_zero_pad);
	//__syncthreads();
	Hd_z.x = adder1.x + adder2.x + adder3.x;
	Hd_z.y = adder1.y + adder2.y + adder3.y;
	
	SetElement(dev_Hd_x_cufft0, x, y, z, lx_zero_pad, ly_zero_pad, Hd_x.x);
	SetElement(dev_Hd_y_cufft0, x, y, z, lx_zero_pad, ly_zero_pad, Hd_y.x);
	SetElement(dev_Hd_z_cufft0, x, y, z, lx_zero_pad, ly_zero_pad, Hd_z.x);

	SetElement(dev_Hd_x_cufft1, x, y, z, lx_zero_pad, ly_zero_pad, Hd_x.y);
	SetElement(dev_Hd_y_cufft1, x, y, z, lx_zero_pad, ly_zero_pad, Hd_y.y);
	SetElement(dev_Hd_z_cufft1, x, y, z, lx_zero_pad, ly_zero_pad, Hd_z.y);
	//__syncthreads();
}

__kernel void Kernel_Hd_set(__global double* dev_Hd_x_cufft0, __global double* dev_Hd_y_cufft0, 
							__global double* dev_Hd_z_cufft0, __global double* dev_Hd_x, 
							__global double* dev_Hd_y, __global double* dev_Hd_z,
                            const int lx_zero_pad, const int ly_zero_pad, const int lz_zero_pad)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	double tempx, tempy, tempz;
	double gd_sum = 1.0; //(double)(lx_zero_pad*ly_zero_pad*lz_zero_pad);

	// if (z == 0)
	// {
	// 	tempx = GetElement(dev_Hd_x_cufft0, x + Nx, y + Ny, z + Nz, lx_zero_pad, ly_zero_pad);
	// 	SetElement(dev_Hd_x, x, y, z, Nx, Ny, tempx/gd_sum);
	// }
	// int idx = (x + Nx) + (y + Ny))*lx_zero_pad + (z + Nz)*lx_zero_pad*ly_zero_pad;
	// int index = z * Nx * Ny + y * Nx + x;
	SetElement(dev_Hd_x, x, y, z, Nx, Ny, GetElement(dev_Hd_x_cufft0, x + Nx, y + Ny, z + Nz, lx_zero_pad, ly_zero_pad)/gd_sum );
	SetElement(dev_Hd_y, x, y, z, Nx, Ny, GetElement(dev_Hd_y_cufft0, x + Nx, y + Ny, z + Nz, lx_zero_pad, ly_zero_pad)/gd_sum );
	SetElement(dev_Hd_z, x, y, z, Nx, Ny, GetElement(dev_Hd_z_cufft0, x + Nx, y + Ny, z + Nz, lx_zero_pad, ly_zero_pad)/gd_sum );
	
	// tempx = GetElement(dev_Hd_x_cufft0, x + Nx, y + Ny, z + Nz, lx_zero_pad, ly_zero_pad);
	// tempy = GetElement(dev_Hd_y_cufft0, x + Nx, y + Ny, z + Nz, lx_zero_pad, ly_zero_pad);
	// tempz = GetElement(dev_Hd_z_cufft0, x + Nx, y + Ny, z + Nz, lx_zero_pad, ly_zero_pad);

	// double gd_sum = 1.0; //(double)(lx_zero_pad*ly_zero_pad*lz_zero_pad);

	// SetElement(dev_Hd_x, x, y, z, Nx, Ny, tempx/gd_sum);
	// SetElement(dev_Hd_y, x, y, z, Nx, Ny, tempy/gd_sum);
	// SetElement(dev_Hd_z, x, y, z, Nx, Ny, tempz/gd_sum);
	// printf("%f\t", dev_Hd_x);
}


double BilinearInterpolation(double q11, double q12, double q21, double q22, double x1, double x2, double y1, double y2, double x, double y) 
{
    double x2x1, y2y1, x2x, y2y, yy1, xx1;
    x2x1 = x2 - x1;
    y2y1 = y2 - y1;
    x2x = x2 - x;
    y2y = y2 - y;
    yy1 = y - y1;
    xx1 = x - x1;
    return 1.0 / (x2x1 * y2y1) * (
        q11 * x2x * y2y +
        q21 * xx1 * y2y +
        q12 * x2x * yy1 +
        q22 * xx1 * yy1
    );
}

__kernel void Kernel_T_ext_set(__global double* dev_T, __global double* dev_extT, 
								const double delta_x, const double delta_t, const double delta_TP,
								const double extTstat_x, const double extTstat_y, const double v, const double t)
{
	#define extT(m, n) dev_extT[(n) + EXTTsize_y * (m)]
	int i = get_global_id(0);
	int j = get_global_id(1);
	int k = get_global_id(2);
	int idx = i + j*Nx + k*Nx*Ny;

	int m = 0,n = 0;
	double m_double = 0, n_double = 0;

	m_double = (i*delta_x + (extTstat_x * 1e-7 - v* t * delta_t)) / delta_TP;
	n_double = (j*delta_x + extTstat_y * 1e-7) / delta_TP;
	m = (int)m_double;
	n = (int)n_double;

	if ((m>=0) && (n>=0) && (m+1< EXTTsize_x)&&(n+1< EXTTsize_y))
	{
		dev_T[idx] = BilinearInterpolation(extT(m, n),extT(m, n+1),extT(m+1, n),extT(m+1, n+1),
								(double)m,(double)(m+1),(double)n,(double)(n+1),m_double,n_double);
	}
	else{
		dev_T[idx]=300.0;
	}
	#undef extT
}


__kernel void Kernel_T_set(__global double* dev_T, 
							const double delta_x, const double delta_t, const double deltaTemp, const double offset,
							const double StdDev_x, const double StdDev_y, const double v, const double t)
{
	int i = get_global_id(0);
	int j = get_global_id(1);
	int k = get_global_id(2);
	int idx = i + j*Nx + k*Nx*Ny;
	dev_T[idx] = 300.0 + deltaTemp*exp((-powf((i*delta_x - v*t*delta_t) - offset*delta_x, 2)/(2*powf(StdDev_x, 2))-powf((j-15)*delta_x, 2)/(2*powf(StdDev_y, 2))));
}

__kernel void Kernel_Happl_ext_set(__global double* dev_Happl_x, __global double* dev_Happl_y, __global double* dev_Happl_z,
								__global double* dev_extH_x, __global double* dev_extH_y, __global double* dev_extH_z,
								const double delta_x, const double delta_t, const double delta_HP,
								const double extHstat_x, const double extHstat_y, 
								const double HPratio, const int NextBit,
								const double v, const double t)
{

	#define extH_x(m, n) dev_extH_x[(n) + EXTHsize_y * (m)]
	#define extH_y(m, n) dev_extH_y[(n) + EXTHsize_y * (m)]
	#define extH_z(m, n) dev_extH_z[(n) + EXTHsize_y * (m)]

	int i = get_global_id(0);
	int j = get_global_id(1);
	int k = get_global_id(2);
	int idx = i + j*Nx + k*Nx*Ny;

	double NextBit_tmp = pown(-1.0f, NextBit);

	int m = 0,n = 0;
	double m_double = 0, n_double = 0;

	m_double = (i*delta_x + extHstat_x * 1e-7 - v* t * delta_t) / delta_HP;
	n_double = (j*delta_x + extHstat_y * 1e-7) / delta_HP;
	m = (int)m_double;
	n = (int)n_double;
	if ((m>=0) && (n>=0) && (m+1< EXTHsize_x)&&(n+1< EXTHsize_y)){
		dev_Happl_x[idx] = BilinearInterpolation(extH_x(m, n),extH_x(m, n+1),extH_x(m+1, n),extH_x(m+1, n+1),
					(double)m,(double)(m+1),(double)n,(double)(n+1),m_double,n_double) * NextBit_tmp *HPratio;
		dev_Happl_y[idx] = BilinearInterpolation(extH_y(m, n),extH_y(m, n+1),extH_y(m+1, n),extH_y(m+1, n+1),
					(double)m,(double)(m+1),(double)n,(double)(n+1),m_double,n_double) * NextBit_tmp *HPratio;
		dev_Happl_z[idx] = BilinearInterpolation(extH_z(m, n),extH_z(m, n+1),extH_z(m+1, n),extH_z(m+1, n+1),
					(double)m,(double)(m+1),(double)n,(double)(n+1),m_double,n_double) * NextBit_tmp *HPratio;
	}
	else{
		dev_Happl_x[idx]=0.0;
		dev_Happl_y[idx]=0.0;
		dev_Happl_z[idx]=0.0;
	}

	#undef extH_x
	#undef extH_y
	#undef extH_z
}

__kernel void Kernel_Happl_set(__global double* dev_Happl_x, __global double* dev_Happl_y, __global double* dev_Happl_z, 
								const double Happl, const double HPratio, const int NextBit)
{
	int i = get_global_id(0);
	int j = get_global_id(1);
	int k = get_global_id(2);
	int idx = i + j*Nx + k*Nx*Ny;

	double NextBit_tmp = pown(-1.0f, NextBit);

	dev_Happl_x[idx] = -Happl *sin(PI/9) * NextBit_tmp * HPratio;
	dev_Happl_y[idx] = 0*HPratio;
	dev_Happl_z[idx] = Happl *cos(PI/9)* NextBit_tmp *HPratio;
}


__kernel void Kernel_MagPara_set(__global double* dev_Ms,  __global double* dev_Ku, __global double* dev_Aex1,
								__global double* dev_Aex2, __global double* dev_alpha, __global double* dev_gamma,
								__global double* dev_D, __global double* dev_T, 
								__global double* dev_std_Aex, __global double* dev_std_Ku, 
								const double delta_x, const double delta_t)
{
	int i = get_global_id(0);
	int j = get_global_id(1);
	int k = get_global_id(2);
	int idx = i + j*Nx + k*Nx*Ny;
	double Aex1_tmp, Aex_tmp, Aratio_tmp, alpha_tmp, Ms_tmp, Ku_tmp;
	if (k<Nz_1)
	{
		double T = dev_T[idx]*(1 + dev_std_Aex[idx]);
		// if (T[idx] > 600 && T[idx] <= 700){
		// 	host_Ms[idx]    = h1_Ms *pow(735-T[idx], h2_Ms);
		// 	host_Ku[idx]    = (h1_Ku*T[idx] + h2_Ku)*(1+std_Ku[idx])*1;  
		// 	host_Aex[idx]   = (h1_Aex*T[idx] + h2_Aex);
		// 	host_alpha[idx] = (h1_alpha*exp(h2_alpha*T[idx]) + h3_alpha*exp(h4_alpha*T[idx]))*2;
		// }
		// else if (T[idx] > 500 && T[idx] <= 600){
		// 	host_Ms[idx]    = l1_Ms*T[idx] + l2_Ms;
		// 	host_Ku[idx]    = (l1_Ku*pow(T[idx], 3) + l2_Ku*pow(T[idx], 2) + l3_Ku*pow(T[idx], 1) + l4_Ku)*(1+std_Ku[idx])*1;
		// 	host_Aex[idx]   = (l1_Aex*pow(T[idx], 3) + l2_Aex*pow(T[idx], 2) + l3_Aex*pow(T[idx], 1) + l4_Aex);
		// 	host_alpha[idx] = l1_alpha*T[idx] + l2_alpha;
		// }
		// else if (T[idx] >= 100 && T[idx] <= 500){
		// 	host_Ms[idx]    = f1_Ms*pow(T[idx], 2) + f2_Ms*pow(T[idx], 1) + f3_Ms;
		// 	host_Ku[idx]    = (f1_Ku*pow(T[idx], 1) + f2_Ku)*(1+std_Ku[idx])*1;
		// 	host_Aex[idx]   = 1.1e-6; //*pow(host_Ms[idx]/1100, 2);
		// 	host_alpha[idx] = l1_alpha*T[idx] + l2_alpha;
		// }
		// else {
		// 	host_Ms[idx]    = 100;
		// 	host_Ku[idx]    = 0;
		// 	host_Aex[idx]   = 0;
		// 	host_alpha[idx] = 0.1;
		// }
		// D[idx] = (2*kb*T[idx]*host_alpha[idx])/(host_Ms[idx]*delta_t*GAMMA*pow(delta_x, 3.0));
		// host_gamma[idx] = (1.76e7)/(1+pow(host_alpha[idx], 2.0));
		/////////Aex///////
		if (T > 700 && T <= 710){
			Aex_tmp   = -3.331e-011 * pow(T,2) + 3.914e-008*(T) -1.077e-005;
			Aratio_tmp = 1.3;
		}
		else if (T > 680 && T <= 700){
			Aex_tmp   = -3.331e-011 * pow(T,2) + 3.914e-008*(T) -1.077e-005;
			Aratio_tmp = -0.05* (T-680)/20 + 1.35;
		}
		else if (T > 660 && T <= 680){
			Aex_tmp   = -3.331e-011 * pow(T,2) + 3.914e-008*(T) -1.077e-005;
			Aratio_tmp = -0.23* (T-660)/20 + 1.58;
		}
		else if (T > 640 && T <= 660){
			Aex_tmp   = -3.331e-011 * pow(T,2) + 3.914e-008*(T) -1.077e-005;
			Aratio_tmp = 0.18* (T-640)/20 + 1.4;
		}
		else if (T > 610 && T <= 640){
			Aex_tmp   = -2.406e-011 * pow(T,2) + 2.574e-008*(T)-5.985e-006;
			Aratio_tmp = 0.1* (T-610)/30 + 1.3;
		}
		else if (T > 580 && T <= 610){
			Aex_tmp   = -2.406e-011 * pow(T,2) + 2.574e-008*(T)-5.985e-006;
			Aratio_tmp = 1.3;
		}
		else if (T > 520 && T <= 580){
			Aex_tmp   = 5.556e-012 * pow(T,2)  -8.611e-009*(T) +3.976e-006;
			Aratio_tmp = 1.3;
		}
		else if (T <= 520){

			Aex_tmp   =  -2.507e-012 * pow(T,2) -1.771e-010 *(T)+1.777e-006; //*pow(host_Ms[idx]/1100, 2);
			Aratio_tmp = 1.3;
		}
		else {
			//host_Ms[idx]    = 100;
			//host_Ku[idx]    = 0;
			Aex_tmp   = 0;
			Aratio_tmp =1.0;
			//host_alpha[idx] = 0.1;
		}
		/////////alpha///////
		if (T <= 600){
			alpha_tmp = 4.0*(2.5e-5*T + 0.02);
		}
		else if (T > 600 && T <= 710){
			alpha_tmp  = (0.005561*exp(0.003078f*T)+ 9.274e-18*exp(0.05185f*T))*8;
		}
		else {
			alpha_tmp = 0.1;
		}
		//D[idx] = (2*kb*T*host_alpha[idx])/(host_Ms[idx]*delta_t*GAMMA*pow(delta_x, 3.0));
		//host_gamma[idx] = (1.76e7)/(1+pow(host_alpha[idx], 2.0));
		//////////Ms////////
		if (T <= 710){
			Ms_tmp= 136.0* pow((739.4f-T),0.3182f);
		}
		else {
			Ms_tmp    = 100;
		}
		//////////Ku/////////
		if (T <= 710){
			Ku_tmp= -6.606e-005*pow(T,4.0f) + 0.2348*pow(T,3.0f) -263.9*pow(T,2.0f) + 2.916e4 *T + 5.32e7;
		}
		else {
			Ku_tmp    = 0;
		}
		Aex1_tmp = 3*Aex_tmp/(1+2*Aratio_tmp);
		dev_Aex1[idx]= Aex1_tmp; //Aex_z
		dev_Aex2[idx]= Aratio_tmp * Aex1_tmp;  //Aex_xy
		dev_alpha[idx] = alpha_tmp;
		dev_Ms[idx]=Ms_tmp;
		dev_D[idx] = (2*KB*T*alpha_tmp)/(Ms_tmp*delta_t*GAMMA*pow(delta_x, 3.0f));
		dev_gamma[idx] = (1.76e7)/(1+pow(alpha_tmp, 2.0f));
		dev_Ku[idx] = Ku_tmp * (1 + dev_std_Ku[idx]);
	}
	else
	{
		double T = dev_T[idx];
		Ms_tmp = 10.0;
		dev_Ms[idx]    = Ms_tmp;
		dev_Ku[idx]    = 0;
		dev_Aex1[idx]   = 0;
		dev_Aex2[idx]   = 0;
		alpha_tmp = 0.1;
		dev_alpha[idx] = alpha_tmp;
		dev_D[idx] = (2*KB*T*alpha_tmp)/(Ms_tmp*delta_t*GAMMA*pow(delta_x, 3.0f));
		dev_gamma[idx] = (1.76e7)/(1+pow(alpha_tmp, 2.0f));
	}
}

/////////////////cl_RNG and cl_ProDist///////////////////
constant cl_double clprobdistNormal_InvP1[] = {
	0.160304955844066229311E2,
	-0.90784959262960326650E2,
	0.18644914861620987391E3,
	-0.16900142734642382420E3,
	0.6545466284794487048E2,
	-0.864213011587247794E1,
	0.1760587821390590
};

constant cl_double clprobdistNormal_InvQ1[] = {
	0.147806470715138316110E2,
	-0.91374167024260313396E2,
	0.21015790486205317714E3,
	-0.22210254121855132366E3,
	0.10760453916055123830E3,
	-0.206010730328265443E2,
	0.1E1
};

constant cl_double clprobdistNormal_InvP2[] = {
	-0.152389263440726128E-1,
	0.3444556924136125216,
	-0.29344398672542478687E1,
	0.11763505705217827302E2,
	-0.22655292823101104193E2,
	0.19121334396580330163E2,
	-0.5478927619598318769E1,
	0.237516689024448000
};

constant cl_double clprobdistNormal_InvQ2[] = {
	-0.108465169602059954E-1,
	0.2610628885843078511,
	-0.24068318104393757995E1,
	0.10695129973387014469E2,
	-0.23716715521596581025E2,
	0.24640158943917284883E2,
	-0.10014376349783070835E2,
	0.1E1
};

constant cl_double clprobdistNormal_InvP3[] = {
	0.56451977709864482298E-4,
	0.53504147487893013765E-2,
	0.12969550099727352403,
	0.10426158549298266122E1,
	0.28302677901754489974E1,
	0.26255672879448072726E1,
	0.20789742630174917228E1,
	0.72718806231556811306,
	0.66816807711804989575E-1,
	-0.17791004575111759979E-1,
	0.22419563223346345828E-2
};

constant cl_double clprobdistNormal_InvQ3[] = {
	0.56451699862760651514E-4,
	0.53505587067930653953E-2,
	0.12986615416911646934,
	0.10542932232626491195E1,
	0.30379331173522206237E1,
	0.37631168536405028901E1,
	0.38782858277042011263E1,
	0.20372431817412177929E1,
	0.1E1
};

constant cl_double clprobdistNormal_SQRT2PI = 2.50662827463100050; // Sqrt(2*Pi)
constant cl_double clprobdistNormal_SQRT2 = 1.4142135623730951;

constant cl_double clprobdistNormalXBIG = 100.0;
constant cl_double clprobdistNormalXBIGM = 1000.0;

typedef enum clprobdistStatus_ {
        CLPROBDIST_SUCCESS = 0,
        CLPROBDIST_INVALID_VALUE = -1
} clprobdistStatus;

#define clprobdistSetErrorString(err, ...) (err)

// clprobdistStatus clprobdistSetErrorString(cl_int err, const char* msg, ...)
// {
// 	char formatted[1024];
// 	const char* base;
// 	switch (err) {
// 		CASE_ERR(SUCCESS);
// 		CASE_ERR(OUT_OF_RESOURCES);
// 		CASE_ERR(INVALID_VALUE);
// 		CASE_ERR(INVALID_ENVIRONMENT);
// 		CASE_ERR(ABSTRACT_FUNCTION);
// 	default: base = MSG_DEFAULT;
// 	}
// 	va_list args;
// 	va_start(args, msg);
// 	vsprintf(formatted, msg, args);
// 	sprintf(errorString, "[%s] %s", base, formatted);
// 	va_end(args);
// 	return (clprobdistStatus)err;
// }


cl_double clprobdistStdNormalInverseCDF(cl_double u, clprobdistStatus* err) {
	/*
	* Returns the inverse of the cdf of the normal distribution.
	* Rational approximations giving 16 decimals of precision.
	* J.M. Blair, C.A. Edwards, J.H. Johnson, "Rational Chebyshev
	* approximations for the Inverse of the Error Function", in
	* Mathematics of Computation, Vol. 30, 136, pp 827, (1976)
	*/

	int i;
	bool negatif;
	cl_double y, z, v, w;
	cl_double x = u;

	if (u < 0.0 || u > 1.0)
	{
		if (err) *err = clprobdistSetErrorString(CLPROBDIST_INVALID_VALUE, "%s(): u is not in [0, 1]", __func__);
		return -1;
	}
	if (u <= 0.0)
		return FLT_MIN;// Double.NEGATIVE_INFINITY;
	if (u >= 1.0)
		return FLT_MAX; // Double.POSITIVE_INFINITY;

	// Transform x as argument of InvErf
	x = 2.0 * x - 1.0;
	if (x < 0.0) {
		x = -x;
		negatif = CL_TRUE;
	}
	else {
		negatif = CL_FALSE;
	}

	if (x <= 0.75) {
		y = x * x - 0.5625;
		v = w = 0.0;
		for (i = 6; i >= 0; i--) {
			v = v * y + clprobdistNormal_InvP1[i];
			w = w * y + clprobdistNormal_InvQ1[i];
		}
		z = (v / w) * x;

	}
    
	else if (x <= 0.9375) {
		y = x * x - 0.87890625;
		v = w = 0.0;
		for (i = 7; i >= 0; i--) {
			v = v * y + clprobdistNormal_InvP2[i];
			w = w * y + clprobdistNormal_InvQ2[i];
		}
		z = (v / w) * x;

	}
    
	else {
		if (u > 0.5)
			y = 1.0 / sqrt(-log(1.0f - x));
		else
			y = 1.0 / sqrt(-log(2.0f * u));
		v = 0.0;
		for (i = 10; i >= 0; i--)
			v = v * y + clprobdistNormal_InvP3[i];
		w = 0.0;
		for (i = 8; i >= 0; i--)
			w = w * y + clprobdistNormal_InvQ3[i];
		z = (v / w) / y;
	}

	if (negatif) {
		if (u < 1.0e-105) {
			cl_double RACPI = 1.77245385090551602729;
			w = exp(-z * z) / RACPI;  // pdf
			y = 2.0 * z * z;
			v = 1.0;
			cl_double term = 1.0;
			// Asymptotic series for erfc(z) (apart from exp factor)
			for (i = 0; i < 6; ++i) {
				term *= -(2 * i + 1) / y;
				v += term;
			}
			// Apply 1 iteration of Newton solver to get last few decimals
			z -= u / w - 0.5 * v / z;

		}
		return -(z * clprobdistNormal_SQRT2);

	}
	else
		return z * clprobdistNormal_SQRT2;
        
}  

                              
                                                                    
__kernel void Kernel_gaussian_distribution(__global clrngMrg31k3pHostStream *streams, 
                          __global double *out)                       
    {                                                                
		int i = get_global_id(0);
		int j = get_global_id(1);
		int k = get_global_id(2);
		int gid = i + j*Nx + k*Nx*Ny;                                 
        double u;                                                            
        clrngMrg31k3pStream workItemStream;                          
        clrngMrg31k3pCopyOverStreamsFromGlobal(1, &workItemStream,   
                                                     &streams[3*gid]); 
                                                                    
        u = clrngMrg31k3pRandomU01(&workItemStream); 
        //out[gid] = rand_tmp;
        out[3*gid] = clprobdistStdNormalInverseCDF(u, NULL);
////////////////////3*gid+1////////////////////////
		// clrngMrg31k3pCopyOverStreamsFromGlobal(1, &workItemStream,   
        //                                              &streams[3*gid+1]); 
                                                                    
        u = clrngMrg31k3pRandomU01(&workItemStream); 
        //out[gid] = rand_tmp;
        out[3*gid+1] = clprobdistStdNormalInverseCDF(u, NULL);  
////////////////////3*gid+2////////////////////////
		// clrngMrg31k3pCopyOverStreamsFromGlobal(1, &workItemStream,   
        //                                              &streams[3*gid+2]); 
                                                                    
        u = clrngMrg31k3pRandomU01(&workItemStream); 
        //out[gid] = rand_tmp;
        out[3*gid+2] = clprobdistStdNormalInverseCDF(u, NULL);  

		clrngMrg31k3pCopyOverStreamsToGlobal(1, &streams[3*gid],
													&workItemStream);       
    }                                                                
                                                                     