//----Document created by Yipeng Jiao (jiaox058@umn.edu) on 01/10/2015-----

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <assert.h>
#include <time.h>
#include <iomanip>

using namespace std;
#include "clRNG/clRNG.h"
#include "clRNG/mrg31k3p.h"
//#include "random.h"
#include "Parameters.h"
#include "Moving_Head1.h"
#include "Ha_field.h"
#include "Hd_FFT1.h"

// int G_tensor(int lx_zero_pad, int ly_zero_pad, int lz_zero_pad, 
//  			cl_context &ctx, cl_command_queue &queue);

bool AllocateDeviceMemory(cl_context &ctx)
{
    cl_int err;

    int lx_zero_pad = 2 * Nx, 
        ly_zero_pad = 2 * Ny, 
        lz_zero_pad = 2 * Nz;

    dev_theta = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_phi = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_a_theta = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_b_theta = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_c_theta = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_d_theta = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_a_phi = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_b_phi = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_c_phi = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_d_phi = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_Ha_x = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_Ha_y = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_Ha_z = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_Hth_x = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_Hth_y = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_Hth_z = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_Hk_x = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_Hk_y = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_Hk_z = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_Hd_x = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_Hd_y = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_Hd_z = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_d_theta_d_t = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_d_phi_d_t = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_indicator1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_int), NULL, &err);
    dev_indicator2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_int), NULL, &err);
    dev_indicator3 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_int), NULL, &err);
    dev_indicator1_temp = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (Nx + 2) * (Ny + 2) * (Nz + 2) * sizeof(cl_int), NULL, &err);
    dev_indicator2_temp = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (Nx + 2) * (Ny + 2) * (Nz + 2) * sizeof(cl_int), NULL, &err);
    dev_Ms = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_Ku = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_Aex1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_Aex2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_alpha = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_gamma = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_D = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_T = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_Happl_x = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_Happl_y = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_Happl_z = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);

    dev_Mx = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_My = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_Mz = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_M_temp_x = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (Nx + 2) * (Ny + 2) * (Nz + 2) * sizeof(cl_double), NULL, &err);
    dev_M_temp_y = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (Nx + 2) * (Ny + 2) * (Nz + 2) * sizeof(cl_double), NULL, &err);
    dev_M_temp_z = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (Nx + 2) * (Ny + 2) * (Nz + 2) * sizeof(cl_double), NULL, &err);
    dev_Ms_temp = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (Nx + 2) * (Ny + 2) * (Nz + 2) * sizeof(cl_double), NULL, &err);
    dev_Ku_temp = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (Nx + 2) * (Ny + 2) * (Nz + 2) * sizeof(cl_double), NULL, &err);
    dev_alpha_temp = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (Nx + 2) * (Ny + 2) * (Nz + 2) * sizeof(cl_double), NULL, &err);
    dev_Aex1_temp = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (Nx + 2) * (Ny + 2) * (Nz + 2) * sizeof(cl_double), NULL, &err);
    dev_Aex2_temp = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (Nx + 2) * (Ny + 2) * (Nz + 2) * sizeof(cl_double), NULL, &err);
  
    dev_std_Aex = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
    dev_std_Ku = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Nx * Ny * Nz * sizeof(cl_double), NULL, &err);
 
    dev_GasArray = clCreateBuffer(ctx, CL_MEM_READ_WRITE, DEG_FREEDOM * sizeof(cl_double), NULL, &err);
 
    dev_Gxx_cufft[0] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);
    dev_Gxx_cufft[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);
    dev_Gxy_cufft[0] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);
    dev_Gxy_cufft[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);
    dev_Gxz_cufft[0] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);
    dev_Gxz_cufft[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);
    
    dev_Gyx_cufft[0] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);
    dev_Gyx_cufft[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);
    dev_Gyy_cufft[0] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);
    dev_Gyy_cufft[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);
    dev_Gyz_cufft[0] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);
    dev_Gyz_cufft[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);

    dev_Gzx_cufft[0] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);
    dev_Gzx_cufft[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);
    dev_Gzy_cufft[0] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);
    dev_Gzy_cufft[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);
    dev_Gzz_cufft[0] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);
    dev_Gzz_cufft[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);
    
    dev_Mx_cufft[0] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);
    dev_Mx_cufft[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);
    dev_My_cufft[0] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);
    dev_My_cufft[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);
    dev_Mz_cufft[0] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);
    dev_Mz_cufft[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);

    dev_Hd_x_cufft[0] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);
    dev_Hd_x_cufft[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);
    dev_Hd_y_cufft[0] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);
    dev_Hd_y_cufft[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);
    dev_Hd_z_cufft[0] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);
    dev_Hd_z_cufft[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, lx_zero_pad * ly_zero_pad * lz_zero_pad * sizeof(cl_double), NULL, &err);

    dev_extT = clCreateBuffer(ctx, CL_MEM_READ_WRITE, EXTTsize_x * EXTTsize_y * sizeof(cl_double), NULL, &err);
    dev_extH_x = clCreateBuffer(ctx, CL_MEM_READ_WRITE, EXTHsize_x * EXTHsize_y * sizeof(cl_double), NULL, &err);
    dev_extH_y = clCreateBuffer(ctx, CL_MEM_READ_WRITE, EXTHsize_x * EXTHsize_y * sizeof(cl_double), NULL, &err);
    dev_extH_y = clCreateBuffer(ctx, CL_MEM_READ_WRITE, EXTHsize_x * EXTHsize_y * sizeof(cl_double), NULL, &err);
  
    return 0;
   
}

void ReleaseMemory()
{
    clReleaseMemObject(dev_theta);
    clReleaseMemObject(dev_phi);
    clReleaseMemObject(dev_a_theta);
    clReleaseMemObject(dev_b_theta);
    clReleaseMemObject(dev_c_theta);
    clReleaseMemObject(dev_d_theta);
    clReleaseMemObject(dev_a_phi);
    clReleaseMemObject(dev_b_phi);
    clReleaseMemObject(dev_c_phi);
    clReleaseMemObject(dev_d_phi);
    clReleaseMemObject(dev_Ha_x);
    clReleaseMemObject(dev_Ha_y);
    clReleaseMemObject(dev_Ha_z);
    clReleaseMemObject(dev_Hth_x);
    clReleaseMemObject(dev_Hth_y);
    clReleaseMemObject(dev_Hth_z);
    clReleaseMemObject(dev_Hk_x);
    clReleaseMemObject(dev_Hk_y);
    clReleaseMemObject(dev_Hk_z);
    clReleaseMemObject(dev_Hd_x);
    clReleaseMemObject(dev_Hd_y);
    clReleaseMemObject(dev_Hd_z);
    clReleaseMemObject(dev_d_theta_d_t);
    clReleaseMemObject(dev_d_phi_d_t);
    clReleaseMemObject(dev_indicator1);
    clReleaseMemObject(dev_indicator2);
    clReleaseMemObject(dev_indicator3);
    clReleaseMemObject(dev_indicator1_temp);
    clReleaseMemObject(dev_indicator2_temp);
    clReleaseMemObject(dev_Ms);
    clReleaseMemObject(dev_Ku);
    clReleaseMemObject(dev_Aex1);
    clReleaseMemObject(dev_Aex2);
    clReleaseMemObject(dev_alpha);
    clReleaseMemObject(dev_gamma);
    clReleaseMemObject(dev_D);
    clReleaseMemObject(dev_T); 
    clReleaseMemObject(dev_Happl_x); 
    clReleaseMemObject(dev_Happl_y);   
    clReleaseMemObject(dev_Happl_z); 

    clReleaseMemObject(dev_Mx);
    clReleaseMemObject(dev_My);
    clReleaseMemObject(dev_Mz);
    clReleaseMemObject(dev_M_temp_x);
    clReleaseMemObject(dev_M_temp_y);
    clReleaseMemObject(dev_M_temp_z);
    clReleaseMemObject(dev_Ms_temp);
    clReleaseMemObject(dev_Ku_temp);
    clReleaseMemObject(dev_alpha_temp);
    clReleaseMemObject(dev_Aex1_temp);
    clReleaseMemObject(dev_Aex2_temp);

    clReleaseMemObject(dev_std_Aex);
    clReleaseMemObject(dev_std_Ku);

    clReleaseMemObject(dev_GasArray);

    clReleaseMemObject(dev_Gxx_cufft[0]);
    clReleaseMemObject(dev_Gxx_cufft[1]);
    clReleaseMemObject(dev_Gxy_cufft[0]);
    clReleaseMemObject(dev_Gxy_cufft[1]);
    clReleaseMemObject(dev_Gxz_cufft[0]);
    clReleaseMemObject(dev_Gxz_cufft[1]);

    clReleaseMemObject(dev_Gyx_cufft[0]);
    clReleaseMemObject(dev_Gyx_cufft[1]);
    clReleaseMemObject(dev_Gyy_cufft[0]);
    clReleaseMemObject(dev_Gyy_cufft[1]);
    clReleaseMemObject(dev_Gyz_cufft[0]);
    clReleaseMemObject(dev_Gyz_cufft[1]);

    clReleaseMemObject(dev_Gzx_cufft[0]);
    clReleaseMemObject(dev_Gzx_cufft[1]);
    clReleaseMemObject(dev_Gzy_cufft[0]);
    clReleaseMemObject(dev_Gzy_cufft[1]);
    clReleaseMemObject(dev_Gzz_cufft[0]);
    clReleaseMemObject(dev_Gzz_cufft[1]);

    clReleaseMemObject(dev_Mx_cufft[0]);
    clReleaseMemObject(dev_Mx_cufft[1]);
    clReleaseMemObject(dev_My_cufft[0]);
    clReleaseMemObject(dev_My_cufft[1]);
    clReleaseMemObject(dev_Mz_cufft[0]);
    clReleaseMemObject(dev_Mz_cufft[1]);

    clReleaseMemObject(dev_Hd_x_cufft[0]);
    clReleaseMemObject(dev_Hd_x_cufft[1]);
    clReleaseMemObject(dev_Hd_y_cufft[0]);
    clReleaseMemObject(dev_Hd_y_cufft[1]);
    clReleaseMemObject(dev_Hd_z_cufft[0]);
    clReleaseMemObject(dev_Hd_z_cufft[1]);

    clReleaseMemObject(dev_extT);
    clReleaseMemObject(dev_extH_x);
    clReleaseMemObject(dev_extH_y);
    clReleaseMemObject(dev_extH_z);

}


int main(int argc, char* argv[])
{
    int idx, idxx;
	float mm;
	int m, is_pow_2, lx_pow2, ly_pow2, lz_pow2,
		G_lx_pow2, G_ly_pow2, G_lz_pow2;
    int lx_zero_pad = 2 * Nx, 
        ly_zero_pad = 2 * Ny, 
        lz_zero_pad = 2 * Nz;
	unsigned long nn[3];

    FILE *fp;

    cl_int err;
    
    //cl_device_id device = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context ctx = 0;
    cl_command_queue queue = 0;
    cl_program program = 0, program_gaussion = 0;
    cl_kernel ker_ini = 0, ker_dev_ind_apron = 0,
              ker_Ha_apron = 0, ker_MA_apron = 0,
              ker_a_qj = 0, ker_x_qj = 0,
              ker_dqj_dt = 0, ker_t_incr = 0, ker_hk_f = 0,
              ker_hth_f = 0, ker_M_ini = 0, ker_fft_M_G = 0, ker_fft_hd = 0,
              ker_T = 0, ker_T_ext = 0, ker_H = 0, ker_H_ext = 0, ker_MgPar = 0,
              ker_gaussian = 0;
    cl_event event = 0;

    char include_str[1024] = {'\0'};
    char build_log[4096]  = {'\0'};
    char *clrng_root;
    clrngMrg31k3pStream *streams = 0;
    size_t streamBufferSize = 0;
    //size_t i = 0;
    size_t source_max_length = 0x100000, source_length;	
    char *source_str;
    //char source_str[0x100000];

    // rfile_kernel.open("LLG_kernel_test.cl");
    // rfile_kernel.seekg(0, ios::end);
    // source_length = rfile_kernel.tellg();
    // if (source_max_length < source_length)
    // {
    //     cout<<"Kernel file is too long!"<<endl;
    //     exit(1);
    // }
    // //source_str = (char*) malloc(source_length);
    // rfile_kernel.read(source_str, source_length);
    // rfile_kernel.close();
    // for (int i = 0; i < source_length; i++)
    // {
    //     printf ("%c", source_str[i]);
    // }
    
    fp = fopen("LLG_kernel_test.cl", "r");	
    if (!fp) {	
        printf("Failed to load kernel.\n");	
        exit(1);	
    }
    source_str = (char *)malloc(source_max_length );
    source_length = fread(source_str, 1, source_max_length , fp);
    fclose( fp );

    rfile2.open("prbs.txt");
    wfile100.open("pattern_o.dat");

	//check how many platforms does the computer have
	char buf[1024];
	cl_platform_id platformlist[10], platform;
	cl_uint numPlatforms;
	//size_t size;
	if ((clGetPlatformIDs(0, NULL, &numPlatforms)) != CL_SUCCESS) {
		cout << "Unable to query the number of platforms" << endl;
		exit(1);
	}
	printf("Found %d platform(s).\n", numPlatforms);
	err = clGetPlatformIDs(numPlatforms, platformlist, NULL);
	if (err != CL_SUCCESS)
	{
		cout << "Error: Failed to find a platform!" << endl;
		exit(1);
	}
	err = 0;
	for (int i = 0; i < numPlatforms; i++) {
		err |= clGetPlatformInfo(platformlist[i], CL_PLATFORM_NAME, sizeof(buf), buf, NULL);
		cout << "Platform " << i <<":"<< buf << endl;
	}
	cout << "Choose a platform:";
	int platform_index;
	cin >> platform_index;
	cout << endl;
	platform = platformlist[platform_index];
	cout << "Platform " << platform_index << " is chosen" << endl;

	//get a device
	cl_device_id devicelist[8], device;
	cl_uint numDevices;
    err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
	if (err != CL_SUCCESS) {
		cout << "Unable to query the number of devices" << endl;
		exit(1);
	}
	printf("Found %d device(s).\n", numDevices);
	//get the list of all devices
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devicelist, NULL);
	if (err != CL_SUCCESS)
	{
		cout << "Error: Failed to find a platform!" << endl;
		exit(1);
	}
	err = 0;
	for (int i = 0; i < numDevices; i++) {
		err |= clGetDeviceInfo(devicelist[i], CL_DEVICE_NAME, sizeof(buf), buf, NULL);
	}
	int device_index;
	cout << "Choose a device:";
	cin >> device_index;
	cout << endl;
	device = devicelist[device_index];
	cout << "Device " << device_index << " is chosen" << endl;


	////////////////////////////////////////


    if (err != CL_SUCCESS)
    {
        cout<<"Error: Failed to create a device group!"<<endl;
        exit(1);
    }
    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
    if (err != CL_SUCCESS)
    {
        cout<<"Error: Failed to create a compute context!"<<endl;
        exit(1);
    }
    queue = clCreateCommandQueue(ctx, device, 0, &err );
    if (err != CL_SUCCESS)
    {
        cout<<"Error: Failed to create a commands queue!"<<endl;
        exit(1);
    }

    // Make sure CLRNG_ROOT is specified to get library path //
    //clrng_root = getenv("CLRNG_ROOT");
    clrng_root = "/Users/water/Documents/vscode/opencl/clRNG/package";
    if(clrng_root == NULL) cout<<"\nSpecify environment variable CLRNG_ROOT as described\n";
    strcpy(include_str, "-I ");
    strcat(include_str, clrng_root);
    strcat(include_str, "/include");

    //program = clCreateProgramWithSource(ctx, 1, (const char**) &source_str, (const size_t*) &source_length, &err);
    program = clCreateProgramWithSource(ctx, 1, (const char**) &source_str, NULL, &err);

    if (!program)
    {
        cout<<"Error: Failed to create a compute program!"<<endl;
        exit(1);
    }
    err = clBuildProgram(program, 1, &device, include_str, NULL, NULL);
    if(err != CL_SUCCESS)
    {
		cout<<"Error: clBuildProgram has failed. Error: "<<err<<endl;
        // for (int i = 0; i < 4096; i++)
        // {
        //     printf("%c ", build_log[i]);
        // }
        // printf("info\n");
        // printf("%d\n", CL_BUILD_NONE);
        // printf("%d\n", CL_BUILD_ERROR);
        // printf("%d\n", CL_BUILD_SUCCESS);
        // printf("%d\n", CL_BUILD_IN_PROGRESS);
        //cl_build_status b_status;
       //clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &b_status, NULL);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 4096, build_log, NULL);
        printf("%s", build_log);
        //cout<<build_log<<endl;
        // for (int i = 0; i < 4096; i++)
        // {
        //     printf("%c", build_log[i]);
        // }
        exit(1);
    }
    ker_ini = clCreateKernel(program, "Kernel_Initialization", &err);
    if (!ker_ini || err != CL_SUCCESS)
    {
        cout<<"Error: Failed to create compute kernel 1!"<<endl;
        exit(1);
    }
    ker_dev_ind_apron = clCreateKernel(program, "Kernel_dev_indicator_with_apron", &err);
    if (!ker_dev_ind_apron || err != CL_SUCCESS)
    {
        cout<<"Error: Failed to create compute kernel 2!"<<endl;
        exit(1);
    }

    ker_Ha_apron = clCreateKernel(program, "Kernel_Ha_with_apron", &err);
    if (!ker_Ha_apron || err != CL_SUCCESS)
    {
        cout<<"Error: Failed to create compute kernel 4!"<<endl;
        exit(1);
    }

    ker_MA_apron = clCreateKernel(program, "Kernel_M_A_with_apron", &err);
    if (!ker_MA_apron || err != CL_SUCCESS)
    {
        cout<<"Error: Failed to create compute kernel 3!"<<endl;
        exit(1);
    }

    ker_a_qj = clCreateKernel(program, "Kernel_a_theta_phi", &err);
    if (!ker_a_qj || err != CL_SUCCESS)
    {
        cout<<"Error: Failed to create compute kernel 5!"<<endl;
        exit(1);
    }


    ker_x_qj = clCreateKernel(program, "Kernel_x_theta_phi", &err);
    if (!ker_x_qj || err != CL_SUCCESS)
    {
        cout<<"Error: Failed to create compute kernel 6!"<<endl;
        exit(1);
    }

    ker_dqj_dt = clCreateKernel(program, "Kernel_d_theta_phi_d_t", &err);
    if (!ker_dqj_dt || err != CL_SUCCESS)
    {
        cout<<"Error: Failed to create compute kernel 7!"<<endl;
        exit(1);
    }

    ker_t_incr = clCreateKernel(program, "Kernel_time_increment", &err);
    if (!ker_t_incr || err != CL_SUCCESS)
    {
        cout<<"Error: Failed to create compute kernel 8!"<<endl;
        exit(1);
    }

    ker_hk_f = clCreateKernel(program, "Kernel_Hk_field", &err);
    if (!ker_hk_f || err != CL_SUCCESS)
    {
        cout<<"Error: Failed to create compute kernel 9!"<<endl;
        exit(1);
    }

    ker_hth_f = clCreateKernel(program, "Kernel_Hth_field", &err);
    if (!ker_hth_f || err != CL_SUCCESS)
    {
        cout<<"Error: Failed to create compute kernel 10!"<<endl;
        exit(1);
    }

    ker_M_ini = clCreateKernel(program, "Kernel_M_cufft_Initialization", &err);
    if (!ker_M_ini || err != CL_SUCCESS)
    {
        cout<<"Error: Failed to create compute kernel 11!"<<endl;
        exit(1);
    }

    ker_fft_M_G = clCreateKernel(program, "Kernel_CUFFT_M_times_G", &err);
    if (!ker_fft_M_G || err != CL_SUCCESS)
    {
        cout<<"Error: Failed to create compute kernel 12!"<<endl;
        exit(1);
    }

    ker_fft_hd = clCreateKernel(program, "Kernel_Hd_set", &err);
    if (!ker_fft_hd || err != CL_SUCCESS)
    {
        cout<<"Error: Failed to create compute kernel 13!"<<endl;
        exit(1);
    }

    ker_T_ext = clCreateKernel(program, "Kernel_T_ext_set", &err);
    if (!ker_T_ext || err != CL_SUCCESS)
    {
        cout<<"Error: Failed to create compute kernel 14!"<<endl;
        exit(1);
    }

    ker_T = clCreateKernel(program, "Kernel_T_set", &err);
    if (!ker_T || err != CL_SUCCESS)
    {
        cout<<"Error: Failed to create compute kernel 15!"<<endl;
        exit(1);
    } 

    ker_H_ext = clCreateKernel(program, "Kernel_Happl_ext_set", &err);
    if (!ker_H_ext || err != CL_SUCCESS)
    {
        cout<<"Error: Failed to create compute kernel 16!"<<endl;
        exit(1);
    } 

    ker_H = clCreateKernel(program, "Kernel_Happl_set", &err);
    if (!ker_H || err != CL_SUCCESS)
    {
        cout<<"Error: Failed to create compute kernel 17!"<<endl;
        exit(1);
    } 

    ker_MgPar = clCreateKernel(program, "Kernel_MagPara_set", &err);
    if (!ker_MgPar || err != CL_SUCCESS)
    {
        cout<<"Error: Failed to create compute kernel 18!"<<endl;
        exit(1);
    } 
    ker_gaussian = clCreateKernel(program, "Kernel_gaussian_distribution", &err);
    if (!ker_gaussian || err != CL_SUCCESS)
    {
        cout<<"Error: Failed to create compute kernel 19!"<<endl;
        exit(1);
    }

    /* Create buffers for the kernel */

    AllocateDeviceMemory(ctx);

    // Write indicator array into device memory 
    //
    int *indicator1 = new int[Nx*Ny*Nz];
    int *indicator2 = new int[Nx*Ny*Nz];
    int *indicator3 = new int[Nx*Ny*Nz];
    double *indicator4 = new double[Nx*Ny*Nz];
    double *indicator5 = new double[Nx*Ny*Nz];
    rfile1.open("indicator.dat");
    for (int i = 0; i < Nx*Ny*Nz; i++){
	//fscanf(rfile1, "%d %d %d %lf %lf", &indicator1[i], &indicator2[i], &indicator3[i], &indicator4[i], &indicator5[i]);
		//printf("%d \n", indicator[i]);
        rfile1>>indicator1[i]>>indicator2[i]>>indicator3[i]>>indicator4[i]>>indicator5[i];
	}
	rfile1.close();
    err = clEnqueueWriteBuffer(queue, dev_indicator1, CL_TRUE, 0, sizeof(cl_int) * Nx * Ny * Nz, indicator1, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, dev_indicator2, CL_TRUE, 0, sizeof(cl_int) * Nx * Ny * Nz, indicator2, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, dev_indicator3, CL_TRUE, 0, sizeof(cl_int) * Nx * Ny * Nz, indicator3, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        cout<<"Error: Failed to write to source array (indicator1 & 2 & 3)!"<<endl;
        exit(1);
    }

    for (int idx = 0; idx < Nx*Ny*Nz; idx++){
		std_Ku[idx] = indicator4[idx];
		std_Aex[idx] = indicator5[idx];
		/*fprintf(wfile5, "%12.3lf \n", std_Ku[idx]);*/
	}

    err = clEnqueueWriteBuffer(queue, dev_std_Ku, CL_TRUE, 0, sizeof(cl_double) * Nx * Ny * Nz, std_Ku, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, dev_std_Aex, CL_TRUE, 0, sizeof(cl_double) * Nx * Ny * Nz, std_Aex, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        cout<<"Error: Failed to write to source array (std_Aex & std_Ku)!"<<endl;
        exit(1);
    }
    
    delete []indicator1;
    delete []indicator2;
    delete []indicator3;
    delete []indicator4;
    delete []indicator5;

    /* Initialize Ha Hth Hk */
    err = clSetKernelArg(ker_ini, 0, sizeof(cl_mem),  &dev_theta);
    err |= clSetKernelArg(ker_ini, 1, sizeof(cl_mem),  &dev_phi);
    err |= clSetKernelArg(ker_ini, 2, sizeof(cl_mem),  &dev_a_theta);
    err |= clSetKernelArg(ker_ini, 3, sizeof(cl_mem),  &dev_b_theta);
    err |= clSetKernelArg(ker_ini, 4, sizeof(cl_mem),  &dev_c_theta);
    err |= clSetKernelArg(ker_ini, 5, sizeof(cl_mem),  &dev_d_theta);
    err |= clSetKernelArg(ker_ini, 6, sizeof(cl_mem),  &dev_a_phi);
    err |= clSetKernelArg(ker_ini, 7, sizeof(cl_mem),  &dev_b_phi);
    err |= clSetKernelArg(ker_ini, 8, sizeof(cl_mem),  &dev_c_phi);
    err |= clSetKernelArg(ker_ini, 9, sizeof(cl_mem),  &dev_d_phi);
    err |= clSetKernelArg(ker_ini, 10, sizeof(cl_mem),  &dev_Ha_x);
    err |= clSetKernelArg(ker_ini, 11, sizeof(cl_mem),  &dev_Ha_y);
    err |= clSetKernelArg(ker_ini, 12, sizeof(cl_mem),  &dev_Ha_z);
    err |= clSetKernelArg(ker_ini, 13, sizeof(cl_mem),  &dev_Hth_x);
    err |= clSetKernelArg(ker_ini, 14, sizeof(cl_mem),  &dev_Hth_y);
    err |= clSetKernelArg(ker_ini, 15, sizeof(cl_mem),  &dev_Hth_z);
    err |= clSetKernelArg(ker_ini, 16, sizeof(cl_mem),  &dev_Hk_x);
    err |= clSetKernelArg(ker_ini, 17, sizeof(cl_mem),  &dev_Hk_y);
    err |= clSetKernelArg(ker_ini, 18, sizeof(cl_mem),  &dev_Hk_z);
    err |= clSetKernelArg(ker_ini, 19, sizeof(cl_mem),  &dev_Hd_x);
    err |= clSetKernelArg(ker_ini, 20, sizeof(cl_mem),  &dev_Hd_y);
    err |= clSetKernelArg(ker_ini, 21, sizeof(cl_mem),  &dev_Hd_z);
    err |= clSetKernelArg(ker_ini, 22, sizeof(cl_mem),  &dev_d_theta_d_t);
    err |= clSetKernelArg(ker_ini, 23, sizeof(cl_mem),  &dev_d_phi_d_t);
    err |= clSetKernelArg(ker_ini, 24, sizeof(cl_mem),  &dev_indicator1);
    err |= clSetKernelArg(ker_ini, 25, sizeof(cl_mem),  &dev_indicator3);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set ker_ini arguments! %d\n", err);
        exit(1);
    }

    /* Execute the kernel and read back results */
    err = clEnqueueNDRangeKernel(queue, ker_ini, 3, NULL, numWorkItems, numWorkItems_local, 0, NULL, &event);
    if (err)
    {
        printf("Error: Failed to execute kernel 1!\n");
        exit(1);
    }
    err = clWaitForEvents(1, &event);

    /////Initialize dev_indicator////

    err = clSetKernelArg(ker_dev_ind_apron, 0, sizeof(cl_mem),  &dev_indicator1);
    err |= clSetKernelArg(ker_dev_ind_apron, 1, sizeof(cl_mem),  &dev_indicator1_temp);
    err |= clSetKernelArg(ker_dev_ind_apron, 2, sizeof(cl_mem),  &dev_indicator2);
    err |= clSetKernelArg(ker_dev_ind_apron, 3, sizeof(cl_mem),  &dev_indicator2_temp);
    if (err != CL_SUCCESS)
    {
        cout<<"Error: Failed to set ker_dev_ind_apron arguments!"<<err<<endl;
        exit(1);
    }
    err = clEnqueueNDRangeKernel(queue, ker_dev_ind_apron, 3, NULL, numWorkItems, numWorkItems_local, 0, NULL, &event);
    if (err)
    {
        cout<<"Error: Failed to execute kernel 2!"<<endl;
        exit(1);
    }
    err = clWaitForEvents(1, &event);


    // Read back the results from the device to verify the output
    // int out[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    // err = clEnqueueReadBuffer(queue, dev_indicator2_temp, CL_TRUE, 0, sizeof(cl_int) * (Nx + 2) * (Ny + 2) * (Nz + 2), out, 0, NULL, NULL);
    // if (err != CL_SUCCESS)
    // {
    //     printf("Error: Failed to read output array! %d\n", err);
    //     exit(1);
    // }

    //watch/////
	// FILE *watch;
	// watch = fopen("p2.dat", "w");
	// //cudaMemcpy(watch2, dev_theta, (Nx)*(Ny)*(Nz)*sizeof(double), cudaMemcpyDeviceToHost);
	// for (int k = 0; k < (Nz + 2); k++){
	// 	for (int j = 0; j < (Ny + 2); j++){
	// 		for (int i = 0; i < (Nx + 2); i++){
	// 			fprintf(watch, " %d", out[i+j*(Nx + 2)+k*(Nx + 2)*(Ny + 2)]);
	// 		}
	// 		fprintf(watch, "\n");
	// 	}
	// 	fprintf(watch, "\n \n");
	// }
	// fclose(watch);
    ////////



    clock_t start, stop;
	double time=0.0;
	assert((start = clock())!=-1);

    ///////G matrix Initialization////////
    // 	for (int k = 0; k < lz_zero_pad; k++){
	// 	for (int j = 0; j < ly_zero_pad; j++){
	// 		for (int i = 0; i < lx_zero_pad; i++){
	// 			idx = i + j*lx_zero_pad + k*lx_zero_pad*ly_zero_pad;
	// 			Gxx_1d_real[idx] = 0.;
	// 			Gxy_1d_real[idx] = 0.;
	// 			Gxz_1d_real[idx] = 0.;
	// 			Gyx_1d_real[idx] = 0.;
	// 			Gyy_1d_real[idx] = 0.;
	// 			Gyz_1d_real[idx] = 0.;
	// 			Gzx_1d_real[idx] = 0.;
	// 			Gzy_1d_real[idx] = 0.;
	// 			Gzz_1d_real[idx] = 0.;
	// 		}
	// 	}
	// }
    if (!G_tensor(lx_zero_pad, ly_zero_pad, lz_zero_pad, ctx, queue))
    { 
        cout<<"!G_tensor() fail!\n"<<endl; 
        exit(1); 
        }
    

    for (unsigned long tt = 0; tt < TOTAL_TIME; tt++)
	{
		Mx_bar1[tt] = 0;
		My_bar1[tt] = 0;
		Mz_bar1[tt] = 0;
		Mx_bar2[tt] = 0;
		My_bar2[tt] = 0;
		Mz_bar2[tt] = 0;
		Mx_bar3[tt] = 0;
		My_bar3[tt] = 0;
		Mz_bar3[tt] = 0;
		M_bar [tt] = 0;
	}
	M_bar_t = 0;

	//if (!Moving_Head(0, queue)) 
    if (int ret = Moving_Head(0, queue, ker_H, ker_H_ext, ker_T, ker_T_ext, ker_MgPar )) 
    { 
        if (ret == 2) 
        {cout<<"Moving_Head(0) failed and terminated!"; exit(1);}
        else {cout<<"Moving_Head(0) failed!";}
    }

    // // Read back the results from the device to verify the output
    // cl_double out[(Nx) * (Ny) * (Nz)];
    // err = clEnqueueReadBuffer(queue, dev_Happl_z, CL_TRUE, 0, sizeof(cl_double) * (Nx) * (Ny) * (Nz), out, 0, NULL, NULL);
    // if (err != CL_SUCCESS)
    // {
    //     printf("Error: Failed to read output array! %d\n", err);
    //     exit(1);
    // }

    // //watch/////
	// FILE *watch;
	// watch = fopen("p2.dat", "w");
	// //cudaMemcpy(watch2, dev_theta, (Nx)*(Ny)*(Nz)*sizeof(double), cudaMemcpyDeviceToHost);
	// for (int k = 0; k < (Nz); k++){
	// 	for (int j = 0; j < (Ny); j++){
	// 		for (int i = 0; i < (Nx); i++){
	// 			fprintf(watch, " %lf", out[i+j*(Nx)+k*(Nx)*(Ny)]);
	// 		}
	// 		fprintf(watch, "\n");
	// 	}
	// 	fprintf(watch, "\n \n");
	// }
	// fclose(watch);
    // ////////

    ////////Normal Distribution Initialization//////////////
    // double gaussArray;
    // MTRand grnd;
    // //int iseed = 100;
    // grnd.seed(iseed);
    // for (int ii = 0; ii < DEG_FREEDOM; ii++)
    // {
    //     gasarray[ii] = mtrandom::gaussian();
    //     //printf("%lf\t", gasarray[ii]);
    // }

    clrngMrg31k3pStreamCreator *RNG_creator = clrngMrg31k3pCopyStreamCreator(NULL, (clrngStatus *)&err);;
    clrngMrg31k3pStreamState seed;
    seed.g1[0] = 101;
    seed.g1[1] = 100;
    seed.g1[2] = 11;
    seed.g2[0] = 203;
    seed.g2[1] = 162;
    seed.g2[2] = 171;

    clrngMrg31k3pSetBaseCreatorState(RNG_creator, &seed);

    streams = clrngMrg31k3pCreateStreams(RNG_creator, DEG_FREEDOM, &streamBufferSize, (clrngStatus *)&err);
    
    dev_streams = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, streamBufferSize, streams, &err);
   
    err = clSetKernelArg(ker_gaussian, 0, sizeof(cl_mem),  &dev_streams);
    err |= clSetKernelArg(ker_gaussian, 1, sizeof(cl_mem), &dev_GasArray);
    if (err != CL_SUCCESS)
    {
        cout<<"Error: Failed to set ker_gaussian arguments!"<<err<<endl;
        exit(1);
    }
//////////////////////Set Hth////////
    err = clSetKernelArg(ker_hth_f, 0, sizeof(cl_mem),  &dev_GasArray);            
    err |= clSetKernelArg(ker_hth_f, 1, sizeof(cl_mem),  &dev_Hth_x);            
    err |= clSetKernelArg(ker_hth_f, 2, sizeof(cl_mem),  &dev_Hth_y);            
    err |= clSetKernelArg(ker_hth_f, 3, sizeof(cl_mem),  &dev_Hth_z);            
    err |= clSetKernelArg(ker_hth_f, 4, sizeof(cl_mem),  &dev_D);                       
                  
    if (err != CL_SUCCESS)
    {
        cout<<"Error: Failed to set ker_hth_f arguments!"<<err<<endl;
        exit(1);
    }

    for (unsigned long tt = 0; tt < TOTAL_TIME; tt++)
    {
        if (tt > 0 && tt < EQUI_START_TIME)
        {
            //if (!Moving_Head(tt, queue)) { printf("Moving_Head() fails!\n"); exit(1); }; 
            if(int ret = Moving_Head(tt, queue, ker_H, ker_H_ext, ker_T, ker_T_ext, ker_MgPar )) 
            {
                if (ret == 2) 
                {cout<<"Moving_Head("<<tt<<") failed and terminated!"; 
                exit(1);}
                else {cout<<"Moving_Head("<<tt<<") failed!";}
            }
///////////////Hd///////s
            if (int ret = Hms(lx_zero_pad, ly_zero_pad, lz_zero_pad, ctx, queue, ker_M_ini, ker_fft_M_G, ker_fft_hd))
            {
                if (ret == 2) 
                {cout<<"Hms failed and terminated!"; exit(1);}
                else {cout<<"Hms fails!";}
            };
  ///////////Hth///////
            err = clEnqueueNDRangeKernel(queue, ker_gaussian, 3, NULL, numWorkItems, numWorkItems_local, 0, NULL, &event);
            if (err)
            {
                cout<<"Error: Failed to execute ker_gaussian!"<<endl;
                exit(1);
            } 
            err = clWaitForEvents(1, &event);
// // ///////watch////////

//             {		
//             cl_event event = 0;
// 			double *out=new double[DEG_FREEDOM];
// 			cl_int err = clEnqueueReadBuffer(queue, dev_GasArray, CL_TRUE, 0, sizeof(cl_double) * DEG_FREEDOM, out, 0, NULL, &event);
// 			if (err != CL_SUCCESS)
// 			{
// 				printf("Error: Failed to read output array! %d\n", err);
// 				exit(1);
// 			}
// 			err = clWaitForEvents(1, &event);			
// 			ofstream wfile_o;
// 			wfile_o.open("dev_GasArray.dat");
// 			// for (int k = 0; k < Nz; k++){
// 			// 	for (int j = 0; j < Ny; j++){
// 			// 		for (int i = 0; i < Nx; i++){
// 			// 						//fprintf(watch, " %lf", out[i+j*(Nx)+k*(Nx)*(Ny)]);
// 			// 			wfile_o<<setw(10)<<out[i+j*Nx+k*Nx*Ny]<<" ";
// 			// 		}
// 			// 					//fprintf(watch, "\n");
// 			// 		wfile_o<<endl;
// 			// 	}
// 			// 				//fprintf(watch, "\n \n");
// 			// wfile_o<<endl<<endl;
// 			// }
//             for (int i = 0; i< DEG_FREEDOM; i++)
//             {
//                 wfile_o<<out[i]<<"\t";
//             }
// 						//fclose(watch);
// 			wfile_o.close();
// 			delete []out;
// 	        }
// // ////////////
            // err = clEnqueueWriteBuffer(queue, dev_GasArray, CL_TRUE, 0, sizeof(cl_double) * DEG_FREEDOM, gasarray, 0, NULL, NULL);
            // if (err != CL_SUCCESS)
            // {
            //     printf("Error: Failed to write to source array (random array)!\n");
            //     exit(1);
            // }
            // err = clSetKernelArg(ker_hth_f, 0, sizeof(cl_mem),  &dev_GasArray);            
            // err |= clSetKernelArg(ker_hth_f, 1, sizeof(cl_mem),  &dev_Hth_x);            
            // err |= clSetKernelArg(ker_hth_f, 2, sizeof(cl_mem),  &dev_Hth_y);            
            // err |= clSetKernelArg(ker_hth_f, 3, sizeof(cl_mem),  &dev_Hth_z);            
            // err |= clSetKernelArg(ker_hth_f, 4, sizeof(cl_mem),  &dev_D); 
            // if (err != CL_SUCCESS)
            // {
            //     cout<<"Error: Failed to set ker_hth_f arguments!"<<err<<endl;
            //     exit(1);
            // }

            err = clEnqueueNDRangeKernel(queue, ker_hth_f, 3, NULL, numWorkItems, numWorkItems_local, 0, NULL, &event);
            if (err)
            {
                cout<<"Error: Failed to execute kernel 10!"<<endl;
                exit(1);
            }
            // for (int ii = 0; ii < DEG_FREEDOM; ii++)
            // {
            //     gasarray[ii] = mtrandom::gaussian();
            //     //printf("%lf\t", gasarray[ii]);
            // }
            err = clWaitForEvents(1, &event);
// ////////////watch//////////
//              	{		cl_event event = 0;
// 			double *out=new double[Nx * Ny * Nz];
// 			cl_int err = clEnqueueReadBuffer(queue, dev_Hth_x, CL_TRUE, 0, sizeof(cl_double) * Nx * Ny * Nz, out, 0, NULL, &event);
// 			if (err != CL_SUCCESS)
// 			{
// 				printf("Error: Failed to read output array! %d\n", err);
// 				exit(1);
// 			}
// 			err = clWaitForEvents(1, &event);			
// 			ofstream wfile_o;
// 			wfile_o.open("dev_Hth_x.dat");
// 			for (int k = 0; k < Nz; k++){
// 				for (int j = 0; j < Ny; j++){
// 					for (int i = 0; i < Nx; i++){
// 									//fprintf(watch, " %lf", out[i+j*(Nx)+k*(Nx)*(Ny)]);
// 						wfile_o<<setw(10)<<out[i+j*Nx+k*Nx*Ny]<<" ";
// 					}
// 								//fprintf(watch, "\n");
// 					wfile_o<<endl;
// 				}
// 							//fprintf(watch, "\n \n");
// 			wfile_o<<endl<<endl;
// 			}
// 						//fclose(watch);
// 			wfile_o.close();
// 			delete []out;
// 	}

// ////////////////////////////////////

            if (!Ha_field(ZERO, dev_a_theta, dev_a_phi, ker_MA_apron, ker_Ha_apron, queue)) 
            { cout<<"Ha_field(ZERO) fail!"<<endl; exit(1); }


            // Kernel_d_theta_phi_d_t<<<grids1, blocks1>>>(dev_d_theta_d_t, dev_d_phi_d_t,
			// 									        dev_theta,  dev_phi, 
			// 										    dev_a_theta, dev_a_phi,
			// 										    dev_Ha_x, dev_Ha_y, dev_Ha_z,
			// 										    dev_Hth_x, dev_Hth_y, dev_Hth_z,
			// 										    dev_Hd_x, dev_Hd_y, dev_Hd_z,
			// 										    dev_Happl_x, dev_Happl_y, dev_Happl_z,
			// 										    dev_Ku, dev_Ms, dev_alpha, dev_gamma);
            double hh = h/2.0;
            err = clSetKernelArg(ker_dqj_dt, 0, sizeof(cl_mem),  &dev_d_theta_d_t);
            err |= clSetKernelArg(ker_dqj_dt, 1, sizeof(cl_mem),  &dev_d_phi_d_t);
            err |= clSetKernelArg(ker_dqj_dt, 2, sizeof(cl_mem),  &dev_theta);
            err |= clSetKernelArg(ker_dqj_dt, 3, sizeof(cl_mem),  &dev_phi);
            err |= clSetKernelArg(ker_dqj_dt, 4, sizeof(cl_mem),  &dev_a_theta);            
            err |= clSetKernelArg(ker_dqj_dt, 5, sizeof(cl_mem),  &dev_a_phi);   
            err |= clSetKernelArg(ker_dqj_dt, 6, sizeof(cl_mem),  &dev_Ha_x);            
            err |= clSetKernelArg(ker_dqj_dt, 7, sizeof(cl_mem),  &dev_Ha_y);            
            err |= clSetKernelArg(ker_dqj_dt, 8, sizeof(cl_mem),  &dev_Ha_z);            
            err |= clSetKernelArg(ker_dqj_dt, 9, sizeof(cl_mem),  &dev_Hth_x);            
            err |= clSetKernelArg(ker_dqj_dt, 10, sizeof(cl_mem),  &dev_Hth_y);            
            err |= clSetKernelArg(ker_dqj_dt, 11, sizeof(cl_mem),  &dev_Hth_z);            
            err |= clSetKernelArg(ker_dqj_dt, 12, sizeof(cl_mem),  &dev_Hd_x);            
            err |= clSetKernelArg(ker_dqj_dt, 13, sizeof(cl_mem),  &dev_Hd_y);            
            err |= clSetKernelArg(ker_dqj_dt, 14, sizeof(cl_mem),  &dev_Hd_z);            
            err |= clSetKernelArg(ker_dqj_dt, 15, sizeof(cl_mem),  &dev_Happl_x);            
            err |= clSetKernelArg(ker_dqj_dt, 16, sizeof(cl_mem),  &dev_Happl_y);            
            err |= clSetKernelArg(ker_dqj_dt, 17, sizeof(cl_mem),  &dev_Happl_z);            
            err |= clSetKernelArg(ker_dqj_dt, 18, sizeof(cl_mem),  &dev_Ku);            
            err |= clSetKernelArg(ker_dqj_dt, 19, sizeof(cl_mem),  &dev_Ms);            
            err |= clSetKernelArg(ker_dqj_dt, 20, sizeof(cl_mem),  &dev_alpha);            
            err |= clSetKernelArg(ker_dqj_dt, 21, sizeof(cl_mem),  &dev_gamma);            
                   
            if (err != CL_SUCCESS)
            {
                cout<<"Error: Failed to set ker_dqj_dt arguments!"<<err<<endl;
                exit(1);
            }
            err = clEnqueueNDRangeKernel(queue, ker_dqj_dt, 3, NULL, numWorkItems, numWorkItems_local, 0, NULL, &event);
            if (err)
            {
                cout<<"Error: Failed to execute kernel 7!"<<endl;
                exit(1);
            }
            err = clWaitForEvents(1, &event);
			// Kernel_a_theta_phi<<<grids1, blocks1>>>(dev_a_theta, dev_a_phi,
			// 									    dev_d_theta_d_t, dev_d_phi_d_t);

            err = clSetKernelArg(ker_a_qj, 0, sizeof(cl_mem),  &dev_a_theta);
            err |= clSetKernelArg(ker_a_qj, 1, sizeof(cl_mem),  &dev_a_phi);
            err |= clSetKernelArg(ker_a_qj, 2, sizeof(cl_mem),  &dev_d_theta_d_t);
            err |= clSetKernelArg(ker_a_qj, 3, sizeof(cl_mem),  &dev_d_phi_d_t);
            if (err != CL_SUCCESS)
            {
                cout<<"Error: Failed to set ker_a_qj arguments!"<<err<<endl;
                exit(1);
            }         

            err = clEnqueueNDRangeKernel(queue, ker_a_qj, 3, NULL, numWorkItems, numWorkItems_local, 0, NULL, &event);
            if (err)
            {
                cout<<"Error: Failed to execute kernel 5!"<<endl;
                exit(1);
            }
            err = clWaitForEvents(1, &event);

			if (!Ha_field(h/2, dev_a_theta, dev_a_phi, ker_MA_apron, ker_Ha_apron, queue)) 
            { cout<<"Ha_field_a(h/2) fail!"<<endl;  exit(1); }



			// Kernel_b_theta_phi<<<grids1, blocks1>>>(dev_b_theta, dev_b_phi,                                   
			// 									    dev_theta, dev_phi,                                      
			// 									    dev_a_theta, dev_a_phi,
			// 									    dev_Ha_x, dev_Ha_y, dev_Ha_z,
			// 									    dev_Hth_x, dev_Hth_y, dev_Hth_z,
			// 									    dev_Hd_x, dev_Hd_y, dev_Hd_z,
			// 									    dev_Happl_x, dev_Happl_y, dev_Happl_z,
			// 									    dev_Ku, dev_Ms, dev_alpha, dev_gamma, h);
            err = clSetKernelArg(ker_x_qj, 0, sizeof(cl_mem),  &dev_b_theta);
            err |= clSetKernelArg(ker_x_qj, 1, sizeof(cl_mem),  &dev_b_phi);
            err |= clSetKernelArg(ker_x_qj, 2, sizeof(cl_mem),  &dev_theta);
            err |= clSetKernelArg(ker_x_qj, 3, sizeof(cl_mem),  &dev_phi);
            err |= clSetKernelArg(ker_x_qj, 4, sizeof(cl_mem),  &dev_a_theta);
            err |= clSetKernelArg(ker_x_qj, 5, sizeof(cl_mem),  &dev_a_phi);
            err |= clSetKernelArg(ker_x_qj, 6, sizeof(cl_mem),  &dev_Ha_x);
            err |= clSetKernelArg(ker_x_qj, 7, sizeof(cl_mem),  &dev_Ha_y);
            err |= clSetKernelArg(ker_x_qj, 8, sizeof(cl_mem),  &dev_Ha_z);
            err |= clSetKernelArg(ker_x_qj, 9, sizeof(cl_mem),  &dev_Hth_x);
            err |= clSetKernelArg(ker_x_qj, 10, sizeof(cl_mem),  &dev_Hth_y);
            err |= clSetKernelArg(ker_x_qj, 11, sizeof(cl_mem),  &dev_Hth_z);
            err |= clSetKernelArg(ker_x_qj, 12, sizeof(cl_mem),  &dev_Hd_x);
            err |= clSetKernelArg(ker_x_qj, 13, sizeof(cl_mem),  &dev_Hd_y);
            err |= clSetKernelArg(ker_x_qj, 14, sizeof(cl_mem),  &dev_Hd_z);
            err |= clSetKernelArg(ker_x_qj, 15, sizeof(cl_mem),  &dev_Happl_x);
            err |= clSetKernelArg(ker_x_qj, 16, sizeof(cl_mem),  &dev_Happl_y);
            err |= clSetKernelArg(ker_x_qj, 17, sizeof(cl_mem),  &dev_Happl_z);
            err |= clSetKernelArg(ker_x_qj, 18, sizeof(cl_mem),  &dev_Ku);
            err |= clSetKernelArg(ker_x_qj, 19, sizeof(cl_mem),  &dev_Ms);
            err |= clSetKernelArg(ker_x_qj, 20, sizeof(cl_mem),  &dev_alpha);
            err |= clSetKernelArg(ker_x_qj, 21, sizeof(cl_mem),  &dev_gamma);
            err |= clSetKernelArg(ker_x_qj, 22, sizeof(cl_double),  &hh);

            if (err != CL_SUCCESS)
            {
                cout<<"Error: Failed to set ker_b_qj arguments!"<<err<<endl;
                exit(1);
            } 
            err = clEnqueueNDRangeKernel(queue, ker_x_qj, 3, NULL, numWorkItems, numWorkItems_local, 0, NULL, &event);
            if (err)
            {
                cout<<"Error: Failed to execute kernel 6b!"<<endl;
                exit(1);
            }
            err = clWaitForEvents(1, &event);
			
			if (!Ha_field(h/2, dev_b_theta, dev_b_phi, ker_MA_apron, ker_Ha_apron, queue)) 
            { cout<<"Ha_field_a(h/2) fail!"<<endl;  exit(1); }
			// Kernel_c_theta_phi<<<grids1, blocks1>>>(dev_c_theta, dev_c_phi,
			// 									    dev_theta, dev_phi, 
	 		// 									    dev_b_theta, dev_b_phi,
			// 									    dev_Ha_x, dev_Ha_y, dev_Ha_z,
			// 									    dev_Hth_x, dev_Hth_y, dev_Hth_z,
			// 									    dev_Hd_x, dev_Hd_y, dev_Hd_z,
			// 									    dev_Happl_x, dev_Happl_y, dev_Happl_z,
			// 									    dev_Ku, dev_Ms, dev_alpha, dev_gamma, h);
            err = clSetKernelArg(ker_x_qj, 0, sizeof(cl_mem),  &dev_c_theta);
            err |= clSetKernelArg(ker_x_qj, 1, sizeof(cl_mem),  &dev_c_phi);
            // err |= clSetKernelArg(ker_x_qj, 2, sizeof(cl_mem),  &dev_theta);
            // err |= clSetKernelArg(ker_x_qj, 3, sizeof(cl_mem),  &dev_phi);
            err |= clSetKernelArg(ker_x_qj, 4, sizeof(cl_mem),  &dev_b_theta);
            err |= clSetKernelArg(ker_x_qj, 5, sizeof(cl_mem),  &dev_b_phi);
            // err |= clSetKernelArg(ker_x_qj, 6, sizeof(cl_mem),  &dev_Ha_x);
            // err |= clSetKernelArg(ker_x_qj, 7, sizeof(cl_mem),  &dev_Ha_y);
            // err |= clSetKernelArg(ker_x_qj, 8, sizeof(cl_mem),  &dev_Ha_z);
            // err |= clSetKernelArg(ker_x_qj, 9, sizeof(cl_mem),  &dev_Hth_x);
            // err |= clSetKernelArg(ker_x_qj, 10, sizeof(cl_mem),  &dev_Hth_y);
            // err |= clSetKernelArg(ker_x_qj, 11, sizeof(cl_mem),  &dev_Hth_z);
            // err |= clSetKernelArg(ker_x_qj, 12, sizeof(cl_mem),  &dev_Hd_x);
            // err |= clSetKernelArg(ker_x_qj, 13, sizeof(cl_mem),  &dev_Hd_y);
            // err |= clSetKernelArg(ker_x_qj, 14, sizeof(cl_mem),  &dev_Hd_z);
            // err |= clSetKernelArg(ker_x_qj, 15, sizeof(cl_mem),  &dev_Happl_x);
            // err |= clSetKernelArg(ker_x_qj, 16, sizeof(cl_mem),  &dev_Happl_y);
            // err |= clSetKernelArg(ker_x_qj, 17, sizeof(cl_mem),  &dev_Happl_z);
            // err |= clSetKernelArg(ker_x_qj, 18, sizeof(cl_mem),  &dev_Ku);
            // err |= clSetKernelArg(ker_x_qj, 19, sizeof(cl_mem),  &dev_Ms);
            // err |= clSetKernelArg(ker_x_qj, 20, sizeof(cl_mem),  &dev_alpha);
            // err |= clSetKernelArg(ker_x_qj, 21, sizeof(cl_mem),  &dev_gamma);
            // err |= clSetKernelArg(ker_x_qj, 22, sizeof(cl_double),  &hh);

            if (err != CL_SUCCESS)
            {
                cout<<"Error: Failed to set ker_c_qj arguments!"<<err<<endl;
                exit(1);
            } 
            err = clEnqueueNDRangeKernel(queue, ker_x_qj, 3, NULL, numWorkItems, numWorkItems_local, 0, NULL, &event);
            if (err)
            {
                cout<<"Error: Failed to execute kernel 6c!"<<endl;
                exit(1);
            }
            err = clWaitForEvents(1, &event);

			if (!Ha_field(h,   dev_c_theta, dev_c_phi, ker_MA_apron, ker_Ha_apron, queue)) 
            { cout<<"Ha_field_a(h/2) fail!"<<endl;  exit(1); }		
			// Kernel_d_theta_phi<<<grids1, blocks1>>>(dev_d_theta, dev_d_phi,
			// 								    	dev_theta, dev_phi, 
			// 									    dev_c_theta, dev_c_phi,
			// 									    dev_Ha_x, dev_Ha_y, dev_Ha_z,
			// 									    dev_Hth_x, dev_Hth_y, dev_Hth_z,
			// 									    dev_Hd_x, dev_Hd_y, dev_Hd_z,
			// 									    dev_Happl_x, dev_Happl_y, dev_Happl_z,
			// 									    dev_Ku, dev_Ms, dev_alpha, dev_gamma, h);

            err = clSetKernelArg(ker_x_qj, 0, sizeof(cl_mem),  &dev_d_theta);
            err |= clSetKernelArg(ker_x_qj, 1, sizeof(cl_mem),  &dev_d_phi);
            // err |= clSetKernelArg(ker_x_qj, 2, sizeof(cl_mem),  &dev_theta);
            // err |= clSetKernelArg(ker_x_qj, 3, sizeof(cl_mem),  &dev_phi);
            err |= clSetKernelArg(ker_x_qj, 4, sizeof(cl_mem),  &dev_c_theta);
            err |= clSetKernelArg(ker_x_qj, 5, sizeof(cl_mem),  &dev_c_phi);
            // err |= clSetKernelArg(ker_x_qj, 6, sizeof(cl_mem),  &dev_Ha_x);
            // err |= clSetKernelArg(ker_x_qj, 7, sizeof(cl_mem),  &dev_Ha_y);
            // err |= clSetKernelArg(ker_x_qj, 8, sizeof(cl_mem),  &dev_Ha_z);
            // err |= clSetKernelArg(ker_x_qj, 9, sizeof(cl_mem),  &dev_Hth_x);
            // err |= clSetKernelArg(ker_x_qj, 10, sizeof(cl_mem),  &dev_Hth_y);
            // err |= clSetKernelArg(ker_x_qj, 11, sizeof(cl_mem),  &dev_Hth_z);
            // err |= clSetKernelArg(ker_x_qj, 12, sizeof(cl_mem),  &dev_Hd_x);
            // err |= clSetKernelArg(ker_x_qj, 13, sizeof(cl_mem),  &dev_Hd_y);
            // err |= clSetKernelArg(ker_x_qj, 14, sizeof(cl_mem),  &dev_Hd_z);
            // err |= clSetKernelArg(ker_x_qj, 15, sizeof(cl_mem),  &dev_Happl_x);
            // err |= clSetKernelArg(ker_x_qj, 16, sizeof(cl_mem),  &dev_Happl_y);
            // err |= clSetKernelArg(ker_x_qj, 17, sizeof(cl_mem),  &dev_Happl_z);
            // err |= clSetKernelArg(ker_x_qj, 18, sizeof(cl_mem),  &dev_Ku);
            // err |= clSetKernelArg(ker_x_qj, 19, sizeof(cl_mem),  &dev_Ms);
            // err |= clSetKernelArg(ker_x_qj, 20, sizeof(cl_mem),  &dev_alpha);
            // err |= clSetKernelArg(ker_x_qj, 21, sizeof(cl_mem),  &dev_gamma);
            err |= clSetKernelArg(ker_x_qj, 22, sizeof(cl_double),  &h);

            if (err != CL_SUCCESS)
            {
                cout<<"Error: Failed to set ker_d_qj arguments!"<<err<<endl;
                exit(1);
            } 
            err = clEnqueueNDRangeKernel(queue, ker_x_qj, 3, NULL, numWorkItems, numWorkItems_local, 0, NULL, &event);
            if (err)
            {
                cout<<"Error: Failed to execute kernel 6d!"<<endl;
                exit(1);
            }
            err = clWaitForEvents(1, &event);
		
			/* time evolution */     
			// Kernel_time_increment<<<grids1, blocks1>>>(dev_theta, dev_phi,
			//      									   dev_a_theta, dev_b_theta, dev_c_theta, dev_d_theta,   
			// 										   dev_a_phi,  dev_b_phi, dev_c_phi, dev_d_phi,   
			// 										   dev_Mx, dev_My, dev_Mz,
			// 										   dev_indicator1,
			// 									       h, dev_Ms);
            err = clSetKernelArg(ker_t_incr, 0, sizeof(cl_mem),  &dev_theta);
            err |= clSetKernelArg(ker_t_incr, 1, sizeof(cl_mem),  &dev_phi);
            err |= clSetKernelArg(ker_t_incr, 2, sizeof(cl_mem),  &dev_a_theta);
            err |= clSetKernelArg(ker_t_incr, 3, sizeof(cl_mem),  &dev_b_theta);
            err |= clSetKernelArg(ker_t_incr, 4, sizeof(cl_mem),  &dev_c_theta);
            err |= clSetKernelArg(ker_t_incr, 5, sizeof(cl_mem),  &dev_d_theta);
            err |= clSetKernelArg(ker_t_incr, 6, sizeof(cl_mem),  &dev_a_phi);
            err |= clSetKernelArg(ker_t_incr, 7, sizeof(cl_mem),  &dev_b_phi);
            err |= clSetKernelArg(ker_t_incr, 8, sizeof(cl_mem),  &dev_c_phi);
            err |= clSetKernelArg(ker_t_incr, 9, sizeof(cl_mem),  &dev_d_phi);
            err |= clSetKernelArg(ker_t_incr, 10, sizeof(cl_mem),  &dev_Mx);
            err |= clSetKernelArg(ker_t_incr, 11, sizeof(cl_mem),  &dev_My);
            err |= clSetKernelArg(ker_t_incr, 12, sizeof(cl_mem),  &dev_Mz);
            err |= clSetKernelArg(ker_t_incr, 13, sizeof(cl_mem),  &dev_indicator1);
            err |= clSetKernelArg(ker_t_incr, 14, sizeof(cl_double),  &h);
            err |= clSetKernelArg(ker_t_incr, 15, sizeof(cl_mem),  &dev_Ms);

            if (err != CL_SUCCESS)
            {
                cout<<"Error: Failed to set ker_t_incr arguments!"<<err<<endl;
                exit(1);
            } 
            err = clEnqueueNDRangeKernel(queue, ker_t_incr, 3, NULL, numWorkItems, numWorkItems_local, 0, NULL, &event);
            if (err)
            {
                cout<<"Error: Failed to execute kernel 8!"<<endl;
                exit(1);
            }
            err = clWaitForEvents(1, &event);


			// Kernel_Hk_field <<<grids1, blocks1>>>(dev_theta, dev_phi, 
			// 									  dev_Hk_x, dev_Hk_y, dev_Hk_z, 
			// 									  dev_Ku, dev_Ms);
            err = clSetKernelArg(ker_hk_f, 0, sizeof(cl_mem),  &dev_theta);
            err |= clSetKernelArg(ker_hk_f, 1, sizeof(cl_mem),  &dev_phi);
            err |= clSetKernelArg(ker_hk_f, 2, sizeof(cl_mem),  &dev_Hk_x);
            err |= clSetKernelArg(ker_hk_f, 3, sizeof(cl_mem),  &dev_Hk_y);
            err |= clSetKernelArg(ker_hk_f, 4, sizeof(cl_mem),  &dev_Hk_z);
            err |= clSetKernelArg(ker_hk_f, 5, sizeof(cl_mem),  &dev_Ku);
            err |= clSetKernelArg(ker_hk_f, 6, sizeof(cl_mem),  &dev_Ms);

            if (err != CL_SUCCESS)
            {
                cout<<"Error: Failed to set ker_hk_f arguments!"<<err<<endl;
                exit(1);
            } 
            err = clEnqueueNDRangeKernel(queue, ker_hk_f, 3, NULL, numWorkItems, numWorkItems_local, 0, NULL, &event);
            if (err)
            {
                cout<<"Error: Failed to execute kernel 9!"<<endl;
                exit(1);
            }
            err = clWaitForEvents(1, &event);
            //-------Calculate spacial avg of magnetization------------------------//
			/*for(int idx = 0; idx < Nx*Ny*Nz; idx++){
				Mx_bar[tt] = Mx_bar[tt] + Mx[idx]/(Nx*Ny*Nz);
				My_bar[tt] = My_bar[tt] + My[idx]/(Nx*Ny*Nz);
				Mz_bar[tt] = Mz_bar[tt] + Mz[idx]/(Nx*Ny*Nz);
			}
			M_bar[tt] = pow((pow(Mx_bar[tt], 2.0) + pow(My_bar[tt], 2.0) + pow(Mz_bar[tt],2.0)), 0.5);*/
			            	//     // // Read back the results from the device to verify the output
           // cl_double out[(Nx) * (Ny) * (Nz)];
            err = clEnqueueReadBuffer(queue, dev_Mz, CL_TRUE, 0, sizeof(cl_double) * (Nx) * (Ny) * (Nz), Mz, 0, NULL, &event);
            if (err != CL_SUCCESS)
            {
                printf("Error: Failed to read output array! %d\n", err);
                exit(1);
            }
            err = clWaitForEvents(1, &event);
            for (int j = 0; j < Ny; j++){
				for (int i = 0; i < Nx; i++){
					for (int k = 0; k < Nz_1; k++){
						idx = i + j*Nx + k*Nx*Ny;
						Mx_bar1[tt] = Mx_bar1[tt] + Mx[idx]/(Nx*Ny*Nz_1);
						My_bar1[tt] = My_bar1[tt] + My[idx]/(Nx*Ny*Nz_1);
						Mz_bar1[tt] = Mz_bar1[tt] + Mz[idx]/(Nx*Ny*Nz_1);
                    }
                }
            }
				//-----------------I/O------------------------------------------------//
			if ( (tt%100) == 0){
                cout<<"t="<<setw(7)<<tt<<setw(12)<<setiosflags(ios::fixed)<<setprecision(3)
                <<Mz_bar1[tt]<<setw(10)<<setiosflags(ios::fixed)<<setprecision(2)<<Happl*powf(-1, NextBit)
                <<setw(10)<<setiosflags(ios::fixed)<<setprecision(2)<<Happl_z[11]<<endl;
                            //watch/////
            //FILE *watch;
                ofstream wfile_Mz;
                wfile_Mz.open("dev_Mz.dat");
                for (int k = 0; k < (Nz); k++){
                    for (int j = 0; j < (Ny); j++){
                        for (int i = 0; i < (Nx); i++){
                            //fprintf(watch, " %lf", out[i+j*(Nx)+k*(Nx)*(Ny)]);
                            wfile_Mz<<Mz[i+j*(Nx)+k*(Nx)*(Ny)]<<" ";
                        }
                        //fprintf(watch, "\n");
                        wfile_Mz<<endl;
                    }
                    //fprintf(watch, "\n \n");
                    wfile_Mz<<endl<<endl;
                }
                //fclose(watch);
                wfile_Mz.close();
            }





           // watch = fopen("dev_Mz.dat", "w");
            //cudaMemcpy(watch2, dev_theta, (Nx)*(Ny)*(Nz)*sizeof(double), cudaMemcpyDeviceToHost);
        }
    }



	stop = clock();
	time = (double) (stop-start)/CLOCKS_PER_SEC;
	cout<<"Run time: "<<time<<endl;

    rfile2.close();
    wfile100.close();

    clReleaseEvent(event);

    ReleaseMemory();
    

    clReleaseKernel(ker_ini);
    clReleaseKernel(ker_dev_ind_apron);
    clReleaseKernel(ker_Ha_apron);
    clReleaseKernel(ker_MA_apron);
    clReleaseKernel(ker_x_qj);
    clReleaseKernel(ker_dqj_dt);
    clReleaseKernel(ker_hk_f);
    clReleaseKernel(ker_a_qj);
    clReleaseKernel(ker_t_incr);
    clReleaseKernel(ker_hth_f);
    clReleaseKernel(ker_M_ini);
    clReleaseKernel(ker_fft_M_G);
    clReleaseKernel(ker_fft_hd);
    clReleaseKernel(ker_T);
    clReleaseKernel(ker_T_ext);
    clReleaseKernel(ker_H);
    clReleaseKernel(ker_H_ext);
    clReleaseKernel(ker_MgPar);

    clrngMrg31k3pDestroyStreamCreator(RNG_creator);

    clReleaseMemObject(dev_streams);

    clReleaseProgram(program);

    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return 0;

    //Kernel_Initialization;
}