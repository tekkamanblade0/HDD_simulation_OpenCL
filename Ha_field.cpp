//#include "cutil.h"
#include "Parameters.h"


bool Ha_field(double hh, cl_mem &dev_x_theta, cl_mem &dev_x_phi, cl_kernel &ker_MA_apron, 
				cl_kernel &ker_Ha_apron, cl_command_queue &queue)
{

	cl_int err = 0;
	cl_event event = 0;
	// Kernel_M_A_with_apron<<<grids1, blocks1>>>(dev_M_temp_x, dev_M_temp_y, dev_M_temp_z, 
	// 									       dev_Ms_temp, dev_Aex1_temp, dev_Aex2_temp,
	// 								   	       dev_theta, dev_x_theta, dev_phi, dev_x_phi,
	// 								  	       dev_Ms, dev_Aex1, dev_Aex2, hh);

    err = clSetKernelArg(ker_MA_apron, 0, sizeof(cl_mem),  &dev_M_temp_x);
    err |= clSetKernelArg(ker_MA_apron, 1, sizeof(cl_mem),  &dev_M_temp_y);
	err |= clSetKernelArg(ker_MA_apron, 2, sizeof(cl_mem),  &dev_M_temp_z);
	err |= clSetKernelArg(ker_MA_apron, 3, sizeof(cl_mem),  &dev_Ms_temp);
	err |= clSetKernelArg(ker_MA_apron, 4, sizeof(cl_mem),  &dev_Aex1_temp);
	err |= clSetKernelArg(ker_MA_apron, 5, sizeof(cl_mem),  &dev_Aex2_temp);
	err |= clSetKernelArg(ker_MA_apron, 6, sizeof(cl_mem),  &dev_theta);
	err |= clSetKernelArg(ker_MA_apron, 7, sizeof(cl_mem),  &dev_x_theta);
	err |= clSetKernelArg(ker_MA_apron, 8, sizeof(cl_mem),  &dev_phi);
	err |= clSetKernelArg(ker_MA_apron, 9, sizeof(cl_mem),  &dev_x_phi);
	err |= clSetKernelArg(ker_MA_apron, 10, sizeof(cl_mem),  &dev_Ms);
	err |= clSetKernelArg(ker_MA_apron, 11, sizeof(cl_mem),  &dev_Aex1);
	err |= clSetKernelArg(ker_MA_apron, 12, sizeof(cl_mem),  &dev_Aex2);
	err |= clSetKernelArg(ker_MA_apron, 13, sizeof(cl_double),  &hh);
	if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set ker_MA_apron arguments! %d\n", err);
        exit(1);
    }

	err = clEnqueueNDRangeKernel(queue, ker_MA_apron, 3, NULL, numWorkItems, NULL, 0, NULL, &event);
    if (err)
    {
        printf("Error: Failed to execute kernel 3!\n");
        exit(1);
    }
    err = clWaitForEvents(1, &event);		





	// Kernel_Ha_with_apron<<<grids1, blocks1>>>(dev_M_temp_x, dev_M_temp_y, dev_M_temp_z,
	// 										  dev_Ms_temp, dev_Aex1_temp, dev_Aex2_temp,
	// 										  dev_Ha_x, dev_Ha_y, dev_Ha_z,
	// 										  dev_indicator1_temp, dev_indicator2_temp,
	// 										  dev_Ms, dev_Aex1, dev_Aex2, delta_x);
	//double delta_x = 1.50e-7;
    err = clSetKernelArg(ker_Ha_apron, 0, sizeof(cl_mem),  &dev_M_temp_x);
    err |= clSetKernelArg(ker_Ha_apron, 1, sizeof(cl_mem),  &dev_M_temp_y);
    err |= clSetKernelArg(ker_Ha_apron, 2, sizeof(cl_mem),  &dev_M_temp_z);
	err |= clSetKernelArg(ker_Ha_apron, 3, sizeof(cl_mem),  &dev_Ms_temp);
    err |= clSetKernelArg(ker_Ha_apron, 4, sizeof(cl_mem),  &dev_Aex1_temp);
    err |= clSetKernelArg(ker_Ha_apron, 5, sizeof(cl_mem),  &dev_Aex2_temp);
    err |= clSetKernelArg(ker_Ha_apron, 6, sizeof(cl_mem),  &dev_Ha_x);
    err |= clSetKernelArg(ker_Ha_apron, 7, sizeof(cl_mem),  &dev_Ha_y);
    err |= clSetKernelArg(ker_Ha_apron, 8, sizeof(cl_mem),  &dev_Ha_z);
    err |= clSetKernelArg(ker_Ha_apron, 9, sizeof(cl_mem),  &dev_indicator1_temp);
    err |= clSetKernelArg(ker_Ha_apron, 10, sizeof(cl_mem),  &dev_indicator2_temp);
    err |= clSetKernelArg(ker_Ha_apron, 11, sizeof(cl_mem),  &dev_Ms);
    err |= clSetKernelArg(ker_Ha_apron, 12, sizeof(cl_mem),  &dev_Aex1);
    err |= clSetKernelArg(ker_Ha_apron, 13, sizeof(cl_mem),  &dev_Aex2);
    err |= clSetKernelArg(ker_Ha_apron, 14, sizeof(cl_double),  &delta_x);
	
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set ker_Ha_apron arguments! %d\n", err);
        exit(1);
    }

	err = clEnqueueNDRangeKernel(queue, ker_Ha_apron, 3, NULL, numWorkItems, NULL, 0, NULL, &event);
    if (err)
    {
        printf("Error: Failed to execute kernel 4!\n");
        exit(1);
    }
    err = clWaitForEvents(1, &event);

	//     // // Read back the results from the device to verify the output
    // cl_double out[(Nx+2) * (Ny+2) * (Nz+2)];
    // err = clEnqueueReadBuffer(queue, dev_Ms_temp, CL_TRUE, 0, sizeof(cl_double) * (Nx+2) * (Ny+2) * (Nz+2), out, 0, NULL, &event);
    // if (err != CL_SUCCESS)
    // {
    //     printf("Error: Failed to read output array! %d\n", err);
    //     exit(1);
    // }
	// err = clWaitForEvents(1, &event);
    // //watch/////
	// FILE *watch;
	// watch = fopen("dev_Ms_temp.dat", "w");
	// //cudaMemcpy(watch2, dev_theta, (Nx)*(Ny)*(Nz)*sizeof(double), cudaMemcpyDeviceToHost);
	// for (int k = 0; k < (Nz+2); k++){
	// 	for (int j = 0; j < (Ny+2); j++){
	// 		for (int i = 0; i < (Nx+2); i++){
	// 			fprintf(watch, " %lf", out[i+j*(Nx+2)+k*(Nx+2)*(Ny+2)]);
	// 		}
	// 		fprintf(watch, "\n");
	// 	}
	// 	fprintf(watch, "\n \n");
	// }
	// fclose(watch);
    ////////

			///// Watch /////
	//if (watchcount==1) {
	//	FILE *p1;
	//	fopen_s(&p1, "p1.dat", "w");
	//	cudaMemcpy(watch1, dev_x_theta, (Nx+2)*(Ny+2)*(Nz+2)*sizeof(double), cudaMemcpyDeviceToHost);
	//	for (int k = 0; k < (Nz+2); k++){
	//		for (int j = 0; j < (Ny+2); j++){
	//			for (int i = 0; i < (Nx+2); i++){
	//				fprintf(p1, " %10.5e", watch1[i+j*(Nx+2)+k*(Nx+2)*(Ny+2)]);
	//			}
	//			fprintf(p1, "\n");
	//		}
	//		fprintf(p1, "\n \n");
	//	}
	//	fclose(p1);
	//}
	
	///// Watch /////
	
	//watchcount++;
	
	
	return true;
}


//bool Hk_field(void)
//{
//	// Setup kernel configuration
//	dim3 blocks1(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);  // dimension of block
//	int  grid_x = (Nx % BLOCK_SIZE_X) ? (Nx/BLOCK_SIZE_X + 1) : (Nx/BLOCK_SIZE_X),
//		 grid_y = (Ny % BLOCK_SIZE_Y) ? (Ny/BLOCK_SIZE_Y + 1) : (Ny/BLOCK_SIZE_Y),
//		 grid_z = (Nz % BLOCK_SIZE_Z) ? (Nz/BLOCK_SIZE_Z + 1) : (Nz/BLOCK_SIZE_Z);
//	dim3 grids1(grid_x, grid_y, grid_z);
//
//	Kernel_Hk_field<<<grids1, blocks1>>>(dev_theta, dev_phi, dev_Hk_x,  dev_Hk_y, dev_Hk_z, Ku, Ms);
//
//	return true;
//}