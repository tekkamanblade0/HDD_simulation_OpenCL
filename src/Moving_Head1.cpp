//----Document created by Yipeng Jiao (jiaox058@umn.edu) on 01/09/2015-----

#include "Parameters.h"

// inline double 
// BilinearInterpolation(double q11, double q12, double q21, double q22, double x1, double x2, double y1, double y2, double x, double y) 
// {
//     double x2x1, y2y1, x2x, y2y, yy1, xx1;
//     x2x1 = x2 - x1;
//     y2y1 = y2 - y1;
//     x2x = x2 - x;
//     y2y = y2 - y;
//     yy1 = y - y1;
//     xx1 = x - x1;
//     return 1.0 / (x2x1 * y2y1) * (
//         q11 * x2x * y2y +
//         q21 * xx1 * y2y +
//         q12 * x2x * yy1 +
//         q22 * xx1 * yy1
//     );
// }

int Moving_Head(int t, cl_command_queue &queue, cl_kernel &ker_H, cl_kernel &ker_H_ext,
				cl_kernel &ker_T, cl_kernel &ker_T_ext, cl_kernel &ker_MgPar)
{
	// Head field profile
	int x_lmt, y_lmt;
	float StdDev_x, StdDev_y, FWHM_x, FWHM_y;
	int idx;
	double HPratio;
	double tt = (double)t;
	cl_event event = 0;
	cl_int err = 0;
	int ret = 0;
	// dist: moving distance of the head

	// Setup the Head Field (or Temperature) Profile
	x_lmt = 256;
	y_lmt = 32;
	FWHM_x = FWHMx*1.0e-7; // in (cm); thus it is 40(nm).
	FWHM_y = FWHMy*1.0e-7; // in (cm); thus it is 40(nm).
	StdDev_x = FWHM_x/(2*powf(2*logf(2),0.5));
	StdDev_y = FWHM_y/(2*powf(2*logf(2),0.5));

	// Temperature Profile
	
	if (extTP){
		err = clSetKernelArg(ker_T_ext, 0, sizeof(cl_mem),  &dev_T);
		err |= clSetKernelArg(ker_T_ext, 1, sizeof(cl_mem),  &dev_extT);
		err |= clSetKernelArg(ker_T_ext, 2, sizeof(cl_double),  &delta_x);
		err |= clSetKernelArg(ker_T_ext, 3, sizeof(cl_double),  &delta_t);
		err |= clSetKernelArg(ker_T_ext, 4, sizeof(cl_double),  &delta_TP);
		err |= clSetKernelArg(ker_T_ext, 5, sizeof(cl_double),  &extTstat_x);
		err |= clSetKernelArg(ker_T_ext, 6, sizeof(cl_double),  &extTstat_y);
		err |= clSetKernelArg(ker_T_ext, 7, sizeof(cl_double),  &v);
		err |= clSetKernelArg(ker_T_ext, 8, sizeof(cl_double),  &tt);

		if (err != CL_SUCCESS)
		{
			cout<<"Error: Failed to set ker_T_ext arguments!"<<err<<endl;
			//ret = 1;
			return 2;
			//exit(1);
		} 
		err = clEnqueueNDRangeKernel(queue, ker_T_ext, 3, NULL, numWorkItems, NULL, 0, NULL, &event);
		if (err)
		{
			cout<<"Error: Failed to execute kernel ker_T_ext!"<<endl;
			return 2;
			//exit(1);
		}
		// err = clWaitForEvents(1, &event);
		// if (err)
		// {
		// 	cout<<"Error: Wait For ker_T_ext!"<<endl;
		// 	ret = 1;
		// 	//exit(1);
		// }
					/*int m = 0,n = 0;
					double m_double = 0, n_double = 0;
					for (int k = 0; k < Nz; k++){
							for (int j = 0; j < Ny; j++){
								for (int i = 0; i < Nx; i++){

									idx = i + j*Nx + k*Nx*Ny;
									m_double = (i*delta_x + (extTstat_x * 1e-7 - v* t * delta_t)) / delta_TP;
									n_double = (j*delta_x + extTstat_y * 1e-7) / delta_TP;
									m = (int)m_double;
									n = (int)n_double;
									if ((m>=0) && (n>=0) && (m+1< EXTTsize_x)&&(n+1< EXTTsize_y)){
										T[idx] = BilinearInterpolation(extT[m][n],extT[m][n+1],extT[m+1][n],extT[m+1][n+1],
										(double)m,(double)(m+1),(double)n,(double)(n+1),m_double,n_double);
									}
									else{
										T[idx]=300.0;
									}
									
								}
							}
						}*/

	}
	else {
		err = clSetKernelArg(ker_T, 0, sizeof(cl_mem),  &dev_T);
		err |= clSetKernelArg(ker_T, 1, sizeof(cl_double),  &delta_x);
		err |= clSetKernelArg(ker_T, 2, sizeof(cl_double),  &delta_t);
		err |= clSetKernelArg(ker_T, 3, sizeof(cl_double),  &deltaTemp);
		err |= clSetKernelArg(ker_T, 4, sizeof(cl_double),  &offset);
		err |= clSetKernelArg(ker_T, 5, sizeof(cl_double),  &StdDev_x);
		err |= clSetKernelArg(ker_T, 6, sizeof(cl_double),  &StdDev_y);
		err |= clSetKernelArg(ker_T, 7, sizeof(cl_double),  &v);
		err |= clSetKernelArg(ker_T, 8, sizeof(cl_double),  &tt);

		if (err != CL_SUCCESS)
		{
			cout<<"Error: Failed to set ker_T arguments!"<<err<<endl;
			//ret = 1;
			return 2;
			//exit(1);
		} 
		err = clEnqueueNDRangeKernel(queue, ker_T, 3, NULL, numWorkItems, NULL, 0, NULL, &event);
		if (err)
		{
			cout<<"Error: Failed to execute kernel ker_T!"<<endl;
			return 2;
			//exit(1);
		}
		// err = clWaitForEvents(1, &event);
		// if (err)
		// {
		// 	cout<<"Error: Wait For ker_T!"<<endl;
		// 	ret = 1;
		// 	//exit(1);
		// }
				/*for (int j = 0; j < Ny; j++){
					for (int i = 0; i < Nx; i++){
						for (int k = 0; k < Nz; k++){
							idx = i + j*Nx + k*Nx*Ny;
							T[idx] = 300 + deltaTemp*exp((-powf((i*delta_x - v*t*delta_t) - offset*delta_x, 2)/(2*powf(StdDev_x, 2))-powf((j-15)*delta_x, 2)/(2*powf(StdDev_y, 2))));
						}
					}
				}*/
	}


	// Head Field Profile

	/*if (powf(-1, NextBit) == -1){Happl = +10000 - 20000/(50e-12)*(t-390000*(NextBit-1)+1)*h;}
	else {Happl = -10000 + 20000/(50e-12)*(t-390000*(NextBit-1)+1)*h;}  
	if (Happl <= -10000){Happl = -10000;}
	if (Happl >= +10000){Happl = +10000;}*/


	//if ((t - rise_time)>=0){
	//	if ((((t + rise_time) %BL_t ) == 0 ) && ((t - rise_time) %BL_t ) == 0)){
	//		if ((t + rise_time) %BL_t ) == 0) {
	//			temp_NextBit=NextBit;
	//			temp_Happl_x=Happl_x[idx];
	//			temp_Happl_y=Happl_y[idx];
	//			temp_Happl_z=Happl_z[idx];
	//		}
	//		if (temp_NextBit==0){
	//			Happl_x[idx]=
	//		
	//		}
	//		else {

	//		}
	//					

	//	}
	//}
	err = clWaitForEvents(1, &event);
	if (err)
	{
		cout<<"Error: Wait For ker_T/ext!"<<endl;
		ret = 1;
		//exit(1);
	}

		/*{		cl_event event = 0;
			double *out=new double[Nx * Ny * Nz];
			cl_int err = clEnqueueReadBuffer(queue, dev_T, CL_TRUE, 0, sizeof(cl_double) * Nx * Ny * Nz, out, 0, NULL, &event);
			if (err != CL_SUCCESS)
			{
				printf("Error: Failed to read output array! %d\n", err);
				exit(1);
			}
			err = clWaitForEvents(1, &event);			
			ofstream wfile_o;
			wfile_o.open("dev_T.dat");
			for (int k = 0; k < Nz; k++){
				for (int j = 0; j < Ny; j++){
					for (int i = 0; i < Nx; i++){
									//fprintf(watch, " %lf", out[i+j*(Nx)+k*(Nx)*(Ny)]);
						wfile_o<<setw(10)<<out[i+j*Nx+k*Nx*Ny]<<" ";
					}
								//fprintf(watch, "\n");
					wfile_o<<endl;
				}
							//fprintf(watch, "\n \n");
			wfile_o<<endl<<endl;
			}
						//fclose(watch);
			wfile_o.close();
			delete []out;
	}*/

	if (extHS){
		HPratio = extH_s[(int)(t %BL_t)];
	}
	else{
		HPratio =1.0;
	}

	if (rise_time==0)
	{
		//         (m,n)11--------21
		//                |      |     ��downtrack
		//                |      |
		//              12________22
		
		if (extHP){
			err = clSetKernelArg(ker_H_ext, 0, sizeof(cl_mem),  &dev_Happl_x);
			err |= clSetKernelArg(ker_H_ext, 1, sizeof(cl_mem),  &dev_Happl_y);
			err |= clSetKernelArg(ker_H_ext, 2, sizeof(cl_mem),  &dev_Happl_z);

			err |= clSetKernelArg(ker_H_ext, 3, sizeof(cl_mem),  &dev_extH_x);
			err |= clSetKernelArg(ker_H_ext, 4, sizeof(cl_mem),  &dev_extH_y);
			err |= clSetKernelArg(ker_H_ext, 5, sizeof(cl_mem),  &dev_extH_z);

			err |= clSetKernelArg(ker_H_ext, 6, sizeof(cl_double),  &delta_x);
			err |= clSetKernelArg(ker_H_ext, 7, sizeof(cl_double),  &delta_t);
			err |= clSetKernelArg(ker_H_ext, 8, sizeof(cl_double),  &delta_HP);
			err |= clSetKernelArg(ker_H_ext, 9, sizeof(cl_double),  &extHstat_x);
			err |= clSetKernelArg(ker_H_ext, 10, sizeof(cl_double),  &extHstat_y);
			err |= clSetKernelArg(ker_H_ext, 11, sizeof(cl_double),  &HPratio);
			err |= clSetKernelArg(ker_H_ext, 12, sizeof(cl_int),  &NextBit);
			err |= clSetKernelArg(ker_H_ext, 13, sizeof(cl_double),  &v);
			err |= clSetKernelArg(ker_H_ext, 14, sizeof(cl_double),  &tt);

			if (err != CL_SUCCESS)
			{
				cout<<"Error: Failed to set ker_H_ext arguments!"<<err<<endl;
				//ret = 1;
				return 2;
				//exit(1);
			} 

			err = clEnqueueNDRangeKernel(queue, ker_H_ext, 3, NULL, numWorkItems, NULL, 0, NULL, &event);
			if (err)
			{
				cout<<"Error: Failed to execute kernel ker_H_ext!"<<endl;
				return 2;
				//exit(1);
			}
			// err = clWaitForEvents(1, &event);
			// if (err)
			// {
			// 	cout<<"Error: Wait For ker_H_ext!"<<endl;
			// 	ret = 1;
			// 	//exit(1);
			// }

			/*int m = 0,n = 0;
			double m_double = 0, n_double = 0;
			for (int k = 0; k < Nz; k++){
					for (int j = 0; j < Ny; j++){
						for (int i = 0; i < Nx; i++){

							idx = i + j*Nx + k*Nx*Ny;
							m_double = (i*delta_x + extHstat_x * 1e-7 - v* t * delta_t) / delta_HP;
							n_double = (j*delta_x + extHstat_y * 1e-7) / delta_HP;
							m = (int)m_double;
							n = (int)n_double;
							if ((m>=0) && (n>=0) && (m+1< EXTHsize_x)&&(n+1< EXTHsize_y)){
							Happl_x[idx] = BilinearInterpolation(extH_x[m][n],extH_x[m][n+1],extH_x[m+1][n],extH_x[m+1][n+1],
								(double)m,(double)(m+1),(double)n,(double)(n+1),m_double,n_double)* powf(-1, NextBit) *HPratio;
							Happl_y[idx] = BilinearInterpolation(extH_y[m][n],extH_y[m][n+1],extH_y[m+1][n],extH_y[m+1][n+1],
								(double)m,(double)(m+1),(double)n,(double)(n+1),m_double,n_double)* powf(-1, NextBit) *HPratio;
							Happl_z[idx] = BilinearInterpolation(extH_z[m][n],extH_z[m][n+1],extH_z[m+1][n],extH_z[m+1][n+1],
								(double)m,(double)(m+1),(double)n,(double)(n+1),m_double,n_double)* powf(-1, NextBit) *HPratio;
							}
							else{
								Happl_x[idx]=0.0;
								Happl_y[idx]=0.0;
								Happl_z[idx]=0.0;
							}

						}
					}
				}
		*/
		
		}
		else {
			err = clSetKernelArg(ker_H, 0, sizeof(cl_mem),  &dev_Happl_x);
			err |= clSetKernelArg(ker_H, 1, sizeof(cl_mem),  &dev_Happl_y);
			err |= clSetKernelArg(ker_H, 2, sizeof(cl_mem),  &dev_Happl_z);

			err |= clSetKernelArg(ker_H, 3, sizeof(cl_double),  &Happl);
			err |= clSetKernelArg(ker_H, 4, sizeof(cl_double),  &HPratio);
			err |= clSetKernelArg(ker_H, 5, sizeof(cl_int),  &NextBit);


			if (err != CL_SUCCESS)
			{
				cout<<"Error: Failed to set ker_H arguments!"<<err<<endl;
				//ret = 1;
				return 2;
				//exit(1);
			} 
			err = clEnqueueNDRangeKernel(queue, ker_H, 3, NULL, numWorkItems, NULL, 0, NULL, &event);
			if (err)
			{
				cout<<"Error: Failed to execute kernel ker_H!"<<endl;
				return 2;
				//exit(1);
			}
			// err = clWaitForEvents(1, &event);
			// if (err)
			// {
			// 	cout<<"Error: Wait For ker_H!"<<endl;
			// 	ret = 1;
			// 	//exit(1);
			// }

			/*for (int k = 0; k < Nz; k++){
					for (int j = 0; j < Ny; j++){
						for (int i = 0; i < Nx; i++){
							idx = i + j*Nx + k*Nx*Ny;
							Happl_x[idx] = -Happl *sin(PI/9)* powf(-1, NextBit)*HPratio;
							Happl_y[idx] = 0*HPratio;
							Happl_z[idx] = Happl *cos(PI/9)* powf(-1, NextBit)*HPratio;
						}
					}
				}*/
		}


		if ((t%BL_t ) == 0) //195000//130000//97500 //BL_t    //BL=13.5nm->135000
		{
			wfile100<<NextBit<<endl;
			rfile2>>NextBit;
			//fprintf(wfile100, "%d\n", NextBit);
			//fscanf(rfile2, "%d", &NextBit);
			//NextBit=NextBit>>1;
			
		}

	}

	err = clWaitForEvents(1, &event);
	if (err)
	{
		cout<<"Error: Wait For ker_H/ext!"<<endl;
		ret = 1;
		//exit(1);
	}
		/*{		cl_event event = 0;
			double *out=new double[Nx * Ny * Nz];
			cl_int err = clEnqueueReadBuffer(queue, dev_Happl_z, CL_TRUE, 0, sizeof(cl_double) * Nx * Ny * Nz, out, 0, NULL, &event);
			if (err != CL_SUCCESS)
			{
				printf("Error: Failed to read output array! %d\n", err);
				exit(1);
			}
			err = clWaitForEvents(1, &event);			
			ofstream wfile_o;
			wfile_o.open("dev_Happl_z.dat");
			for (int k = 0; k < Nz; k++){
				for (int j = 0; j < Ny; j++){
					for (int i = 0; i < Nx; i++){
									//fprintf(watch, " %lf", out[i+j*(Nx)+k*(Nx)*(Ny)]);
						wfile_o<<setw(10)<<out[i+j*Nx+k*Nx*Ny]<<" ";
					}
								//fprintf(watch, "\n");
					wfile_o<<endl;
				}
							//fprintf(watch, "\n \n");
			wfile_o<<endl<<endl;
			}
						//fclose(watch);
			wfile_o.close();
			delete []out;
	}*/
	/*else
	{
		if (t==0) {
			//fscanf(rfile2, "%d", &temp_NextBit);
			rfile2>>temp_NextBit;
			NextBit=temp_NextBit;
		}

		if (((t - rise_time)> 0)&&((t %BL_t ) == (BL_t -rise_time)))  { //����rise_time��0 ��Ҫ����ʽ�ұ߸�Ϊ0//(BL_t -rise_time)
			//fscanf(rfile2, "%d", &temp_NextBit);
			rfile2>>temp_NextBit;
			//printf("OK1 %d\t%d\n", NextBit,temp_NextBit);
			cout<<"OK1 "<<NextBit<<"\t"<<temp_NextBit<<endl;
			//fprintf(wfile100, "%d\t%d\n", NextBit,temp_NextBit);
			wfile100<<NextBit<<"\t"<<temp_NextBit<<endl;
			if (temp_NextBit^NextBit) {
				for (int k = 0; k < Nz; k++){
					for (int j = 0; j < Ny; j++){
						for (int i = 0; i < Nx; i++){
							idx = i + j*Nx + k*Nx*Ny;
							temp_Happl_x[idx]=Happl_x[idx];
							temp_Happl_y[idx]=Happl_y[idx];
							temp_Happl_z[idx]=Happl_z[idx];
						}
					}
				}
				temp_time=t;
			}

		}
		if ((((t %BL_t ) <= rise_time) || ((t %BL_t ) >= BL_t -rise_time))&& (temp_NextBit^NextBit)){
			for (int k = 0; k < Nz; k++){
				for (int j = 0; j < Ny; j++){
					for (int i = 0; i < Nx; i++){
						idx = i + j*Nx + k*Nx*Ny;
						Happl_x[idx]=temp_Happl_x[idx]+(-2.0*temp_Happl_x[idx])/(2.0*rise_time)*(t-temp_time) ;
						Happl_y[idx]=temp_Happl_y[idx]+(-2.0*temp_Happl_y[idx])/(2.0*rise_time)*(t-temp_time) ;
						Happl_z[idx]=temp_Happl_z[idx]+(-2.0*temp_Happl_z[idx])/(2.0*rise_time)*(t-temp_time) ;
					}
				}
			}
			//printf("OK2 \n");
			//printf("%d %f\n",temp_time, temp_Happl_z[111]);
		}
		else 	{
				//printf("z");
				for (int k = 0; k < Nz; k++){
					for (int j = 0; j < Ny; j++){
						for (int i = 0; i < Nx; i++){
							idx = i + j*Nx + k*Nx*Ny;
							Happl_x[idx] = -Happl *sin(0.)* powf(-1, NextBit);
							Happl_y[idx] = 0.;
							Happl_z[idx] = Happl *cos(0.)* powf(-1, NextBit);
						}
					}
				}
		}

		if ((t %BL_t ) == rise_time) //195000//130000//97500 //BL_t    //BL=13.5nm->135000
		{
			//printf("OK4");
			cout<<"OK4"<<endl;
			NextBit=temp_NextBit;
					//NextBit=NextBit>>1;
			}
	}
	*/
	// Temperature-dependent input parameters for 2nm cells (h: high temperature, l: low temperature)
/*	double h1_Ms  = 142.9,        h2_Ms  = 0.3111,
		   h1_Ku  = -9.369e4,     h2_Ku  = 7.449e7, 
		   h1_Aex = -5.255e-9,    h2_Aex = 3.899e-6,
		   h1_alpha = 0.005561,   h2_alpha = 0.003078,  h3_alpha = 9.274e-18, h4_alpha = 0.05185,
		   l1_Ms = -1.3,          l2_Ms = 1436,
		   l1_Ku = 5.333,         l2_Ku = -9080,        l3_Ku = 5.058e6,      l4_Ku = -8.998e8,
		   l1_Aex = 1.333e-13,    l2_Aex = -2.371e-10,  l3_Aex = 1.368e-7,    l4_Aex = -2.474e-5,
		   l1_alpha = 2.5e-5,     l2_alpha = 0.02,
		   f1_Ms = -0.00101,      f2_Ms = -0.056,       f3_Ms = 1030,
		   f1_Ku = -9.374e4,      f2_Ku = 6.928e7;*/
	// for (int k = 0; k < Nz_1; k++){
	// 	for (int j = 0; j < Ny; j++){
	// 		for (int i = 0; i < Nx; i++){
	// 			idx = i + j*Nx + k*Nx*Ny;
	// 			T[idx] = T[idx]*(1 + std_Aex[idx]);
	// 			// if (T[idx] > 600 && T[idx] <= 700){
	// 			// 	host_Ms[idx]    = h1_Ms *pow(735-T[idx], h2_Ms);
	// 			// 	host_Ku[idx]    = (h1_Ku*T[idx] + h2_Ku)*(1+std_Ku[idx])*1;  
	// 			// 	host_Aex[idx]   = (h1_Aex*T[idx] + h2_Aex);
	// 			// 	host_alpha[idx] = (h1_alpha*exp(h2_alpha*T[idx]) + h3_alpha*exp(h4_alpha*T[idx]))*2;
	// 			// }
	// 			// else if (T[idx] > 500 && T[idx] <= 600){
	// 			// 	host_Ms[idx]    = l1_Ms*T[idx] + l2_Ms;
	// 			// 	host_Ku[idx]    = (l1_Ku*pow(T[idx], 3) + l2_Ku*pow(T[idx], 2) + l3_Ku*pow(T[idx], 1) + l4_Ku)*(1+std_Ku[idx])*1;
	// 			// 	host_Aex[idx]   = (l1_Aex*pow(T[idx], 3) + l2_Aex*pow(T[idx], 2) + l3_Aex*pow(T[idx], 1) + l4_Aex);
	// 			// 	host_alpha[idx] = l1_alpha*T[idx] + l2_alpha;
	// 			// }
	// 			// else if (T[idx] >= 100 && T[idx] <= 500){
	// 			// 	host_Ms[idx]    = f1_Ms*pow(T[idx], 2) + f2_Ms*pow(T[idx], 1) + f3_Ms;
	// 			// 	host_Ku[idx]    = (f1_Ku*pow(T[idx], 1) + f2_Ku)*(1+std_Ku[idx])*1;
	// 			// 	host_Aex[idx]   = 1.1e-6; //*pow(host_Ms[idx]/1100, 2);
	// 			// 	host_alpha[idx] = l1_alpha*T[idx] + l2_alpha;
	// 			// }
	// 			// else {
	// 			// 	host_Ms[idx]    = 100;
	// 			// 	host_Ku[idx]    = 0;
	// 			// 	host_Aex[idx]   = 0;
	// 			// 	host_alpha[idx] = 0.1;
	// 			// }
	// 			// D[idx] = (2*kb*T[idx]*host_alpha[idx])/(host_Ms[idx]*delta_t*GAMMA*pow(delta_x, 3.0));
	// 			// host_gamma[idx] = (1.76e7)/(1+pow(host_alpha[idx], 2.0));
	// 			if (T[idx] > 700 && T[idx] <= 710){
	// 				host_Aex[idx]   = -3.331e-011 * pow(T[idx],2) + 3.914e-008*(T[idx]) -1.077e-005;
	// 				host_Aratio[idx] = 1.3;
	// 			}
	// 			else if (T[idx] > 680 && T[idx] <= 700){
	// 				host_Aex[idx]   = -3.331e-011 * pow(T[idx],2) + 3.914e-008*(T[idx]) -1.077e-005;
	// 				host_Aratio[idx] = -0.05* (T[idx]-680)/20 + 1.35;
	// 			}
	// 			else if (T[idx] > 660 && T[idx] <= 680){
	// 				host_Aex[idx]   = -3.331e-011 * pow(T[idx],2) + 3.914e-008*(T[idx]) -1.077e-005;
	// 				host_Aratio[idx] = -0.23* (T[idx]-660)/20 + 1.58;
	// 			}
	// 			else if (T[idx] > 640 && T[idx] <= 660){
	// 				host_Aex[idx]   = -3.331e-011 * pow(T[idx],2) + 3.914e-008*(T[idx]) -1.077e-005;
	// 				host_Aratio[idx] = 0.18* (T[idx]-640)/20 + 1.4;
	// 			}
	// 			else if (T[idx] > 610 && T[idx] <= 640){
	// 				host_Aex[idx]   = -2.406e-011 * pow(T[idx],2) + 2.574e-008*(T[idx])-5.985e-006;
	// 				host_Aratio[idx] = 0.1* (T[idx]-610)/30 + 1.3;
	// 			}
	// 			else if (T[idx] > 580 && T[idx] <= 610){
	// 				host_Aex[idx]   = -2.406e-011 * pow(T[idx],2) + 2.574e-008*(T[idx])-5.985e-006;
	// 				host_Aratio[idx] = 1.3;
	// 			}
	// 			else if (T[idx] > 520 && T[idx] <= 580){
	// 				host_Aex[idx]   = 5.556e-012 * pow(T[idx],2)  -8.611e-009*(T[idx]) +3.976e-006;
	// 				host_Aratio[idx] = 1.3;
	// 			}
	// 			else if (T[idx] <= 520){

	// 				host_Aex[idx]   =  -2.507e-012 * pow(T[idx],2) -1.771e-010 *(T[idx])+1.777e-006; //*pow(host_Ms[idx]/1100, 2);
	// 				host_Aratio[idx] = 1.3;
	// 			}
	// 			else {
	// 				//host_Ms[idx]    = 100;
	// 				//host_Ku[idx]    = 0;
	// 				host_Aex[idx]   = 0;
	// 				host_Aratio[idx] =1.0;
	// 				//host_alpha[idx] = 0.1;
	// 			}
	// 			if (T[idx] <= 600){
	// 				host_alpha[idx] = 4.0*(2.5e-5*T[idx] + 0.02);
	// 			}
	// 			else if (T[idx] > 600 && T[idx] <= 710){

	// 				host_alpha[idx]  = (0.005561*exp(0.003078*T[idx]) + 9.274e-18*exp(0.05185*T[idx]))*8;
	// 			}
	// 			else {
	// 				host_alpha[idx] = 0.1;
	// 			}
	// 			D[idx] = (2*kb*T[idx]*host_alpha[idx])/(host_Ms[idx]*delta_t*GAMMA*pow(delta_x, 3.0));
	// 			host_gamma[idx] = (1.76e7)/(1+pow(host_alpha[idx], 2.0));
	// 			if (T[idx] <= 710){
	// 				host_Ms[idx]= 136.0* pow((739.4-T[idx]),0.3182);
	// 			}
	// 			else {
	// 				host_Ms[idx]    = 100;
	// 			}
	// 			if (T[idx] <= 710){
	// 				host_Ku[idx]= -6.606e-005*pow(T[idx],4) + 0.2348*pow(T[idx],3) -263.9*pow(T[idx],2) + 2.916e4 *T[idx] + 5.32e7;
	// 			}
	// 			else {
	// 				host_Ku[idx]    = 0;
	// 			}
				
	// 			host_Aex1[idx]= 3*host_Aex[idx]/(1+2*host_Aratio[idx]); //Aex_z
	// 			host_Aex2[idx]= host_Aratio[idx]*host_Aex1[idx];  //Aex_xy
	// 			D[idx] = (2*kb*T[idx]*host_alpha[idx])/(host_Ms[idx]*delta_t*GAMMA*pow(delta_x, 3.0));
	// 			host_gamma[idx] = (1.76e7)/(1+pow(host_alpha[idx], 2.0));
	// 		}
	// 	}
	// }
	// for (int k = Nz_1; k < Nz_2; k++){
	// 	for (int j = 0; j < Ny; j++){
	// 		for (int i = 0; i < Nx; i++){
	// 			idx = i + j*Nx + k*Nx*Ny;
	// 			host_Ms[idx]    = 10;
	// 			host_Ku[idx]    = 0;
	// 			host_Aex1[idx]   = 0;
	// 			host_Aex2[idx]   = 0;
	// 			host_alpha[idx] = 0.1;
	// 			D[idx] = (2*kb*T[idx]*host_alpha[idx])/(host_Ms[idx]*delta_t*GAMMA*pow(delta_x, 3.0));
	// 			host_gamma[idx] = (1.76e7)/(1+pow(host_alpha[idx], 2.0));
	// 		}
	// 	}
	// }
	
	err = clSetKernelArg(ker_MgPar, 0, sizeof(cl_mem),  &dev_Ms);
    err |= clSetKernelArg(ker_MgPar, 1, sizeof(cl_mem),  &dev_Ku);
    err |= clSetKernelArg(ker_MgPar, 2, sizeof(cl_mem),  &dev_Aex1);
    err |= clSetKernelArg(ker_MgPar, 3, sizeof(cl_mem),  &dev_Aex2);
    err |= clSetKernelArg(ker_MgPar, 4, sizeof(cl_mem),  &dev_alpha);
    err |= clSetKernelArg(ker_MgPar, 5, sizeof(cl_mem),  &dev_gamma);
    err |= clSetKernelArg(ker_MgPar, 6, sizeof(cl_mem),  &dev_D);
    err |= clSetKernelArg(ker_MgPar, 7, sizeof(cl_mem),  &dev_T);
    err |= clSetKernelArg(ker_MgPar, 8, sizeof(cl_mem),  &dev_std_Aex);
    err |= clSetKernelArg(ker_MgPar, 9, sizeof(cl_mem),  &dev_std_Ku);
    err |= clSetKernelArg(ker_MgPar, 10, sizeof(cl_double),  &delta_x);
    err |= clSetKernelArg(ker_MgPar, 11, sizeof(cl_double),  &delta_t);

    if (err != CL_SUCCESS)
    {
        cout<<"Error: Failed to set ker_MgPar arguments!"<<err<<endl;
		//ret = 1;
		return 2;
        //exit(1);
    } 
    err = clEnqueueNDRangeKernel(queue, ker_MgPar, 3, NULL, numWorkItems, NULL, 0, NULL, &event);
    if (err)
    {
         cout<<"Error: Failed to execute kernel ker_MgPar!"<<endl;
		 return 2;
         //exit(1);
    }
    err = clWaitForEvents(1, &event);
	if (err)
    {
         cout<<"Error: Wait For ker_MgPar!"<<endl;
		 ret = 1;
         //exit(1);
    }

	/*{		cl_event event = 0;
			double *out=new double[Nx * Ny * Nz];
			cl_int err = clEnqueueReadBuffer(queue, dev_Aex2, CL_TRUE, 0, sizeof(cl_double) * Nx * Ny * Nz, out, 0, NULL, &event);
			if (err != CL_SUCCESS)
			{
				printf("Error: Failed to read output array! %d\n", err);
				exit(1);
			}
			err = clWaitForEvents(1, &event);			
			ofstream wfile_o;
			wfile_o.open("dev_Aex2.dat");
			for (int k = 0; k < Nz; k++){
				for (int j = 0; j < Ny; j++){
					for (int i = 0; i < Nx; i++){
									//fprintf(watch, " %lf", out[i+j*(Nx)+k*(Nx)*(Ny)]);
						wfile_o<<setw(10)<<out[i+j*Nx+k*Nx*Ny]<<" ";
					}
								//fprintf(watch, "\n");
					wfile_o<<endl;
				}
							//fprintf(watch, "\n \n");
			wfile_o<<endl<<endl;
			}
						//fclose(watch);
			wfile_o.close();
			delete []out;
	}*/
	/*for (int k = Nz_2; k < Nz; k++){
		for (int j = 0; j < Ny; j++){
			for (int i = 0; i < Nx; i++){
				idx = i + j*Nx + k*Nx*Ny;
				host_Ms[idx]    = 100;
				host_Ku[idx]    = 0;
				host_Aex[idx]   = 0;
				host_alpha[idx] = 0.1;
				D[idx] = (2*kb*T[idx]*host_alpha[idx])/(host_Ms[idx]*delta_t*GAMMA*pow(delta_x, 3.0));
				host_gamma[idx] = (1.76e7)/(1+pow(host_alpha[idx], 2.0));
			}
		}
	}*/


	/////// Watch /////
	/*FILE *p1;
	fopen_s(&p1, "p1.dat", "w");
	for (int i = 0; i < Nx*Ny*Nz; i++){
	if((i)%(Nx) == 0 && i != 0){ fprintf(p1, "\n"); }
	if((i)%(Nx*Ny) == 0 && i != 0){ fprintf(p1, "\n\n"); }
	fprintf(p1, "%15.3e", T[i]);
	}
	fclose(p1);*/
	/////// Watch /////

	// err = clEnqueueWriteBuffer(queue, dev_Ms, CL_TRUE, 0, sizeof(cl_double) * Nx * Ny * Nz, host_Ms, 0, NULL, NULL);
	// err |= clEnqueueWriteBuffer(queue, dev_Ku, CL_TRUE, 0, sizeof(cl_double) * Nx * Ny * Nz, host_Ku, 0, NULL, NULL);
	// err |= clEnqueueWriteBuffer(queue, dev_Aex1, CL_TRUE, 0, sizeof(cl_double) * Nx * Ny * Nz, host_Aex1, 0, NULL, NULL);
	// err |= clEnqueueWriteBuffer(queue, dev_Aex2, CL_TRUE, 0, sizeof(cl_double) * Nx * Ny * Nz, host_Aex2, 0, NULL, NULL);
	// err |= clEnqueueWriteBuffer(queue, dev_alpha, CL_TRUE, 0, sizeof(cl_double) * Nx * Ny * Nz, host_alpha, 0, NULL, NULL);
	// err |= clEnqueueWriteBuffer(queue, dev_gamma, CL_TRUE, 0, sizeof(cl_double) * Nx * Ny * Nz, host_gamma, 0, NULL, NULL);
	// err |= clEnqueueWriteBuffer(queue, dev_D, CL_TRUE, 0, sizeof(cl_double) * Nx * Ny * Nz, D, 0, NULL, NULL);
	// err |= clEnqueueWriteBuffer(queue, dev_T, CL_TRUE, 0, sizeof(cl_double) * Nx * Ny * Nz, T, 0, NULL, NULL);
	// err |= clEnqueueWriteBuffer(queue, dev_Happl_x, CL_TRUE, 0, sizeof(cl_double) * Nx * Ny * Nz, Happl_x, 0, NULL, NULL);
	// err |= clEnqueueWriteBuffer(queue, dev_Happl_y, CL_TRUE, 0, sizeof(cl_double) * Nx * Ny * Nz, Happl_y, 0, NULL, NULL);
	// err |= clEnqueueWriteBuffer(queue, dev_Happl_z, CL_TRUE, 0, sizeof(cl_double) * Nx * Ny * Nz, Happl_z, 0, NULL, NULL);
    // if (err != CL_SUCCESS)
    // {
    //     printf("Error: Failed to write to source array (Moving_head)!\n");
    //     exit(1);
    // }

	// cudaMemcpy(dev_Ms, host_Ms,       (Nx)*(Ny)*(Nz)*sizeof(double), cudaMemcpyHostToDevice);
	// cudaMemcpy(dev_Ku, host_Ku,       (Nx)*(Ny)*(Nz)*sizeof(double), cudaMemcpyHostToDevice);
	// cudaMemcpy(dev_Aex1, host_Aex1,     (Nx)*(Ny)*(Nz)*sizeof(double), cudaMemcpyHostToDevice);
	// cudaMemcpy(dev_Aex2, host_Aex2,     (Nx)*(Ny)*(Nz)*sizeof(double), cudaMemcpyHostToDevice);
	// cudaMemcpy(dev_alpha, host_alpha, (Nx)*(Ny)*(Nz)*sizeof(double), cudaMemcpyHostToDevice);
	// cudaMemcpy(dev_gamma, host_gamma, (Nx)*(Ny)*(Nz)*sizeof(double), cudaMemcpyHostToDevice);
	// cudaMemcpy(dev_D, D,			  (Nx)*(Ny)*(Nz)*sizeof(double), cudaMemcpyHostToDevice);
	// cudaMemcpy(dev_T, T,			  (Nx)*(Ny)*(Nz)*sizeof(double), cudaMemcpyHostToDevice);
	// cudaMemcpy(dev_Happl_x, Happl_x,  (Nx)*(Ny)*(Nz)*sizeof(double), cudaMemcpyHostToDevice);
	// cudaMemcpy(dev_Happl_y, Happl_y,  (Nx)*(Ny)*(Nz)*sizeof(double), cudaMemcpyHostToDevice);
	// cudaMemcpy(dev_Happl_z, Happl_z,  (Nx)*(Ny)*(Nz)*sizeof(double), cudaMemcpyHostToDevice);
	// /////// Watch /////
	/*for (int i = 0; i < Nx*Ny*Nz; i++){
	if((i)%(Nx) == 0 && i != 0){ fprintf(wfile1, "\n"); }
	if((i)%(Nx*Ny) == 0 && i != 0){ fprintf(wfile1, "\n\n"); }
	fprintf(wfile1, "%15.3e", host_Ms[i]);
	}
	fclose(wfile1);*/
	/////// Watch /////
	/////// Watch /////
	/*FILE *p2;
	fopen_s(&p2, "p2.dat", "w");
	for (int i = 0; i < Nx*Ny*Nz; i++){
	if((i)%(Nx) == 0 && i != 0){ fprintf(p2, "\n"); }
	if((i)%(Nx*Ny) == 0 && i != 0){ fprintf(p2, "\n\n"); }
	fprintf(p2, "%15.3e", host_Aex[i]);
	}
	fclose(p2);*/
	/////// Watch /////
	///// Watch /////
	/*FILE *p3;
	fopen_s(&p3, "p3.dat", "w");
	cudaMemcpy(watch3, dev_Aex, (Nx)*(Ny)*(Nz)*sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < Nx*Ny*Nz; i++){
	if((i)%(Nx) == 0 && i != 0){ fprintf(p3, "\n"); }
	if((i)%(Nx*Ny) == 0 && i != 0){ fprintf(p3, "\n\n"); }
	fprintf(p3, "%15.3e", watch3[i]);
	}
	fclose(p3);*/
	///// Watch /////

	return ret;
}