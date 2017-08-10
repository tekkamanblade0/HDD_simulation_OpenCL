#include "clFFT.h"
#include "Parameters.h"


int FFT_3D_OpenCL(cl_mem buffersIn[], //cl_mem buffersOut[], 
				const char* direction, int sizex, int sizey, int sizez,
				cl_context &ctx, cl_command_queue &queue);

int G_tensor(int lx_zero_pad, int ly_zero_pad, int lz_zero_pad, 
			cl_context &ctx, cl_command_queue &queue);

int Hms(int lx_zero_pad, int ly_zero_pad, int lz_zero_pad,
		cl_context &ctx, cl_command_queue &queue,
		cl_kernel &ker_M_ini, cl_kernel &ker_fft_M_G, cl_kernel &ker_fft_hd);
        