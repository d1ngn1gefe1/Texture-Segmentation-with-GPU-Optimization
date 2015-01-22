#ifndef FILTERKERNEL_CUH
#define FILTERKERNEL_CUH

#include "Support.h"

//Class definition for filter response kernels
class FilterKernel
{
public:
	unsigned int x_dim, y_dim;//Dimension of the filtering kernel
	unsigned int half_x, half_y;
	unsigned char dir_idx;
	float std_x, std_y;
	float *pdata_h, *pdata_d;				    //Pointer to the kernel
	bool gpuCompute;

	FilterKernel(unsigned int _x_dim, unsigned int _y_dim, float _std_x, unsigned char order, bool _gpuCompute):
		x_dim(_x_dim),y_dim(_y_dim),half_x((x_dim-1)/2),half_y((y_dim-1)/2),std_x(_std_x),gpuCompute(_gpuCompute)
	{
		//Constructor for the Gaussian kernel (order = 0) or Laplacian of Gaussian (order = 2)
		if (x_dim%2==0||y_dim%2==0)
		{
			printf("Cannot generate filter size with odd number of elements");
			exit(-1);
		}
		switch (gpuCompute)
		{
		case false:
			{
				pdata_h = (float*)malloc(sizeof(float)*x_dim*y_dim);
				if (!pdata_h)
				{
					printf("[Error] Cannot initialize the memory for filtering kernel\n");
					exit(-1);
				}
			}
			break;
		case true:
			{
				cudaError_t memErr = cudaMalloc((void**)&pdata_d, sizeof(float)*x_dim*y_dim);
				if(memErr != cudaSuccess)
				{
					printf("CUDA error: %s\n", cudaGetErrorString(memErr));
					exit(-1);
				}
			}
			break;
		}
		switch (order)
		{
		case 0:
			generateGaussianKernel(std_x, std_x);
			break;
		case 2:
			generateLaplacianOfGaussianKernel(std_x,std_x);
			break;
		default:
			printf("[Error] Unsupported mode for symmetric filters...\n");
			break;
		}
		normalization();
	}

	//For derivative of directional Gaussian
	FilterKernel(unsigned int _x_dim, unsigned int _y_dim, unsigned char _dir_idx,
		float _std_x, float _std_y, unsigned char order, bool _gpuCompute):
	x_dim(_x_dim),y_dim(_y_dim),half_x((x_dim-1)/2),half_y((y_dim-1)/2),
		std_x(_std_x),std_y(_std_y),dir_idx(_dir_idx),gpuCompute(_gpuCompute)
	{
		//Constructor for the filtering kernel
		if (x_dim%2==0||y_dim%2==0)
		{
			printf("Cannot generate filter size with odd number of elements");
			exit(-1);
		}
		if (gpuCompute)
		{
			cudaError_t memErr = cudaMalloc((void**)&pdata_d, sizeof(float)*x_dim*y_dim);
			if(memErr != cudaSuccess)
			{
				printf("CUDA error: %s\n", cudaGetErrorString(memErr));
				exit(-1);
			}
		}
		else
		{
			pdata_h = (float*)malloc(sizeof(float)*x_dim*y_dim);
			if (!pdata_h)
			{
				printf("[Error] Cannot initialize the memory for filtering kernel\n");
				exit(-1);
			}
		}
		switch (order)
		{
		case 1:
			generate1stOrderDerivDirectionalGaussian(std_x, std_y,dir_idx);
			break;
		case 2:
			generate2ndOrderDerivDirectionalGaussian(std_x, std_y,dir_idx);
			break;
		}
		normalization();
	}

	~FilterKernel()
	{
		switch (gpuCompute)
		{
		case true:
			cudaFree(pdata_d);
			break;
		case false:
			free(pdata_h);
			break;
		}
	}

	virtual void generateFilterData(float param1, float param2){}

	void generateGaussianKernel(float std_x, float std_y);

	void generateLaplacianOfGaussianKernel(float std_x,float std_y);

	void generate1stOrderDerivDirectionalGaussian(float std_x, float std_y,unsigned char dir_idx);

	void generate2ndOrderDerivDirectionalGaussian(float std_x, float std_y,unsigned char dir_idx);

	void normalization();

	void kernelVerfication();

private:
	//helper function for normalization
	float *prefixSum(int choice);
	void doArith(float* in_d, int choice);
};

void generateFilterKernels(FilterKernel ** h_arr, bool gpuCompute);

#endif //FILTERKERNEL_CUH