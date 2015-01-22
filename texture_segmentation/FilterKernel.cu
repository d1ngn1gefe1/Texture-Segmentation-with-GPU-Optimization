#include "FilterKernel.cuh"

//

__global__ void reduction(float *out, float *in, unsigned int x_dim, unsigned int y_dim, int choice)
{
	extern __shared__ float partialSum[]; // shared memory: 49152 bytes per block
	unsigned int t = threadIdx.x;
	unsigned int start = 2*blockIdx.x*blockDim.x;
	if (start+t < x_dim*y_dim) partialSum[t] = ((choice == 0)?in[start+t]:abs(in[start+t]));
	else partialSum[t] = 0; 
	if (start+blockDim.x+t < x_dim*y_dim) partialSum[blockDim.x+t] = ((choice == 0)?in[start+blockDim.x+t]:abs(in[start+blockDim.x+t]));
	else partialSum[blockDim.x+t] = 0;
	for (unsigned int stride = 1; stride <= blockDim.x; stride*=2)
	{
		__syncthreads();
		if (t%stride == 0)
			partialSum[2*t] += partialSum[2*t+stride];
	}
	if (t == 0) out[blockIdx.x] = partialSum[0];
}

float *FilterKernel::prefixSum(int choice)
{
	int blockSize = BLOCK_SIZE_FILTER;
	int gridSize = (x_dim*y_dim/2-1)/blockSize+1;
	//
	float* partialSum;
	cudaMalloc((void**)&partialSum, gridSize*sizeof(float));
	float* out_d;
	cudaMalloc((void**)&out_d, sizeof(float));
	//
	reduction<<<gridSize, blockSize, 2*blockSize*sizeof(float)>>>(partialSum, pdata_d, x_dim, y_dim, choice);
	reduction<<<1, (gridSize-1)/2+1, gridSize*sizeof(float)>>>(out_d, partialSum, gridSize, 1, 0);
	return out_d;
}

__global__ void arith(float *out, float *in, unsigned int x_dim, unsigned int y_dim, int choice)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i >= x_dim*y_dim) return;
	if (choice == 0)
		out[i] -= *in/x_dim/y_dim;
	else
		out[i] /= *in;
}

void FilterKernel::doArith(float* in_d, int choice)
{
	int blockSize = BLOCK_SIZE_FILTER;
	int gridSize = (x_dim*y_dim-1)/blockSize+1;
	arith<<<gridSize, blockSize>>>(pdata_d, in_d, x_dim, y_dim, choice);
	//printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
}

//

__global__ void generateGaussianKernelCUDA(float* pdata_d,float std_x,float std_y,
										   unsigned int half_x,unsigned int half_y,
										   unsigned int x_dim,unsigned int y_dim)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i >= x_dim*y_dim) return;
	int x_coord = i%x_dim;
	int y_coord = i/x_dim;
	float cur_x = x_coord - (float) half_x;
	float cur_y = y_coord - (float) half_y;
	pdata_d[y_coord*x_dim+x_coord] = (float)exp(-(cur_x*cur_x/std_x/std_x+cur_y*cur_y/std_y/std_y)/2);
}

void FilterKernel::generateGaussianKernel(float std_x, float std_y)
{  //Generate symmetric Gaussian kernel
	if (gpuCompute) // 49*49
	{
		int blockSize = BLOCK_SIZE_FILTER;
		int gridSize = (x_dim*y_dim-1)/blockSize+1;
		generateGaussianKernelCUDA<<<gridSize,blockSize>>>(pdata_d,std_x,std_y,half_x,half_y,x_dim,y_dim);
	}
	else
	{
		float cur_x, cur_y;
		for (int y_coord=0; y_coord < y_dim; y_coord++)
		{
			for (int x_coord = 0; x_coord < x_dim; x_coord++)
			{
				cur_x = x_coord - (float) half_x;
				cur_y = y_coord - (float) half_y;
				pdata_h[y_coord*x_dim+x_coord] = (float)exp(-(cur_x*cur_x/std_x/std_x+cur_y*cur_y/std_y/std_y)/2);
			}
		}
	}
}

__global__ void generateLaplacianOfGaussianKernelCUDA(float* pdata_d,float std_x,float std_y,
													  unsigned int half_x,unsigned int half_y,
													  unsigned int x_dim,unsigned int y_dim)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int n = x_dim*y_dim;
	if (i >= n) return;
	int x_coord = i%x_dim;
	int y_coord = i/x_dim;

	float cur_x = x_coord - (float) half_x;
	float cur_y = y_coord - (float) half_y;
	pdata_d[y_coord*x_dim+x_coord] = -(float)(exp(-(cur_x*cur_x/std_x/std_x+cur_y*cur_y/std_y/std_y)/2))*
		((1/pow(std_x,2)-pow(cur_x,2)/pow(std_x,4))+(1/pow(std_y,2)-pow(cur_y,2)/pow(std_y,4)));
}

void FilterKernel::generateLaplacianOfGaussianKernel(float std_x,float std_y)
{
	if (gpuCompute) // 49*49
	{
		int blockSize = BLOCK_SIZE_FILTER;
		int	gridSize = (x_dim*x_dim-1)/blockSize+1;
		generateLaplacianOfGaussianKernelCUDA<<<gridSize,blockSize>>>(pdata_d,std_x,std_y,half_x,half_y,x_dim,y_dim);
	}
	else
	{
		float cur_x, cur_y;
		for (int x_coord=0; x_coord<x_dim; x_coord++)
		{
			for (int y_coord = 0;y_coord<y_dim;y_coord++)
			{
				cur_x = x_coord - (float) half_x;
				cur_y = y_coord - (float) half_y;
				pdata_h[y_coord*x_dim+x_coord] = -(float)(exp(-(cur_x*cur_x/std_x/std_x+cur_y*cur_y/std_y/std_y)/2))*
					((1/pow(std_x,2)-pow(cur_x,2)/pow(std_x,4))+(1/pow(std_y,2)-pow(cur_y,2)/pow(std_y,4)));
			}
		}
	}
}

__global__ void generate1stOrderDerivDirectionalGaussianCUDA(float* pdata_d,float std_x,float std_y,
															 unsigned int half_x,unsigned int half_y,
															 unsigned int x_dim,unsigned int y_dim,
															 float costheta, float sintheta)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int n = x_dim*y_dim;
	if (i >= n) return;
	int x_coord = i%x_dim;
	int y_coord = i/x_dim;

	float cur_x = x_coord - (float) half_x;
	float cur_y = y_coord - (float) half_y;
	float x_coord_new = costheta*cur_x - sintheta*cur_y; //Compute the rotated coordinates
	float y_coord_new = sintheta*cur_x + costheta*cur_y;
	pdata_d[y_coord*x_dim+x_coord] = (float)-exp(-(x_coord_new*x_coord_new/std_x/std_x+y_coord_new*y_coord_new/std_y/std_y)/2)*
		x_coord_new/(std_x*std_x);
}

void FilterKernel::generate1stOrderDerivDirectionalGaussian(float std_x, float std_y,unsigned char dir_idx)
{	//Genereate the 1st order derivative of the Gaussian kernel along the x direction
	//_dir_idx: index of the directions from 0 to 7 corresponding to the angle from 0 to 180
	float costheta, sintheta;
	float angle = ((float)dir_idx/float(DIRECTION_NUM)*PI);
	costheta = cos(angle);
	sintheta = sin(angle);
	if (gpuCompute)
	{
		int blockSize = BLOCK_SIZE_FILTER;
		int	gridSize = (x_dim*y_dim-1)/blockSize+1;
		generate1stOrderDerivDirectionalGaussianCUDA<<<gridSize,blockSize>>>(pdata_d,std_x,std_y,half_x,half_y,x_dim,y_dim,costheta,sintheta);
		//printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
	}
	else
	{
		float x_coord_new, y_coord_new;
		float cur_x, cur_y;
		for (int x_coord=0; x_coord<x_dim; x_coord++)
		{
			for (int y_coord = 0;y_coord<y_dim;y_coord++)
			{
				cur_x = x_coord - (float) half_x;
				cur_y = y_coord - (float) half_y;
				x_coord_new = costheta*cur_x - sintheta*cur_y; //Compute the rotated coordinates
				y_coord_new = sintheta*cur_x + costheta*cur_y;
				pdata_h[y_coord*x_dim+x_coord] = (float)-exp(-(x_coord_new*x_coord_new/std_x/std_x+y_coord_new*y_coord_new/std_y/std_y)/2)*
					x_coord_new/(std_x*std_x);
			}
		}
	}
}

__global__ void generate2ndOrderDerivDirectionalGaussianCUDA(float* pdata_d,float std_x,float std_y,
															 unsigned int half_x,unsigned int half_y,
															 unsigned int x_dim,unsigned int y_dim,
															 float costheta, float sintheta)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int n = x_dim*y_dim;
	if (i >= n) return;
	int x_coord = i%x_dim;
	int y_coord = i/x_dim;

	float cur_x = x_coord - (float) half_x;
	float cur_y = y_coord - (float) half_y;
	float x_coord_new = costheta*cur_x - sintheta*cur_y; //Compute the rotated coordinates
	float y_coord_new = sintheta*cur_x + costheta*cur_y;
	pdata_d[y_coord*x_dim+x_coord] = (float)(exp(-(x_coord_new*x_coord_new/std_x/std_x+y_coord_new*y_coord_new/std_y/std_y)/2))*
		(1/pow(std_x,2)-pow(x_coord_new,2)/pow(std_x,4));
}

void FilterKernel::generate2ndOrderDerivDirectionalGaussian(float std_x, float std_y,unsigned char dir_idx)
{
	//Genereate the 2nd order derivative of the Gaussian kernel along the x direction
	//_dir_idx: index of the directions from 0 to 7 corresponding to the angle from 0 to 180
	float costheta, sintheta;
	float angle = ((float)dir_idx/float(DIRECTION_NUM)*PI);
	costheta = cos(angle);
	sintheta = sin(angle);
	if (gpuCompute)
	{
		int blockSize = BLOCK_SIZE_FILTER;
		int	gridSize = (x_dim*x_dim-1)/blockSize+1;
		generate2ndOrderDerivDirectionalGaussianCUDA<<<gridSize,blockSize>>>(pdata_d,std_x,std_y,half_x,half_y,x_dim,y_dim,costheta,sintheta);
	}
	else
	{
		float x_coord_new, y_coord_new;
		float cur_x, cur_y;
		for (int x_coord=0; x_coord<x_dim; x_coord++)
		{
			for (int y_coord = 0;y_coord<y_dim;y_coord++)
			{
				cur_x = x_coord - (float) half_x;
				cur_y = y_coord - (float) half_y;
				x_coord_new = costheta*cur_x - sintheta*cur_y; //Compute the rotated coordinates
				y_coord_new = sintheta*cur_x + costheta*cur_y;
				pdata_h[y_coord*x_dim+x_coord] = (float)(exp(-(x_coord_new*x_coord_new/std_x/std_x+y_coord_new*y_coord_new/std_y/std_y)/2))*
					(1/pow(std_x,2)-pow(x_coord_new,2)/pow(std_x,4));
			}
		}
	}
}

//

void FilterKernel::normalization()
{	//Normalize all the filter kernel to have zero-mean and l1-norm
	if (gpuCompute)
	{
		float *mean_val = prefixSum(0);
		doArith(mean_val, 0);
		float *l1_norm = prefixSum(1);
		doArith(l1_norm, 1);
	}
	else
	{
		//First, mean-subtraction
		float mean_val = 0.0;
		float l1_norm =0.0;
		for (int x_coord = 0;x_coord<x_dim;x_coord++)	
			for (int y_coord = 0;y_coord<y_dim;y_coord++)
				mean_val += pdata_h[y_coord*x_dim+x_coord];

		mean_val = mean_val/x_dim/y_dim;

		for (int x_coord = 0;x_coord<x_dim;x_coord++)
			for (int y_coord = 0;y_coord<y_dim;y_coord++)
				pdata_h[y_coord*x_dim+x_coord] -= mean_val;

		//Next, L1 minimization
		for (int x_coord = 0;x_coord<x_dim;x_coord++)	
			for (int y_coord = 0;y_coord<y_dim;y_coord++)
				l1_norm += abs(pdata_h[y_coord*x_dim+x_coord]);

		for (int x_coord = 0;x_coord<x_dim;x_coord++)
			for (int y_coord = 0;y_coord<y_dim;y_coord++)
				pdata_h[y_coord*x_dim+x_coord] /= l1_norm;
	}
}

//

void FilterKernel::kernelVerfication()
{
	//Verification of the filtering kernel
	printf("\n");
	printf("Filtering kernel");
	float cur_sum = 0.0;
	float l1_norm = 0.0;
	for (int x_coord = 0;x_coord<x_dim;x_coord++)
	{
		for (int y_coord= 0;y_coord<y_dim;y_coord++)
		{
			//printf("%2.3f ",pdata[y_coord*x_dim+x_coord]);
			cur_sum +=pdata_h[y_coord*x_dim+x_coord];
			l1_norm +=abs(pdata_h[y_coord*x_dim+x_coord]);
		}
		//printf("\n");
	}
	//Display the kernel filter, just for verification
	printf("Size of float: %d",sizeof(float));
	Mat tempIm = cv::Mat(x_dim,y_dim,CV_32FC1,(void*)pdata_h,sizeof(float)*x_dim);//Step should be imagewidth*number of bytes per pixels
	//Note that we need to resize the image since imshow just display all pixels from [0,1]
	double minval,maxval;
	cv::minMaxLoc(tempIm,&minval,&maxval,NULL,NULL);
	tempIm.convertTo(tempIm,CV_32FC1,1.0/(maxval-minval),-minval/(maxval-minval));
	cv::namedWindow("Kernel",WINDOW_NORMAL);
	cv::imshow("Kernel",tempIm);
	waitKey(0);
	printf("Mean= %3.3f, L1-norm=%3.3f \n",cur_sum/(float)x_dim/(float)y_dim,l1_norm);
}

//

void generateFilterKernels(FilterKernel ** h_arr, bool gpuCompute)
{
	//Next, generate filter responses
	for (int nonsymscaleidx = 0;nonsymscaleidx<NONSYMSCALES;nonsymscaleidx++)
	{
		float scalex;
		scalex = pow((float)sqrt(2.0),nonsymscaleidx+1); //This is the scale of the small axis at current scale
		for (int diridx=0;diridx<DIRECTION_NUM;diridx++)
		{
			//Generate the 1st derivative of directional Gaussian filters (non symmetric)	
			h_arr[nonsymscaleidx*DIRECTION_NUM+diridx]=new FilterKernel(FILTER_KERNEL_SIZE,FILTER_KERNEL_SIZE,diridx,scalex,3.0*scalex,1,gpuCompute);
			//Generate the 2nd Derivative of Gaussian filters (non-symmetric)
			h_arr[nonsymscaleidx*DIRECTION_NUM+diridx+NONSYMSCALES*DIRECTION_NUM]=new FilterKernel(FILTER_KERNEL_SIZE,FILTER_KERNEL_SIZE,diridx,scalex,3.0*scalex,2,gpuCompute);

			//---Uncomment the following lines to see the filters--
			//h_arr[nonsymscaleidx*DIRECTION_NUM+diridx]->kernelVerfication();
			//h_arr[nonsymscaleidx*DIRECTION_NUM+diridx+NONSYMSCALES*DIRECTION_NUM]->kernelVerfication();
		}

	}

	//Next, generate the symmetric filter responses
	for (int symscaleidx = 0;symscaleidx<SYMSCALES;symscaleidx++)
	{
		float scalex;
		scalex = pow((float)sqrt(2.0),symscaleidx);
		h_arr[NONSYMSCALES*DIRECTION_NUM*2+2*symscaleidx]= new FilterKernel(FILTER_KERNEL_SIZE,FILTER_KERNEL_SIZE,scalex,0,gpuCompute);//Gaussian filters
		h_arr[NONSYMSCALES*DIRECTION_NUM*2+2*symscaleidx+1] = new FilterKernel(FILTER_KERNEL_SIZE,FILTER_KERNEL_SIZE,scalex,2,gpuCompute);//Laplacian of Gaussian filters

		//---Uncomment the following lines to see the filters--
		//h_arr[NONSYMSCALES*DIRECTION_NUM*2+2*symscaleidx]->kernelVerfication();
		//h_arr[NONSYMSCALES*DIRECTION_NUM*2+2*symscaleidx+1]->kernelVerfication();
	}	
}