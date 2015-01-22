#include "FilterResponse.cuh"

// Complex multiplication
static __device__ __host__ inline cufftComplex ComplexMul(const cufftComplex a, const cufftComplex b, int size)
{
	cufftComplex c;
	c.x = (a.x * b.x - a.y * b.y)/size;
	c.y = (a.x * b.y + a.y * b.x)/size;
	return c;
}

// Complex pointwise multiplication
static __global__ void ComplexPointwiseMul(const cufftComplex* src1, const cufftComplex* src2, cufftComplex* dst, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
		dst[i] = ComplexMul(src1[i], src2[i], size);     
} 

// Real multiplication
static __device__ __host__ inline float RealMul(const float a, const float b)
{
	float c = a*b;
	return c;
}

// Real pointwise multiplication
static __global__ void RealPointwiseMul(const float *src1, const float *src2, float *dst, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
		dst[i] = RealMul(src1[i], src2[i]);     
} 

// Real addition
static __device__ __host__ inline float RealAdd(const float a, const float b)
{
	float c = a+b;
	return c;
}

// Real pointwise addition
static __global__ void RealPointwiseAdd(const float *src1, const float *src2, float *dst, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
		dst[i] = RealAdd(src1[i], src2[i]);     
} 

__global__ void paddingKernel(float *expandedKernel_d, const float *kernel_d, int filterKernelSize, int xmax, int ymax)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= xmax*ymax) return;
	expandedKernel_d[i] = 0;
	int xcoord = i%xmax;
	int ycoord = i/xmax;
	int filterRadius = filterKernelSize/2;
	//top-left
	if (ycoord >= 0 && ycoord < filterRadius + 1 && xcoord >= 0 && xcoord < filterRadius + 1)
		expandedKernel_d[i] = kernel_d[(ycoord+filterRadius)*filterKernelSize+(xcoord+filterRadius)];
	//top-right
	if (ycoord >= 0 && ycoord < filterRadius + 1 && xcoord >= xmax - filterRadius && xcoord < xmax)
		expandedKernel_d[i] = kernel_d[(ycoord+filterRadius)*filterKernelSize+(xcoord-xmax+filterRadius)];
	//bot-left
	if (ycoord >= ymax - filterRadius && ycoord < ymax && xcoord >= 0 && xcoord < filterRadius + 1)
		expandedKernel_d[i] = kernel_d[(ycoord-ymax+filterRadius)*filterKernelSize+(xcoord+filterRadius)];
	//bot-right
	if (ycoord >= xmax - filterRadius && ycoord < ymax && xcoord >= xmax - filterRadius && xcoord < xmax)
		expandedKernel_d[i] = kernel_d[(ycoord-ymax+filterRadius)*filterKernelSize+(xcoord-xmax+filterRadius)];
}

void cv2cuda::padding(float *expandedKernel_d, const float *kernel_d, int filterKernelSize)
{
	/* naive way
	//top-left
	for (int k = 0; k < filterRadius + 1; k++)
	{
	for (int l = 0; l < filterRadius + 1; l++)
	{
	expandedKernel_h[(k)*DIM_IM_X+(l)] =
	kernel_h[(k+filterRadius)*FILTER_KERNEL_SIZE+(l+filterRadius)];
	}
	}
	//top-right
	for (int m = 0; m < filterRadius + 1; m++)
	{
	for (int n = DIM_IM_X - filterRadius; n < DIM_IM_X; n++)
	{
	expandedKernel_h[(m)*DIM_IM_X+(n)] =
	kernel_h[(m+filterRadius)*FILTER_KERNEL_SIZE+(n-DIM_IM_X+filterRadius)];
	}
	}
	//bot-left
	for (int o = DIM_IM_Y - filterRadius; o < DIM_IM_Y; o++)
	{
	for (int p = 0; p < filterRadius + 1; p++)
	{
	expandedKernel_h[(o)*DIM_IM_X+(p)] = 
	kernel_h[(o-DIM_IM_Y+filterRadius)*FILTER_KERNEL_SIZE+(p+filterRadius)];
	}
	}
	//bot-right
	for (int q = DIM_IM_X - filterRadius; q < DIM_IM_Y; q++)
	{
	for (int r = DIM_IM_X - filterRadius; r < DIM_IM_X; r++)
	{
	expandedKernel_h[(q)*DIM_IM_X+(r)] =
	kernel_h[(q-DIM_IM_Y+filterRadius)*FILTER_KERNEL_SIZE+(r-DIM_IM_X+filterRadius)];
	}
	}
	*/
	paddingKernel<<<(DIM_IM_X*DIM_IM_Y-1)/BLOCK_SIZE_FILTER+1, BLOCK_SIZE_FILTER>>>(expandedKernel_d, kernel_d, filterKernelSize, DIM_IM_X, DIM_IM_Y);
}

void cv2cuda::filter2D(const float *src, float *dst, const float *kernel, float *expandedKernel_d,
					   cufftHandle *plan1, cufftHandle *plan2,
					   cufftComplex *src_c, cufftComplex *expandedKernel_d_c) //image: 2048*2048, kernel: 49*49
{
	// Pad the kernel to right size
	padding(expandedKernel_d, kernel, FILTER_KERNEL_SIZE);

	// Transform image and kernel
	cufftExecR2C(*plan1, (cufftReal*)src, (cufftComplex*)src_c);
	cufftExecR2C(*plan1, (cufftReal*)expandedKernel_d, (cufftComplex*)expandedKernel_d_c);

	// Multiply the coefficients together
	ComplexPointwiseMul<<<(DIM_IM_X*DIM_IM_Y-1)/BLOCK_SIZE_FILTER+1, BLOCK_SIZE_FILTER>>>(src_c, expandedKernel_d_c, src_c, DIM_IM_X*DIM_IM_Y);
	//printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));

	// Inverse
	cufftExecC2R(*plan2, (cufftComplex*)src_c, (cufftReal*)dst);
}

void cv2cuda::multiply(const float *src1, const float *src2, float *dst, int size)
{
	RealPointwiseMul<<<(size-1)/BLOCK_SIZE_FILTER+1, BLOCK_SIZE_FILTER>>>(src1, src2, dst, size);
	//printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
}

void cv2cuda::add(const float *src1, const float *src2, float *dst, int size)
{
	RealPointwiseAdd<<<(size-1)/BLOCK_SIZE_FILTER+1, BLOCK_SIZE_FILTER>>>(src1, src2, dst, size);
	//printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
}

__global__ void helperKernel1(float *tempOut_d, float *maxRes_d, unsigned char *maxidx_d, unsigned char diridx, int xmax, int ymax)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= xmax*ymax) return;
	if (tempOut_d[i] > maxRes_d[i])
	{
		maxRes_d[i] = tempOut_d[i];
		maxidx_d[i] = diridx;
	}
}

void FilterResponse::helperFunc1(float *tempOut_d, float *maxRes_d, unsigned char *maxidx_d, unsigned char diridx)
{
	helperKernel1<<<(DIM_IM_X*DIM_IM_X-1)/BLOCK_SIZE_FILTER+1, BLOCK_SIZE_FILTER>>>(tempOut_d, maxRes_d, maxidx_d, diridx, DIM_IM_X, DIM_IM_Y);
}

__global__ void helperKernel2(float *maxfres_d, float *maxRes_d, unsigned char *maxdiridx_d, unsigned char *maxidx_d, int xmax, int ymax, int nonsymscaleidx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= xmax*ymax) return;
	maxfres_d[i+nonsymscaleidx*xmax*ymax] = maxRes_d[i];
	maxdiridx_d[i+nonsymscaleidx*xmax*ymax] = maxidx_d[i];
}

void FilterResponse::helperFunc2(float *maxfres_d, float *maxRes_d, unsigned char *maxdiridx_d, unsigned char *maxidx_d, int nonsymscaleidx)
{
	helperKernel2<<<(DIM_IM_X*DIM_IM_X-1)/BLOCK_SIZE_FILTER+1, BLOCK_SIZE_FILTER>>>(maxfres_d, maxRes_d, maxdiridx_d, maxidx_d, DIM_IM_X, DIM_IM_Y, nonsymscaleidx);
}

__global__ void helperKernel3(float *maxfres_d, float *tempOut_d, int symscaleidx, unsigned char extraidx, int nonsymscales, int xmax, int ymax)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= xmax*ymax) return;
	maxfres_d[i+(2*symscaleidx+nonsymscales+extraidx)*xmax*ymax] = tempOut_d[i];
}

void FilterResponse::helperFunc3(float *maxfres_d, float *tempOut_d, int symscaleidx, unsigned char extraidx)
{
	helperKernel3<<<(DIM_IM_X*DIM_IM_X-1)/BLOCK_SIZE_FILTER+1, BLOCK_SIZE_FILTER>>>(maxfres_d, tempOut_d, symscaleidx, extraidx, NONSYMSCALES, DIM_IM_X, DIM_IM_Y);
}

void cpucode::filter2D(const float *src, float *dst, const float *kernel) //can't be in-place
{
	int r = FILTER_KERNEL_SIZE/2;
	for (int i = 0; i < DIM_IM_Y; i++)
	{ 
		for (int j = 0; j < DIM_IM_X; j++)
		{
			float sum = 0.0f;
			for (int k = -r; k <= r; k++)
			{
				for (int l = -r; l <= r; l++)
				{
					int y = i+k;
					int x = j+l;
					if (y >= 0 && y < DIM_IM_Y && x >= 0 && x < DIM_IM_X)
					{
						sum += src[y*DIM_IM_X+x]*kernel[k*FILTER_KERNEL_SIZE+l];
					}
				}
			}
			dst[i*DIM_IM_X+j] = sum;
		}
	}
}