#include "VectorQ.h"
#include "support.h"
#include "kmeans.h"

__constant__ float devicecluscent[CLUSTER_COUNT*(2*SYMSCALES+NONSYMSCALES)];  // frenum is not defined in this scope

__global__ void cudaVQ(	unsigned char *textonIdxIm,
						float *maxfres, 
						float *minSqDist,
						int DIM_X, int DIM_Y, int fresnum, unsigned char textonidx){

	// Declare local variables for calculation within a thread
	float temp;
	float curdist = 0.0f;

	// Get thread indexes
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int index = row*DIM_X + col;

	// Boundary check within the matrix
	if((row < DIM_X) && (col < DIM_Y)){

		// Access maxfres and devicecluscent(CONSTANT) and store in register
		for(unsigned char fresidx = 0; fresidx<fresnum; fresidx++){	
			temp = maxfres[index + fresidx*DIM_X*DIM_Y] - devicecluscent[textonidx*fresnum + fresidx];
			temp = temp*temp;
			curdist += temp;
		}
		
		// Do the vector checking with registers
		if (textonidx == 0)
			minSqDist[index] = curdist;
		else
		{
			if (curdist < minSqDist[index])
			{
				minSqDist[index] = curdist;
				textonIdxIm[index] = textonidx;
			}
		}
	}
}

void vectorQuantization(unsigned char *textonIdxIm,
						float *minSqDist, 
						float *maxfres,
						float *cluscent,
						int DIM_X, int DIM_Y, unsigned char fresnum, int clstnum,
						int textonidx
						)
{
	// Declare device variables and copy data into them
    float *devicemaxfres;
    float *deviceminSqDist;
    unsigned char *devicetextonIdxIm;

	checkCuda(cudaMalloc(&devicemaxfres, fresnum*DIM_X*DIM_Y*sizeof(float)));
	checkCuda(cudaMalloc(&deviceminSqDist, DIM_X*DIM_Y*sizeof(float)));
	checkCuda(cudaMalloc(&devicetextonIdxIm, DIM_X*DIM_Y*sizeof(unsigned char)));

	checkCuda(cudaMemcpy(devicemaxfres, maxfres,
              fresnum*DIM_X*DIM_Y*sizeof(float), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(devicetextonIdxIm, textonIdxIm,
              DIM_X*DIM_Y*sizeof(unsigned char), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(deviceminSqDist, minSqDist,
              DIM_X*DIM_Y*sizeof(float), cudaMemcpyHostToDevice));

	// Copy to const memory for better performance
	checkCuda(cudaMemcpyToSymbol(devicecluscent, cluscent, CLUSTER_COUNT*fresnum*sizeof(float)));

	// Calling cuda function with correct Dims
	dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
	dim3 gridSize((DIM_X-1)/BLOCK_SIZE_X + 1, (DIM_Y-1)/BLOCK_SIZE_Y + 1, 1);
	cudaVQ<<<gridSize, blockSize>>>(devicetextonIdxIm, devicemaxfres, 
		deviceminSqDist, DIM_X, DIM_Y, fresnum, textonidx);
	cudaDeviceSynchronize(); checkLastCudaError();

	// Get the result back
	checkCuda(cudaMemcpy(textonIdxIm, devicetextonIdxIm,
              DIM_X*DIM_Y*sizeof(unsigned char), cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(minSqDist, deviceminSqDist,
              DIM_X*DIM_Y*sizeof(float), cudaMemcpyDeviceToHost));

	// Free all device variables
    checkCuda(cudaFree(devicemaxfres));
    checkCuda(cudaFree(deviceminSqDist));
    checkCuda(cudaFree(devicetextonIdxIm));	
}