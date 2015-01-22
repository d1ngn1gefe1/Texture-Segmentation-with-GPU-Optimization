#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

void vectorQuantization(unsigned char *textonIdxIm,
						float *minSqDist, 
						float *maxfres,
						float *cluscent,
						int DIM_X, int DIM_Y, unsigned char fresnum, int clstnum,
						int textonidx
						);