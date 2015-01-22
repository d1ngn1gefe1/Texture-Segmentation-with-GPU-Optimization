#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define msg(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); } while (0)
#define err(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); exit(1); } while (0)

#define malloc2D(name, xDim, yDim, type) do {						 \
    name = (type **)malloc(xDim * sizeof(type *));					 \
    assert(name != NULL);													 \
    name[0] = (type *)malloc(xDim * yDim * sizeof(type));			 \
    assert(name[0] != NULL);												 \
    for (size_t i = 1; i < xDim; i++)											 \
        name[i] = name[i-1] + yDim;										 \
} while (0)		

inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        err("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
    }
}

inline void checkLastCudaError() {
    checkCuda(cudaGetLastError());
}

void cuda_kmeans(float **objects,      /* in: [numObjs][numCoords] */
                   int     numDims,    /* no. features */
                   int     numObjs,      /* no. objects */
                   int     numClusters,  /* no. clusters */
                   float   threshold,    /* % objects change membership */
                   int    *membership,   /* out: [numObjs] */
                   int    *loop_iterations,
				   float  **clusters
				   );
