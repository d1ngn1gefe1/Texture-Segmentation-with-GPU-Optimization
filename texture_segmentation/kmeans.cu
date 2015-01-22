#include "kmeans.h"

// Helpoer functions for Kmean
static inline int nextPowerOfTwo(int n) {
    n--;
    n = n >>  1 | n;
    n = n >>  2 | n;
    n = n >>  4 | n;
    n = n >>  8 | n;
    n = n >> 16 | n;
//  n = n >> 32 | n;    //  For 64-bit ints
    return ++n;
}

__host__ __device__ inline static
float euclid_dist_2(int    numCoords,
                    int    numObjs,
                    int    numClusters,
                    float *objects,     // [numDims][numObjs]
                    float *clusters,    // [numDims][numClusters]
                    int    objectId,
                    int    clusterId)
{
    float ans=0.0f;
    for (int i = 0; i < numCoords; i++) {
        ans += (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]) *
               (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]);
    }
    return(ans);
}


__global__ static
void findNearestCluster(int numDims, int numObjs, int numClusters,
                          float *objects,           //  [numDims][numObjs]
                          float *deviceClusters,    //  [numDims][numClusters]
                          int *membership,          //  [numObjs]
                          int *intermediates)
{
    extern __shared__ char sharedMemory[];

    //  The type chosen for membershipChanged must be large enough to support
    //  reductions! There are blockDim.x elements, one for each thread in the
    //  block. See numThreadsPerClusterBlock in cuda_kmeans().
    unsigned char *membershipChanged = (unsigned char *)sharedMemory;

    float *clusters = (float *)(sharedMemory + blockDim.x);


    membershipChanged[threadIdx.x] = 0;

    //  BEWARE: We can overrun our shared memory here if there are too many
    //  clusters or too many coordinates! For reference, a Tesla C1060 has 16
    //  KiB of shared memory per block, and a GeForce GTX 480 has 48 KiB of
    //  shared memory per block.
    for (int i = threadIdx.x; i < numClusters; i += blockDim.x) {
        for (int j = 0; j < numDims; j++) {
            clusters[numClusters * j + i] = deviceClusters[numClusters * j + i];
        }
    }
    __syncthreads();


    int objectId = blockDim.x * blockIdx.x + threadIdx.x;

    if (objectId < numObjs) {
        int   index, i;
        float dist, min_dist;

        /* find the cluster id that has min distance to object */
        index    = 0;
        min_dist = euclid_dist_2(numDims, numObjs, numClusters,
                                 objects, clusters, objectId, 0);

        for (i=1; i<numClusters; i++) {
            dist = euclid_dist_2(numDims, numObjs, numClusters,
                                 objects, clusters, objectId, i);
            /* no need square root */
            if (dist < min_dist) { /* find the min and its array index */
                min_dist = dist;
                index    = i;
            }
        }

        if (membership[objectId] != index) {
            membershipChanged[threadIdx.x] = 1;
        }

        /* assign the membership to object objectId */
        membership[objectId] = index;

        __syncthreads();    //  For membershipChanged[]

        //  blockDim.x *must* be a power of two!
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                membershipChanged[threadIdx.x] +=
                    membershipChanged[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            intermediates[blockIdx.x] = membershipChanged[0];
        }
    }
}


__global__ static
void computeDelta(int *deviceIntermediates,
                   int numIntermediates,    //  The actual number of intermediates
                   int numIntermediates2)   //  The next power of two
{
    //  The number of elements in this array should be equal to
    //  numIntermediates2, the number of threads launched. It *must* be a power
    //  of two!
    extern __shared__ unsigned int intermediates[];

    //  Copy global intermediate values into shared memory.
    intermediates[threadIdx.x] =
        (threadIdx.x < numIntermediates) ? deviceIntermediates[threadIdx.x] : 0;

    __syncthreads();

    //  numIntermediates2 *must* be a power of two!
    for (unsigned int s = numIntermediates2 / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            intermediates[threadIdx.x] += intermediates[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        deviceIntermediates[0] = intermediates[0];
    }
}

/* return an array of cluster centers of size [numClusters][numCoords]       */
void cuda_kmeans(float **objects,      /* in: [numObjs][numCoords] */
                   int     numDims,    /* no. features */
                   int     numObjs,      /* no. objects */
                   int     numClusters,  /* no. clusters */
                   float   threshold,    /* % objects change membership */
                   int    *membership,   /* out: [numObjs] */
                   int    *loop_iterations,
				   float  **clusters
				   )
{
    int i, j, index, loop=0;
    int *newClusterSize; /* [numClusters]: no. objects assigned in each
                                new cluster */
    float delta;          /* % of objects change their clusters */
    float  **dimObjects;
    //float  **clusters;       /* out: [numClusters][numCoords] */
    float  **dimClusters;
    float  **newClusters;    /* [numCoords][numClusters] */

    float *deviceObjects;
    float *deviceClusters;
    int *deviceMembership;
    int *deviceIntermediates;

    //  Copy objects given in [numObjs][numCoords] layout to new
    //  [numDims][numObjs] layout
    malloc2D(dimObjects, numDims, numObjs, float);
    for (i = 0; i < numDims; i++) {
        for (j = 0; j < numObjs; j++) {
            dimObjects[i][j] = objects[j][i];
        }
    }

    /* pick first numClusters elements of objects[] as initial cluster centers*/
    malloc2D(dimClusters, numDims, numClusters, float);
    for (i = 0; i < numDims; i++) {
        for (j = 0; j < numClusters; j++) {
            dimClusters[i][j] = clusters[j][i];
        }
    }

    /* initialize membership[] */
    for (i=0; i<numObjs; i++) membership[i] = -1;

    /* need to initialize newClusterSize and newClusters[0] to all 0 */
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);

    malloc2D(newClusters, numDims, numClusters, float);
    memset(newClusters[0], 0, numDims * numClusters * sizeof(float));

    //  To support reduction, numThreadsPerClusterBlock *must* be a power of
    //  two, and it *must* be no larger than the number of bits that will
    //  fit into an unsigned char, the type used to keep track of membership
    //  changes in the kernel.
    const unsigned int numThreadsPerClusterBlock = 128;
    const unsigned int numClusterBlocks =
        (numObjs + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;

    const unsigned int clusterBlockSharedDataSize =
        numThreadsPerClusterBlock * sizeof(unsigned char) +
        numClusters * numDims * sizeof(float);

    cudaDeviceProp deviceProp;
    int deviceNum;
    cudaGetDevice(&deviceNum);
    cudaGetDeviceProperties(&deviceProp, deviceNum);

    if (clusterBlockSharedDataSize > deviceProp.sharedMemPerBlock) {
        err("WARNING: Your CUDA hardware has insufficient block shared memory. "
            "You need to recompile with BLOCK_SHARED_MEM_OPTIMIZATION=0. "
            "See the README for details.\n");
    }

    const unsigned int numReductionThreads =
        nextPowerOfTwo(numClusterBlocks);
    const unsigned int reductionBlockSharedDataSize =
        numReductionThreads * sizeof(unsigned int);

    checkCuda(cudaMalloc(&deviceObjects, numObjs*numDims*sizeof(float)));
    checkCuda(cudaMalloc(&deviceClusters, numClusters*numDims*sizeof(float)));
    checkCuda(cudaMalloc(&deviceMembership, numObjs*sizeof(int)));
    checkCuda(cudaMalloc(&deviceIntermediates, numReductionThreads*sizeof(unsigned int)));

    checkCuda(cudaMemcpy(deviceObjects, dimObjects[0],
              numObjs*numDims*sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(deviceMembership, membership,
              numObjs*sizeof(int), cudaMemcpyHostToDevice));

    do {
        checkCuda(cudaMemcpy(deviceClusters, dimClusters[0],
                  numClusters*numDims*sizeof(float), cudaMemcpyHostToDevice));

        findNearestCluster
            <<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
            (numDims, numObjs, numClusters,
             deviceObjects, deviceClusters, deviceMembership, deviceIntermediates);

        cudaDeviceSynchronize(); checkLastCudaError();

        computeDelta <<< 1, numReductionThreads, reductionBlockSharedDataSize >>>
            (deviceIntermediates, numClusterBlocks, numReductionThreads);

        cudaDeviceSynchronize(); checkLastCudaError();

        int d;
        checkCuda(cudaMemcpy(&d, deviceIntermediates,
                  sizeof(int), cudaMemcpyDeviceToHost));
        delta = (float)d;

        checkCuda(cudaMemcpy(membership, deviceMembership,
                  numObjs*sizeof(int), cudaMemcpyDeviceToHost));

        for (i=0; i<numObjs; i++) {
            /* find the array index of nestest cluster center */
            index = membership[i];

            /* update new cluster centers : sum of objects located within */
            newClusterSize[index]++;
            for (j=0; j<numDims; j++)
                newClusters[j][index] += objects[i][j];
        }

        //  TODO: Flip the nesting order
        //  TODO: Change layout of newClusters to [numClusters][numCoords]
        /* average the sum and replace old cluster centers with newClusters */
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numDims; j++) {
                if (newClusterSize[i] > 0)
                    dimClusters[j][i] = newClusters[j][i] / newClusterSize[i];
                newClusters[j][i] = 0.0;   /* set back to 0 */
            }
            newClusterSize[i] = 0;   /* set back to 0 */
        }

        delta /= numObjs;
    } while (delta > threshold && loop++ < 100);

    *loop_iterations = loop + 1;

    /* allocate a 2D space for returning variable clusters[] (coordinates
       of cluster centers) */
    //malloc2D(clusters, numClusters,numDims, float);
    for (i = 0; i < numClusters; i++) {
        for (j = 0; j < numDims; j++) {
            clusters[i][j] = dimClusters[j][i];
        }
    }

    checkCuda(cudaFree(deviceObjects));
    checkCuda(cudaFree(deviceClusters));
    checkCuda(cudaFree(deviceMembership));
    checkCuda(cudaFree(deviceIntermediates));

    free(dimObjects[0]);
    free(dimObjects);
    free(dimClusters[0]);
    free(dimClusters);
    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    //return clusters;
}
