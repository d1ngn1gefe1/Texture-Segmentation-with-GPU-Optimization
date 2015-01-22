#include <stdio.h>
#include <conio.h>
#include <windows.h>
#include <math.h>
#include <string.h>
#include <iostream>
#define BLOCK_SIZE	32//Number of threads per block working together to create a tree
#define BLOCK_SIZE_VAL 256
#define MAX_LABEL_NUM 3//Maximum number of labels


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=false)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"\nGPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


struct NodeDataGPU{
	int varnum;	//Index of the variable to determine the split
	float thresh;//Value of the threshold
	int numClass; //Number of class;
	float fraction_arr[MAX_LABEL_NUM]; //This is the fractional array of instances for each class in the node
	long nsamples;
	int majorclassidx;//Major class index in the node
	int level;		  //Level of the node
	//The following variables ar for the tree-pruning process, which will not be initialized in the training process
	long valnsamples;
	long valtruecount; //Fractional array of the evaluation instances for the pruning purpose
};


struct TreeNodeGPU{
	//This is a struct for each note in the tree
	NodeDataGPU data;
	unsigned long key;
	TreeNodeGPU *parent;
	TreeNodeGPU *leftChild;
	TreeNodeGPU *rightChild;	
};

void generateFeatureToBeUsedByRandomForest(int *&subfeatarr, int ntrees, int nfeateach, int nfeat);

//Kernels for generating the random forest
__global__ void generateRandomForestOnGPU(double * data, int *label, int *feat_d, TreeNodeGPU *nodeList_d, long *nodeIndex_d, int nlabels, int nfeatures,long nsamples);//This kernel creates a Binary Tree on GPU
__device__ void treeLearnID3(double *data, int *label,int nlabels, int nfeatures, long nsamples); //Train the classification tree
__device__ void splitNode(TreeNodeGPU *curNode, double*data,int *label, int nlables,int nfeatures, long nsamples);	//Recursively splitting data at a node
__device__ void findBestSplit(TreeNodeGPU *curNode,double *data,int *label, int nlabels, int nfeatures, int &vartosplit, float &thresh, int &nspl1, int &nspl2, float *fraction_arr);//Find the best split at current node
__global__ void randomForestEval(double *data,int *label, float *fraction_arr, int nlabels, int nfeatures, long nsamples);//Evaluate the label given the input data, produce the fractional result for the output confidence
__device__ void treeEval(double *data, int *fraction_arr, int treeIdx, int sampleIdxInBlock, int nlabels, int nfeatures);					//Evaluate the tree index and save the output in the fraction array
__device__ void preScan(TreeNodeGPU *p);//Do preOrder scan and display the key 
__device__ void inScan(TreeNodeGPU *p);//Do inOrder scan and display the key and subkey of a node
__device__ void postScan(TreeNodeGPU *p);//Do post scan
__device__ bool isFeatureUsedInAncesstor(TreeNodeGPU *p, int featureIdx);	//Check if a feature has been previously used in the ancessor


