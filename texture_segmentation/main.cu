/***********************************************************************
This program implements the texture segmentation algorithm based on texton
features.
Author: Tan H. Nguyen
Advisers: Dr. Minh N. Do and Dr. Gabriel Popescu
IFP & QLI Labs - University of Illinois at Urbana-Champaign
************************************************************************/

#include <stdio.h>
#include "kernel.h"
#include "support.h"
#include "tree.h"
#include "treeGPU.h"
#include <math.h>
#include "kmeans.h"
#include "VectorQ.h"
#include "FilterResponse.cuh"

//Random Forest defined on the GPU
//__device__ extern BinaryTreeGPU *pTrees[TREE_NUM]; //This is an array of trees used for the random forest
extern int * subfeatarr_d;
 
int main()
{
	//First, print the information of the GPU device (if available)
	printf("**-------Texture segmentation with Textons-------**\n");
	printf("**--------------Author: Tan H. Nguyen - Xiaotan Duan - Zelun Luo------------**\n");
	printf("**--University of Illinois at Urbana-Champaign---**\n");
	printf("**--------------ECE 408 class project------------**\n\n");

	int inImHeight=0, inImWidth=0;
	float *inImData = NULL;
	auto impath = "../InputData/";
	char imfilename[10] = "D1.tif";
	Mat inIm; //This structure is for holding the input image
	//Filters informations....
	unsigned int filternum;
	unsigned int numIm=0; //Number of images
	int runMode;
	Timer curTimer;
	//Get device information
	getDeviceInfo(); 
	
	//Load the image into data structure
	bool gpuCompute = false;
	printf("2. Read input data\n");
	//Generate filters
	startTime(&curTimer);
	filternum = (unsigned int)NONSYMSCALES*2*DIRECTION_NUM+SYMSCALES*2;
	FilterKernel **h_arr=(FilterKernel**)malloc(sizeof(FilterKernel)*filternum);		//This is a pointer, points to 
	generateFilterKernels(h_arr, gpuCompute);
	stopTime(&curTimer);
	printf("[Generate filters]	Time: %f [ms]\n", elapsedTime(curTimer));

	
	//Step 1: filter the input image with all of the filters in the kernel set
	//FilterResponse *f1 = new FilterResponse(impath,imfilename,h_arr,DIM_IM_X,DIM_IM_Y,DIRECTION_NUM,SYMSCALES,NONSYMSCALES,0);
	FilterResponse *f1 = new FilterResponse(impath,imfilename,h_arr,DIM_IM_X,DIM_IM_Y,DIRECTION_NUM,SYMSCALES,NONSYMSCALES,gpuCompute);



	//Step 2: subsample the data size and do k-means to retrieve textons, index image
	//First, create data for k-means clustering where each row is 1 data sample.
	float *kmeansdata = (float*)calloc(SAMPLE_PER_IM*(2*SYMSCALES+NONSYMSCALES),sizeof(float));
	unsigned int samplingstep = floor(DIM_IM_X/sqrt((float)SAMPLE_PER_IM));
	unsigned long sampleIdx=0;
	unsigned char fresnum = 2*SYMSCALES+NONSYMSCALES;
	for (unsigned int x_coord=0;x_coord<DIM_IM_X;x_coord+=samplingstep){
		for (unsigned int y_coord=0;y_coord<DIM_IM_Y;y_coord+=samplingstep){
			for (unsigned char fresidx=0;fresidx<fresnum;fresidx++)
			{
				kmeansdata[sampleIdx*fresnum+fresidx]=f1->maxfres[y_coord*DIM_IM_X+x_coord+fresidx*DIM_IM_X*DIM_IM_Y];
			}
			sampleIdx++;
		}
	}
	
	//Step 3. Find textons
	printf("3. Texton searching...");
	//Convert the data into a format supported by opencv
	cv::Mat kmeansdataim = cv::Mat(SAMPLE_PER_IM,fresnum,CV_32FC1,(void*)kmeansdata,fresnum*sizeof(float));
	cv::Mat outlbl, cluscent;
	
	
	//CPU code
	cv::TermCriteria stopcriteria;
	stopcriteria.epsilon = K_MEANS_MIN_EPS; 
	startTime(&curTimer);
	cv::kmeans(kmeansdataim,CLUSTER_COUNT,outlbl,stopcriteria,2,KMEANS_RANDOM_CENTERS,cluscent);
	stopTime(&curTimer);
	printf("\n[Finding textons - CPU]	Time: %f [ms]",elapsedTime(curTimer));
	

	//GPU mode
	float** objects;
	malloc2D(objects, SAMPLE_PER_IM,  fresnum, float);
    for (int i = 0; i < SAMPLE_PER_IM; i++)
	{
        for (int j = 0; j < fresnum; j++)
		{
            objects[i][j] = kmeansdata[i*fresnum + j];
        }
    }
	float** clusters;
	malloc2D(clusters, CLUSTER_COUNT, fresnum, float);
	for (unsigned int textonidx=0;textonidx<CLUSTER_COUNT;textonidx++)
	{
		for (unsigned char fresidx=0;fresidx<fresnum;fresidx++)
		{	
			clusters[textonidx][fresidx] =  cluscent.at<float>(textonidx,fresidx); // Using openCV kmean
		}
	}
	int* clusterID;
	clusterID = (int *)malloc(SAMPLE_PER_IM*sizeof(int));
	int loop_iterations;
	startTime(&curTimer);
	cuda_kmeans(objects, fresnum, SAMPLE_PER_IM, CLUSTER_COUNT, K_MEANS_MIN_EPS, clusterID, &loop_iterations, clusters);
	stopTime(&curTimer);
	printf("\n[Finding textons - GPU]	Time: %f ms",elapsedTime(curTimer));
	
	
	
	//Step 4: Vector quantization on each input image
	//printf("\n4. Vector quantization...");
	/*
	//CPU code
	unsigned char *textonIdxIm = (unsigned char*)calloc(DIM_IM_X*DIM_IM_Y,sizeof(unsigned char)); //Texton index image
	float *minSqDist = (float*)calloc(DIM_IM_X*DIM_IM_Y,sizeof(float));
	startTime(&curTimer);
	//Searching for best texton in each image
	for (unsigned int textonidx=0;textonidx<CLUSTER_COUNT;textonidx++){
		for (unsigned int x_coord=0;x_coord<DIM_IM_X;x_coord++){
			for (unsigned int y_coord=0;y_coord<DIM_IM_Y;y_coord++){
					float curdistsq = 0;
					//Compute the distance from current textons to all filter responses in the image.
					for (unsigned char fresidx=0;fresidx<fresnum;fresidx++)			
						curdistsq +=  pow((float)f1->maxfres[y_coord*DIM_IM_X+x_coord+fresidx*DIM_IM_X*DIM_IM_Y]-cluscent.at<float>(textonidx,fresidx),2);					
					if (textonidx==0)
						minSqDist[y_coord*DIM_IM_X+x_coord]=curdistsq;
					else
					{
						if (curdistsq<minSqDist[y_coord*DIM_IM_X+x_coord])
						{
							minSqDist[y_coord*DIM_IM_X+x_coord]=curdistsq;
							textonIdxIm[y_coord*DIM_IM_X+x_coord]=textonidx;
						}
					}				
			
			}
		}
		cv::Mat textonIdxImVis = cv::Mat(DIM_IM_Y,DIM_IM_X,CV_8UC1,(void*)textonIdxIm,DIM_IM_X*sizeof(unsigned char));
		imagesc("Texton index",textonIdxImVis);
	}
	stopTime(&curTimer);
	printf("\n[VQ - CPU]	Elapsed time: %f ms\n",elapsedTime(curTimer));
	*/

	//---------------------------------------------------------------------------------------------------------------------

	float *cluscentTemp1 = (float*)calloc(CLUSTER_COUNT*fresnum,sizeof(float));
	float *cluscentTemp2 = (float*)calloc(CLUSTER_COUNT*fresnum,sizeof(float));
	for (unsigned int textonidx=0;textonidx<CLUSTER_COUNT;textonidx++)
	{
		for (unsigned char fresidx=0;fresidx<fresnum;fresidx++)
		{	
			cluscentTemp1[textonidx*fresnum + fresidx] = cluscent.at<float>(textonidx,fresidx); // Using openCV kmean
			cluscentTemp2[textonidx*fresnum + fresidx] = clusters[textonidx][fresidx]; // Using cuda_kmean clusters, CLUSTER_COUNT, fresnum,
			if((cluscentTemp1[textonidx*fresnum + fresidx]>cluscentTemp2[textonidx*fresnum + fresidx]+100)||
				(cluscentTemp1[textonidx*fresnum + fresidx]<cluscentTemp2[textonidx*fresnum + fresidx]-100)) 
				printf( "\nCLusters shifted more than 100" );
		}
	}
	unsigned char *textonIdxIm_mdf = (unsigned char*)calloc(DIM_IM_X*DIM_IM_Y,sizeof(unsigned char)); //Texton index image
	float *minSqDist_mdf = (float*)calloc(DIM_IM_X*DIM_IM_Y,sizeof(float));
	for (unsigned int textonidx=0;textonidx<CLUSTER_COUNT;textonidx++){	
		// Call the cuda function
		vectorQuantization(textonIdxIm_mdf,
							minSqDist_mdf, 
							f1->maxfres,
							cluscentTemp1,
							DIM_IM_X, DIM_IM_Y, fresnum, CLUSTER_COUNT,
							textonidx
							);
		cv::Mat textonIdxImVis = cv::Mat(DIM_IM_Y,DIM_IM_X,CV_8UC1,(void*)textonIdxIm_mdf,DIM_IM_X*sizeof(unsigned char));
		imagesc("Texton index",textonIdxImVis);	
	}
	free(minSqDist_mdf);
	stopTime(&curTimer);
	printf("\n[VQ - GPU]	Elapsed time: %f ms",elapsedTime(curTimer));
	
	//Step 4: Compute the histogram of textons for each image in the database
	float *hot; //Histogram of textons
	printf("\n5. Histogram computing...");	
	startTime(&curTimer);
	computeRegionDescriptor(textonIdxIm_mdf, hot);
	stopTime(&curTimer);
	printf("\n[HOT] Elapsed time: %f ms",elapsedTime(curTimer));
	
	
	//Step 5: do random forest on the training set and validation set.
	//Prepare the data for the Random Forest training
	int *label_arr = NULL ,*eval_label_arr = NULL;
	double *score_arr = NULL;
	//unsigned long ninputsamples = 4601;
	//unsigned int feature_dim = 57;
	unsigned long ninputsamples = 268000;
	unsigned int feature_dim = 51;
	
	unsigned int nlabels = 2;
	long ntrain,ntest,nval;
	double *trainscore,*valscore,*testscore;
	int *trainlabel,*vallabel,*testlabel;

	label_arr = (int*) calloc(ninputsamples,sizeof(int));
	score_arr = (double*) calloc(ninputsamples*feature_dim,sizeof(double));
	char datafilename[100]="../../data/textondata.txt";
	//char datafilename[100]="../../data/spamdata.txt";
	
	readDataFile(datafilename,label_arr,score_arr,ninputsamples,feature_dim);//Read in the text data file. label_arr is an array in [0,1,...num_arr-1]
	//Prepare the data for training a the random forest
	prepareDataForRandomForest(score_arr, label_arr, feature_dim,ninputsamples, trainscore, trainlabel,valscore,vallabel,
		ntrain,nval);
	
	printf("\n6. Training the random forest");	
	int *out_label_arr;
	float *out_fract_arr;
	float *cf_mat;

	
	//Train and test the Random Forest - CPU code
	long nsamplespertree = 200000;
	startTime(&curTimer);
	RandomForest *pForest = new RandomForest(TREE_NUM,SUP_FEAT_NUM,feature_dim,ntrain);
	pForest->randomForestLearn(trainscore,trainlabel,nlabels,ntrain);
	stopTime(&curTimer);
	printf("\n[Training RF- CPU] time: %1.4f [ms]",elapsedTime(curTimer));

	startTime(&curTimer);
	pForest->randomForestEval(valscore,out_label_arr,out_fract_arr,nlabels,feature_dim,nval);
	stopTime(&curTimer);
	printf("\n[Testing RF- CPU] time: %1.4f [ms]",elapsedTime(curTimer));
	confusionMatrix(vallabel, out_label_arr, cf_mat, nlabels, nval);
	
	
	//Train and test the Random Forest -  GPU code
	int *subfeatarr_h, *subfeatarr_d;
	size_t newheapsize = 1024*1024*1024;//New heap size
	dim3 blockdim(BLOCK_SIZE,1,1);//1 block is for 1 tree
	dim3 griddim(ceil((float)TREE_NUM/BLOCK_SIZE),1,1); //Use 1 thread for 1 tree
	double *score_d;		//Data for the training on GPU
	int *label_d;			//An array for the training label
	TreeNodeGPU *nodeList_d;  //An array for all the nodes in the tree
	long *nodeIndex_d;			//This is an array for training the node index of size [TREE_NUM*NSAMPLETRAIN]. Each tree has a different row
	size_t avail_mem, total_mem;
	cudaEvent_t start, stop;
	float kernelTime;
	gpuErrchk(cudaDeviceSetLimit(cudaLimitMallocHeapSize,newheapsize)); //Larger heap size is needed for a larger forest..
	size_t newstacksize = 128*1000;
	gpuErrchk(cudaDeviceSetLimit(cudaLimitStackSize,newstacksize));
	generateFeatureToBeUsedByRandomForest(subfeatarr_h, TREE_NUM, SUP_FEAT_NUM, feature_dim);
	
	//Static dynamic memory allocation for CUDA RANDOM FOREST
	gpuErrchk(cudaMalloc((void**)&nodeIndex_d,sizeof(long)*TREE_NUM*ntrain));
	gpuErrchk(cudaMalloc((void**)&nodeList_d,TREE_NUM*sizeof(TreeNodeGPU)*MAX_NODE_COUNT_PER_TREE));
	gpuErrchk(cudaMalloc((void**)&subfeatarr_d,sizeof(int)*SUP_FEAT_NUM*TREE_NUM)); //Featured to be used by different tree
	gpuErrchk(cudaMemcpy((void*)subfeatarr_d,(void*)subfeatarr_h,sizeof(int)*SUP_FEAT_NUM*TREE_NUM,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**)&score_d,sizeof(double)*feature_dim*ntrain));       //Initialize training data on the GPU
	gpuErrchk(cudaMemcpy((void*)score_d,(void*)trainscore,sizeof(double)*feature_dim*ntrain,cudaMemcpyHostToDevice));//Copy the data onto the GPU
	gpuErrchk(cudaMalloc((void**)&label_d,sizeof(int)*ntrain));
	gpuErrchk(cudaMemcpy((void*)label_d,(void*)trainlabel,sizeof(int)*ntrain,cudaMemcpyHostToDevice));				//Copy training label onto the GPU
	cudaMemGetInfo(&avail_mem,&total_mem);
	printf("\nUsed memory (kernel called): %ld [Mbytes]",(long)(total_mem-avail_mem)>>20);

	//Measuring timing performance
	gpuErrchk(cudaEventCreate(&start));
	gpuErrchk(cudaEventCreate(&stop));
	gpuErrchk(cudaEventRecord(start,0));
	generateRandomForestOnGPU<<<griddim,blockdim>>>(score_d,label_d,subfeatarr_d,nodeList_d,nodeIndex_d,nlabels,feature_dim,ntrain);
	gpuErrchk(cudaEventRecord(stop,0));
	gpuErrchk(cudaEventSynchronize(stop));
	gpuErrchk(cudaEventElapsedTime(&kernelTime,start,stop));
	printf("\n[Training RF - GPU] time: %1.4f [ms]",kernelTime);
	cudaFree(score_d); 
	cudaFree(label_d);
	

	cudaDeviceSynchronize();//Synchronize CPU & GPU for output
	

	//Prepare the data for validation
	int *val_label_d, *val_label_h;			//An array for the validation set label
	float *val_fraction_arr_d,*val_fraction_arr_h;
	double *val_score_d;					//Data for validation, exist on the GPU
	gpuErrchk(cudaMalloc((void**)&val_label_d,sizeof(int)*nval));
	gpuErrchk(cudaMalloc((void**)&val_score_d,sizeof(double)*nval*feature_dim));
	gpuErrchk(cudaMalloc((void**)&val_fraction_arr_d,sizeof(float)*nval*nlabels));
	gpuErrchk(cudaMemcpy((void*)val_score_d,(void*)valscore,sizeof(double)*nval*feature_dim,cudaMemcpyHostToDevice));
	val_label_h = (int*)malloc(sizeof(int)*nval);
	val_fraction_arr_h = (float*)malloc(sizeof(float)*nval*nlabels);
	
	//Measuring timing performance
	//blockdim.x = 256;
	//griddim.x = 16;
	gpuErrchk(cudaEventCreate(&start));
	gpuErrchk(cudaEventCreate(&stop));
	gpuErrchk(cudaEventRecord(start,0));
	blockdim.x = BLOCK_SIZE_VAL;
	griddim.x = 8;
	randomForestEval<<<griddim,blockdim>>>(val_score_d,val_label_d, val_fraction_arr_d, nlabels,feature_dim,nval); //Testing directly on the training data
	gpuErrchk(cudaEventRecord(stop,0));
	gpuErrchk(cudaEventSynchronize(stop));
	gpuErrchk(cudaEventElapsedTime(&kernelTime,start,stop));
	printf("\n[Testing RF - GPU]: %1.4f [ms]",kernelTime);
	
	cudaMemcpy((void*)val_label_h,(void*)val_label_d,sizeof(int)*nval,cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)val_fraction_arr_h,(void*)val_fraction_arr_d,sizeof(float)*nval*nlabels,cudaMemcpyDeviceToHost);
	confusionMatrix(vallabel,val_label_h, cf_mat, nlabels, nval);
	cudaFree(subfeatarr_d);
	cudaFree(nodeList_d);
	cudaFree(nodeIndex_d);
	cudaDeviceSynchronize();//Synchronize CPU & GPU for output
	cudaDeviceReset();
	

	getchar();	
	
}