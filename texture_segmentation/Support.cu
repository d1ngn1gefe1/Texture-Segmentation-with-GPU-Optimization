#include "Support.h"
#include "kernel.h"
#include "treeGPU.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
//These functions are for measuring the timing
void startTime(Timer* ptimer)
{
	ptimer->startTime = clock();
}

void stopTime(Timer* ptimer)
{
	ptimer->endTime = clock();
}

float elapsedTime(Timer timer)
{
	float timemSec = 0.0;
	timemSec = (float) (timer.endTime - timer.startTime); 
	return timemSec;//www.union.uiuc.edu/involvement/sorf/forms/SORF-Allocation-Meeting-Dates-6-27-13.pdf;
}

int getDeviceInfo(void)
{
	int gpuCount;	//Number of GPUs
	int runMode;	//0: GPU, 1: GPU
	cudaGetDeviceCount(&gpuCount);	//Get GPU information
	if (!gpuCount){
		runMode = 1; printf("Found no GPU, the program will run on GPU");
	}
	else {		
		runMode = 0; 
		cudaDeviceProp gpuProp;	//Device properties
		printf("1.GPU info:\n");
		printf("	# GPUs: %d\n",gpuCount);
		cudaGetDeviceProperties(&gpuProp,0);//Get the properties of the devices
		printf("	# Multiprocessors: %d\n",gpuProp.multiProcessorCount);
		printf("	Total amount of global memory: %u [GB]\n",gpuProp.totalGlobalMem>>30);
		printf("	Total amount of const memory: %d [Kbytes]\n",gpuProp.totalConstMem>>10);
		printf("	Total amount of shared memory per block: %d [bytes]\n",gpuProp.sharedMemPerBlock);
		printf("	# register per block: %d\n",gpuProp.regsPerBlock);
		printf("	Warp size: %d\n",gpuProp.warpSize);
		printf("	Max num of threads per block: %d\n",gpuProp.maxThreadsPerBlock);
		printf("	Max num of threads per SM: %d\n",gpuProp.maxThreadsPerMultiProcessor);
		printf("	Max dimensions per block: [%d, %d, %d]\n",gpuProp.maxThreadsDim[0],gpuProp.maxThreadsDim[1],gpuProp.maxThreadsDim[2]);
		printf("	Max dimensions per grid: [%u, %u, %u]\n",gpuProp.maxGridSize[0],gpuProp.maxGridSize[1],gpuProp.maxGridSize[2]);
		printf("	Clock rate: %.2f GHz\n",gpuProp.clockRate*1e-6f);
		printf("	\n");
	}
	return runMode;
}

Mat readResizeInputImage(char *impath, char *imfilename)
{
	char *fullname = strncat(impath,imfilename,20);
	printf("	Loading %s ",fullname);
	Mat inIm = cv::imread(fullname,CV_LOAD_IMAGE_GRAYSCALE);
	Mat resizedIm;
	resizedIm.data = NULL;
	if (!inIm.data)
	{
		printf("Cannot read the image \n");
	}
	else
	{
		printf("...successful\n");
		cv::resize(inIm,resizedIm,cv::Size(DIM_IM_Y,DIM_IM_X),0,0,CV_INTER_LINEAR); //Resize the image to match the new size
		resizedIm.convertTo(resizedIm,CV_32F); //Convert into 32-bit floating point image before doing filtering
	}
	return resizedIm;
}

float *readResizeInputImageGPU(Mat mat_h)
{
	float* img_h = (float*)mat_h.data;
	float* img_d;
	cudaMalloc((void**)&img_d, mat_h.rows*mat_h.cols*sizeof(float));
	cudaMemcpy(img_d, img_h, mat_h.rows*mat_h.cols*sizeof(float), cudaMemcpyHostToDevice);
	return img_d;
}

void imagesc(string title,Mat dispIm)
{
	namedWindow(title,WINDOW_NORMAL);
	double minval,maxval;
	minMaxLoc(dispIm,&minval,&maxval);
	dispIm.convertTo(dispIm,CV_32FC1,1.0/(maxval-minval), -(minval)/(maxval-minval));//Normalize to the interval [0,1]
	imshow(title,dispIm);
	waitKey(10);
}

bool checkEqual(float *image_h, float *image_d, int size)
{
	float *image_d2h = (float*)malloc(size*sizeof(float));
	cudaMemcpy(image_d2h, image_d, size*sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; i++)
	{
		float diff = image_h[i] - image_d2h[i];
		if (diff > FLT_EPSILON || -diff > FLT_EPSILON)
		{
			printf("False at %d, host: %f, device: %f, dif: %f\n", i, image_h[i], image_d2h[i], diff);
			free(image_d2h);
			return false;
		}
	}
	free(image_d2h);
	return true;
}


void readDataFile(char *datafilepath, int *label_arr, double *data, long nsamples, int nfeatures)
{
	FILE *file2read;
	char curline[MAX_READ_LENGTH];
	file2read = fopen(datafilepath,"rt"); //Read the input file
	int lineidx = 0;
	int label_val; //Label of each data
	float *array_val=(float*)calloc(nfeatures,sizeof(float)); //This is the score of each data sample
	printf("\nReading traing data from textfile...");
	long * backspaceloc = (long*)calloc(nfeatures,sizeof(long));
	long searchingloc, backspacecount;
	while (fgets(curline,MAX_READ_LENGTH,file2read)!=NULL)
	{
		searchingloc=0;
		backspacecount=0;
		//Search for all location of the backspace characters
		while (searchingloc<MAX_READ_LENGTH)
		{
			char curchar = curline[searchingloc];
			if (curchar==' ')
			{
				backspaceloc[backspacecount++]=searchingloc;
			}
			searchingloc++;
		}
		char extractedstr[30];
		//First, copy the label
		for (int charidx =0;charidx<backspaceloc[0];charidx++)
		{
			extractedstr[charidx]=curline[charidx];
		}
		extractedstr[backspaceloc[0]]='\0';
		label_val = atoi(extractedstr);
		label_arr[lineidx]=label_val;
		
		//Next, copy the feature
		for (int featureidx=0;featureidx<nfeatures;featureidx++)
		{
			int firstidx = backspaceloc[featureidx]+1;
			int lastidx = ((backspaceloc[featureidx+1]-1)<strlen(curline)-1)?(backspaceloc[featureidx+1]-1):strlen(curline)-1;
			int substringlen = lastidx-firstidx+1;
			for (int charidx = 0;charidx<substringlen;charidx++)
				extractedstr[charidx]=curline[charidx+firstidx];
			extractedstr[substringlen]='\0';
			array_val[featureidx]=atof(extractedstr);	
			//printf("\n: A[%d] =  %1.4f",featureidx,array_val[featureidx]);
		}
		
		for (int featureidx = 0;featureidx<nfeatures;featureidx++)
			data[nfeatures*lineidx+featureidx]=(double)array_val[featureidx];
		
		lineidx++;
	}
	printf("\nDone!...");
}

void prepareDataForTrainValandTest(double *inputdata, int* input_label, int nfeatures, long nsamples, double *&traindata, int *&train_label, double *&valdata, int *&val_label,
	double *&testdata, int *&test_label, long &ntrain, long &nval, long &ntest)
{
	long traincount=0,valcount=0,testcount=0;
	bool trainfull=FALSE,valfull=FALSE,testfull=FALSE;
	//Prepare the data for training from an input dataset
	ntrain = floor((double)nsamples/2);
	nval = floor((double)nsamples/4);
	ntest = floor((double)nsamples/4);
	traindata = (double*) calloc(ntrain*nfeatures,sizeof(double));
	valdata = (double*)calloc(nval*nfeatures,sizeof(double));
	testdata = (double*)calloc(ntest*nfeatures,sizeof(double));
	//Go through the data and randomly put it into these 4 different dataset
	train_label = (int*)calloc(ntrain,sizeof(int));
	val_label = (int*)calloc(nval,sizeof(int));
	test_label = (int*)calloc(ntest,sizeof(int));
	
	for (long sampleidx=0;sampleidx<nsamples;sampleidx++)
	{
		//printf("\n %ld",sampleidx);
		bool newsampleassigned = FALSE;
		while ((!newsampleassigned))
			{
				float rannum =(float)rand()/100.0;
				if (rannum<0.5)
				{
					if (traincount<ntrain)
					{
						for (int featureidx=0;featureidx<nfeatures;featureidx++)
						{
							traindata[traincount*nfeatures+featureidx] = inputdata[sampleidx*nfeatures+featureidx];
							train_label[traincount] = input_label[sampleidx];
						}
						traincount++;
						newsampleassigned = TRUE;
						//printf("\nTraining %ld",traincount);
					}
					else
					{
						trainfull=TRUE;
					}
				}
				else if ((rannum>=0.5)&(rannum<0.75))
				{
					if (valcount<nval)
					{
						for (int featureidx=0;featureidx<nfeatures;featureidx++)
						{
							valdata[valcount*nfeatures+featureidx] = inputdata[sampleidx*nfeatures+featureidx];
							val_label[valcount] = input_label[sampleidx];
						}
						valcount++;
						newsampleassigned = TRUE;
						//printf("\nEvaluation %ld",valcount);
					}
					else
					{
						valfull=TRUE;
					}
				}
				else
				{
					if (testcount<ntest)
					{
						for (int featureidx=0;featureidx<nfeatures;featureidx++)
						{
							testdata[testcount*nfeatures+featureidx] = inputdata[sampleidx*nfeatures+featureidx];
							test_label[testcount] = input_label[sampleidx];
						}
						testcount++;
						newsampleassigned = TRUE;
						//printf("\nTesting %ld",testcount);
					}
					else
					{
						testfull = TRUE;
					}
				}
				if ((trainfull==TRUE)&(testfull==TRUE)&(valfull==TRUE))
					break;
			}
		
	}
	printf("\n Sample assigned. Train: %ld, Evaluation: %ld, Testing: %ld",traincount,valcount,testcount);
}


void prepareDataForRandomForest(double *inputdata, int*input_label, int nfeatures, long nsamples, double *&traindata, int *&train_label, double *&valdata, int *&val_label,
	long &ntrain, long &nval)
{
	//Prepare the data for random forest, just divide the data into training and testing. Train :75%, testing: 25%

	long traincount=0,valcount=0;
	bool trainfull=FALSE,valfull=FALSE;
	//Prepare the data for training from an input dataset
	ntrain = floor((double)0.75*nsamples);
	nval = floor((double)0.25*nsamples);
	traindata = (double*) calloc(ntrain*nfeatures,sizeof(double));
	valdata = (double*)calloc(nval*nfeatures,sizeof(double));
	//Go through the data and randomly put it into these 4 different dataset
	train_label = (int*)calloc(ntrain,sizeof(int));
	val_label = (int*)calloc(nval,sizeof(int));

	for (long sampleidx=0;sampleidx<nsamples;sampleidx++)
	{
		//printf("\n %ld",sampleidx);
		bool newsampleassigned = FALSE;
		while ((!newsampleassigned))
			{
				float rannum =(float)(rand()%100)/100.0;
				if (rannum<0.75)
				{
					if (traincount<ntrain)
					{
						for (int featureidx=0;featureidx<nfeatures;featureidx++)
						{
							traindata[traincount*nfeatures+featureidx] = inputdata[sampleidx*nfeatures+featureidx];
						}
						train_label[traincount] = input_label[sampleidx];
						traincount++;
						newsampleassigned = TRUE;
						//printf("\nTraining %ld",traincount);
					}
					else
					{
						trainfull=TRUE;
					}
				}
				else 
				{
					if (valcount<nval)
					{
						for (int featureidx=0;featureidx<nfeatures;featureidx++)
						{
							valdata[valcount*nfeatures+featureidx] = inputdata[sampleidx*nfeatures+featureidx];
						}
						val_label[valcount] = input_label[sampleidx];
						//printf("%d ", input_label[sampleidx]);
						valcount++;
						newsampleassigned = TRUE;
						//printf("\nEvaluation %ld",valcount);
					}
					else
					{
						valfull=TRUE;
					}
				}
				if ((trainfull==TRUE)&(valfull==TRUE))
					break;
			}
		
	}
	printf("\n Sample assigned. Train: %ld, Evaluation: %ld" ,traincount,valcount);
	
}

void confusionMatrix(int *gtlabel, int *elabel, float *&cf_mat, int nlabels, long nsamples)
{

	
		//Compute the confusion matrix. The confusion matrix is store as a row-based vector in cf_mat. Prediction (e) is for each column. Actual (gt) is for each row
		//Inputs: gtlabel, elabel: label matrix
		printf("\nConfusion matrix:..\n");
		cf_mat = (float*)calloc(nlabels*nlabels,sizeof(float));
		float *rowsum = (float*) calloc(nlabels,sizeof(float));
		
		for (long sampleidx=0;sampleidx<nsamples;sampleidx++)
		{
			cf_mat[gtlabel[sampleidx]*nlabels+elabel[sampleidx]]++;
			rowsum[gtlabel[sampleidx]]++;
		}
	
		//Now, normalize with respect to the ground truth (l1-normalization over each row to unit norm)
		for (int labelidx=0;labelidx<nlabels;labelidx++)
		{
			printf("Class: %d [%1.0f]: ",labelidx,rowsum[labelidx]);
			for (int elabelidx=0;elabelidx<nlabels;elabelidx++)
			{
				cf_mat[labelidx*nlabels+elabelidx]/=(float)(rowsum[labelidx]+1e-3);
				printf("%1.2f ",cf_mat[labelidx*nlabels+elabelidx]);
			}
			printf("\n");
		}
		free(cf_mat);
}

bool computeRegionDescriptor(unsigned char* textonIdxIm_mdf, float *&hot)
{
	//Compute the region descriptor given the texton index in textonIdxIm_mdf, output the results in to a hot matrix of dimension IM_X*IM_Y*NCLUSTER
	hot = (float*) calloc(DIM_IM_Y*DIM_IM_X*CLUSTER_COUNT,sizeof(float));
	if (hot==NULL)
	{
		printf("Can't allocate enough memory for the histogram\n");
		return FALSE;
	}
	char *indexim = (char*)calloc(DIM_IM_Y*DIM_IM_X*CLUSTER_COUNT,sizeof(unsigned char));
	char *curChunk = (char*)calloc(DIM_IM_Y*DIM_IM_X,sizeof(unsigned char));
	
	//First, compute the histogram of texton for each image using the integral image
	for (long pixIdx=0; pixIdx<DIM_IM_Y*DIM_IM_X; pixIdx++)
	{
		char curIdx = textonIdxIm_mdf[pixIdx];
		indexim[curIdx*DIM_IM_Y*DIM_IM_X+pixIdx]++;
	}
	free(textonIdxIm_mdf);

	int radius = 40;//Pixels
	float x1,x2,x3,x4;
	
	int *tl_y,*tl_x,*bl_x,*bl_y,*tr_x,*tr_y,*br_x,*br_y;
	tl_y = (int*)calloc(DIM_IM_Y*DIM_IM_X,sizeof(int));
	tl_x = (int*)calloc(DIM_IM_Y*DIM_IM_X,sizeof(int));
	bl_x = (int*)calloc(DIM_IM_Y*DIM_IM_X,sizeof(int));
	bl_y = (int*)calloc(DIM_IM_Y*DIM_IM_X,sizeof(int));
	tr_x = (int*)calloc(DIM_IM_Y*DIM_IM_X,sizeof(int));
	tr_y = (int*)calloc(DIM_IM_Y*DIM_IM_X,sizeof(int));
	br_x = (int*)calloc(DIM_IM_Y*DIM_IM_X,sizeof(int));
	br_y = (int*)calloc(DIM_IM_Y*DIM_IM_X,sizeof(int));

	//Pre-compute the pixel index for accessing integral image
	for (unsigned int y_coord=0; y_coord<DIM_IM_Y;y_coord++)
	{
		for (unsigned int x_coord=0;x_coord<DIM_IM_X;x_coord++)
		{
			int curcoord = y_coord*DIM_IM_X+x_coord; 
			tl_x[curcoord] = x_coord - radius; tl_y[curcoord] = y_coord - radius;
	     	bl_x[curcoord] = x_coord - radius; bl_y[curcoord] = y_coord + radius;
			tr_x[curcoord] = x_coord + radius; tr_y[curcoord] = y_coord - radius;
		    br_x[curcoord] = x_coord + radius; br_y[curcoord] = y_coord + radius;
				
			tl_x[curcoord] = (tl_x[curcoord]>=0)?tl_x[curcoord]:0; 
			tl_y[curcoord] = (tl_y[curcoord]>=0)?tl_y[curcoord]:0;
			bl_x[curcoord] = (bl_x[curcoord]>=0)?bl_x[curcoord]:0;
			bl_y[curcoord] = (bl_y[curcoord]<DIM_IM_Y)?bl_y[curcoord]:(DIM_IM_Y-1);
			tr_x[curcoord] = (tr_x[curcoord]<DIM_IM_X)?tr_x[curcoord]:(DIM_IM_X-1);
			tr_y[curcoord] = (tr_y[curcoord]>=0)?tr_y[curcoord]:0;
			br_x[curcoord] = (br_x[curcoord]<DIM_IM_X)?br_x[curcoord]:(DIM_IM_X-1);
			br_y[curcoord] = (br_y[curcoord]<DIM_IM_Y)?br_y[curcoord]:(DIM_IM_Y-1);				
		}
	}
	cv::Mat curIntIm; //Current integral image
	float *prow1,*prow2;
	for (int textonIdx = 0;textonIdx<CLUSTER_COUNT;textonIdx++)
	{
		//Find all the pixel corresponding to a specific texton type
		memcpy((void*)curChunk,(void*)&indexim[textonIdx*DIM_IM_Y*DIM_IM_X],sizeof(unsigned char)*DIM_IM_Y*DIM_IM_X);
		cv::Mat curIdxIm = cv::Mat(DIM_IM_Y,DIM_IM_X,CV_8UC1,(void*)curChunk,DIM_IM_X*sizeof(unsigned char));
		cv::integral(curIdxIm,curIntIm,CV_32F);
		//Go through each pixel and compute the histogram
		for (unsigned int y_coord=0; y_coord<DIM_IM_Y;y_coord++)
		{
			prow1 = curIntIm.ptr<float>(tl_y[y_coord*DIM_IM_X]); //Pointer is faster to access the row instead of using the at operator
			prow2 = curIntIm.ptr<float>(bl_y[y_coord*DIM_IM_X]);
			for (unsigned int x_coord=0;x_coord<DIM_IM_X;x_coord++)
			{
				int curcoord = y_coord*DIM_IM_X+x_coord; 			
				x1 = prow1[tl_x[curcoord]];
				x2 = prow2[br_x[curcoord]];
				x3 = prow1[tr_x[curcoord]];
				x4 = prow2[bl_x[curcoord]];
				hot[textonIdx*DIM_IM_Y*DIM_IM_X+y_coord*DIM_IM_X+x_coord] = (x1 + x2 - x3 - x4);
			}
		}
		/*cv::Mat curhistIm = cv::Mat(DIM_IM_Y,DIM_IM_X,CV_32FC1,(void*)&hot[textonIdx*DIM_IM_Y*DIM_IM_X],DIM_IM_X*sizeof(float));
		imagesc("HistIm",curhistIm);*/		
	}
	free(tl_x);
	free(tl_y);
	free(bl_x);
	free(bl_y);
	free(tr_x);
	free(tr_y);
	free(br_x);
	free(br_y);
	return TRUE;
}

