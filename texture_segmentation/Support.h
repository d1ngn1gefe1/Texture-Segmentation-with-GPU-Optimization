#ifndef SUPPORT_H
#define SUPPORT_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h> //for FLT_EPSILON
#include <time.h>
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv\cxcore.h>
#include <string.h>
#include <iostream>
#include <math.h>


#define NONSYMSCALES		5				//Number of non symmetry filters
#define SYMSCALES			5				//Number of symmetry filters
#define FILTER_KERNEL_SIZE	49				//This is the size of the filtering kernel
#define DIRECTION_NUM		4				//Number of scales for the 
#define PI					3.14159265
#define DIM_IM_X			2048			//Dimension of the image in the x-dimension
#define DIM_IM_Y			2048			//Dimension of the image in the y-dimension
#define SAMPLE_PER_IM		65536			//Number of samples per image that is used for k-means
#define CLUSTER_COUNT		51				//Number of cluster
#define K_MEANS_MIN_EPS		1e-3			//Minimum value of the shifted center to soft update the k-means	
#define BLOCK_SIZE_FILTER	512				//Block size of cuda kernel
#define MAX_READ_LENGTH		500	//Maximum length for reading a line in the data

/*
#define DebugNormalization 0
#define DebugGenerateGaussianKernel 0
#define DebugLaplacianOfGaussianKernel 0
#define Debug1stOrderDerivDirectionalGaussian 0
#define Debug2ndOrderDerivDirectionalGaussian 0
*/

using namespace cv;
using namespace gpu;

#define cufftSafeCall( err ) __cufftSafeCall(err, __FILE__, __LINE__)
inline void __cufftSafeCall(cufftResult err, const char *file, const int line)
{
	if ( CUFFT_SUCCESS != err )
	{
		printf("cufftSafeCall() failed at\n %s \n line: %i\n error: %d", file, line, err);
		if (err == CUFFT_INVALID_PLAN) printf("CUFFT was passed an invalid plan handle\n");
		if (err == CUFFT_ALLOC_FAILED) printf("CUFFT failed to allocate GPU or CPU memory\n");
		if (err == CUFFT_INVALID_TYPE) printf("No longer used\n");
		if (err == CUFFT_INVALID_VALUE) printf("User specified an invalid pointer or parameter\n");
		if (err == CUFFT_INTERNAL_ERROR) printf("Used for all driver and internal CUFFT library errors\n");
		if (err == CUFFT_EXEC_FAILED) printf("CUFFT failed to execute an FFT on the GPU\n");
		if (err == CUFFT_SETUP_FAILED) printf("The CUFFT library failed to initialize\n");
		if (err == CUFFT_INVALID_SIZE) printf("User specified an invalid transform size\n");
		if (err == CUFFT_UNALIGNED_DATA) printf("No longer used\n");
		while(1);
		exit(20);
	}
	return;
}

//This struct defines a struct for measuring the elapse time 
typedef struct {
	clock_t startTime;
	clock_t endTime;
} Timer;

//Time measuring functions
void startTime(Timer* ptimer);
void stopTime(Timer* ptimer);
float elapsedTime(Timer timer); //Return elapsed time in miliseconds

//Get CPU/GPU parameters
int getDeviceInfo(void); 

//Read input
Mat readResizeInputImage(char *impath, char *imfilename);
float *readResizeInputImageGPU(Mat mat_h);

//Scale the image and display it
void imagesc(string title,Mat dispIm);

//Compare host image with device image
bool checkEqual(float *image_h, float *image_d, int size);

void readDataFile(char *datafilepath, int *label_arr, double *data, long nsamples, int nfeatures); //Read the data file for the random forest training
void prepareDataForTrainValandTest(double *inputdata, int*input_label, int nfeatures, long nsamples, double *&traindata, int *&train_label, double *&valdata, int *&val_label,
	double *&testdata, int *&test_label, long &ntrain, long &nval, long &ntest);
void prepareDataForRandomForest(double *inputdata, int*input_label, int nfeatures, long nsamples, double *&traindata, int *&train_label, double *&valdata, int *&val_label,
	long &ntrain, long &nval);
//Prepare the data for training from an input dataset
void confusionMatrix(int *gtlabel, int *elabel, float *&cf_mat, int nlabels, long nsamples);//Compute the confusion matrix

bool computeRegionDescriptor(unsigned char* textonIdxIm_mdf, float *&hot);
#endif //SUPPORT_H