#ifndef FILTERRESPONSE_CUH

#include "FilterKernel.cuh"

class cv2cuda
{
public:
	static void filter2D(const float *src, float *dst, const float *kernel, float *expandedKernel_d,
		cufftHandle *plan1, cufftHandle *plan2, 
		cufftComplex *src_c, cufftComplex *expandedKernel_d_c); //image: 2048*2048, kernel: 49*49
	static void multiply(const float *src1, const float *src2, float *dst, int size);
	static void add(const float *src1, const float *src2, float *dst, int size);
private:
	static void padding(float *expandedKernel_d, const float *kernel_d, int filterKernelSize); //helper function of filter2D
};

class cpucode
{
public:
	static void filter2D(const float *src, float *dst, const float *kernel);
};

class FilterResponse
{
public:
	unsigned int x_dim,y_dim, ndirs,nfsym,nfnonsym;	//Dimension of the image
	char fileName[20];
	char filePath[50];
	float * maxfres; //Maximum filter response all over the scale and directions
	unsigned char *maxdiridx; //Indices of maximum response over the scales
	float * maxfres_d; //Maximum filter response all over the scale and directions
	unsigned char *maxdiridx_d; //Indices of maximum response over the scales
	bool gpuCompute;

	FilterResponse(const char * _filePath,const char *_fileName, FilterKernel **h_arr, 
		unsigned int _x_dim, unsigned int _y_dim, unsigned int _ndirs,
		unsigned int _nfsym, unsigned int _nfnonsym, bool _gpuCompute):
	x_dim(_x_dim),y_dim(_y_dim),ndirs(_ndirs),nfsym(_nfsym),nfnonsym(_nfnonsym),gpuCompute(_gpuCompute)
	{
		//Generate data responses
		strcpy(filePath,_filePath);
		strcpy(fileName,_fileName);

		cv::Mat inIm; //Temporary OpenCV Mat structure for image storage
		float *inIm_d; //Temporary array for image storage
		Timer curTimer;
		switch (gpuCompute)
		{
		case false: //CPU computation
			{
				//Compute the anisotropic filter response
				startTime(&curTimer);
				printf("Scale: ");
				maxfres=(float*)calloc(x_dim*y_dim*(2*nfsym+nfnonsym),sizeof(float));//Data for the maximum response data. 
				if (!maxfres) printf("\n[Error] Do not have enough memory for maximum filter response\n");
				maxdiridx = (unsigned char*)calloc(x_dim*y_dim*nfnonsym,sizeof(unsigned char));
				if (!maxdiridx) printf("\n[Error]Cannot allocate memory for storing the max index images\n");
				//Measuring the loaded time
				startTime(&curTimer);
				inIm = readResizeInputImage(filePath,fileName);
				stopTime(&curTimer);
				printf("[Image reading to CPU]	Elapsed time: %f ms\n",elapsedTime(curTimer));
				for (int nonsymscaleidx = 0;nonsymscaleidx<NONSYMSCALES;nonsymscaleidx++)
				{
					printf("%d..",nonsymscaleidx);
					float *maxRes = (float*)calloc(DIM_IM_X*DIM_IM_Y,sizeof(float));
					unsigned char *maxidx = (unsigned char *)calloc(DIM_IM_X*DIM_IM_Y,sizeof(unsigned char));
					//Now, compare	
					for (int diridx=0;diridx<DIRECTION_NUM;diridx++)
					{
						cv::Mat OddRes,EvenRes,tempOut;
						cv::Mat Oddkernel = cv::Mat(FILTER_KERNEL_SIZE,FILTER_KERNEL_SIZE,CV_32FC1,(void*)h_arr[nonsymscaleidx*DIRECTION_NUM+diridx]->pdata_h,sizeof(float)*FILTER_KERNEL_SIZE);
						cv::Mat Evenkernel = cv::Mat(FILTER_KERNEL_SIZE,FILTER_KERNEL_SIZE,CV_32FC1,(void*)h_arr[nonsymscaleidx*DIRECTION_NUM+diridx+NONSYMSCALES*DIRECTION_NUM]->pdata_h,
							sizeof(float)*FILTER_KERNEL_SIZE);
						cv::filter2D(inIm,OddRes,-1,Oddkernel);
						cv::filter2D(inIm,EvenRes,-1,Evenkernel);
						cv::multiply(OddRes,OddRes,OddRes);	//Square each image and sum up
						cv::multiply(EvenRes,EvenRes,EvenRes);
						cv::add(OddRes,EvenRes,tempOut);	
						tempOut.convertTo(tempOut,CV_32FC1);
						for (unsigned int x_coord=0; x_coord<DIM_IM_X;x_coord++)
						{
							for (unsigned int y_coord=0;y_coord<DIM_IM_Y;y_coord++)
							{
								float curPixelVal = tempOut.at<float>(y_coord,x_coord);
								if (curPixelVal>maxRes[y_coord*DIM_IM_X+x_coord])
								{
									maxRes[y_coord*DIM_IM_X+x_coord]= curPixelVal;
									maxidx[y_coord*DIM_IM_X+x_coord] = diridx;
								}
							}
						}
					}

					//Now, copy the maximum response data into the filter response array
					for (unsigned int x_coord = 0;x_coord<DIM_IM_X;x_coord++)
					{
						for (unsigned int y_coord = 0;y_coord<DIM_IM_Y;y_coord++)
						{
							maxfres[y_coord*DIM_IM_X+x_coord+nonsymscaleidx*DIM_IM_X*DIM_IM_Y]=maxRes[y_coord*DIM_IM_X+x_coord];
							maxdiridx[y_coord*DIM_IM_X+x_coord+nonsymscaleidx*DIM_IM_X*DIM_IM_Y]=maxidx[y_coord*DIM_IM_X+x_coord];
						}
					}
#if WRITEOUTPUT
					char name1[30];
					sprintf(name1, "Disp1_%d.tif", nonsymscaleidx);
					writeFromCPU(maxRes, name1, DIM_IM_X, DIM_IM_Y);
#endif
#if DISPLAYOUTPUT
					//This section of the code is for visualization the filter response...
					cv::Mat maxResIm, maxIdxIm;
					maxResIm = cv::Mat(DIM_IM_Y,DIM_IM_X,CV_32FC1,(void*)maxRes,DIM_IM_X*sizeof(float));
					maxIdxIm = cv::Mat(DIM_IM_Y,DIM_IM_X,CV_8UC1,(void*)maxidx,DIM_IM_X*sizeof(unsigned char));
					imagesc("Disp",maxResIm);
					imagesc("Index",maxIdxIm);
#endif
					free(maxRes);
					free(maxidx);
				}
				stopTime(&curTimer);
				printf("\n[Anisotropic]	Elapsed time: %f ms\n",elapsedTime(curTimer));

				startTime(&curTimer);
				printf("Scale: ");
				//Continue to the isotropic part
				for (unsigned int symscaleidx=0;symscaleidx<SYMSCALES;symscaleidx++)
				{
					printf("%d..",symscaleidx);
					//Just compute and save the filter response without taking maximum over the direction
					cv::Mat tempOut;
					//Go with the the Gaussian and LOG separately
					for (unsigned char extraidx = 0;extraidx<2;extraidx++)
					{
						cv::Mat Filterkernel = cv::Mat(FILTER_KERNEL_SIZE,FILTER_KERNEL_SIZE,CV_32FC1,(void*)h_arr[2*NONSYMSCALES*DIRECTION_NUM+2*symscaleidx+extraidx]->pdata_h,
							sizeof(float)*FILTER_KERNEL_SIZE);
						cv::filter2D(inIm,tempOut,-1,Filterkernel);
						tempOut.convertTo(tempOut,CV_32FC1);//Make sure the output array is 32 bit floating point format
						//Copy to the data for storing...
						for (unsigned int x_coord = 0;x_coord<DIM_IM_X;x_coord++)
						{
							for (unsigned int y_coord = 0;y_coord<DIM_IM_Y;y_coord++)
							{
								maxfres[y_coord*DIM_IM_X+x_coord+(2*symscaleidx+NONSYMSCALES+extraidx)*DIM_IM_X*DIM_IM_Y]=tempOut.at<float>(y_coord,x_coord);
								char name1[30];
							}
						}
#if WRITEOUTPUT
						char name2[30];
						sprintf(name2, "Disp2_%d_%d.tif", symscaleidx, extraidx);
						writeFromCPU((float*)tempOut.data, name2, DIM_IM_X, DIM_IM_Y);
#endif
#if DISPLAYOUTPUT
						imagesc("Disp",tempOut);
						#endif
					}
				}
				stopTime(&curTimer);
				printf("\n[Isotropic filtering]	Elapsed time: %f ms\n",elapsedTime(curTimer));
			}
			break;
		case true: //GPU computing
			{
				//Memory allocations
				startTime(&curTimer);
				cudaMalloc((void**)&maxfres_d, x_dim*y_dim*(2*nfsym+nfnonsym)*sizeof(float)); //Data for the maximum response data. 
				if (!maxfres_d) printf("\n[Error] Do not have enough memory for maximum filter response\n");
				cudaMalloc((void**)&maxdiridx_d, x_dim*y_dim*nfnonsym*sizeof(unsigned char));
				if (!maxdiridx_d) printf("\n[Error]Cannot allocate memory for storing the max index images\n");
				float *OddRes_d, *EvenRes_d, *tempOut_d;
				cudaMalloc((void**)&OddRes_d, DIM_IM_X*DIM_IM_Y*sizeof(float));
				cudaMalloc((void**)&EvenRes_d, DIM_IM_X*DIM_IM_Y*sizeof(float));
				cudaMalloc((void**)&tempOut_d, DIM_IM_X*DIM_IM_Y*sizeof(float));
				float *maxRes_d;
				cudaMalloc((void**)&maxRes_d, DIM_IM_X*DIM_IM_Y*sizeof(float)); //2048*2048
				unsigned char *maxidx_d;
				cudaMalloc((void**)&maxidx_d, DIM_IM_X*DIM_IM_Y*sizeof(unsigned char)); //2048*2048
				float *expandedKernel_d;
				cudaMalloc((void**)&expandedKernel_d, DIM_IM_X*DIM_IM_Y*sizeof(float));
				stopTime(&curTimer);
				printf("[Memory allocation]	Elapsed time: %f ms\n",elapsedTime(curTimer));
				// CUFFT
				startTime(&curTimer);
				cufftHandle plan1;
				cufftSafeCall(cufftPlan2d(&plan1, DIM_IM_Y, DIM_IM_X, CUFFT_R2C));
				cufftHandle plan2;
				cufftSafeCall(cufftPlan2d(&plan2, DIM_IM_Y, DIM_IM_X, CUFFT_C2R));
				cufftComplex *src_c;
				cufftComplex *expandedKernel_d_c;
				cudaMalloc((void**)&src_c, DIM_IM_X*DIM_IM_Y*sizeof(cufftComplex));
				cudaMalloc((void**)&expandedKernel_d_c, DIM_IM_X*DIM_IM_Y*sizeof(cufftComplex));
				stopTime(&curTimer);
				printf("[CUFFT Plans and Buffers]	Elapsed time: %f ms\n",elapsedTime(curTimer));
				//Measuring the loaded time
				startTime(&curTimer);
				inIm = readResizeInputImage(filePath,fileName);
				stopTime(&curTimer);
				printf("[Image reading to CPU]	Elapsed time: %f ms\n",elapsedTime(curTimer));
				startTime(&curTimer);
				inIm_d = readResizeInputImageGPU(inIm);
				stopTime(&curTimer);
				printf("[Image reading from CPU to GPU]	Elapsed time: %f ms\n",elapsedTime(curTimer));
				//Compute the anisotropic filter response
				startTime(&curTimer);
				printf("Scale: ");
				for (int nonsymscaleidx = 0; nonsymscaleidx < NONSYMSCALES; nonsymscaleidx++)
				{
					printf("%d..",nonsymscaleidx);
					cudaMemset(maxRes_d, 0, DIM_IM_X*DIM_IM_Y*sizeof(float));
					cudaMemset(maxidx_d, 0, DIM_IM_X*DIM_IM_Y*sizeof(unsigned char));
					//Now, compare	
					for (unsigned char diridx = 0; diridx < DIRECTION_NUM; diridx++)
					{
						float *Oddkernel_d = h_arr[nonsymscaleidx*DIRECTION_NUM+diridx]->pdata_d; //49*49
						float *Evenkernel_d = h_arr[nonsymscaleidx*DIRECTION_NUM+diridx+NONSYMSCALES*DIRECTION_NUM]->pdata_d; //49*49
						cv2cuda::filter2D(inIm_d, OddRes_d, Oddkernel_d, expandedKernel_d, &plan1, &plan2, src_c, expandedKernel_d_c);
						cv2cuda::filter2D(inIm_d, EvenRes_d, Evenkernel_d, expandedKernel_d, &plan1, &plan2, src_c, expandedKernel_d_c);
						cv2cuda::multiply(OddRes_d, OddRes_d, OddRes_d, DIM_IM_X*DIM_IM_Y);	//Square each image and sum up
						cv2cuda::multiply(EvenRes_d, EvenRes_d, EvenRes_d, DIM_IM_X*DIM_IM_Y);
						cv2cuda::add(OddRes_d, EvenRes_d, tempOut_d, DIM_IM_X*DIM_IM_Y);
						helperFunc1(tempOut_d, maxRes_d, maxidx_d, diridx);
					}
					//Now, copy the maximum response data into the filter response array
					helperFunc2(maxfres_d, maxRes_d, maxdiridx_d, maxidx_d, nonsymscaleidx);
#if WRITEOUTPUT
					char name1[30];
					sprintf(name1, "Disp1d_%d.tif", nonsymscaleidx);
					writeFromGPU(maxRes_d, name1, DIM_IM_X, DIM_IM_Y);
#endif
#if DISPLAYOUTPUT
					//This section of the code is for visualization the filter response...
					float *maxRes = (float*)malloc(DIM_IM_X*DIM_IM_Y*sizeof(float));
					cudaMemcpy(maxRes, maxRes_d, DIM_IM_X*DIM_IM_Y*sizeof(float), cudaMemcpyDeviceToHost);
					float *maxidx = (float*)malloc(DIM_IM_X*DIM_IM_Y*sizeof(unsigned char));
					cudaMemcpy(maxidx, maxidx_d, DIM_IM_X*DIM_IM_Y*sizeof(unsigned char), cudaMemcpyDeviceToHost);
					cv::Mat maxResIm, maxIdxIm;
					maxResIm = cv::Mat(DIM_IM_Y,DIM_IM_X,CV_32FC1,(void*)maxRes,DIM_IM_X*sizeof(float));
					maxIdxIm = cv::Mat(DIM_IM_Y,DIM_IM_X,CV_8UC1,(void*)maxidx,DIM_IM_X*sizeof(unsigned char));
					imagesc("Disp",maxResIm);
					imagesc("Index",maxIdxIm);
#endif
				}
				stopTime(&curTimer);
				printf("\n[Anisotropic]	Elapsed time: %f ms\n",elapsedTime(curTimer));

				/////////////////////////////////

				startTime(&curTimer);
				printf("Scale: ");
				//Continue to the isotropic part
				for (int symscaleidx = 0; symscaleidx < SYMSCALES; symscaleidx++)
				{
					printf("%d..",symscaleidx);
					//Just compute and save the filter response without taking maximum over the direction
					//Go with the the Gaussian and LOG separately
					for (unsigned char extraidx = 0; extraidx < 2; extraidx++)
					{
						float *Filterkernel = h_arr[2*NONSYMSCALES*DIRECTION_NUM+2*symscaleidx+extraidx]->pdata_d;
						cv2cuda::filter2D(inIm_d, tempOut_d, Filterkernel, expandedKernel_d, &plan1, &plan2, src_c, expandedKernel_d_c);
						//Copy to the data for storing...
						helperFunc3(maxfres_d, tempOut_d, symscaleidx, extraidx);
#if WRITEOUTPUT
						char name2[30];
						sprintf(name2, "Disp2d_%d_%d.tif", symscaleidx, extraidx);
						writeFromGPU(tempOut_d, name2, DIM_IM_X, DIM_IM_Y);
#endif
#if DISPLAYOUTPUT
						float *tempOut = (float*)malloc(DIM_IM_X*DIM_IM_Y*sizeof(float));
						cudaMemcpy(tempOut, tempOut_d, DIM_IM_X*DIM_IM_Y*sizeof(float), cudaMemcpyDeviceToHost);
						Mat tempOutIm(DIM_IM_X, DIM_IM_Y, CV_32FC1, (void*)tempOut, DIM_IM_X*sizeof(float));
						imagesc("Disp", tempOutIm);
#endif
					}
				}
				stopTime(&curTimer);
				printf("\n[Isotropic filtering]	Elapsed time: %f ms\n",elapsedTime(curTimer));

				maxfres = (float*)malloc(DIM_IM_X*DIM_IM_Y*(2*nfsym+nfnonsym)*sizeof(float));
				maxdiridx = (unsigned char*)malloc(DIM_IM_X*DIM_IM_Y*nfnonsym*sizeof(unsigned char));
				cudaMemcpy(maxfres, maxfres_d, DIM_IM_X*DIM_IM_Y*(2*nfsym+nfnonsym)*sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(maxdiridx, maxdiridx_d, DIM_IM_X*DIM_IM_Y*nfnonsym*sizeof(unsigned char), cudaMemcpyDeviceToHost);

				startTime(&curTimer);
				cudaFree(OddRes_d);
				cudaFree(EvenRes_d);
				cudaFree(tempOut_d);
				cudaFree(expandedKernel_d);
				cudaFree(maxRes_d);
				cudaFree(maxidx_d);
				cufftDestroy(plan1);
				cufftDestroy(plan2);
				stopTime(&curTimer);
				printf("[GPU Memory Free]	Elapsed time: %f ms\n",elapsedTime(curTimer));
			}
			break;
		}
		saveFilterResponse(filePath, fileName);
	}

	void helperFunc1(float *tempOut_d, float *maxRes_d, unsigned char *maxidx_d, unsigned char diridx);
	void helperFunc2(float *maxfres_d, float *maxRes_d, unsigned char *maxdiridx_d, unsigned char *maxidx_d, int nonsymscaleidx);
	void helperFunc3(float *maxfres_d, float *tempOut_d, int symscaleidx, unsigned char extraidx);

	~FilterResponse()
	{
		free(maxfres);
		free(maxdiridx);
	}

	void saveFilterResponse(char * filePath, char *fileName)
	{
		//Save the filter response into *.txt file
		//Will be added by Tan later
	}
};

#endif //FILTERRESPONSE_CUH

