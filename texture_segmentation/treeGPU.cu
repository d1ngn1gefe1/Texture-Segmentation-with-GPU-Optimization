#include "treeGPU.h"
#include "cuda_runtime.h"
#include "stdio.h"
#include "tree.h"
#include "math.h"
__device__ TreeNodeGPU *nodeList; //An array storing all nodes information of all trees
__device__ long* nodedataIndex;		  //An array of temporal data index for each node
__device__ long ntrainsamples;	//Number of training samples

//Define following global variables for accessing in all __device__ and __global__ function. This step is very imporant since the __device__ function can't access the address passed by the __global__ kernel.
//We need to define a variable that store the address...
__device__ double *train_d;
__device__ int *label_d;
__device__ int *selected_feat;
__device__ long maxkeyval[TREE_NUM];	//Maximum values of the key for each tree

__device__ void preScan(TreeNodeGPU *p)
{
	//Do prescan sarch and print out the key value at each node
	if (p!=NULL)
	{
		printf("%d..",p->key);
		//Traverse to the left subtree
		if (p->leftChild!=NULL)
		{
			preScan(p->leftChild);
		}
		if (p->rightChild!=NULL)
		{
			preScan(p->rightChild);
		}
	}
	else
	{
		return;
	}
}

__device__ void inScan(TreeNodeGPU *p)
{
	if (p!=NULL)
	{
		if (p->leftChild!=NULL)
		{
			inScan(p->leftChild);
		}
		printf("%d..",p->key);
		if (p->rightChild!=NULL)
		{
			inScan(p->rightChild);
		}
	}
	else
		return;
}

__device__ void postScan(TreeNodeGPU *p)
{
	if (p!=NULL)
	{
		if (p->leftChild!=NULL)
		{
			postScan(p->leftChild);
		}
		if (p->rightChild!=NULL)
		{
			postScan(p->rightChild);
		}
		printf("%d..",p->key);
	}
	else
		return;
}

__device__ void treeLearnID3(double *data, int *label, int nlabels, int nfeatures, long nsamples)
{
	/*Train the classification tree.
	Inputs:
	data: a 1D array of dimension 1 x (nfeatures*nsamples) in which each  training vector is a row vector with dimension of nfeatures
	label: a 1D array of dimension 1 x nsamples with elements in range [0,1,..,nlabels-1] correspond to different classes.
	nlabels: number of labels
	nfeatures: number of features
	*/
	long curnodeAccessIdx;
	int count;		
	long sampleidx;
	long treeIdx = blockIdx.x*BLOCK_SIZE+threadIdx.x;
	if (treeIdx<TREE_NUM)
	{
		//	printf("\n[Error]: can't create new tree");
		maxkeyval[treeIdx] = 0;				  //Maximum value of the key for each tree
	
		//Generate the sample list for all the nodes...
		for (sampleidx =0; sampleidx<nsamples;sampleidx++)
			nodedataIndex[treeIdx*nsamples+sampleidx]=0;//All samples start at node 0 for all trees
		
		//Prepare the root node for each tree
		curnodeAccessIdx = ((long)treeIdx*MAX_NODE_COUNT_PER_TREE)+maxkeyval[treeIdx];
		
		nodeList[curnodeAccessIdx].key = 0;
		nodeList[curnodeAccessIdx].parent = NULL;
		nodeList[curnodeAccessIdx].leftChild = NULL;
		nodeList[curnodeAccessIdx].rightChild = NULL;
		nodeList[curnodeAccessIdx].data.varnum = -1;
		nodeList[curnodeAccessIdx].data.thresh = 0.0;
		nodeList[curnodeAccessIdx].data.level = 0;
		nodeList[curnodeAccessIdx].data.valnsamples = 0;
		nodeList[curnodeAccessIdx].data.nsamples = nsamples;
		splitNode(&nodeList[curnodeAccessIdx],data,label,nlabels,nfeatures,nodeList[curnodeAccessIdx].data.nsamples);	
	}
}

__device__ void splitNode(TreeNodeGPU *curNode, double*data,int *label, int nlables,int nfeatures, long nsamples)
{
		float besthresh;					//These variables are needed to update by all threads in blocks
		int vartosplit,nsleft,nsright;
		bool newNodeGen;
		float largestfraction;
		long curLeftNodeAccessIdx, curRightNodeAccessIdx;
		long leftNodeCount, rightNodeCount;
		
		//Go through every features and find the best split
		long treeIdx = blockIdx.x*BLOCK_SIZE +threadIdx.x;
		if (treeIdx<TREE_NUM)
		{
			findBestSplit(curNode,data,label,nlables,nfeatures,vartosplit,besthresh,nsleft,nsright,curNode->data.fraction_arr);
		 	//printf("\n TreeIdx: %ld, vartosplit: %d, thresh: %1.4f",treeIdx,vartosplit,besthresh);	
			newNodeGen = FALSE;
			largestfraction=0.0;
			for (int classidx=0;classidx<nlables;classidx++)
			{
				if (largestfraction<=(curNode->data.fraction_arr[classidx]))
				{
					largestfraction =curNode->data.fraction_arr[classidx];
					curNode->data.majorclassidx = classidx;
				}
			}
			
			if ((threadIdx.x==0)&(blockIdx.x==0))
			{
			/*
				if (curNode->key!=0) //If this is not the root node
					printf("\nT: %d, N.:%d, Par.: %d, Depth: %d, Var: %d, Thre.: %1.2f, Major class: %d ,#samples: %ld, f[0]: %1.5f, f[1]: %1.5f, split: %d/%d",treeIdx,curNode->key,
					curNode->parent->key, curNode->data.level, vartosplit,besthresh,curNode->data.majorclassidx,curNode->data.nsamples,curNode->data.fraction_arr[0],
					curNode->data.fraction_arr[1],nsleft,nsright);
				else
					printf("\nT: %d, N.: %d, Depth: %d, Var: %d, Thre.: %1.2f, Major class: %d ,#samples: %ld, f[0]: %1.5f, f[1]: %1.5f, split: %d/%d",treeIdx,curNode->key,
					curNode->data.level, vartosplit,besthresh,curNode->data.majorclassidx,curNode->data.nsamples,curNode->data.fraction_arr[0],
					curNode->data.fraction_arr[1],nsleft,nsright);
			*/
			}
			
			//Next, split the node if the condition is satisfied...
			if ((curNode->data.nsamples<MIN_NODE_SIZE)|(vartosplit==-1)|(curNode->data.level>=MAX_DEPTH)|(maxkeyval[treeIdx]>=MAX_NODE_COUNT_PER_TREE-2)) //Do not further divide if all the features have been used/ the number of instances at each node is too small/maximum depth reach
			{
				return;
			}
			else //Perform splitting
			{
				newNodeGen=TRUE; //New node is generated...
				curNode->data.varnum = vartosplit;
				curNode->data.thresh = besthresh;										
				curLeftNodeAccessIdx = treeIdx*MAX_NODE_COUNT_PER_TREE + (++maxkeyval[treeIdx]);
				nodeList[curLeftNodeAccessIdx].parent = curNode;
				nodeList[curLeftNodeAccessIdx].key = maxkeyval[treeIdx];
		
				nodeList[curLeftNodeAccessIdx].data.nsamples = nsleft;
				nodeList[curLeftNodeAccessIdx].data.varnum = -1;
				nodeList[curLeftNodeAccessIdx].data.thresh = 0.0;
				nodeList[curLeftNodeAccessIdx].data.level = curNode->data.level+1;
				nodeList[curLeftNodeAccessIdx].data.valnsamples= 0;
				nodeList[curLeftNodeAccessIdx].leftChild = NULL;//Make these node the leaf node
				nodeList[curLeftNodeAccessIdx].rightChild= NULL;
	
				curRightNodeAccessIdx = treeIdx*MAX_NODE_COUNT_PER_TREE + (++maxkeyval[treeIdx]);
				nodeList[curRightNodeAccessIdx].parent = curNode;
				nodeList[curRightNodeAccessIdx].key = maxkeyval[treeIdx];
				nodeList[curRightNodeAccessIdx].data.nsamples = nsright;
				nodeList[curRightNodeAccessIdx].data.varnum = -1;
				nodeList[curRightNodeAccessIdx].data.thresh = 0.0;
				nodeList[curRightNodeAccessIdx].data.level = curNode->data.level+1;
				nodeList[curRightNodeAccessIdx].data.valnsamples= 0;
				nodeList[curRightNodeAccessIdx].leftChild = NULL;
				nodeList[curRightNodeAccessIdx].rightChild = NULL;

				curNode->leftChild = &nodeList[curLeftNodeAccessIdx];
				curNode->rightChild =&nodeList[curRightNodeAccessIdx];
				//Update the node index for each training sample
				for (long sampleidx=0;sampleidx<ntrainsamples;sampleidx++)
				{
					if (nodedataIndex[treeIdx*ntrainsamples+sampleidx]==curNode->key)//Replace the key of the parent by thoses of the children
					{
						float curfeat = (float)data[sampleidx*nfeatures+curNode->data.varnum]; //Get current value of the feature
						if (curfeat<curNode->data.thresh)
							nodedataIndex[treeIdx*ntrainsamples+sampleidx]=nodeList[curLeftNodeAccessIdx].key;
						else
							nodedataIndex[treeIdx*ntrainsamples+sampleidx]=nodeList[curRightNodeAccessIdx].key;		
					}
				}
			}
			//Begin splitting the child node
			if (newNodeGen)
			{
				splitNode(curNode->leftChild,data,label,nlables,nfeatures,curNode->leftChild->data.nsamples);
				splitNode(curNode->rightChild,data,label,nlables,nfeatures,curNode->rightChild->data.nsamples);
			}
		}
}

__device__ void findBestSplit(TreeNodeGPU *curNode,double *data,int *label, int nlabels, int nfeatures, int &vartosplit, float &thresh, int &nspl1, int &nspl2, float *fraction_arr)
{
	//Find the best split at current node of the current tree. IF all the features have been used, return the varidx = -1, which means we do not split.
	vartosplit = -1;
	thresh = 0;
	nspl1 = 0;
	nspl2 = 0;
	__shared__ float minval[BLOCK_SIZE*SUP_FEAT_NUM];//Put to shared memory to save space
	__shared__ float maxval[BLOCK_SIZE*SUP_FEAT_NUM];
	
	__shared__ long ns1[BLOCK_SIZE*MAX_LABEL_NUM];
	__shared__ long ns2[BLOCK_SIZE*MAX_LABEL_NUM];
	__shared__ long ns[BLOCK_SIZE*MAX_LABEL_NUM];

	__shared__ long histbin[BLOCK_SIZE*MAX_LABEL_NUM*(THRESHOLD_NUM+1)];//Histogram for 1 features of differnt classes

	float curfeat;
	float curthresh;
	long ssum=0,s1sum=0,s2sum=0;		
	float es=0.0,es1=0.0,es2=0.0,ig=0.0; //Entropies and information gain
	float bestig;
	float curbestthresh = -1.0;
	int bestsplitvar=-1;
	long curNodeIndexOfData;
	long sampleidx;
	bool proceedwiththisfeat = FALSE; 
	int featidx;
	int thresholdidx;
	float curmin, curmax;
	int curlabel, curbin;
	float deltax;
	long bestnspl1=0;
	long bestnspl2=0;
	int labelidx;
	long treeIdx = blockIdx.x*BLOCK_SIZE+threadIdx.x;
	
	if (treeIdx<TREE_NUM)
	{
		vartosplit = -1;
		thresh = 0.0;
		nspl1 = 0;
		nspl2 = 0;
		//First, search for a list of data points that ends in the current node
		long *dataidxincurrentnode = (long*)malloc(sizeof(long)*curNode->data.nsamples);// You can't do this on the GPU!!!
		long datacount=0;
		for (sampleidx=0;sampleidx<ntrainsamples;sampleidx++)
		{				
				curNodeIndexOfData = nodedataIndex[treeIdx*ntrainsamples+sampleidx]; //Get the current Node index of a sample in the database
				if (curNodeIndexOfData==(curNode->key))
				{
					dataidxincurrentnode[datacount++]=sampleidx;//Add the index of the current node to the searching list
				}			
		}
		
		//Modified searching for the data
		ssum = curNode->data.nsamples;  //Get current number of samples
		for (int varidx=0;varidx<SUP_FEAT_NUM;varidx++)
		{
			featidx = selected_feat[treeIdx*SUP_FEAT_NUM + varidx];
			minval[threadIdx.x*SUP_FEAT_NUM+varidx]=1e+20;
			maxval[threadIdx.x*SUP_FEAT_NUM+varidx]=-1e+20;
			for (sampleidx=0;sampleidx<datacount;sampleidx++)
			{
				{
				 	curfeat = (float)data[dataidxincurrentnode[sampleidx]*nfeatures+featidx]; //Get current value of the feature
					if (curfeat<minval[threadIdx.x*SUP_FEAT_NUM+varidx])
						minval[threadIdx.x*SUP_FEAT_NUM + varidx] = curfeat;
					if (curfeat>maxval[threadIdx.x*SUP_FEAT_NUM+varidx])
						maxval[threadIdx.x*SUP_FEAT_NUM + varidx] = curfeat;
				}				
			}
		}
		
		bestig = -1e+20;
		
		
		//**Faster computational speed*** but there are some problems with the memory here
		for (int varidx=0;varidx<SUP_FEAT_NUM;varidx++)
		{
			
			featidx = selected_feat[treeIdx*SUP_FEAT_NUM + varidx];
			curmin = minval[threadIdx.x*SUP_FEAT_NUM + varidx];
			curmax = maxval[threadIdx.x*SUP_FEAT_NUM + varidx];
			deltax = (curmax-curmin)/(THRESHOLD_NUM-1);

			if (curmin==curmax)
				continue;
	
			if ((isFeatureUsedInAncesstor(&nodeList[treeIdx*MAX_NODE_COUNT_PER_TREE],featidx)==FALSE)|(FEAT_MULTIPLE_OCCURENCE)) //If a feature is allowed to be used multiple times or ut has not bean used
			{
				//Restart the histogram bin for each feature for each threshold
				for (thresholdidx=0;thresholdidx<THRESHOLD_NUM+1;thresholdidx++)
				{
					for (labelidx =0;labelidx<nlabels;labelidx++)
					{
						histbin[threadIdx.x*MAX_LABEL_NUM*(THRESHOLD_NUM+1) + labelidx*(THRESHOLD_NUM+1)+ thresholdidx]=0;
					}
				}
			
						
				//Compute the intergral for each features
				for (sampleidx=0;sampleidx<datacount;sampleidx++)
				{				
					curfeat = (float)data[dataidxincurrentnode[sampleidx]*nfeatures+featidx]; //Get current value of the feature
					curbin = roundf((curfeat-curmin)/deltax+0.5);
					curlabel = label[dataidxincurrentnode[sampleidx]];						 //Get the current label
					histbin[threadIdx.x*MAX_LABEL_NUM*(THRESHOLD_NUM+1)+curlabel*(THRESHOLD_NUM+1) + curbin]++;
				}
				
				for (int threshidx=1;threshidx<THRESHOLD_NUM+1;threshidx++)
				{
					for (int labelidx=0;labelidx<nlabels;labelidx++)
					{
						histbin[threadIdx.x*MAX_LABEL_NUM*(THRESHOLD_NUM+1)+labelidx*(THRESHOLD_NUM+1) + threshidx]+=
							histbin[threadIdx.x*MAX_LABEL_NUM*(THRESHOLD_NUM+1)+labelidx*(THRESHOLD_NUM+1) + threshidx-1];
					}
				}
								
				//Compute the information gain for each thresh hold
				for (int threshidx=0;threshidx<THRESHOLD_NUM;threshidx++)
				{	
					
					s1sum = 0;
					s2sum = 0;
					ssum = 0;
					for (int labelidx=0;labelidx<nlabels;labelidx++)
					{
						ns1[threadIdx.x*MAX_LABEL_NUM+labelidx]=histbin[threadIdx.x*MAX_LABEL_NUM*(THRESHOLD_NUM+1)+labelidx*(THRESHOLD_NUM+1) +threshidx]; //Number of samples on the left sides
						ns2[threadIdx.x*MAX_LABEL_NUM+labelidx]=histbin[threadIdx.x*MAX_LABEL_NUM*(THRESHOLD_NUM+1)+labelidx*(THRESHOLD_NUM+1) + THRESHOLD_NUM]-
																	ns1[threadIdx.x*MAX_LABEL_NUM+labelidx];//Number of sample on the right side
						ns[threadIdx.x*MAX_LABEL_NUM+labelidx]=ns1[threadIdx.x*MAX_LABEL_NUM+labelidx]+ns2[threadIdx.x*MAX_LABEL_NUM+labelidx];
						s1sum +=ns1[threadIdx.x*MAX_LABEL_NUM+labelidx];
						s2sum +=ns2[threadIdx.x*MAX_LABEL_NUM+labelidx];
						ssum +=ns[threadIdx.x*MAX_LABEL_NUM+labelidx];
					
					}
				
					//Next, compute the information gain for each split
					es1 = 0.0;
					es2 = 0.0;
					es = 0.0;
				
					for (int labelidx = 0;labelidx<nlabels;labelidx++)
					{
						if (ns1[threadIdx.x*MAX_LABEL_NUM+labelidx]!=0)
						{
							float p = (float)ns1[threadIdx.x*MAX_LABEL_NUM+labelidx]/(s1sum+1e-3);
							es1 -= p*__logf((float)p);
						}
						if (ns2[threadIdx.x*MAX_LABEL_NUM+labelidx]!=0)
						{
							float p = (float)ns2[threadIdx.x*MAX_LABEL_NUM+labelidx]/(s2sum+1e-3);
							es2 -= p*__logf((float)p);
						}
						if (ns[threadIdx.x*MAX_LABEL_NUM+labelidx]!=0)
						{
							float p = (float)ns[threadIdx.x*MAX_LABEL_NUM+labelidx]/(ssum+1e-3);
							es -=p*__logf((float)p);
						}
						fraction_arr[labelidx] = (float)ns[threadIdx.x*MAX_LABEL_NUM+labelidx]/(ssum+1e-3);		
					}
			
				
					ig = es-(float)s1sum/(ssum+1e-3)*es1-(float)s2sum/(ssum+1e-3)*es2; //Need to be modified....
								

					if (ig>bestig)
					{
						bestig = ig;
						bestsplitvar=featidx;
						curbestthresh=(float)threshidx*deltax+curmin;		
						bestnspl1 = s1sum;
						bestnspl2 = s2sum;
						////For debugging-Find the best features of all possible split
						//if ((treeIdx==0))
						//	printf("\nVar: %d. Ig: %1.4f, Thresh: %1.4f, curNodeAddr: %p, s1: %ld, s2:%ld",bestsplitvar,bestig,curbestthresh,curNode,bestnspl1,bestnspl2);
					}				
				}
			}
		}
		//-------------------------------------------------------------------------

		
		
		/*
		//-----------Longer computational step but less memory required-------
		ssum = curNode->data.nsamples;
		for (int varidx=0;varidx<SUP_FEAT_NUM;varidx++)
		{
			featidx = selected_feat[treeIdx*SUP_FEAT_NUM + varidx];
			if ((isFeatureUsedInAncesstor(&nodeList[treeIdx*MAX_NODE_COUNT_PER_TREE],featidx)==FALSE)|(FEAT_MULTIPLE_OCCURENCE)) //If a feature is allowed to be used multiple times or ut has not bean used
			{
				curmin = minval[threadIdx.x*SUP_FEAT_NUM + varidx];
				curmax = maxval[threadIdx.x*SUP_FEAT_NUM + varidx];
				//Restart the histogram bin for each feature
			
				for (int threshidx=0;threshidx<THRESHOLD_NUM;threshidx++)
				{

				
					curthresh = (float)(threshidx)/THRESHOLD_NUM*(curmax-curmin)+curmin;
					s1sum = 0;
					s2sum = 0;
				
					for (int labelidx=0;labelidx<nlabels;labelidx++)
					{
						ns1[threadIdx.x*MAX_LABEL_NUM+labelidx] = 0;
						ns2[threadIdx.x*MAX_LABEL_NUM+labelidx] = 0;
						ns[threadIdx.x*MAX_LABEL_NUM+labelidx] = 0;
					}
					for (sampleidx=0;sampleidx<datacount;sampleidx++)
					{
						curfeat = (float)data[dataidxincurrentnode[sampleidx]*nfeatures+featidx]; //Get current value of the feature
						curlabel = label[dataidxincurrentnode[sampleidx]];						 //Get the current label
						ns[threadIdx.x*MAX_LABEL_NUM+curlabel]++;
						if (curfeat<curthresh)
						{
							ns1[threadIdx.x*MAX_LABEL_NUM+curlabel]++;
							s1sum++;
						}
						else
						{
							ns2[threadIdx.x*MAX_LABEL_NUM+curlabel]++;
							s2sum++;
						}
					}
					//Next, compute the information gain for each split
					es1 = 0.0;
					es2 = 0.0;
					es = 0.0;
					for (int labelidx = 0;labelidx<nlabels;labelidx++)
					{
						if (ns1[threadIdx.x*MAX_LABEL_NUM+labelidx]!=0)
						{
							float p = (float)ns1[threadIdx.x*MAX_LABEL_NUM+labelidx]/s1sum;
							es1 -= p*__logf((float)p);
						}
						if (ns2[threadIdx.x*MAX_LABEL_NUM+labelidx]!=0)
						{
							float p = (float)ns2[threadIdx.x*MAX_LABEL_NUM+labelidx]/s2sum;
							es2 -= p*__logf((float)p);
						}
						if (ns[threadIdx.x*MAX_LABEL_NUM+labelidx]!=0)
						{
							float p = (float)ns[threadIdx.x*MAX_LABEL_NUM+labelidx]/ssum;
							es -=p*__logf((float)p);
						}
						fraction_arr[labelidx] = (float)ns[threadIdx.x*MAX_LABEL_NUM+labelidx]/ssum;		
					}
					ig = es-(float)s1sum/ssum*es1-(float)s2sum/ssum*es2; //Need to be modified....
				
				
					if (ig>bestig)
					{
						bestig = ig;
						bestsplitvar=featidx;
						curbestthresh=curthresh;				
						bestnspl1 = s1sum;
						bestnspl2 = s2sum;
					
						//For debugging-Find the best features of all possible split
						//if ((treeIdx==0))
						//	printf("\nVar: %d. Ig: %1.4f, Thresh: %1.4f, curNodeAddr: %p, s1: %ld, s2:%ld",bestsplitvar,bestig,curbestthresh,curNode,bestnspl1,bestnspl2);
					}
				}
			}
		}
		free(dataidxincurrentnode);	
		//--------------------------------------------------------------------------------------------------------------------------------
		*/

		vartosplit = bestsplitvar;
		thresh = curbestthresh;
		nspl1 = bestnspl1;
		nspl2 = bestnspl2;
		if ((nspl1==0)|(nspl2==0))
		{
			vartosplit=-1;
		}
	}	
}


__device__ bool isFeatureUsedInAncesstor(TreeNodeGPU *p, int featureIdx)	
{
	//Check if a feature has been previously used in the ancessor
	if ((p->parent)==NULL)	//If we are standing at the rootnode
		return FALSE;
	else
	{
		if (p->parent->data.varnum==featureIdx)
			return TRUE;
		else
			isFeatureUsedInAncesstor(p->parent,featureIdx);
	}
}

//Host function to generate the selected feature for each data


void generateFeatureToBeUsedByRandomForest(int *&subfeatarr, int ntrees, int nfeateach, int nfeat)
{
	subfeatarr = (int*)calloc(nfeateach*ntrees,sizeof(int));
	 //Generate featured to be used by different trees
	for (int treeidx=0;treeidx<ntrees;treeidx++)
	{
		for (int featidx=0;featidx<nfeateach;featidx++)
		{
			int randnum = (rand()%100);
			int selectedfeature = (randnum % nfeat);
			bool skipFeat=FALSE;
			for (int innerfeatidx=0;innerfeatidx<featidx;innerfeatidx++)
			{
				if (selectedfeature==subfeatarr[treeidx*nfeateach+innerfeatidx])
				{
					skipFeat = TRUE;
					break;
				}
			}
			if (!skipFeat)
			{
				subfeatarr[treeidx*nfeateach+featidx]=selectedfeature;
			}
		}
	}
}

//Main kernel called to generate a binary tree...
__global__ void generateRandomForestOnGPU(double * data, int *label, int *feat_d, TreeNodeGPU *nodeList_d, long *nodeIndex_d, int nlabels, int nfeatures,long nsamples)
{
	//Train a set of trees for the random forest...
	
	//Get the pointer to the global memory area
	train_d = data;
	label_d = label;
	selected_feat = feat_d;
	nodedataIndex = nodeIndex_d;
	nodeList= nodeList_d;
	ntrainsamples =nsamples;
	treeLearnID3(train_d, label_d, nlabels, nfeatures, nsamples); 
	__syncthreads();
}

__global__ void randomForestEval(double *data,int *label, float *fraction_arr, int nlabels, int nfeatures, long nsamples)
{
	//Evaluate the label given the input data, produce the fractional result for the output confidence
	int stride = blockDim.x * gridDim.x;
	long sampleIdx = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ float fractionarray[BLOCK_SIZE_VAL*MAX_LABEL_NUM]; //Each row is for 1 sample, each column is for 1 class
	TreeNodeGPU *pcurNode; //Current nodepTrees[treeIdx]->proot
	TreeNodeGPU *temp;
	int nstride = 0;
	int treeIdx;
	float curval;
	while (sampleIdx<nsamples)
	{
		//Reset the fractional count for each sample
		for (int labelIdx =0;labelIdx<nlabels;labelIdx++)
		{
			fractionarray[threadIdx.x*MAX_LABEL_NUM + labelIdx] = 0.0f;
		}
		__syncthreads();
	    for (treeIdx=0;treeIdx<TREE_NUM;treeIdx++)
		{
			pcurNode = &(nodeList[treeIdx*MAX_NODE_COUNT_PER_TREE]);
			while ((pcurNode->leftChild!=NULL)&(pcurNode->rightChild!=NULL))
			{
				curval = (float)data[sampleIdx*nfeatures + pcurNode->data.varnum]; 
				/*
				//For debugging
				if ((nstride==0)&(treeIdx==0)&(sampleIdx==0))
					printf("\n[Before]Sample Idx: %ld, treeIdx: %d, Node: %d, FeatureIdx: %d, Threshval: %1.4f, Var value: %1.4f", sampleIdx,treeIdx,
					pcurNode->key,pcurNode->data.varnum,pcurNode->data.thresh,curval);
				*/

				if (curval<((float)pcurNode->data.thresh))
				{
					//Go to the left node
					temp = (pcurNode->leftChild); //We just need to copy the address of the pointer here...Not the whole node (*pcurNode) = (*pcurNode)->leftChild
				}
				else
				{
					//Go to the right node
					temp = (pcurNode->rightChild);
				}
				pcurNode = temp;												
			}
			//Get the fraction of the at the leaf node
			for (int labelIdx=0;labelIdx<nlabels;labelIdx++)
			{
				fractionarray[threadIdx.x*MAX_LABEL_NUM + labelIdx] += (float)pcurNode->data.fraction_arr[labelIdx]/TREE_NUM;
			}
		}
		int bestlabel = 0;
		float bestfraction_arr_val = -1e+20;
		//Accumulate to the final fractional array
		for (int labelIdx=0;labelIdx<nlabels;labelIdx++)
		{
				fraction_arr[sampleIdx*nlabels+labelIdx] = fractionarray[threadIdx.x*MAX_LABEL_NUM + labelIdx];
				if (bestfraction_arr_val<fractionarray[threadIdx.x*MAX_LABEL_NUM + labelIdx])
				{
					bestfraction_arr_val=fractionarray[threadIdx.x*MAX_LABEL_NUM + labelIdx];
					bestlabel = labelIdx;
				}
		}
		label[sampleIdx]=bestlabel;
		sampleIdx +=stride;
		nstride ++;
	}
}
