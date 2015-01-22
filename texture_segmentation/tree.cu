#include "tree.h"
#include "support.h"

BinaryTree::BinaryTree()
{
	proot = NULL;
	maxkeyval = 0;
	usedsubfeatures = FALSE;
	nsubfeat = 0;
}

BinaryTree::BinaryTree(int *selectedFeats, int nfeateach)
{
	//Allow training binary tree with subfeatures
	proot = NULL;
	maxkeyval = 0;
	usedsubfeatures = TRUE;
	subfeatarr = (int*)calloc(nfeateach,sizeof(int));
	memcpy((void*)subfeatarr,(void*)selectedFeats,sizeof(int)*nfeateach); //Copy teh allow featured
	nsubfeat = nfeateach;

}
void BinaryTree::searchParentNode(int key, bool &found, TreeNode *&parent)
{
	//Search a node with current key, if not found, return the pointer to the parent of the key else, bool is set to true
	found = FALSE;
	TreeNode *resparent =  proot; //Pointer to the parent node
	parent = resparent;
	while (resparent!=NULL)				//Keep going down if the current node is not a leaf node
	{
		if ((resparent->key==key))
		{
			found = TRUE;			
			break;
		}
		else
		{
			parent = resparent;
			//Next, if not found, then, go left or right to return the parent
			if (resparent->key>key)
			{
				resparent = parent->leftChild;
			}
			else
			{
				resparent = parent->rightChild;
			}
		}
	}

}

void BinaryTree::children(int key, TreeNode **leftChild, TreeNode **rightChild)
{
	bool found = FALSE;
	(*leftChild) = NULL;
	(*rightChild) = NULL;
	TreeNode *tempNode = proot;
	if (tempNode==NULL)
		return;
	else
	{
		while (tempNode!=NULL)
		{
			if ((tempNode->key)==key)
			{
				*leftChild = tempNode->leftChild;
				*rightChild = tempNode->rightChild;
				break;
			}
			if ((tempNode->key)>key)
			{
				tempNode = tempNode->leftChild;
			}
			else
			{
				tempNode = tempNode ->rightChild;
			}
		}
	}
}

void BinaryTree::preScan(TreeNode *p)
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
}

void BinaryTree::inScan(TreeNode *p)
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
}

void BinaryTree::postScan(TreeNode *p)
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
}

bool BinaryTree::insertNode(int key, NodeData d)
{
	//Insert data a into the tree and assign the node key to it.
	TreeNode *ptempNode = new TreeNode();
	ptempNode->leftChild = NULL;
	ptempNode->rightChild = NULL;
	ptempNode->parent = NULL;
	ptempNode->data = d;
	ptempNode->key = key;
	if (proot==NULL)
	{	//If the tree is just get initialized
		proot = ptempNode;
		return TRUE;
	}
	else
	{
		bool keyfound = FALSE;
		TreeNode *parent;
		//Search for the parent node
		this->searchParentNode(key,keyfound,parent);
		if (keyfound==TRUE)
		{
			printf("Key is already existed...Cannot assign data to the same key");
			return FALSE;
		}
		else
		{
			if (parent->key>key)
			{
				ptempNode->parent = parent; 
				parent->leftChild = ptempNode;
			}
			else
			{
				ptempNode->parent = parent;
				parent->rightChild = ptempNode;
			}
			return TRUE;
		}
	}
	return FALSE;
}

void BinaryTree::deleteNode(int key)
{
	//Get the pointer to the parent node of the current node with key
	TreeNode *parentNode;
	bool parentFound;
	searchParentNode(key,parentFound,parentNode);
	//Delete the link to the child node from the parent node
	if (key!=parentNode->key) //If this is not the root node
	{
		int parentkey = parentNode->key;
		if (key>parentkey) //If current node is the right subtree
		{
			delete parentNode->rightChild;
			parentNode->rightChild = NULL;
		}
		else
		{
			delete parentNode->leftChild;
			parentNode->leftChild = NULL;
		}
	}
	else //If this is a root node
	{
		parentNode->rightChild = NULL;
		parentNode->leftChild = NULL;
	}
}

BinaryTree::~BinaryTree()
{
	//deleteNode(proot);
}

void BinaryTree::elements()
{
}

int BinaryTree::size()
{
	return 0;
}

void BinaryTree::treeLearnID3(double *data, int *label, int nlabels, int nfeatures, long nsamples, bool gpuen)
{
	/*Train the classification tree.
	Inputs:
	data: a 1D array of dimension 1 x (nfeatures*nsamples) in which each  training vector is a row vector with dimension of nfeatures
	label: a 1D array of dimension 1 x nsamples with elements in range [0,1,..,nlabels-1] correspond to different classes.
	nlabels: number of labels
	nfeatures: number of features
	*/
	if (gpuen==FALSE) //CPU training
	{
		long *nodeIndex; //Node indices of the training data
		nodeIndex = (long*) calloc(nsamples,sizeof(long));//This array stores the indices of the samples for all samples
		for (int sampleidx =0; sampleidx<nsamples;sampleidx++)
			nodeIndex[sampleidx]=sampleidx;
		curkeyval = 0;
		//Initialize data for the root node
		TreeNode *curNode = new TreeNode();
		//Search for the threshold that best split the data for the rootnode
		curNode->key = curkeyval;
		curNode->parent = NULL;
		curNode->leftChild  =NULL;
		curNode->rightChild = NULL;
		curNode->data.nsamples = nsamples;
		curNode->data.varnum = -1;
		curNode->data.thresh = 0.0;
		curNode->data.level = 0;
		curNode->data.valnsamples = 0; //Reset the number of evaluation samples for training
		curNode->data.dataidx = (long*) calloc(curNode->data.nsamples,sizeof(long));
		memcpy((void*)curNode->data.dataidx,(void*)nodeIndex,curNode->data.nsamples*sizeof(long));//Copy the indices of all data belongs to the current node
		
		//Copy data to the root node
		proot = curNode;
		splitNode(proot,data,label,nlabels,nfeatures,nsamples,gpuen);
		

	}
	else
	{
	}
}

void BinaryTree::splitNode(TreeNode *curNode, double*data,int *label,int nlables,int nfeatures, long nsamples,bool gpuen)
{
		float besthresh = 0.0;
		int vartosplit = 0, nsleft, nsright;
		//Go through every features and find the best split
		curNode->data.fraction_arr = (float*)calloc(nlables,sizeof(float));
		findBestSplitCPU(curNode,data,label,nlables,nfeatures,vartosplit,besthresh,nsleft,nsright,curNode->data.fraction_arr);
		//Compute the assigned class at the current node
		float largestfraction = 0.0;
		for (int classidx=0;classidx<nlables;classidx++)
		{
			if (largestfraction<(curNode->data.fraction_arr[classidx]))
			{
				largestfraction =curNode->data.fraction_arr[classidx];
				curNode->data.majorclassidx = classidx;
			}
		}

		/*
		if (curNode!=this->proot)
			printf("\nN.:%d, Par.: %d, Depth: %d, Var: %d, Thre.: %1.2f, Major class: %d ,#samples: %ld, f[0]: %1.5f, f[1]: %1.5f",curNode->key,curNode->parent->key, curNode->data.level, vartosplit,besthresh,curNode->data.majorclassidx,curNode->data.nsamples,curNode->data.fraction_arr[0],curNode->data.fraction_arr[1]);
		else
			printf("\nN.: %d, Depth: %d, Var: %d, Thre.: %1.2f, Major class: %d ,#samples: %ld, f[0]: %1.5f, f[1]: %1.5f",curNode->key, curNode->data.level, vartosplit,besthresh,curNode->data.majorclassidx,curNode->data.nsamples,curNode->data.fraction_arr[0],curNode->data.fraction_arr[1]);
		*/

		if ((curNode->data.nsamples<MIN_NODE_SIZE)|(vartosplit==-1)|(curNode->data.level==MAX_DEPTH)) //Do not further divide if all the features have been used/ the number of instances at each node is too small/maximum depth reach
		{
			return;
		}
		else
		{
			curNode->data.varnum = vartosplit;
			curNode->data.thresh = besthresh;
			//Split the node  
			TreeNode *leftNode = new TreeNode();
			TreeNode *rightNode = new TreeNode();
			curNode->leftChild = leftNode;
			curNode->rightChild = rightNode;

			leftNode->parent = curNode;
			leftNode->key = ++maxkeyval;
			leftNode->data.nsamples = nsleft;
			leftNode->data.varnum = -1;
			leftNode->data.thresh = 0.0;
			leftNode->data.dataidx = (long*) calloc(leftNode->data.nsamples,sizeof(long));
			leftNode->data.level = curNode->data.level+1;
			leftNode->data.valnsamples = 0;

			rightNode->parent = curNode;
			rightNode->key = ++maxkeyval;
			rightNode->data.nsamples = nsright;
			rightNode->data.varnum = -1;
			rightNode->data.thresh = 0.0;
			rightNode->data.dataidx = (long*) calloc(rightNode->data.nsamples,sizeof(long));
			rightNode->data.level = curNode->data.level+1;
			rightNode->data.valnsamples = 0;

			//Update the sample index of each node
			long leftNodeCount =0, rightNodeCount = 0;
			for (long sampleidx=0;sampleidx<curNode->data.nsamples;sampleidx++)
			{
				int sampleidxintrainingset = curNode->data.dataidx[sampleidx]; //Get the sample indices at current node [a values from 0->(number of samples-1)]
				float curfeat = (float)data[sampleidxintrainingset*nfeatures+curNode->data.varnum]; //Get current value of the feature
				if (curfeat<curNode->data.thresh)
					leftNode->data.dataidx[leftNodeCount++]=sampleidxintrainingset;
				else
					rightNode->data.dataidx[rightNodeCount++]=sampleidxintrainingset;
			}
			splitNode(leftNode,data,label,nlables,nfeatures,nsleft,gpuen);
			splitNode(rightNode,data,label,nlables,nfeatures,nsright,gpuen);
		}
		
}

void BinaryTree::findBestSplitCPU(TreeNode *curNode,double *data,int *label, int nlabels, int nfeatures, int &vartosplit, float &thresh, int &nspl1, int &nspl2, float *fraction_arr)
{
	//Find the best split at current node. IF all the features have been used, return the varidx = -1, which means we do not split.
	float bestig=0.0;
	vartosplit = -1;
	thresh = 0;
	nspl1 = 0;
	nspl2 = 0;
	for (int varidx=0;varidx<nfeatures;varidx++)
	{
		bool proceedwiththisfeat = FALSE;
		if (this->usedsubfeatures==TRUE) //If we are only allowed to used a
		{
			//Check if this feature is allowed in the featur set
			for (int subfeatidx=0;subfeatidx<nsubfeat;subfeatidx++)
			{
				if (varidx==subfeatarr[subfeatidx])
				{
					proceedwiththisfeat = TRUE;
					break;
				}
			}	
		}
		else
		{
			proceedwiththisfeat = TRUE;
		}
		if (proceedwiththisfeat)
		{
			if ((isFeatureUsedInAncesstor(curNode,varidx)==FALSE)|(FEAT_MULTIPLE_OCCURENCE)) //Just go with the feature that hasn't been used
			{
				//Look for the min and maximum values of the current feature at the current node
				float minval=+1e20;
				float maxval=-1e20;
				//printf("\n Feat: %d", varidx);
				for (long sampleidx=0;sampleidx<curNode->data.nsamples;sampleidx++)
				{
					int sampleidxintrainingset = curNode->data.dataidx[sampleidx]; //Get the sample indices at current node [a values from 0->(number of samples-1)]
					float curfeat = (float)data[sampleidxintrainingset*nfeatures+varidx]; //Get current value of the feature
					if (curfeat<minval)
						minval = curfeat;
					if (curfeat>maxval)
						maxval = curfeat;
				//	printf("%1.4f ",curfeat);
				}

				for (int thresholdidx=0;thresholdidx<THRESHOLD_NUM;thresholdidx++) //Choose different value for the threshold
				{
					float curthresh = (thresholdidx)*(maxval-minval)/(THRESHOLD_NUM-1) + minval;
					//This section is for computing the information gain of each possible split
					long *ns, *ns1,*ns2;
					long ssum=0,s1sum=0,s2sum=0;
					float es=0.0,es1=0.0,es2=0.0,ig=0.0; //Entropies and information gain
					ns = (long*)calloc(nlabels,sizeof(long));
					ns1 = (long*)calloc(nlabels,sizeof(long));
					ns2 = (long*)calloc(nlabels,sizeof(long));
					ssum = curNode->data.nsamples;
		
					//Compute the number of samples belongs to for each class corresponding to each split
					for (long sampleidx=0;sampleidx<curNode->data.nsamples;sampleidx++)
					{
						int sampleidxintrainingset = curNode->data.dataidx[sampleidx]; //Get the sample indices at current node [a values from 0->(number of samples-1)]
						float curfeat = (float)data[sampleidxintrainingset*nfeatures+varidx]; //Get current value of the feature
						int curlabel = label[sampleidxintrainingset];
						ns[curlabel]++;
						if (curfeat<curthresh)
						{
							ns1[curlabel]++;
							s1sum++;
						}
						else
						{
							ns2[curlabel]++;
							s2sum++;
						}
					}
					//Next, compute the information gain for each split
					for (int labelidx = 0;labelidx<nlabels;labelidx++)
					{
						if (ns1[labelidx]!=0)
						{
							float p = (float)ns1[labelidx]/s1sum;
							es1 -= p*log((float)p);
						}
						if (ns2[labelidx]!=0)
						{
							float p = (float)ns2[labelidx]/s2sum;
							es2 -= p*log((float)p);
						}
						if (ns[labelidx]!=0)
						{
							float p = (float)ns[labelidx]/ssum;
							es -=p*log((float)p);
						}
						fraction_arr[labelidx] = (float)ns[labelidx]/ssum;
					}
					ig = es-(float)s1sum/ssum*es1-(float)s2sum/ssum*es2; //Need to be modified....

					if (ig>bestig)
					{ 
						bestig = ig;
						vartosplit=varidx;
						thresh = curthresh;
						nspl1 = s1sum; //Return the number of elements in each set
						nspl2 = s2sum;
					}
					free(ns1);
					free(ns2);
					free(ns);
				}
			}
		}
	}
}

bool BinaryTree::isFeatureUsedInAncesstor(TreeNode *p, int featureIdx)	
{
	//Check if a feature has been previously used in the ancessor
	if (p==proot)	//If we are standing at the rootnode
		return FALSE;
	else
	{
		if (p->parent->data.varnum==featureIdx)
			return TRUE;
		else
			isFeatureUsedInAncesstor(p->parent,featureIdx);
	}
}

void BinaryTree::treeEval(double *data,int *&label, float *&fraction_arr, long *&key_arr, long *&nsampleatleaf_arr, int nlabels, int nfeatures, long nsamples, bool gpuen)//Evaluate the label given the input data
{
	//Traverse through the classification tree and produce an output for each input data sample
	float *curvect=(float*)calloc(nfeatures,sizeof(float));
	float *cur_fraction_arr = (float*)calloc(nlabels,sizeof(float));
	fraction_arr = (float*)calloc(nsamples*nlabels,sizeof(float));
	label = (int*)calloc(nsamples,sizeof(int));
	key_arr = (long*)calloc(nsamples,sizeof(long));
	nsampleatleaf_arr = (long*)calloc(nsamples,sizeof(long));
	for (long sampleidx=0;sampleidx<nsamples;sampleidx++)
	{
		
		for (int featureidx=0;featureidx<nfeatures;featureidx++)
			curvect[featureidx]=(float)data[sampleidx*nfeatures+featureidx];
		evalNode(proot,curvect,label[sampleidx],cur_fraction_arr,key_arr[sampleidx],nsampleatleaf_arr[sampleidx],nlabels,nfeatures,gpuen);
	
		//Return the fractional array
		for (int labelidx=0;labelidx<nlabels;labelidx++)
			fraction_arr[sampleidx*nlabels+labelidx]=cur_fraction_arr[labelidx];
	
		//printf("\nSpl: %ld Label: %d A[0]=%1.2f ,A[1]= %1.2f",sampleidx,label[sampleidx],cur_fraction_arr[0],cur_fraction_arr[1]);
	}
	free(curvect);
	free(cur_fraction_arr);
}

void BinaryTree::evalNode(TreeNode *curNode, float *curvect, int &label, float *cur_fraction_arr, long &key, long &nsampleatleaf, int nlabels, int nfeatures, bool gpuen)
{
	//Evaluate the output of the classifier at current node, return the output of the classification (label,fraction_array, key of leaf node)
	//Note that fraction_array here is the fractional array of the trainign set
	//Also put count the increase the number of node

	float curfeat = curvect[curNode->data.varnum];
	//Check for condition of the leaf node
	if (((curNode->leftChild)==NULL)&((curNode->rightChild)==NULL))
	{
		key = curNode->key;
		nsampleatleaf = curNode->data.nsamples;
		label = curNode->data.majorclassidx;
		memcpy((void*)cur_fraction_arr,(void*)curNode->data.fraction_arr,sizeof(float)*nlabels);
		return;
	}
	else if ((curfeat<curNode->data.thresh)&((curNode->leftChild)!=NULL))
	{
		evalNode(curNode->leftChild, curvect, label, cur_fraction_arr, key, nsampleatleaf, nlabels, nfeatures, gpuen);
	}
	else if ((curNode->rightChild)!=NULL)
	{
		evalNode(curNode->rightChild, curvect, label, cur_fraction_arr, key, nsampleatleaf, nlabels, nfeatures, gpuen);
	}
	else
	{
		printf("\n[Error]Can't find a correct condition for traversing...");
	}
}

void BinaryTree::prepareTreeForPruning(TreeNode * curNode)
{
	//Recursively reset the truecount and datacount at each node for validation
	if (curNode==NULL) //If we reach the leaf node
		return;
	else
	{
		curNode->data.valnsamples = 0;
		curNode->data.valtruecount = 0;
		prepareTreeForPruning(curNode->leftChild);
		prepareTreeForPruning(curNode->rightChild);
	}
}

void BinaryTree::evalNodeForPruning(TreeNode *curNode,float *curvect,int gtlabel, int nlabels, int nfeatures)
{
	//Update a node and its descentdant for a validation sample
	curNode->data.valnsamples++;
	//Once we visit a node, then we assign a label
	if ((curNode->data.majorclassidx)==gtlabel)
			curNode->data.valtruecount++;
	if ((curNode->leftChild==NULL)&(curNode->rightChild==NULL)) //If this is a leaf node, then, just compare
		return;
	else
	{
		if (curvect[curNode->data.varnum]<curNode->data.thresh)
			evalNodeForPruning(curNode->leftChild,curvect,gtlabel,nlabels,nfeatures);
		else
			evalNodeForPruning(curNode->rightChild,curvect,gtlabel,nlabels,nfeatures);
	}
}

void BinaryTree::treePrune(double *valdata, int *vallabel, int nfeatures, int nlabels, long nsamples,bool gpuen)
{
	/*Tree pruning
	Inputs:
		valdata: validatation data
		vallabel: groundtruth of the validatiation data.
	Methods: for each sample, we traverse down the tree, accumulate the number of datasample passing through the node,
	compare the label produce by the node to the groundtruth label.
	If they are different, increase the error count by 1
	*/
	prepareTreeForPruning(this->proot);
	float *curvect = (float*)calloc(nfeatures,sizeof(float));
	for (long valsampleidx = 0;valsampleidx<nsamples;valsampleidx++)
	{
		for (int featidx=0;featidx<nfeatures;featidx++)
		{
			curvect[featidx]=(float)valdata[valsampleidx*nfeatures+featidx];
		}
		evalNodeForPruning(proot,curvect,vallabel[valsampleidx],nlabels,nfeatures); //Update the tree with all the labels
	}
	free(curvect);
	
	//Display node accuracy
	//displayNodeValAccuracy(proot);
	
	pruneNode(proot);
	//printf("\nTree after pruning");
	
	//displayNodeValAccuracy(proot);
}

void BinaryTree::displayNodeValAccuracy(TreeNode *curNode)
{
	//Use inorder traversering and display node accurracy
	if (curNode==NULL)
		return;
	else
	{
		displayNodeValAccuracy(curNode->leftChild);
		if (curNode->parent!=NULL)
			printf("\nNode %d, Parent %d, #Samples: %ld, Accuracy: %1.4f",curNode->key,curNode->parent->key,curNode->data.valnsamples, (float)curNode->data.valtruecount/((float)curNode->data.valnsamples+1e-3));
		displayNodeValAccuracy(curNode->rightChild);
	}
}

void BinaryTree::pruneNode(TreeNode *curNode)
{
	//Compare a node with its two child. If its performances is better than its two child, then, make it a leaf node
	//Prune the tree starting from the current node
	if ((curNode->leftChild==NULL)&(curNode->rightChild==NULL)) //If we are at the leaf node, nothing to do
	{
		return;
	}
	else
	{
		if ((curNode->data.valtruecount)>(curNode->leftChild->data.valtruecount+curNode->rightChild->data.valtruecount)) //if the parent is better than the children :)
		{
			deleteNode(curNode->leftChild);
			deleteNode(curNode->rightChild);
			curNode->leftChild = NULL;
			curNode->rightChild = NULL;
		}
		else
		{
			//Keep going down and check with the child
			pruneNode(curNode->leftChild);
			pruneNode(curNode->rightChild);
		}
	}
}

void BinaryTree::deleteNode(TreeNode *curNode)
{
	//Delete all the descendant of a current node
	if (curNode==NULL)
	{
		//If the current pointer is already NULL
		return;
	}
	else
	{
		if ((curNode->leftChild!=NULL)&(curNode->rightChild!=NULL))
		{
			deleteNode(curNode->leftChild); //Delete all descentdant of current descendant
			deleteNode(curNode->rightChild); 
			delete curNode->leftChild;
			delete curNode->rightChild;
			curNode->leftChild = NULL;
			curNode->rightChild = NULL;
		}
	}
}

RandomForest::RandomForest(int _ntree,int _nfeateach, int _nfeat, long _nsampleeach)
{
	ntree = _ntree;
	nfeateach = _nfeateach;
	nfeat = _nfeat;
	nsampleeach = _nsampleeach; //Number of boostrapped samples
	ppTree = (BinaryTree**)calloc(ntree,sizeof(BinaryTree*));
}

void RandomForest::randomForestLearn(double *data, int *label, int nlabels, long _nsamples)
{
	//Train the random forest
	/*
	Inputs:
		data: an array for the input data
		label: groundtruth for training
		nlabels: number of labels for the data
		_nsamples: number of training sample for the large data set
	*/
	nsamples = _nsamples;
	double * curinputsamples = (double*)calloc(nfeat*nsampleeach,sizeof(double));
	int * curlabels = (int*)calloc(nsampleeach,sizeof(int));
	int * curfeats = (int*)calloc(nfeateach,sizeof(int));	
	for (int treeidx = 0; treeidx<ntree;treeidx++)
	{
		bool * inbags = (bool*)calloc(nsamples,sizeof(bool));
		//printf("\nTraining tree [%d]..",treeidx);
		//Prepare the training data and corresponding label
		for (long sampleidx=0;sampleidx<nsampleeach;sampleidx++)
		{
			int narr[RAND_GEN_NUM];
			long randnum=0;
			for (int narridx=0;narridx<RAND_GEN_NUM;narridx++)
			{
				narr[narridx]=rand() % 100;//Generate a random number from 0->99
				randnum += pow((float)10,(float)2*narridx)*narr[narridx];
			}
			randnum = randnum % nsamples; //Sampling with replacement
			
			//Copy the data and the label
			for (int featidx=0;featidx<nfeat;featidx++)
			{
				curinputsamples[sampleidx*nfeat+featidx]=(double)data[randnum*nfeat+featidx];
				curlabels[sampleidx]=(int)label[randnum];
			}
			//Mark this sample has been used
			inbags[randnum]=TRUE;
		}
		
		//Generate a subset of selected features
		for (int featidx=0;featidx<nfeateach;featidx++)
		{
			int randnum = (rand()%100);
			int selectedfeature = (randnum % nfeat);
			bool skipFeat=FALSE;
			for (int innerfeatidx=0;innerfeatidx<featidx;innerfeatidx++)
			{
				if (selectedfeature==curfeats[innerfeatidx])
				{
					skipFeat = TRUE;
					break;
				}
			}
			if (!skipFeat)
				curfeats[featidx]=selectedfeature;
		}

		//Initialize the trees on each with each subset of features
		ppTree[treeidx] = new BinaryTree(curfeats,nfeateach);
		//Train the trees
		ppTree[treeidx]->treeLearnID3(curinputsamples,curlabels,nlabels,nfeat,nsampleeach,0);
		
		//Next-prepare OOB samples for tree pruning
		long noobsamples=0; //Number of outofback samples
		for (long sampleidx=0;sampleidx<nsamples;sampleidx++)
		{
			if (!inbags[sampleidx])
				noobsamples++;
		}
		
		//Generate the out of bags samples for tree pruning
		//Generate an array and label for OOB samples
		double * oobinputsamples = (double*)calloc(nfeat*noobsamples,sizeof(double));
		int * ooblabels = (int*)calloc(noobsamples,sizeof(int));
		long oobcount=0;
		for (long sampleidx=0;sampleidx<this->nsamples;sampleidx++)
		{
			if (!inbags[sampleidx])
			{
				//Copy the current feature vector and it labels
				memcpy((void*)&oobinputsamples[oobcount*nfeat],(void*)&data[sampleidx*nfeat],sizeof(double)*nfeat);
				ooblabels[oobcount++]=label[sampleidx];
			}
		}
		if (PRUNING_EN)
		{
			//printf("\nOOB pruning...");
			ppTree[treeidx]->treePrune(oobinputsamples,ooblabels,nfeat,nlabels,oobcount,0);
		}

		//printf("\n OOB Accuracy:...");
		int *out_ooblabels;
		float *out_fr_arr;
		long *nsampleatleaf_arr;
		long *key_arr;
		float *cf_mat;
		//Evaluate oob after prunning
		ppTree[treeidx]->treeEval(oobinputsamples,out_ooblabels,out_fr_arr,key_arr,nsampleatleaf_arr,nlabels,nfeat,oobcount,0);
		confusionMatrix(ooblabels, out_ooblabels,cf_mat,nlabels,oobcount);//Compute the confusion matrix
		

		free(oobinputsamples);
		free(ooblabels);
		free(inbags);
		

	}
	//printf("\nDone...");
	free(curinputsamples);
	free(curlabels);
	free(curfeats);
}

void RandomForest::randomForestEval(double *data,int *&label, float *&fraction_arr, int nlabels, int nfeatures, long nsamples)
{
	label = (int*)calloc(nsamples,sizeof(int));//An array for storing the output labels
	fraction_arr = (float*)calloc(nsamples*nlabels,sizeof(float));

	float *cur_fraction_arr;
	int *curlabel;
	long *cur_key_arr;
	long *curnsampleatleaf_arr;
	for (int treeIdx=0;treeIdx<this->ntree;treeIdx++)
	{
		ppTree[treeIdx]->treeEval(data,curlabel,cur_fraction_arr,cur_key_arr,curnsampleatleaf_arr,nlabels,nfeatures,nsamples,0);
		for (long sampleidx=0;sampleidx<nsamples;sampleidx++)
		{
			for (int labelidx=0;labelidx<nlabels;labelidx++)
			{
				fraction_arr[sampleidx*nlabels+labelidx]+=cur_fraction_arr[sampleidx*nlabels+labelidx];
			}
		}
	}

	for (long sampleidx=0;sampleidx<nsamples;sampleidx++)
	{
		int bestlabel = 0; 
		float largest_fraction = 0.0;
		for (int labelidx=0;labelidx<nlabels;labelidx++)
		{
			fraction_arr[sampleidx*nlabels+labelidx]/=ntree;
			if (fraction_arr[sampleidx*nlabels+labelidx]>largest_fraction)
			{
				largest_fraction = fraction_arr[sampleidx*nlabels+labelidx];
				bestlabel = labelidx;
			}
			
		}
		label[sampleidx]=bestlabel;
	}
	free(curlabel);
	free(curnsampleatleaf_arr);
	free(cur_fraction_arr);
	free(cur_key_arr);
}

/*
void RandomForest::randomForestLearnGPU(double *data, int *label, int nlabels, long nsamples)
{
	//Copy the training data onto the cpu
	float *data_d;
	int *label_d;
	float *data_h = (float*)calloc(nsamples*nfeat,sizeof(float));
	printf("\nAllocating device memory area...");
	cudaMalloc((void**)&data_d,sizeof(float)*nsamples*nfeat);
	cudaMalloc((void**)&label_d,sizeof(int)*nsamples);
	//Convert the input data into float and copy to the device
	for (long sampleidx=0;sampleidx<nsamples;sampleidx++)
	{
		for (int featidx=0;featidx<nfeat;featidx++)
			data_h[sampleidx*nfeat+featidx]=(float)data[sampleidx*nfeat+featidx];
	}
	cudaMemcpy((void*)data_d,(void*)data_h,sizeof(float)*nsamples*nfeat,cudaMemcpyHostToDevice);
	cudaMemcpy((void*)label_d,(void*)label,sizeof(int)*nsamples,cudaMemcpyHostToDevice);
	
	
	cudaFree(data_d);
	cudaFree(label_d);
	free(data_h);
	printf("\nDone...");
}
*/