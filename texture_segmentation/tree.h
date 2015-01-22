#include <stdio.h>
#include <conio.h>
#include <windows.h>
#include <math.h>
#include <string.h>
#include <iostream>

#define THRESHOLD_NUM	25 //Number of threshold values
#define MIN_NODE_SIZE	5	//Minimum node size
#define SUP_FEAT_NUM	20	//Number of features used for each tree
#define MAX_DEPTH		10	//Maximum depth of the tree
#define TREE_NUM	    32	//Number of tree in a random forest
#define RAND_GEN_NUM	3	//Number of random number we use in training the random forest. The index of the sample will be as max as 10^(2*RAN_GEN_NUM)-1
#define FEAT_MULTIPLE_OCCURENCE	1//Allow a feature to appear multiple times
#define PRUNING_EN		1	//Enable the prunning of trees in the random forest
#define MAX_NODE_COUNT_PER_TREE	300	//Maximum number of nodes per tree
#define MAX_NODE_TOTAL (long)(MAX_NODE_COUNT_PER_TREE*TREE_NUM)	//MAXIMUM NUMBER OF NODES FOR ALL TREES

struct NodeData{
	int varnum;	//Index of the variable to determine the split
	float thresh;//Value of the threshold
	int numClass; //Number of class;
	float *fraction_arr; //This is the fractional array of instances for each class in the node
	long * dataidx;	//This is the index of the data that reach this node
	long nsamples;
	int majorclassidx;//Major class index in the node
	int level;		  //Level of the node
	//The following variables ar for the tree-pruning process, which will not be initialized in the training process
	long valnsamples;
	long valtruecount; //Fractional array of the evaluation instances for the pruning purpose
};


struct TreeNode{
	//This is a struct for each note in the tree
	NodeData data;
	unsigned long key;
	TreeNode *parent;
	TreeNode *leftChild;
	TreeNode *rightChild;	
};

//Functions for binary tree
class BinaryTree{
//Abstract datatype for BinarySearchTree
public:
	BinaryTree();//Constructor with all of the features is allow to used
	BinaryTree(int *selectedFeats, int nfeateach);//Constructor with only allow features allowed
	~BinaryTree();
	void elements();
	void deleteNode(int key);//Delete a node with key = [key]
	int size();	//Return the size of the tree
	void searchParentNode(int key, bool &found, TreeNode *&parent);//Search a node with current key, if not found, return the pointer to the parent of the key
	bool insertNode(int key, NodeData d);//Insert the data d into a child node of key = [key]. If the 
	void preScan(TreeNode *p);//Do preOrder scan and display the key 
	void inScan(TreeNode *p);//Do inOrder scan and display the key
	void postScan(TreeNode *p);//Do post scan
	//void printLeafNode(TreeNode *p);//Print all leaf nodes that are descendant of node p
	bool isFeatureUsedInAncesstor(TreeNode *p, int featureIdx);	//Check if a feature has been previously used in the ancessor

	//Ancessor methods
	TreeNode * root(){return proot;}	//Return the value of the pointers points to the data of the rootnode
	bool isroot(int key){return ((key==proot->key)?TRUE:FALSE);} //Check if a given key is of the root node or not
	void children(int key, TreeNode **leftChild, TreeNode **rightChild);//Return the pointer to the left and right childrens of a node with key = key	
	
	
	//Functions for learning the classfication tree;
	void treeLearnID3(double *data, int *label, int nlabels, int nfeatures, long nsamples, bool gpuen); //Train the classification tree
	void splitNode(TreeNode *curNode, double*data,int *label,int nlables,int nfeatures, long nsamples,bool gpuen);	//Recursively splitting data at a node
	void findBestSplitCPU(TreeNode *curNode,double *data,int *label, int nlabels, int nfeatures, int &vartosplit, float &thresh, int &nspl1, int &nspl2, float *fraction_arr);//Find the best split at current node
	void treeEval(double *data,int *&label, float *&fraction_arr, long *&key_arr, long *&nsampleatleaf_arr, int nlabels, int nfeatures, long nsamples, bool gpuen);//Evaluate the label given the input data, produce the fractional result for the output confidence
	void evalNode(TreeNode *curNode, float *curvect, int &label, float *cur_fraction_arr, long &key, long &nsampleatleaf, int nlabels, int nfeatures, bool gpuen); //Evaluate the output of the classifier at current node
	
	//Functions for tree pruning
	void treePrune(double *valdata, int *vallabel, int nfeatures, int nlabels, long nsamples,bool gpuen); //Tree pruning
	void prepareTreeForPruning(TreeNode * curNode);	//Prepare the tree for pruning
	void evalNodeForPruning(TreeNode *curNode,float *curvect,int gtlabel, int nlabels, int nfeatures);	  //Update a node and its descentdant for a validation sample
	void displayNodeValAccuracy(TreeNode *curNode);	//Display current node accuracy;
	void pruneNode(TreeNode *curNode); //Prune the tree starting from the current node
	void deleteNode(TreeNode *curNode); //Delete all desendant of a current node
public:
	long maxkeyval,curkeyval;		//Maximum key value
	
private:
	TreeNode *proot;	//Pointer to the root node
	bool usedsubfeatures;//Allow the used of subfeatures
	int *subfeatarr;
	int nsubfeat;
};

class RandomForest
{
public:
	int ntree,nfeateach,nfeat;			//Number of trees/number of selected feature per tree/total number of features
	long nsampleeach,nsamples;					//Number of samples used for training per tree
	BinaryTree **ppTree;				//This is array of pointers. Each pointer refers to a binary tree
	RandomForest(int _ntree,int _nfeateach, int _nfeat, long _nsampleeach);
	~RandomForest(){};
	
	//Functions for training the Random Forest on CPU
	void randomForestLearn(double *data, int *label, int nlabels, long nsamples);//Train the random forest
	void randomForestEval(double *data,int *&label, float *&fraction_arr, int nlabels, int nfeatures, long nsamples);//Evaluate the label given the input data, produce the fractional result for the output confidence

};
