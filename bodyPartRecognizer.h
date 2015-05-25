#ifndef __bodyPartRecognizer_h_
#define __bodyPartRecognizer_h_

#include "opencv2/opencv.hpp"
#include "depthSample.h"
#include "depthFeature.h"
#include <list>
#include "denseDefine.h"
#include "CapgSkeleton.h"
#include "Macros.h"
//#include <XnOS.h>

extern int g_ImgIndex;

using namespace std;
using namespace cv;

#define DEPTH_FEATURE_GSUB 0
#define DEPTH_FEATURE_GADD 1
#define DEPTH_FEATURE_GHALF 2

#define OutRangeBody 45
#define OutRangeOther 30
#define OutRangeNone 500

#define NotErode 255
#define ErodeForBackground 0
#define ErodeForError 128

#ifdef LINC_DEBUG
struct Vector3
{
	int x;
	int y;
	int z;

	Vector3(): x(0), y(0), z(0){}
	Vector3(int xx, int yy, int zz)
		: x(xx), y(yy), z(zz){}

	Vector3 operator+ (Vector3& v){
		return Vector3(x+v.x, y+v.y, z+v.z);
	}

	Vector3 operator/ (double s){
		return Vector3(x/s, y/s, z/s);
	}
};
#else
#include "Vector3.h"
#endif


class DepthSampleGenerator;
class BPRecognizer;
class BPRTree;

void generateMaskImage( const IplImage* pDepthImg,IplImage* pMaskImg );
IplImage* convertSkeletonToImage(const CapgSkeleton&s);

struct BPRecogPara
{
	int tree_depth; // depth one decision tree should reach
	int min_sample_count; // minimum sample count that one splittable node should have
	int tree_num; // terminate critical: maximum tree count the forests could have

	int feature_type; // feature type: DEPTH_FEATURE_COMPARE or DEPTH_FEATURE_COMPARE_THRESHOLD
	int feature_high_bound; // high bound of u and v

	int img_per_tree; // training depth sample for each tree
	int pixel_per_img; // randomly selected pixel count for each sample image 
	int theta_per_tree; // theta candidate for each tree

	int node_active_var; // active variable number of every tree node

	int merge_flag; // merge flag

	BPRecogPara(){
		tree_depth = 20; min_sample_count = 10; tree_num = 5;
		feature_type = DEPTH_FEATURE_GHALF;
		feature_high_bound = 60;
		img_per_tree = 1000;
		pixel_per_img = 100;
		theta_per_tree = 1500;
		node_active_var = 300;
		merge_flag = Leg_Merge | Arm_Merge;
	}

	BPRecogPara(BPRecogPara& para){
		tree_depth = para.tree_depth; min_sample_count = para.min_sample_count;
		tree_num = para.tree_num; feature_type = para.feature_type;
		feature_high_bound = para.feature_high_bound;
		img_per_tree = para.img_per_tree; pixel_per_img = para.pixel_per_img;
		theta_per_tree = para.theta_per_tree;
		node_active_var = para.node_active_var;
		merge_flag = para.merge_flag;
	}
};

struct BPRTreeBestSplitFinder
{
	BPRTreeBestSplitFinder(){ tree = 0; node = 0; }
	BPRTreeBestSplitFinder(BPRTree* _tree, CvDTreeNode* _node);
	BPRTreeBestSplitFinder(const BPRTreeBestSplitFinder& finder, CvDTreeSplit* split);
	virtual ~BPRTreeBestSplitFinder(){}
	void find(int var_start, int var_end);
	void join(BPRTreeBestSplitFinder& rhs);
	Ptr<CvDTreeSplit> bestSplit;
	Ptr<CvDTreeSplit> split;
	int splitSize;
	BPRTree* tree;
	CvDTreeNode* node;
};

class BPRTree : public CvDTree
{
public:
	BPRTree();
	virtual ~BPRTree();

	virtual void read(CvFileStorage* fs, CvFileNode* node, BPRecognizer* recognizer);
	virtual void clear();

protected:
	friend class BPRecognizer;
	friend struct BPRTreeBestSplitFinder;

	virtual bool train(BPRecognizer* recognizer);
	virtual CvDTreeSplit* find_best_split(CvDTreeNode* node);
	//virtual void calc_node_value(CvDTreeNode* node);
	virtual void try_split_node(CvDTreeNode* node);

	virtual CvDTreeSplit* find_split_ord_class(CvDTreeNode* n, int vi, 
		float init_quality = 0, CvDTreeSplit* _split = 0, uchar* ext_buf = 0);
	virtual CvDTreeSplit* find_split_cat_class(CvDTreeNode* n, int vi,
		float init_quality = 0, CvDTreeSplit* _split = 0, uchar* ext_buf = 0);
	virtual CvDTreeSplit* find_split_ord_reg(CvDTreeNode* n, int vi, 
		float init_quality = 0, CvDTreeSplit* _split = 0, uchar* ext_buf = 0);
	virtual CvDTreeSplit* find_split_cat_reg(CvDTreeNode* n, int vi, 
		float init_quality = 0, CvDTreeSplit* _split = 0, uchar* ext_buf = 0);

	void rebuildValueMat();

	CvDTreeNode* predict(IplImage* img, CvPoint pixel, vector<DepthFeature*>& fset) const;

	BPRecognizer* _recognizer;

	list<int> _rImg;
	CvMat* _value_mat;
	CvMat* _response_mat;
	CvMat* _var_type_mat;
	CvMat* _active_var_mask;

	int _nodeCount;
	int _featureCount; // non-null feature

private:
	vector<pair<DepthFeature*, bool> > _curFeatures;
};

//***************************
//** BPRecognizer **
//** Body part recognizer from depth image using random forests
//***************************
class BPRecognizer
{
public:
	BPRecognizer();
	~BPRecognizer();

	bool load(string& pathname);
	bool save(string& pathname);

	BPRecogPara& getPara();
	void setPara(BPRecogPara& para);

	void setSampleGenerator(DepthSampleGenerator* trainGenerator = 0,
		DepthSampleGenerator* testGenerator = 0){
			_trainSamples = trainGenerator;
			_testSamples = testGenerator;
	}

	void test(bool save = false);
	void testRuntime(bool save = false);

	IplImage* predict(IplImage* depthImg, IplImage* maskImg = 0, bool useDense = true, bool usePre = true);

	void mergeJoint(IplImage* labelImg, IplImage* depthImg, CapgSkeleton& sklt,
		bool useErode = true, bool usePre = true);

	void predictAndMergeJoint(IplImage* depthImg, CapgSkeleton& sklt, IplImage* maskImg = 0, bool usePre = false, bool useErode = true,bool showPic=false);
// 	{
// 		IplImage* pLabelImg = predict(depthImg, maskImg, false, usePre);
// 
// 		////////////////////////////////////////////////////edit by mao
// 		IplImage* pHandImg = cvCreateImage(cvSize(IMAGE_WIDTH, IMAGE_HEIGHT), IPL_DEPTH_8U, 1);
// 		////////////////////////////////////////////////////edit by mao
// 		
// #ifdef DEBUG
// 		if(showPic){
// 			IplImage* pLabelImg2 = cvCreateImage(cvSize(IMAGE_WIDTH, IMAGE_HEIGHT), IPL_DEPTH_8U, 3);
// 			for (size_t y = 0; y < pLabelImg2->height; y++)
// 			{
// 					for (size_t x = 0; x < pLabelImg2->width; x++)
// 				{
// 					uchar* ptr1 = (uchar*)(pLabelImg->imageData+ y * pLabelImg->widthStep + x * sizeof(uchar) * 1);
// 					uchar* ptr2 = (uchar*)(pLabelImg2->imageData + y * pLabelImg2->widthStep + x * sizeof(uchar) * 3);
// 					Color c = DepthSample::getColor(ptr1[0]);
// 					ptr2[0] = c.get_b();ptr2[1] = c.get_g();ptr2[2] = c.get_r();
// 
// 					////////////////////////////////////////////////////edit by mao
// 					uchar* ptr3 = (uchar*)(pHandImg->imageData+ y * pHandImg->widthStep + x * sizeof(uchar) * 1);
// 					uchar* depthMap = (uchar*)(depthImg->imageData+ y * depthImg->widthStep + x * sizeof(uchar) * 1);
// 					if (*ptr1 == BodyLabel_L_Hand || *ptr1 == BodyLabel_R_Hand)
// 					{
// 						*ptr3 = *depthMap;
// 					}
// 					else{
// 						*ptr3 = 255;
// 					}
// 					////////////////////////////////////////////////////edit by mao
// 
// 
// 				}
// 			}
// 
// 			cvShowImage("p",pLabelImg2);
// 			cvShowImage("HandShowImage", pHandImg);
// 			char tBuffer[100] = {'\0'};
// 			string fileName = string("e:/imagetest/") + string(itoa(g_ImgIndex++, tBuffer, 10)) + string(".jpg");
// 			cvSaveImage(fileName.c_str(), pHandImg);
// 
// 			cvReleaseImage(&pHandImg);
// /*
// 			char tBuffer[100] = {'\0'};
// 			string fileName = string(itoa(g_ImgIndex, tBuffer, 10)) + string("p.jpg");
// 			cvSaveImage(fileName.c_str(), pLabelImg2);*/
// 
// 			cvReleaseImage(&pLabelImg2);
// 		}
// #endif
// 		mergeJoint(pLabelImg, depthImg, sklt, usePre, useErode);
// 
// 		/*int nPrint = 5;
// 		while (nPrint--)
// 		{
// 			CapgPrintf("&sklet = %u, sklt.size() is %d\n", &sklt, sklt.size());
// 			xnOSSleep(50);
// 		}*/
// 
// #ifdef DEBUG
// 		if(showPic){
// 			IplImage* simag=convertSkeletonToImage(sklt);
// 			cvShowImage("sk",simag);
// /*
// 			char tBuffer[100] = {'\0'};
// 			string fileName = string(itoa(g_ImgIndex, tBuffer, 10)) + string("sk.jpg");
// 			cvSaveImage(fileName.c_str(), simag);*/
// 
// 			//g_ImgIndex++;
// 			cvReleaseImage(&simag);
// 		}
// #endif
// 		//return skeleton;
// 	}

	void clear();

	void clearPre();

	static cv::RNG* getRng(){
		if(_rng == 0)
			_rng = new RNG(cvGetTickCount());
		return _rng;
	}

protected:
	void clearTrainMat();

	void addNewBestFeature(DepthFeature* f);
	void addNewNullFeature();

	void buideDenseForest();
	ushort buildDenseNode(CvDTreeNode* node);
	BodyLabel predicDense(IplImage* img, CvPoint p, int treeNum);

protected:
	friend class BPRTree;

	vector<vector<DepthFeature*>*> _features;

	vector<BPRTree*> _forest;

	CapgSkeleton _skeleton;
	vector<CvPoint> _searchWin;
	CvPoint _uniSearchStart;
	CvPoint _uniSearchEnd;
	bool _preAvailable;
	IplImage* _preErode;

	int _preErodeForError;

	IplImage* _preLabelImg;
	IplImage* _curLabelImg;

	DepthSampleGenerator* _trainSamples;
	DepthSampleGenerator* _testSamples;

	BPRecogPara _para;
	int _treeNum;

	int _moduleID;

	static RNG* _rng;

	CvMat* _value_mat;
	CvMat* _response_mat;
	CvMat* _var_type_mat;
	CvMat* _active_var_mask;
	CvDTreeParams _dPara;

	int _sample_count;
	int _value_count;

	int _erodeWin;

	int _predictStep;

private:
	ushort* _denseRoot;
	DenseNode* _denseForest;
	DenseFeature* _denseFeature;
	int _featureSize;
	int _nodeSize;
	int _treeCount;
	int* _featuremap;
	char debugbuffer[100];
};

//***************************
//** DepthSampleGenerator **
//** Sample generation tool of body part recognizer
//** The primary function is training / test data collection
//***************************
#define DATA_SOURCE_DATABASE 0
#define DATA_SOURCE_FILE 1
class DepthSampleGenerator
{
public:
	DepthSampleGenerator(string& name){
		_moduleID = LogCat::getInstancePtr()->registerModule(name.data());
	}
	~DepthSampleGenerator();

	bool generateDataFromFile();
	vector<DepthSample*>& getSampleSet(){ return _samples; }

	void setDataSource(int data_source, const string& path = string());
	int getDataSource(){ return _data_source; }

private:
	vector<DepthSample*> _samples;

	int _data_source;
	string _data_path;

	int _moduleID;

};

#endif
