#include <iostream>
#include <vector>
#include <queue>
#include <opencv2/opencv.hpp>
#include <ctime>

#include "CapgSkeleton.h"
#include "bodyPartRecognizer.h"

#include "sgf_segment.h"
//using namespace sgf;

using namespace std;
using namespace cv;

const int QVGA_WIDTH = 320,
	QVGA_HEIGHT = 240,
	MAX_VALID_DEPTH = 10000;

//zhangxaochen: 假定 320*240 数据对应的焦距
#define XTION_FOCAL_XY 300

#define MIN_VALID_HEIGHT_PHYSICAL_SCALE 600
#define MIN_VALID_HW_RATIO 1.0 //height-width ratio

static cv::RNG rng;

extern const int thickLimitDefault;// = 1500;
extern int thickLimit;// = thickLimitDefault; //毫米


namespace zc{
	class HumanFg;

#ifdef CV_VERSION_EPOCH
//#if CV_VERSION_MAJOR < 3

//lincccc's code below:
	//BPRecognizer* getBprAndLoadFeature();
	//BPRecognizer* getBprAndLoadFeature(const char *featurePath = nullptr);
	BPRecognizer* getBprAndLoadFeature(const string &featurePath);
#endif

	Mat simpleMask(const Mat &curMat, bool debugDraw = false);

	Point seedSimple(Mat dmat, int *outVeryDepth = 0, bool debugDraw = false);

	//尝试从 findFgMasksUseBbox 剥离解耦, √
	//用正视图、俯视图两种 bbox 求交，判定人体轮廓位置
	//注：
	// 1. 若 debugDraw = true, 则 _debug_mat 必须传实参
	// 2. 返回值 vector<vector<Point>> 实际就是挑选过的 contours
	vector<vector<Point>> seedUseBbox(Mat dmat, bool debugDraw = false, OutputArray _debug_mat = noArray());

	//@deprecated, 语义不明，弃用；其实包含两步：去背景 & findFgMasksUseBbox
	//1. 找背景大墙面；
	//2. 若没墙，说明背景空旷，物理高度判定剔除(<2500mm)
	//3. 剩余最高点做种子点
	vector<Mat> findFgMasksUseWallAndHeight(Mat dmat, /*bool usePre = false, */bool debugDraw = false);

	//根据高度or大平面增长出背景：
	Mat getBgMskUseWallAndHeight(Mat dmat);
	Mat fetchBgMskUseWallAndHeight(Mat dmat);


	//Mat simpleRegionGrow(const Mat &dmat, Point seed, int thresh, bool debugDraw = false);

	//Point seed, Rect roi
	Mat _simpleRegionGrow(const Mat &dmat, Point seed, int thresh, const Rect roi, bool debugDraw = false);

	//vector<Point> seeds, Rect roi
	Mat _simpleRegionGrow(const Mat &dmat, vector<Point> seeds, int thresh, const Rect roi, bool debugDraw = false);

	//mask 为是否有效增长 flag， 若某像素黑(0), 则此处终止，去看别处
	Mat _simpleRegionGrow_core_pt2mat(const Mat &dmat, Point seed, int thresh, const Mat &validMask, bool debugDraw = false);

	//_simpleRegionGrow 核心函数：
	Mat _simpleRegionGrow_core_mat2mat(const Mat &dmat, Mat sdsMat, int thresh, const Mat &validMask, bool debugDraw = false);
	Mat _simpleRegionGrow_core_vec2mat(const Mat &dmat, vector<Point> sdsVec, int thresh, const Mat &validMask, bool debugDraw = false);

	//先 seedsMask -> vector<Point> seeds，再调用重载
	Mat _simpleRegionGrow(const Mat &dmat, Mat seedsMask, int thresh, const Mat &mask, bool debugDraw = false);

	//N个种子点， 
	//1. getMultiMasks = false, 尽可能增长为多个mask
	//2. getMultiMasks = true, 增长为一个mask, 不管是否连成片，存在位置[0]。
	vector<Mat> simpleRegionGrow(const Mat &dmat, vector<Point> seedsVec, int thresh, const Mat &mask, bool getMultiMasks = false, bool debugDraw = false);

	//先 sdsMat -> vector<Point> seeds, 再调用重载
	vector<Mat> simpleRegionGrow(const Mat &dmat, Mat sdsMat, int thresh, const Mat &mask, bool getMultiMasks = false, bool debugDraw = false);

	//N个种子点vector， 增长为N个mask
	vector<Mat> simpleRegionGrow(const Mat &dmat, vector<vector<Point>> seedsVecOfVec, int thresh, const Mat &mask, bool debugDraw = false);

	//N个种子点mask， 增长为N个mask
	vector<Mat> simpleRegionGrow(const Mat &dmat, vector<Mat> sdMats, int thresh, const Mat &mask, bool debugDraw = false);
	
	//---------------@deprecated, 错误思路，不该有特定经验性限制！
// 	vector<Mat> simpleRegionGrow(Mat dmat, Mat seedsMask, int thresh, Mat mask);

	//返回一个 mask-mat; type: 8uc1, 用白色 UCHAR_MAX 点表示有效点
	Mat pts2maskMat(const vector<Point> pts, Size matSz);

	//maskMat: type 8uc1, 白色 UCHAR_MAX 点表示有效点
	vector<Point> maskMat2pts(Mat maskMat, int step = 1);

	//三个默认参数(flrKrnl, mskThresh, morphRadius) 版：
	//flrKrnl = { 1, 1, 1, -1, -1, -1 };
	//mskThresh = 100;
	//morphRadius = 3;
	Mat getFloorApartMask(Mat dmat, bool debugDraw = false);
	//返回：用于去除地板的mask
	Mat getFloorApartMask(Mat dmat, Mat flrKrnl, int mskThresh, int morphRadius, bool debugDraw = false);

	Mat fetchFloorApartMask(Mat dmat, bool debugDraw = false);

	//计算物理尺度宽度X-mat
	//centerX： 中心点x坐标，默认值0，即mat左边缘
	Mat calcWidthMap(Mat dmat, int centerX = 0, bool debugDraw = false);
	Mat fetchWidthMap(Mat dmat, int centerX = 0, bool debugDraw = false);

	cv::Mat calcHeightMap0(Mat dmat, bool debugDraw = false);

	//计算物理尺度高度Mat：
	//以mat下边缘为0高度，越往上越高
	Mat calcHeightMap1(Mat dmat, bool debugDraw = false);
	//与 calcHeightMap 区别在于： 不是每次算， 只有 dmat 内容变了，才更新
	Mat fetchHeightMap(Mat dmat, bool debugDraw = false);

	//截取物理尺度高度<limitMs(毫米)像素点做mask
	Mat getHeightMask(Mat dmat, int limitMs = 2500);

	//直方图方式寻找大墙面深度值（大峰值）
	//适用于正对墙面的用例
	//RETURN: -1 表示没找到足够大的墙面；正值为墙面深度值
	int getWallDepth(Mat &dmat);
	int fetchWallDepth(Mat &dmat);

	void drawOneSkeleton(Mat &img, CapgSkeleton &sk);
	void drawSkeletons(Mat &img, const vector<CapgSkeleton> &sklts, int skltIdx);

	//@Overload HumanFg-vec 版
	void drawSkeletons(Mat &img, vector<HumanFg> &humObjVec, int skltIdx);


	void eraseNonHumanContours(vector<vector<Point> > &contours);
	bool isHumanContour(const vector<Point> &cont);
	bool isHumanMask(const Mat &msk, int fgPxCntThresh = 1000);

	//没用 normalize， 因为会导致不同帧灰度比不同；用 convertTo -> 1. * UCHAR_MAX / MAX_VALID_DEPTH
	Mat getDmatGrayscale(const Mat &dmat);
	Mat fetchDmatGrayscale(const Mat &dmat);
	//上半身、脚部分别canny，求和，得到人体较清晰的轮廓。白色描边
	Mat getHumanEdge(const Mat &dmat, bool debugDraw = false);


	//@deprecated
	Mat distMap2contoursDebug(const Mat &dmat, bool debugDraw = false);

	//1. 对 distMap 二值化得到 contours； 2. 对 contours bbox 判断，得到人体区域
	//注：
	// 1. 若 debugDraw = true, 则 _debug_mat 必须传实参
	vector<vector<Point> > distMap2contours(const Mat &dmat, bool debugDraw = false, OutputArray _debug_mat = noArray());

	//@deprecated
	Mat dmat2TopDownViewDebug(const Mat &dmat, bool debugDraw = false);

	//dmat 砍掉下半屏, 转为 top-down-view, 膨胀, 根据bbox判断, 提取合适的轮廓
	//注:
	// 1. Z轴缩放比为定值： UCHAR_MAX/MAX_VALID_DEPTH, 即 top-down-view 图高度为 256
	// 2. debugDraw, dummy variable
	// 3. 若 debugDraw = true, 则 _debug_mat 必须传实参
	vector<vector<Point> > dmat2TopDownView(const Mat &dmat, double ratio = 1. * UCHAR_MAX / MAX_VALID_DEPTH, bool debugDraw = false, OutputArray _debug_mat = noArray());

	//【注】
	//1. 返回mat.type==ushort
	//2. 内部最后有row0填充零操作，防止无效区域干扰
	//3. squash用了 convertTo，float值round，非floor；所以反投影小心
	Mat dmat2tdview_core(const Mat &dmat, double ratio = 1. * UCHAR_MAX / MAX_VALID_DEPTH, bool debugDraw = false);

	//用正视图、俯视图两种 bbox 求交，判定人体轮廓位置
	//注：
	// 1. 若 debugDraw = true, 则 _debug_mat 必须传实参
	vector<Mat> findFgMasksUseBbox(Mat &dmat, /*bool usePre = false, */bool debugDraw = false, OutputArray _debug_mat = noArray());

	vector<Mat> trackingNoMove(Mat dmat, const vector<Mat> &prevFgMaskVec, const vector<Mat> &currFgMskVec, bool debugDraw = false);

	//若某mask中包含多个孤立连通区域，则将其打散为多个mask
	//e.g., 两人拉手时tracking成为一个mask，【XY视图】分离时mask也应打散；
	//人靠近（握持）某物体时，增长为一个mask，分离时，应打散，并根据bbox etc. 判定是否跟踪“某物”
	vector<Mat> separateMasksXYview(Mat dmat, vector<Mat> &inMaskVec, bool debugDraw = false);

	//类似separateMasksXYview，但变换视角为【XZ视图】
	vector<Mat> separateMasksXZview(Mat dmat, vector<Mat> &inMaskVec, bool debugDraw = false);


	//返回 contours[contIdx] 对应的top-down-view 上的 XZ-bbox
	Rect contour2XZbbox(Mat dmat, vector<vector<Point>> &contours, int contIdx);

#if CV_VERSION_MAJOR >= 3
	//尝试从findFgMasksUseBbox 剥离解耦
	//用 MOG2 背景减除，得到前景点， erode 得到比较大的前景区域
	//注：
	//1. 用到时序信息(history)！一次循环中尽量避免多次调用；如必须，置位 isNewFrame = false
	Mat seedUseBGS(Mat &dmat, bool isNewFrame = true, bool usePre = false, bool debugDraw = false);

	//用 opencv300 background-subtraction 方法提取运动物体（不必是人,e.g.:转椅）轮廓
	//1. bgs -> roi; 2. region-grow -> vector<Mat>; 3. bbox etc. 判定; 4. usePre 时序上，重心判定
	//
	vector<Mat> findFgMasksUseBGS(Mat &dmat, bool usePre = false, bool debugDraw = false, OutputArray _debug_mat = noArray());
#endif

	//在前一帧找到的前景mask范围内，前后帧深度值变化不大(diff < thresh)的像素点，作为新候选点。
	//返回新候选点mask
	Mat seedNoMove(Mat dmat, Mat mask, int thresh = 50);
	vector<Mat> seedNoMove(Mat dmat, vector<Mat> masks, int thresh = 50);

	//@Overload
	vector<Point> seedNoMove(Mat dmat, vector<Point> sdVec, int thresh = 50);

	vector<vector<Point>> seedNoMove(Mat dmat, vector<vector<Point>> sdVov, int thresh = 50);
		

	//返回不同灰度标记前景的mat
	Mat getHumansMask(vector<Mat> masks, Size sz);

	//用全局的 vector<HumanFg> 画一个彩色 mask mat:
	Mat getHumansMask(Mat dmat, const vector<HumanFg> &humVec);

	//应有粘性跟踪能力，多人场景下，uid不应突变
	//跟踪(更新)策略为：
	//某蒙板与当前某 humObj蒙板求交区域深度变化均值较小
	void getHumanObjVec(Mat &dmat, vector<Mat> fgMasks, vector<HumanFg> &outHumVec);

	class HumanFg
	{
	public:
		HumanFg(const Mat &dmat_, Mat currMask_, int humId);

		//若成功， 更新 this各项； 否则，返回false，用于表示跟丢，准备删除此对象
		bool updateDmatAndMask(const Mat &dmat, const vector<Mat> &fgMaskVec, vector<bool> &mskUsedFlags);

		Scalar getColor();

		Mat getCurrMask();

		void setCurrMask(Mat newMask);//setCurrMask

		void setPrevMask(Mat newMask);//setPrevMask

		int getHumId();

#ifdef CV_VERSION_EPOCH
		//用内部 _dmat, _currMask 做预测
		void calcSkeleton();
#endif //CV_VERSION_EPOCH

		CapgSkeleton getSkeleton();

		//---------------

	protected:
	private:
		Mat _dmat;

		Mat _prevMask;
		Point _prevCenter;

		Mat _currMask;
		int _currMaskArea;
		Point _currCenter;

		Scalar _humColor; //颜色不能作为id，无法确保唯一性
		//此对象唯一id序号，从资源池中取、放
		int _humId;
		//vec-heap 做资源池
		//static vector<int> idResPool;
		
		CapgSkeleton _sklt;

		Point getContMassCenter(vector<Point> cont_){
			//Point3i mc;
			
			Point mc2i;
			Moments mu = moments(cont_);

			//防止面积为零。之前 simpleRegionGrow 确保前景mask较大，此处保险起见：
			if (abs(mu.m00) < 1e-8) //area is zero
				mc2i = cont_[0];
			else
				mc2i = Point(mu.m10 / mu.m00, mu.m01 / mu.m00);

			// 			mc.x = mc2i.x;
			// 			mc.y = mc2i.y;
			// 			mc.z=
			return mc2i;
		}//getContMassCenter

		//fgMask必须有且仅有一个连通前景
		Point getContMassCenter(Mat fgMask){
			vector<vector<Point> > contours;
			findContours(fgMask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
			//有且仅有一个连通前景：
			//CV_Assert(contours.size() == 1);

			if (contours.size() > 0)
				return getContMassCenter(contours[0]);
			else
				return Point(-1, -1);
		}//getContMassCenter

	};//class HumanFg

	//假定origMasks里没有全黑mat，否则出错！
	vector<Mat> bboxFilter(Mat dmat, const vector<Mat> &origMasks);

	//假定 mask 不是全黑，否则出错！假定 mask中有唯一一个连通区域
	bool bboxIsHuman(Mat dmat, Mat mask);
// 	bool bboxIsHuman(Mat dmat, vector<Point> cont);

	//radius: kernel size is (2*radius+1)^2
	//shape: default MORPH_RECT
	Mat getMorphKrnl(int radius = 1, int shape = MORPH_RECT);//getMorphKrnl

	//缓存上一帧深度数据，以便某些需要时序信息的算法使用
	//主线程中一次循环更新一次：
	void setPrevDmat(Mat currDmat);
	void initPrevDmat(Mat currDmat);
	Mat getPrevDmat();

	//@param fgMsk: 前景蒙板， 白色(uchar_max)为有效区域
	CapgSkeleton calcSkeleton(const Mat &dmat, const Mat &fgMsk);


#pragma region //孙国飞头部种子点

	//单例模式，返回单例指针
	sgf::segment* loadSeedHeadConf(const char *confFn, const char *templFn);

	//孙国飞实现的头部种子点方法
	vector<Point> seedHead(const Mat &dmat, bool debugDraw = false);

#pragma endregion //孙国飞头部种子点


}//namespace zc

using zc::HumanFg;
