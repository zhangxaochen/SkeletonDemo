#ifndef _SIMPLE_SILHOUETTE_
#define _SIMPLE_SILHOUETTE_

#include <iostream>
#include <vector>
#include <queue>
#include <opencv2/opencv.hpp>
#include <ctime>

#include "CapgSkeleton.h"
#include "bodyPartRecognizer.h"

#include "./sgf_seed/sgf_segment.h"
//using namespace sgf;

using namespace std;
using namespace cv;

const int QVGA_WIDTH = 320,
	QVGA_HEIGHT = 240,
	MAX_VALID_DEPTH = 10000;

//zhangxaochen: 假定 320*240 数据对应的焦距
#define XTION_FOCAL_XY 300

#define MIN_VALID_HEIGHT_PHYSICAL_SCALE 600
#define MIN_VALID_HW_RATIO 1.5 //height-width ratio

//static cv::RNG rng;

extern const int thickLimitDefault;// = 1500;
extern int thickLimit;// = thickLimitDefault; //毫米

//一些调试颜色：
extern Scalar cwhite, cred, cgreen, cblue, cyellow;


namespace zc{
	class HumanObj;

#ifdef CV_VERSION_EPOCH
//#if CV_VERSION_MAJOR < 3

//lincccc's code below:
	//BPRecognizer* getBprAndLoadFeature();
	//BPRecognizer* getBprAndLoadFeature(const char *featurePath = nullptr);
	BPRecognizer* getBprAndLoadFeature(const string &featurePath);
#endif //CV_VERSION_EPOCH

	//存 mat-vec 到 FileStorage
	void saveVideo(const vector<Mat> &matVec, const char *fname);

	//读 FileStorage
	//@return mat-vec
	vector<Mat> loadVideo(const char *fname);

	Mat simpleMask(const Mat &curMat, bool debugDraw = false);

	Point seedSimple(Mat dmat, int *outVeryDepth = 0, bool debugDraw = false);

	//尝试从 findFgMasksUseBbox 剥离解耦, √
	//用正视图、俯视图两种 bbox 求交，判定人体轮廓位置
	//注：
	// 1. 若 debugDraw = true, 则 _debug_mat 必须传实参
	// 2. 返回值 vector<vector<Point>> 实际就是挑选过的 contours
	vector<vector<Point>> seedUseBboxXyXz(Mat dmat, bool debugDraw = false, OutputArray _debug_mat = noArray());

	//用正视图+头部种子点联合判定人体轮廓位置，
	//@return 正视图contours
	vector<vector<Point>> seedUseHeadAndBodyCont(Mat dmat, bool debugDraw = false, OutputArray _debug_mat = noArray());

	//@note 2015年7月12日00:15:52	【未完成】
	//@brief 用头部种子点+MOG2运动联合判定可信种子点
	//@return N个离散的头部种子点
	vector<Point> seedUseMovingHead(Mat dmat, bool isNewFrame = true, bool debugDraw = false, OutputArray _debug_mat = noArray());

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

	//若镜头倾斜俯视怎么办？
	//@return 用于去除地板的mask
	Mat getFloorApartMask(Mat dmat, Mat flrKrnl, int mskThresh, int morphRadius, bool debugDraw = false);

	//三个默认参数(flrKrnl, mskThresh, morphRadius) 版：
	//flrKrnl = { 1, 1, 1, -1, -1, -1 };
	//mskThresh = 100;
	//morphRadius = 3;
	Mat getFloorApartMask(Mat dmat, bool debugDraw = false);


	Mat fetchFloorApartMask(Mat dmat, bool debugDraw = false);

	//计算物理尺度宽度X-mat
	//centerX： 中心点x坐标，默认值0，即mat左边缘
	Mat calcWidthMap(Mat dmat, int centerX = 0, bool debugDraw = false);
	Mat fetchWidthMap(Mat dmat, int centerX = 0, bool debugDraw = false);

	//计算物理尺度高度Mat：
	//以mat下边缘为0高度，逐【行】越往上越高，同一行越深“高”越大，故错！
	//不过暂时能用，实际用于条件过滤： (过高 || 过深)
	cv::Mat calcHeightMap0(Mat dmat, bool debugDraw = false);
	//与 calcHeightMap 区别在于： 不是每次算， 只有 dmat 内容变了，才更新
	Mat fetchHeightMap0(Mat dmat, bool debugDraw = false);

	//计算物理尺度高度Mat：
	//之前的实现错误，改为中轴线为0高度，最后统一偏移 +hmin
	Mat calcHeightMap1(Mat dmat, bool debugDraw = false);

	//截取物理尺度高度<limitMs(毫米)像素点做mask
	Mat getFakeHeightMask(Mat dmat, int limitMs = 2500);

	//直方图方式寻找大墙面深度值（大峰值）
	//适用于正对墙面的用例
	//RETURN: -1 表示没找到足够大的墙面；正值为墙面深度值
	int getWallDepth(Mat &dmat);
	int fetchWallDepth(Mat &dmat);

	void drawOneSkeleton(Mat &img, CapgSkeleton &sk);
	void drawSkeletons(Mat &img, const vector<CapgSkeleton> &sklts, int skltIdx);

	//@Overload HumanObj-vec 版
	void drawSkeletons(Mat &img, vector<HumanObj> &humObjVec, int skltIdx);

	//@deprecated, no use
	//@param contours 引用， 会被修改
	void eraseNonHumanContours(vector<vector<Point> > &contours);

	//@return cont.size() > 222;
	bool isHumanContour(const vector<Point> &cont);

	//目前仅通过蒙板前景像素点个数判断
	//@return countNonZero(msk==UCHAR_MAX) > fgPxCntThresh
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

	//@brief 重构了distMap2contours，另外debugDraw彩色
	//@return 正视图中 bboxIsHuman 的 contours
	vector<vector<Point> > distMap2contours_new(const Mat &dmat, bool debugDraw = false, OutputArray _debug_mat = noArray());

	//distMap2contours_new 的别名
	extern vector<vector<Point> >(*getHumanContoursXY)(const Mat &, bool, OutputArray);

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

	//@deprecated 因为仅等价于 seedUseBboxXyXz + simpleRegionGrow
	//用正视图、俯视图两种 bbox 求交，判定人体轮廓位置
	//注：
	// 1. 若 debugDraw = true, 则 _debug_mat 必须传实参
	vector<Mat> findFgMasksUseBbox(Mat &dmat, /*bool usePre = false, */bool debugDraw = false, OutputArray _debug_mat = noArray());

	//@deprecated 未实现，因为将仅等价于 seedUseHeadAndBodyCont + simpleRegionGrow
	vector<Mat> findFgMasksUseHeadAndBodyCont(Mat &dmat, bool debugDraw = false);

	//@brief 
	//@param prevFgMaskVec: 前一帧的最终 fgMaskVec
	//@param currFgMskVec: 当前帧步骤(B)找到的初始前景(可能是新人,可能不)
	//@param 新的 fgMaskVec
	//@param moveMaskMat, 运动前景蒙板，目前用 MOG2
	vector<Mat> trackingNoMove(Mat dmat, const vector<Mat> &prevFgMaskVec, const vector<Mat> &currFgMskVec, int noMoveThresh = 55, Mat moveMaskMat = Mat(), bool debugDraw = false);

	//若某mask中包含多个孤立连通区域，则将其打散为多个mask
	//e.g., 两人拉手时tracking成为一个mask，【XY视图】分离时mask也应打散；
	//人靠近（握持）某物体时，增长为一个mask，分离时，应打散，并根据bbox etc. 判定是否跟踪“某物”
	vector<Mat> separateMasksXYview(Mat dmat, vector<Mat> &inMaskVec, bool debugDraw = false);

	//类似separateMasksXYview，但变换视角为【XZ视图】
	vector<Mat> separateMasksXZview(Mat dmat, vector<Mat> &inMaskVec, bool debugDraw = false);


	//返回 contours[contIdx] 对应的top-down-view 上的 XZ-bbox
	Rect contour2XZbbox(Mat dmat, vector<vector<Point>> &contours, int contIdx);

	vector<Mat> separateMasksMoving(Mat dmat, vector<Mat> &inMaskVec, bool debugDraw = false);

#if CV_VERSION_MAJOR >= 3
	//@deprecated 改用 seedBgsMOG2
	//尝试从findFgMasksUseBbox 剥离解耦
	//用 MOG2 背景减除，得到前景点， erode 得到比较大的前景区域
	//注：
	//1. 用到时序信息(history)！一次循环中尽量避免多次调用；如必须，置位 isNewFrame = false
	Mat seedUseBGS(Mat &dmat, bool isNewFrame = true, bool usePre = false, bool debugDraw = false);

	//@deprecated 因为 isNewFrame传参很可能用错，设计更改为主循环中仅做一次MOG2
	//@brief seedUseBGS 重写
	//@param erodeRadius: if >0, erode; if ==0, don't erode
	//@note 1. 用到时序信息(history)！一次循环中尽量避免多次调用；如必须，置位 isNewFrame = false
	Mat seedBgsMOG2(const Mat &dmat, bool isNewFrame = true, int erodeRadius = 2, bool debugDraw = false, OutputArray _debug_mat = noArray());

	//用 opencv300 background-subtraction 方法提取运动物体（不必是人,e.g.:转椅）轮廓
	//1. bgs -> roi; 2. region-grow -> vector<Mat>; 3. bbox etc. 判定; 4. usePre 时序上，重心判定
	//
	vector<Mat> findFgMasksUseBGS(Mat &dmat, bool usePre = false, bool debugDraw = false, OutputArray _debug_mat = noArray());
#endif

	//在【前一帧】找到的前景mask范围内，前后帧深度值变化不大(diff < thresh)的像素点，作为新候选点。
	//@return 新候选点mask
	Mat seedNoMove(Mat dmat, Mat mask, int thresh = 50);
	vector<Mat> seedNoMove(Mat dmat, vector<Mat> maskVec, int thresh = 50);

	//@Overload
	vector<Point> seedNoMove(Mat dmat, vector<Point> sdVec, int thresh = 50);

	vector<vector<Point>> seedNoMove(Mat dmat, vector<vector<Point>> sdVov, int thresh = 50);

	Mat testMOG2Func(Mat dmat, int history = 100, double varThresh = 1, double learnRate = -1);

	//@deprecated	---------------2015年7月10日23:17:59 有问题！ 帧循环过程中不应多次调用！设计问题。不应static pMOG2
	//@brief 用 (MOG2运动检测 | seedNoMove不动点检测)，产生一个有效前景蒙板
	//@param prevFgMaskVec 【前一帧】找到的前景们; from seedNoMove
	//@param noMoveThresh: 不动阈值(mm), 默认50mm; from seedNoMove
	//@param history: opencv-MOG2 相关
	//@param varThresh: opencv-MOG2 相关
	//@param learnRate: opencv-MOG2 相关
	Mat maskMoveAndNoMove(Mat dmat, vector<Mat> prevFgMaskVec, int noMoveThresh = 50, int history = 100, double varThresh = 1, double learnRate = -1, bool debugDraw = false, OutputArray _debug_mat = noArray());

	//@brief moveMask 传入运动检测得到的FG; 目前暂用 MOG2结果
	//实际上应该每个mask单独长成一个new-mask, 此处简化测试, N个长成一个:
	Mat maskMoveAndNoMove(Mat dmat, Mat moveMask, vector<Mat> prevFgMaskVec, int noMoveThresh = 50, bool debugDraw = false, OutputArray _debug_mat = noArray());

	//@debugging
	//@brief 返回不同灰度标记前景的mat
	Mat getHumansMask(vector<Mat> masks, Size sz);

	//@debugging
	//@brief 用全局的 vector<HumanObj> 画一个彩色 mask mat:
	Mat getHumansMask(Mat dmat, const vector<HumanObj> &humVec);

	//@brief 类似getHumansMask， 但转到tdview， 彩色绘制：
	Mat getHumansMask2tdview(Mat dmat, const vector<HumanObj> &humVec);

	vector<Mat> humVec2tdviewVec(Mat dmat, const vector<HumanObj> &humVec);

	//应有粘性跟踪能力，多人场景下，uid不应突变
	//跟踪(更新)策略为：
	//某蒙板与当前某 humObj蒙板求交区域深度变化均值较小
	void getHumanObjVec(Mat &dmat, vector<Mat> fgMasks, vector<HumanObj> &outHumVec);

	//@brief 输入单帧原始深度图， 得到最终 vec-mat-mask. process/run-a-frame
	vector<Mat> getFgMaskVec(Mat &dmat, int fid, bool debugDraw = false);

	class HumanObj
	{
	public:
		HumanObj(const Mat &dmat_, Mat currMask_, int humId);

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

	};//class HumanObj

	//@deprecated
	//假定origMasks里没有全黑mat，否则出错！
	vector<Mat> bboxFilter(Mat dmat, const vector<Mat> &origMasks);

	//假定 mask 不是全黑，否则出错！mask中连通区域不必唯一
	//用到 mask 说明必然用到深度信息, e.g., dmax-dmin < thresh
	bool fgMskIsHuman(Mat dmat, Mat mask);
// 	bool fgMskIsHuman(Mat dmat, vector<Point> cont);

	//用cont，说明仅使用XY信息，无深度信息, e.g., bbox
	//@return 目前等价于bboxIsHuman
	bool contIsHuman(Size matSize, vector<Point> cont);

	//仅使用XY信息，无深度信息, 
	//@return pxHeightEnough && narrowEnough && feetLowEnough;
	bool bboxIsHuman(Size matSize, Rect bbox);

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
	sgf::segment* loadSeedHeadConf(const char *confFn = "./sgf_seed/config.txt", const char *templFn = "./sgf_seed/headtemplate.bmp");

	//孙国飞实现的头部种子点方法，的包装方法
	vector<Point> seedHeadTempMatch(const Mat &dmat, bool debugDraw = false);

	//孙国飞获取头部大小的包装方法
	vector<double> getHeadSizes();

#pragma endregion //孙国飞头部种子点

#pragma region //从 opencv300 拷贝 boundingRect 
	cv::Rect boundingRect(InputArray array);
#pragma endregion //从 opencv300 拷贝 boundingRect 

}//namespace zc
using zc::HumanObj;

//从 opencv300 拷贝 Point_.operator/, 兼容 opencv2.x
namespace zc{
#ifdef CV_VERSION_EPOCH //if opencv2.x
	//@types.hpp L1165
	template<typename _Tp> static inline
		Point_<_Tp>& operator /= (Point_<_Tp>& a, int b)
	{
		a.x = saturate_cast<_Tp>(a.x / b);
		a.y = saturate_cast<_Tp>(a.y / b);
		return a;
	}

	template<typename _Tp> static inline
		Point_<_Tp>& operator /= (Point_<_Tp>& a, float b)
	{
		a.x = saturate_cast<_Tp>(a.x / b);
		a.y = saturate_cast<_Tp>(a.y / b);
		return a;
	}

	template<typename _Tp> static inline
		Point_<_Tp>& operator /= (Point_<_Tp>& a, double b)
	{
		a.x = saturate_cast<_Tp>(a.x / b);
		a.y = saturate_cast<_Tp>(a.y / b);
		return a;
	}

	//@types.hpp L1275
	template<typename _Tp> static inline
		Point_<_Tp> operator / (const Point_<_Tp>& a, int b)
	{
		Point_<_Tp> tmp(a);
		tmp /= b;
		return tmp;
	}

	template<typename _Tp> static inline
		Point_<_Tp> operator / (const Point_<_Tp>& a, float b)
	{
		Point_<_Tp> tmp(a);
		tmp /= b;
		return tmp;
	}

	template<typename _Tp> static inline
		Point_<_Tp> operator / (const Point_<_Tp>& a, double b)
	{
		Point_<_Tp> tmp(a);
		tmp /= b;
		return tmp;
	}
#elif CV_VERSION_MAJOR >= 3 //if opencv3
	//do-nothing
#endif //CV_VERSION_EPOCH
}//namespace zc-从 opencv300 拷贝 Point_.operator/, 兼容 opencv2.x



//---------------测试代码放这里
namespace zc{
	Mat getLaplaceEdgeKrnl(size_t krnlSz = 5);

	//@brief 以 krnlSz 为核， getLaplaceEdgeKrnl， 然后卷积
	//然并卵！
	Mat getLaplaceEdgeFilter2D(const Mat &dmat, size_t krnlSz = 5);

	//@brief 根据krnlSz邻域内非零点个数 N 填充空洞, 若 N > countThresh, 填充邻域均值
	//@param hasSideEffect 逐行扫描时, 前面填充的像素是否会影响后面像素. 若 true, 则最终必然再无非零像素
	//@return filled-dmat
	//@note 效率： 1. 默认参数, qvga~8ms; 2. hasSideEffect==true, qvga~11ms; 3. krnlSz=3, countThresh=2 几乎无效率影响
	Mat holeFillNbor(const Mat &dmat, bool hasSideEffect = false, size_t krnlSz = 5, int countThresh = 3);

#pragma region //接要求, 自己实现帧差法背景减除MyBGSubtractor:

	//用history长度的历史窗口（不含当前帧）做平滑，与当前帧做差，若 >diffThresh，算作前景
	class MyBGSubtractor
	{
	public:
		MyBGSubtractor();
		MyBGSubtractor(int history, int diffThresh);
		~MyBGSubtractor();


		Mat apply(const Mat &currFrame);

		Mat getBgMat();

	private:
		//@brief frame 入队， 然后重新计算avg-frame做背景
		void addToHistory(const Mat &frame32f);

		queue<Mat> _historyFrames;

		//@brief _historyFrames 长度上限，即capacity
		int _history;
		int _diffThresh;

		//@brief cv8u, or 16u?
		//Mat _bgMat;

		//@brief cv32f
		Mat _bgMat32f;

		//@brief cv8u
		Mat _fgMask;
	};//MyBGSubtractor
#pragma endregion //接要求, 自己实现帧差法背景减除MyBGSubtractor

	//@brief 对每个move-mask, X轴向上Y值统计直方图
	//@param humVec: vec-HumanObj, 仅用HumanObj的颜色信息
	//@param moveMaskVec： vec-mask-mat, 每个HumanObj对应区域内的“运动区域”mask
	//@return 彩色 histo-mat， N个直方图画在同一个mat上
	Mat getHumVecMaskHisto(Mat dmat, vector<HumanObj> humVec, vector<Mat> moveMaskVec, bool debugDraw = false);


	//@brief 对 maskMat 统计X轴向上Y值统计直方图, 用color绘制
	//@return 一个彩色 histo-mat
	Mat getMaskXyHisto(Mat dmat, Mat maskMat, Scalar color, bool debugDraw = false);

	//@brief 各个像素历史最大深度值
	//@return max-depth-mat
	Mat& getMaxDmat(Mat &dmat, bool debugDraw = false);

	//@brief 使用新一帧【更新】各个像素历史最大深度值
	//@return max-depth-mat
	Mat updateMaxDmat(Mat &dmat, bool debugDraw = false);

	//@brief 利用maxDmat，稳定背景做diff，扣除鬼影；利用maxDmat-MOG，扣除伪鬼影
	Mat  getMaxDepthBgMask(Mat dmat, bool debugDraw = false);

	vector<Mat> separateMasksMovingHead(Mat dmat, vector<Mat> &inMaskVec, Mat &mogMask, bool debugDraw = false);

	//@brief 孙国飞V形分割方案的包装方法，接口改为 inMaskVec，非单一mask
	vector<Mat> separateMasksContValley(Mat dmat, vector<Mat> &inMaskVec, bool debugDraw = false);
}//zc

using zc::MyBGSubtractor;

#endif //_SIMPLE_SILHOUETTE_
