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

//zhangxaochen: �ٶ� 320*240 ���ݶ�Ӧ�Ľ���
#define XTION_FOCAL_XY 300

#define MIN_VALID_HEIGHT_PHYSICAL_SCALE 600
#define MIN_VALID_HW_RATIO 1.5 //height-width ratio

//static cv::RNG rng;

extern const int thickLimitDefault;// = 1500;
extern int thickLimit;// = thickLimitDefault; //����

//һЩ������ɫ��
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

	//�� mat-vec �� FileStorage
	void saveVideo(const vector<Mat> &matVec, const char *fname);

	//�� FileStorage
	//@return mat-vec
	vector<Mat> loadVideo(const char *fname);

	Mat simpleMask(const Mat &curMat, bool debugDraw = false);

	Point seedSimple(Mat dmat, int *outVeryDepth = 0, bool debugDraw = false);

	//���Դ� findFgMasksUseBbox �������, ��
	//������ͼ������ͼ���� bbox �󽻣��ж���������λ��
	//ע��
	// 1. �� debugDraw = true, �� _debug_mat ���봫ʵ��
	// 2. ����ֵ vector<vector<Point>> ʵ�ʾ�����ѡ���� contours
	vector<vector<Point>> seedUseBboxXyXz(Mat dmat, bool debugDraw = false, OutputArray _debug_mat = noArray());

	//������ͼ+ͷ�����ӵ������ж���������λ�ã�
	//@return ����ͼcontours
	vector<vector<Point>> seedUseHeadAndBodyCont(Mat dmat, bool debugDraw = false, OutputArray _debug_mat = noArray());

	//@note 2015��7��12��00:15:52	��δ��ɡ�
	//@brief ��ͷ�����ӵ�+MOG2�˶������ж��������ӵ�
	//@return N����ɢ��ͷ�����ӵ�
	vector<Point> seedUseMovingHead(Mat dmat, bool isNewFrame = true, bool debugDraw = false, OutputArray _debug_mat = noArray());

	//@deprecated, ���岻�������ã���ʵ����������ȥ���� & findFgMasksUseBbox
	//1. �ұ�����ǽ�棻
	//2. ��ûǽ��˵�������տ�������߶��ж��޳�(<2500mm)
	//3. ʣ����ߵ������ӵ�
	vector<Mat> findFgMasksUseWallAndHeight(Mat dmat, /*bool usePre = false, */bool debugDraw = false);

	//���ݸ߶�or��ƽ��������������
	Mat getBgMskUseWallAndHeight(Mat dmat);
	Mat fetchBgMskUseWallAndHeight(Mat dmat);


	//Mat simpleRegionGrow(const Mat &dmat, Point seed, int thresh, bool debugDraw = false);

	//Point seed, Rect roi
	Mat _simpleRegionGrow(const Mat &dmat, Point seed, int thresh, const Rect roi, bool debugDraw = false);

	//vector<Point> seeds, Rect roi
	Mat _simpleRegionGrow(const Mat &dmat, vector<Point> seeds, int thresh, const Rect roi, bool debugDraw = false);

	//mask Ϊ�Ƿ���Ч���� flag�� ��ĳ���غ�(0), ��˴���ֹ��ȥ����
	Mat _simpleRegionGrow_core_pt2mat(const Mat &dmat, Point seed, int thresh, const Mat &validMask, bool debugDraw = false);

	//_simpleRegionGrow ���ĺ�����
	Mat _simpleRegionGrow_core_mat2mat(const Mat &dmat, Mat sdsMat, int thresh, const Mat &validMask, bool debugDraw = false);
	Mat _simpleRegionGrow_core_vec2mat(const Mat &dmat, vector<Point> sdsVec, int thresh, const Mat &validMask, bool debugDraw = false);

	//�� seedsMask -> vector<Point> seeds���ٵ�������
	Mat _simpleRegionGrow(const Mat &dmat, Mat seedsMask, int thresh, const Mat &mask, bool debugDraw = false);

	//N�����ӵ㣬 
	//1. getMultiMasks = false, ����������Ϊ���mask
	//2. getMultiMasks = true, ����Ϊһ��mask, �����Ƿ�����Ƭ������λ��[0]��
	vector<Mat> simpleRegionGrow(const Mat &dmat, vector<Point> seedsVec, int thresh, const Mat &mask, bool getMultiMasks = false, bool debugDraw = false);

	//�� sdsMat -> vector<Point> seeds, �ٵ�������
	vector<Mat> simpleRegionGrow(const Mat &dmat, Mat sdsMat, int thresh, const Mat &mask, bool getMultiMasks = false, bool debugDraw = false);

	//N�����ӵ�vector�� ����ΪN��mask
	vector<Mat> simpleRegionGrow(const Mat &dmat, vector<vector<Point>> seedsVecOfVec, int thresh, const Mat &mask, bool debugDraw = false);

	//N�����ӵ�mask�� ����ΪN��mask
	vector<Mat> simpleRegionGrow(const Mat &dmat, vector<Mat> sdMats, int thresh, const Mat &mask, bool debugDraw = false);
	
	//---------------@deprecated, ����˼·���������ض����������ƣ�
// 	vector<Mat> simpleRegionGrow(Mat dmat, Mat seedsMask, int thresh, Mat mask);

	//����һ�� mask-mat; type: 8uc1, �ð�ɫ UCHAR_MAX ���ʾ��Ч��
	Mat pts2maskMat(const vector<Point> pts, Size matSz);

	//maskMat: type 8uc1, ��ɫ UCHAR_MAX ���ʾ��Ч��
	vector<Point> maskMat2pts(Mat maskMat, int step = 1);

	//����ͷ��б������ô�죿
	//@return ����ȥ���ذ��mask
	Mat getFloorApartMask(Mat dmat, Mat flrKrnl, int mskThresh, int morphRadius, bool debugDraw = false);

	//����Ĭ�ϲ���(flrKrnl, mskThresh, morphRadius) �棺
	//flrKrnl = { 1, 1, 1, -1, -1, -1 };
	//mskThresh = 100;
	//morphRadius = 3;
	Mat getFloorApartMask(Mat dmat, bool debugDraw = false);


	Mat fetchFloorApartMask(Mat dmat, bool debugDraw = false);

	//��������߶ȿ��X-mat
	//centerX�� ���ĵ�x���꣬Ĭ��ֵ0����mat���Ե
	Mat calcWidthMap(Mat dmat, int centerX = 0, bool debugDraw = false);
	Mat fetchWidthMap(Mat dmat, int centerX = 0, bool debugDraw = false);

	//��������߶ȸ߶�Mat��
	//��mat�±�ԵΪ0�߶ȣ����С�Խ����Խ�ߣ�ͬһ��Խ��ߡ�Խ�󣬹ʴ�
	//������ʱ���ã�ʵ�������������ˣ� (���� || ����)
	cv::Mat calcHeightMap0(Mat dmat, bool debugDraw = false);
	//�� calcHeightMap �������ڣ� ����ÿ���㣬 ֻ�� dmat ���ݱ��ˣ��Ÿ���
	Mat fetchHeightMap0(Mat dmat, bool debugDraw = false);

	//��������߶ȸ߶�Mat��
	//֮ǰ��ʵ�ִ��󣬸�Ϊ������Ϊ0�߶ȣ����ͳһƫ�� +hmin
	Mat calcHeightMap1(Mat dmat, bool debugDraw = false);

	//��ȡ����߶ȸ߶�<limitMs(����)���ص���mask
	Mat getFakeHeightMask(Mat dmat, int limitMs = 2500);

	//ֱ��ͼ��ʽѰ�Ҵ�ǽ�����ֵ�����ֵ��
	//����������ǽ�������
	//RETURN: -1 ��ʾû�ҵ��㹻���ǽ�棻��ֵΪǽ�����ֵ
	int getWallDepth(Mat &dmat);
	int fetchWallDepth(Mat &dmat);

	void drawOneSkeleton(Mat &img, CapgSkeleton &sk);
	void drawSkeletons(Mat &img, const vector<CapgSkeleton> &sklts, int skltIdx);

	//@Overload HumanObj-vec ��
	void drawSkeletons(Mat &img, vector<HumanObj> &humObjVec, int skltIdx);

	//@deprecated, no use
	//@param contours ���ã� �ᱻ�޸�
	void eraseNonHumanContours(vector<vector<Point> > &contours);

	//@return cont.size() > 222;
	bool isHumanContour(const vector<Point> &cont);

	//Ŀǰ��ͨ���ɰ�ǰ�����ص�����ж�
	//@return countNonZero(msk==UCHAR_MAX) > fgPxCntThresh
	bool isHumanMask(const Mat &msk, int fgPxCntThresh = 1000);

	//û�� normalize�� ��Ϊ�ᵼ�²�ͬ֡�ҶȱȲ�ͬ���� convertTo -> 1. * UCHAR_MAX / MAX_VALID_DEPTH
	Mat getDmatGrayscale(const Mat &dmat);
	Mat fetchDmatGrayscale(const Mat &dmat);
	//�ϰ����Ų��ֱ�canny����ͣ��õ��������������������ɫ���
	Mat getHumanEdge(const Mat &dmat, bool debugDraw = false);


	//@deprecated
	Mat distMap2contoursDebug(const Mat &dmat, bool debugDraw = false);

	//1. �� distMap ��ֵ���õ� contours�� 2. �� contours bbox �жϣ��õ���������
	//ע��
	// 1. �� debugDraw = true, �� _debug_mat ���봫ʵ��
	vector<vector<Point> > distMap2contours(const Mat &dmat, bool debugDraw = false, OutputArray _debug_mat = noArray());

	//@brief �ع���distMap2contours������debugDraw��ɫ
	//@return ����ͼ�� bboxIsHuman �� contours
	vector<vector<Point> > distMap2contours_new(const Mat &dmat, bool debugDraw = false, OutputArray _debug_mat = noArray());

	//distMap2contours_new �ı���
	extern vector<vector<Point> >(*getHumanContoursXY)(const Mat &, bool, OutputArray);

	//@deprecated
	Mat dmat2TopDownViewDebug(const Mat &dmat, bool debugDraw = false);

	//dmat �����°���, תΪ top-down-view, ����, ����bbox�ж�, ��ȡ���ʵ�����
	//ע:
	// 1. Z�����ű�Ϊ��ֵ�� UCHAR_MAX/MAX_VALID_DEPTH, �� top-down-view ͼ�߶�Ϊ 256
	// 2. debugDraw, dummy variable
	// 3. �� debugDraw = true, �� _debug_mat ���봫ʵ��
	vector<vector<Point> > dmat2TopDownView(const Mat &dmat, double ratio = 1. * UCHAR_MAX / MAX_VALID_DEPTH, bool debugDraw = false, OutputArray _debug_mat = noArray());

	//��ע��
	//1. ����mat.type==ushort
	//2. �ڲ������row0������������ֹ��Ч�������
	//3. squash���� convertTo��floatֵround����floor�����Է�ͶӰС��
	Mat dmat2tdview_core(const Mat &dmat, double ratio = 1. * UCHAR_MAX / MAX_VALID_DEPTH, bool debugDraw = false);

	//@deprecated ��Ϊ���ȼ��� seedUseBboxXyXz + simpleRegionGrow
	//������ͼ������ͼ���� bbox �󽻣��ж���������λ��
	//ע��
	// 1. �� debugDraw = true, �� _debug_mat ���봫ʵ��
	vector<Mat> findFgMasksUseBbox(Mat &dmat, /*bool usePre = false, */bool debugDraw = false, OutputArray _debug_mat = noArray());

	//@deprecated δʵ�֣���Ϊ�����ȼ��� seedUseHeadAndBodyCont + simpleRegionGrow
	vector<Mat> findFgMasksUseHeadAndBodyCont(Mat &dmat, bool debugDraw = false);

	//@brief 
	//@param prevFgMaskVec: ǰһ֡������ fgMaskVec
	//@param currFgMskVec: ��ǰ֡����(B)�ҵ��ĳ�ʼǰ��(����������,���ܲ�)
	//@param �µ� fgMaskVec
	//@param moveMaskMat, �˶�ǰ���ɰ壬Ŀǰ�� MOG2
	vector<Mat> trackingNoMove(Mat dmat, const vector<Mat> &prevFgMaskVec, const vector<Mat> &currFgMskVec, int noMoveThresh = 55, Mat moveMaskMat = Mat(), bool debugDraw = false);

	//��ĳmask�а������������ͨ���������ɢΪ���mask
	//e.g., ��������ʱtracking��Ϊһ��mask����XY��ͼ������ʱmaskҲӦ��ɢ��
	//�˿������ճ֣�ĳ����ʱ������Ϊһ��mask������ʱ��Ӧ��ɢ��������bbox etc. �ж��Ƿ���١�ĳ�
	vector<Mat> separateMasksXYview(Mat dmat, vector<Mat> &inMaskVec, bool debugDraw = false);

	//����separateMasksXYview�����任�ӽ�Ϊ��XZ��ͼ��
	vector<Mat> separateMasksXZview(Mat dmat, vector<Mat> &inMaskVec, bool debugDraw = false);


	//���� contours[contIdx] ��Ӧ��top-down-view �ϵ� XZ-bbox
	Rect contour2XZbbox(Mat dmat, vector<vector<Point>> &contours, int contIdx);

	vector<Mat> separateMasksMoving(Mat dmat, vector<Mat> &inMaskVec, bool debugDraw = false);

#if CV_VERSION_MAJOR >= 3
	//@deprecated ���� seedBgsMOG2
	//���Դ�findFgMasksUseBbox �������
	//�� MOG2 �����������õ�ǰ���㣬 erode �õ��Ƚϴ��ǰ������
	//ע��
	//1. �õ�ʱ����Ϣ(history)��һ��ѭ���о��������ε��ã�����룬��λ isNewFrame = false
	Mat seedUseBGS(Mat &dmat, bool isNewFrame = true, bool usePre = false, bool debugDraw = false);

	//@deprecated ��Ϊ isNewFrame���κܿ����ô���Ƹ���Ϊ��ѭ���н���һ��MOG2
	//@brief seedUseBGS ��д
	//@param erodeRadius: if >0, erode; if ==0, don't erode
	//@note 1. �õ�ʱ����Ϣ(history)��һ��ѭ���о��������ε��ã�����룬��λ isNewFrame = false
	Mat seedBgsMOG2(const Mat &dmat, bool isNewFrame = true, int erodeRadius = 2, bool debugDraw = false, OutputArray _debug_mat = noArray());

	//�� opencv300 background-subtraction ������ȡ�˶����壨��������,e.g.:ת�Σ�����
	//1. bgs -> roi; 2. region-grow -> vector<Mat>; 3. bbox etc. �ж�; 4. usePre ʱ���ϣ������ж�
	//
	vector<Mat> findFgMasksUseBGS(Mat &dmat, bool usePre = false, bool debugDraw = false, OutputArray _debug_mat = noArray());
#endif

	//�ڡ�ǰһ֡���ҵ���ǰ��mask��Χ�ڣ�ǰ��֡���ֵ�仯����(diff < thresh)�����ص㣬��Ϊ�º�ѡ�㡣
	//@return �º�ѡ��mask
	Mat seedNoMove(Mat dmat, Mat mask, int thresh = 50);
	vector<Mat> seedNoMove(Mat dmat, vector<Mat> maskVec, int thresh = 50);

	//@Overload
	vector<Point> seedNoMove(Mat dmat, vector<Point> sdVec, int thresh = 50);

	vector<vector<Point>> seedNoMove(Mat dmat, vector<vector<Point>> sdVov, int thresh = 50);

	Mat testMOG2Func(Mat dmat, int history = 100, double varThresh = 1, double learnRate = -1);

	//@deprecated	---------------2015��7��10��23:17:59 �����⣡ ֡ѭ�������в�Ӧ��ε��ã�������⡣��Ӧstatic pMOG2
	//@brief �� (MOG2�˶���� | seedNoMove��������)������һ����Чǰ���ɰ�
	//@param prevFgMaskVec ��ǰһ֡���ҵ���ǰ����; from seedNoMove
	//@param noMoveThresh: ������ֵ(mm), Ĭ��50mm; from seedNoMove
	//@param history: opencv-MOG2 ���
	//@param varThresh: opencv-MOG2 ���
	//@param learnRate: opencv-MOG2 ���
	Mat maskMoveAndNoMove(Mat dmat, vector<Mat> prevFgMaskVec, int noMoveThresh = 50, int history = 100, double varThresh = 1, double learnRate = -1, bool debugDraw = false, OutputArray _debug_mat = noArray());

	//@brief moveMask �����˶����õ���FG; Ŀǰ���� MOG2���
	//ʵ����Ӧ��ÿ��mask��������һ��new-mask, �˴��򻯲���, N������һ��:
	Mat maskMoveAndNoMove(Mat dmat, Mat moveMask, vector<Mat> prevFgMaskVec, int noMoveThresh = 50, bool debugDraw = false, OutputArray _debug_mat = noArray());

	//@debugging
	//@brief ���ز�ͬ�Ҷȱ��ǰ����mat
	Mat getHumansMask(vector<Mat> masks, Size sz);

	//@debugging
	//@brief ��ȫ�ֵ� vector<HumanObj> ��һ����ɫ mask mat:
	Mat getHumansMask(Mat dmat, const vector<HumanObj> &humVec);

	//@brief ����getHumansMask�� ��ת��tdview�� ��ɫ���ƣ�
	Mat getHumansMask2tdview(Mat dmat, const vector<HumanObj> &humVec);

	vector<Mat> humVec2tdviewVec(Mat dmat, const vector<HumanObj> &humVec);

	//Ӧ��ճ�Ը������������˳����£�uid��Ӧͻ��
	//����(����)����Ϊ��
	//ĳ�ɰ��뵱ǰĳ humObj�ɰ���������ȱ仯��ֵ��С
	void getHumanObjVec(Mat &dmat, vector<Mat> fgMasks, vector<HumanObj> &outHumVec);

	//@brief ���뵥֡ԭʼ���ͼ�� �õ����� vec-mat-mask. process/run-a-frame
	vector<Mat> getFgMaskVec(Mat &dmat, int fid, bool debugDraw = false);

	class HumanObj
	{
	public:
		HumanObj(const Mat &dmat_, Mat currMask_, int humId);

		//���ɹ��� ���� this��� ���򣬷���false�����ڱ�ʾ������׼��ɾ���˶���
		bool updateDmatAndMask(const Mat &dmat, const vector<Mat> &fgMaskVec, vector<bool> &mskUsedFlags);

		Scalar getColor();

		Mat getCurrMask();

		void setCurrMask(Mat newMask);//setCurrMask

		void setPrevMask(Mat newMask);//setPrevMask

		int getHumId();

#ifdef CV_VERSION_EPOCH
		//���ڲ� _dmat, _currMask ��Ԥ��
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

		Scalar _humColor; //��ɫ������Ϊid���޷�ȷ��Ψһ��
		//�˶���Ψһid��ţ�����Դ����ȡ����
		int _humId;
		//vec-heap ����Դ��
		//static vector<int> idResPool;
		
		CapgSkeleton _sklt;

		Point getContMassCenter(vector<Point> cont_){
			//Point3i mc;
			
			Point mc2i;
			Moments mu = moments(cont_);

			//��ֹ���Ϊ�㡣֮ǰ simpleRegionGrow ȷ��ǰ��mask�ϴ󣬴˴����������
			if (abs(mu.m00) < 1e-8) //area is zero
				mc2i = cont_[0];
			else
				mc2i = Point(mu.m10 / mu.m00, mu.m01 / mu.m00);

			// 			mc.x = mc2i.x;
			// 			mc.y = mc2i.y;
			// 			mc.z=
			return mc2i;
		}//getContMassCenter

		//fgMask�������ҽ���һ����ͨǰ��
		Point getContMassCenter(Mat fgMask){
			vector<vector<Point> > contours;
			findContours(fgMask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
			//���ҽ���һ����ͨǰ����
			//CV_Assert(contours.size() == 1);

			if (contours.size() > 0)
				return getContMassCenter(contours[0]);
			else
				return Point(-1, -1);
		}//getContMassCenter

	};//class HumanObj

	//@deprecated
	//�ٶ�origMasks��û��ȫ��mat���������
	vector<Mat> bboxFilter(Mat dmat, const vector<Mat> &origMasks);

	//�ٶ� mask ����ȫ�ڣ��������mask����ͨ���򲻱�Ψһ
	//�õ� mask ˵����Ȼ�õ������Ϣ, e.g., dmax-dmin < thresh
	bool fgMskIsHuman(Mat dmat, Mat mask);
// 	bool fgMskIsHuman(Mat dmat, vector<Point> cont);

	//��cont��˵����ʹ��XY��Ϣ���������Ϣ, e.g., bbox
	//@return Ŀǰ�ȼ���bboxIsHuman
	bool contIsHuman(Size matSize, vector<Point> cont);

	//��ʹ��XY��Ϣ���������Ϣ, 
	//@return pxHeightEnough && narrowEnough && feetLowEnough;
	bool bboxIsHuman(Size matSize, Rect bbox);

	//radius: kernel size is (2*radius+1)^2
	//shape: default MORPH_RECT
	Mat getMorphKrnl(int radius = 1, int shape = MORPH_RECT);//getMorphKrnl

	//������һ֡������ݣ��Ա�ĳЩ��Ҫʱ����Ϣ���㷨ʹ��
	//���߳���һ��ѭ������һ�Σ�
	void setPrevDmat(Mat currDmat);
	void initPrevDmat(Mat currDmat);
	Mat getPrevDmat();

	//@param fgMsk: ǰ���ɰ壬 ��ɫ(uchar_max)Ϊ��Ч����
	CapgSkeleton calcSkeleton(const Mat &dmat, const Mat &fgMsk);


#pragma region //�����ͷ�����ӵ�

	//����ģʽ�����ص���ָ��
	sgf::segment* loadSeedHeadConf(const char *confFn = "./sgf_seed/config.txt", const char *templFn = "./sgf_seed/headtemplate.bmp");

	//�����ʵ�ֵ�ͷ�����ӵ㷽�����İ�װ����
	vector<Point> seedHeadTempMatch(const Mat &dmat, bool debugDraw = false);

	//����ɻ�ȡͷ����С�İ�װ����
	vector<double> getHeadSizes();

#pragma endregion //�����ͷ�����ӵ�

#pragma region //�� opencv300 ���� boundingRect 
	cv::Rect boundingRect(InputArray array);
#pragma endregion //�� opencv300 ���� boundingRect 

}//namespace zc
using zc::HumanObj;

//�� opencv300 ���� Point_.operator/, ���� opencv2.x
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
}//namespace zc-�� opencv300 ���� Point_.operator/, ���� opencv2.x



//---------------���Դ��������
namespace zc{
	Mat getLaplaceEdgeKrnl(size_t krnlSz = 5);

	//@brief �� krnlSz Ϊ�ˣ� getLaplaceEdgeKrnl�� Ȼ����
	//Ȼ���ѣ�
	Mat getLaplaceEdgeFilter2D(const Mat &dmat, size_t krnlSz = 5);

	//@brief ����krnlSz�����ڷ������� N ���ն�, �� N > countThresh, ��������ֵ
	//@param hasSideEffect ����ɨ��ʱ, ǰ�����������Ƿ��Ӱ���������. �� true, �����ձ�Ȼ���޷�������
	//@return filled-dmat
	//@note Ч�ʣ� 1. Ĭ�ϲ���, qvga~8ms; 2. hasSideEffect==true, qvga~11ms; 3. krnlSz=3, countThresh=2 ������Ч��Ӱ��
	Mat holeFillNbor(const Mat &dmat, bool hasSideEffect = false, size_t krnlSz = 5, int countThresh = 3);

#pragma region //��Ҫ��, �Լ�ʵ��֡���������MyBGSubtractor:

	//��history���ȵ���ʷ���ڣ�������ǰ֡����ƽ�����뵱ǰ֡����� >diffThresh������ǰ��
	class MyBGSubtractor
	{
	public:
		MyBGSubtractor();
		MyBGSubtractor(int history, int diffThresh);
		~MyBGSubtractor();


		Mat apply(const Mat &currFrame);

		Mat getBgMat();

	private:
		//@brief frame ��ӣ� Ȼ�����¼���avg-frame������
		void addToHistory(const Mat &frame32f);

		queue<Mat> _historyFrames;

		//@brief _historyFrames �������ޣ���capacity
		int _history;
		int _diffThresh;

		//@brief cv8u, or 16u?
		//Mat _bgMat;

		//@brief cv32f
		Mat _bgMat32f;

		//@brief cv8u
		Mat _fgMask;
	};//MyBGSubtractor
#pragma endregion //��Ҫ��, �Լ�ʵ��֡���������MyBGSubtractor

	//@brief ��ÿ��move-mask, X������Yֵͳ��ֱ��ͼ
	//@param humVec: vec-HumanObj, ����HumanObj����ɫ��Ϣ
	//@param moveMaskVec�� vec-mask-mat, ÿ��HumanObj��Ӧ�����ڵġ��˶�����mask
	//@return ��ɫ histo-mat�� N��ֱ��ͼ����ͬһ��mat��
	Mat getHumVecMaskHisto(Mat dmat, vector<HumanObj> humVec, vector<Mat> moveMaskVec, bool debugDraw = false);


	//@brief �� maskMat ͳ��X������Yֵͳ��ֱ��ͼ, ��color����
	//@return һ����ɫ histo-mat
	Mat getMaskXyHisto(Mat dmat, Mat maskMat, Scalar color, bool debugDraw = false);

	//@brief ����������ʷ������ֵ
	//@return max-depth-mat
	Mat& getMaxDmat(Mat &dmat, bool debugDraw = false);

	//@brief ʹ����һ֡�����¡�����������ʷ������ֵ
	//@return max-depth-mat
	Mat updateMaxDmat(Mat &dmat, bool debugDraw = false);

	//@brief ����maxDmat���ȶ�������diff���۳���Ӱ������maxDmat-MOG���۳�α��Ӱ
	Mat  getMaxDepthBgMask(Mat dmat, bool debugDraw = false);

	vector<Mat> separateMasksMovingHead(Mat dmat, vector<Mat> &inMaskVec, Mat &mogMask, bool debugDraw = false);

	//@brief �����V�ηָ���İ�װ�������ӿڸ�Ϊ inMaskVec���ǵ�һmask
	vector<Mat> separateMasksContValley(Mat dmat, vector<Mat> &inMaskVec, bool debugDraw = false);
}//zc

using zc::MyBGSubtractor;

#endif //_SIMPLE_SILHOUETTE_
