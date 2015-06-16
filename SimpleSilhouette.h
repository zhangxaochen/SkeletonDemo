#include <iostream>
#include <vector>
#include <queue>
#include <opencv2/opencv.hpp>
#include <ctime>

#include "CapgSkeleton.h"
#include "bodyPartRecognizer.h"

using namespace std;
using namespace cv;

const int QVGA_WIDTH = 320,
	QVGA_HEIGHT = 240,
	MAX_VALID_DEPTH = 10000;

//zhangxaochen: �ٶ� 320*240 ���ݶ�Ӧ�Ľ���
#define XTION_FOCAL_XY 300

#define MIN_VALID_HEIGHT_PHYSICAL_SCALE 600
#define MIN_VALID_HW_RATIO 1.0 //height-width ratio

static cv::RNG rng;

namespace zc{
#if CV_VERSION_MAJOR < 3
//lincccc's code below:
	//BPRecognizer* getBprAndLoadFeature();
	//BPRecognizer* getBprAndLoadFeature(const char *featurePath = nullptr);
	BPRecognizer* getBprAndLoadFeature(const string &featurePath);
#endif

	Mat simpleMask(const Mat &curMat, bool debugDraw = false);

	Point simpleSeed(const Mat &dmat, int *outVeryDepth = 0, bool debugDraw = false);

	//1. �ұ�����ǽ�棻
	//2. ��ûǽ��˵�������տ�������߶��ж��޳�(<2500mm)
	//3. ʣ����ߵ������ӵ�
	vector<Mat> findFgMasksUseWallAndHeight(Mat dmat, bool debugDraw = false);


	//Mat simpleRegionGrow(const Mat &dmat, Point seed, int thresh, bool debugDraw = false);
	Mat _simpleRegionGrow(const Mat &dmat, Point seed, int thresh, const Rect roi, bool debugDraw = false);
	Mat _simpleRegionGrow(const Mat &dmat, vector<Point> seeds, int thresh, const Rect roi, bool debugDraw = false);

	Mat _simpleRegionGrow(const Mat &dmat, Point seed, int thresh, const Mat &mask, bool debugDraw = false);
	Mat _simpleRegionGrow(const Mat &dmat, vector<Point> seeds, int thresh, const Mat &mask, bool debugDraw = false);
	vector<Mat> simpleRegionGrow(const Mat &dmat, vector<Point> seeds, int thresh, const Mat &mask, bool getMultiMasks = false, bool debugDraw = false);


	//����Ĭ�ϲ���(flrKrnl, mskThresh, morphRadius) �� 
	Mat getFloorApartMask(Mat dmat, bool debugDraw = false);
	//���أ�����ȥ���ذ��mask
	Mat getFloorApartMask(Mat dmat, Mat flrKrnl, int mskThresh, int morphRadius, bool debugDraw = false);

	Mat fetchFloorApartMask(Mat dmat, bool debugDraw = false);

	Mat calcHeightMap(Mat dmat, bool debugDraw = false);
	Mat fetchHeightMap(Mat dmat, bool debugDraw = false);

	Mat getHeightMask(Mat dmat, int limitMs = 2500);

	//ֱ��ͼ��ʽѰ�Ҵ�ǽ�����ֵ�����ֵ��
	//����������ǽ�������
	//RETURN: -1 ��ʾû�ҵ��㹻���ǽ�棻��ֵΪǽ�����ֵ
	int getWallDepth(Mat &dmat);

	void drawOneSkeleton(Mat &img, CapgSkeleton &sk);
	void drawSkeletons(Mat &img, const vector<CapgSkeleton> &sklts, int skltIdx);

	void eraseNonHumanContours(vector<vector<Point> > &contours);
	bool isHumanContour(const vector<Point> &cont);
	bool isHumanMask(const Mat &msk, int fgPxCntThresh = 1000);

	//@deprecated
	Mat distMap2contoursDebug(const Mat &dmat, bool debugDraw = false);

	//1. �� distMap ��ֵ���õ� contours�� 2. �� contours bbox �жϣ��õ���������
	//ע��
	// 1. �� debugDraw = true, �� _debug_mat ���봫ʵ��
	vector<vector<Point> > distMap2contours(const Mat &dmat, bool debugDraw = false, OutputArray _debug_mat = noArray());

	//@deprecated
	Mat dmat2TopDownViewDebug(const Mat &dmat, bool debugDraw = false);

	//dmat �����°���, תΪ top-down-view, ����, ����bbox�ж�, ��ȡ���ʵ�����
	//ע:
	// 1. Z�����ű�Ϊ��ֵ�� UCHAR_MAX/MAX_VALID_DEPTH, �� top-down-view ͼ�߶�Ϊ 256
	// 2. debugDraw, dummy variable
	// 3. �� debugDraw = true, �� _debug_mat ���봫ʵ��
	vector<vector<Point> > dmat2TopDownView(const Mat &dmat, bool debugDraw = false, OutputArray _debug_mat = noArray());

	//������ͼ������ͼ���� bbox �󽻣��ж���������λ��
	//ע��
	// 1. �� debugDraw = true, �� _debug_mat ���봫ʵ��
	vector<Mat> findHumanMasksUseBbox(Mat &dmat, bool debugDraw = false, OutputArray _debug_mat = noArray());

#if CV_VERSION_MAJOR >= 3
	//�� opencv300 background-subtraction ������ȡ�˶����壨��������,e.g.:ת�Σ�����
	//1. bgs -> roi; 2. region-grow -> vector<Mat>; 3. bbox etc. �ж�; 4. usePre ʱ���ϣ������ж�
	//
	vector<Mat> findFgMasksUseBGS(Mat &dmat, bool usePre = false, bool debugDraw = false, OutputArray _debug_mat = noArray());
#endif

	//���ز�ͬ�Ҷȱ��ǰ����mat
	Mat getHumansMask(vector<Mat> masks, Size sz);


	class HumanFg;

	//��ȫ�ֵ� vector<HumanFg> ��һ����ɫ mask mat:
	Mat getHumansMask(Mat dmat, const vector<HumanFg> &humVec);

	//Ӧ��ճ�Ը������������˳����£�uid��Ӧͻ��
	//����(����)����Ϊ��
	//1. _prevCenterλ��ǰ��֡��ȱ仯��С
	vector<HumanFg> getHumanObjVec(Mat &dmat, vector<Mat> fgMasks);

	class HumanFg
	{
	public:
		HumanFg(const Mat &dmat_, Mat currMask_)
			:_dmat(dmat_)
			//,_currMask(currMask_)
		{
			//_currCenter = getContMassCenter(_currMask);
			setCurrMask(currMask_);

			if (_prevMask.empty()){
// 				_currMask.copyTo(_prevMask);
// 				_prevCenter = _currCenter;
				setPrevMask(_currMask);
			}

			for (int i = 0; i < 3; i++)
				_humColor[i] = rng.uniform(UCHAR_MAX/2, UCHAR_MAX);
		}//HumanFg-ctor

// 		void updateMask(Mat currMask_){
// 
// 		}//updateMask

		//���ɹ��� ���� this��� ���򣬷���false�����ڱ�ʾ������׼��ɾ���˶���
		bool updateMask(const Mat &dmat, const vector<Mat> &fgMasks_, vector<bool> &mskUsedFlags){
			//��������
// 			uchar currMcDepth = _dmat.at<ushort>(_currCenter),
// 				newMcDepth = dmat.at<ushort>(_currCenter);
// 
// 			//������λ��ǡ��Ϊ��Ч�㣬 ��Ǹ�����
// 			if (currMcDepth == 0 || newMcDepth == 0)
// 				return false;
			
			//���������� _currCenter �� fgMasks_ ĳ�����ڣ�//����
			//����ǰ��֡��ɫ�����󽻣�
			size_t fgMskSize = fgMasks_.size();
			bool foundNewMask = false;
			for (size_t i = 0; i < fgMskSize; i++){
				Mat fgMsk = fgMasks_[i];
				//�󽻣�
				Mat currNewIntersect = _currMask & fgMsk;
				int intersectArea = countNonZero(currNewIntersect != 0),
					fgMskArea = countNonZero(fgMsk != 0);
				double percent = 0.5;
				if (mskUsedFlags[i] == false
					//&& fgMsk.at<uchar>(_currCenter) == UCHAR_MAX){
					&& (intersectArea > _currMaskArea * percent 
						|| intersectArea > fgMskArea * percent)
					){

// 					_prevMask = _currMask;
// 					_prevCenter = _currCenter;
// 
// 					_currMask = fgMsk;
// 					_currCenter = getContMassCenter(_currMask);

					setPrevMask(_currMask);
					setCurrMask(fgMsk);

					foundNewMask = true;
					//���ĳmask�ѱ��ù���
					mskUsedFlags[i] = true;
					
					break;
				}
			}

			//��������
// 			//��һ����λ��ǡ��Ϊ��Ч����ô�죿��δ�������Ӧ�ñ��Ϊ��ʧ
// 			int depthDiff = abs((int)_dmat.at<ushort>(_currCenter)-(int)dmat.at<ushort>(_currCenter));
// 
// 			//�������徲ֹ�� _currCenter ���� fgMasks_�κ������ڣ����ܿգ���
// 			//�� _currCenter λ��ǰ��֡��Ȳ���΢С(<200mm)�����Դ�Ϊ���ӵ���������һ������
// 			if (!foundNewMask 
// 				//&& (_dmat.at<ushort>(_currCenter) == 0 || depthDiff < 200))
// 				&& depthDiff < 200)
// 			{
// 
// 				_prevMask = _currMask;
// 				_prevCenter = _currCenter;
// 
// 				bool debugDraw = false;
// 				//Mat flrApartMsk = getFloorApartMask(dmat, debugDraw);
// 				Mat flrApartMsk = fetchFloorApartMask(dmat, debugDraw);
// 				int rgThresh = 55;
// 				_currMask = _simpleRegionGrow(dmat, _currCenter, rgThresh, flrApartMsk, debugDraw);
// 				_currCenter = getContMassCenter(_currMask);
// 			}

			//���� fgMasks_�ص���̫��δ�ҵ�����
			if (!foundNewMask){
				//ȡ�� mask & ��һ֡��������
				Mat tmp_msk = _currMask & (dmat != 0);
				//�Լ��¾�֡΢С�仯�󽻣�
				tmp_msk &= (abs(dmat - _dmat) < 100);
				//��Ϊ�µ�������ѡ����
				bool debugDraw_ = false;
				Mat flrApartMsk = fetchFloorApartMask(dmat, debugDraw_);
				int rgThresh = 550;

				Mat sdPts;
				cv::findNonZero(tmp_msk, sdPts);
				if (sdPts.empty()){
					cout << "sdPts.empty()" << endl;
					//return false;
				}
				else{
					Mat newMask = _simpleRegionGrow(dmat, sdPts.at<Point>(0), rgThresh, flrApartMsk, debugDraw_);

					setCurrMask(newMask);
				}
			}
			//���ͼ���£�
			_dmat = dmat;


			//���� distMap2contours ��bbox �ж����ˣ�
			//1. ����̫��
			double dmin, dmax;
			minMaxLoc(dmat, &dmin, &dmax, nullptr, nullptr, _currMask);
			if (dmax - dmin > 1500)
				return false;

			vector<vector<Point> > contours;
			findContours(_currMask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

			//CV_Assert(contours.size() > 0); //��Щʱ������Ҳ�������δ�����fake��
			//fake:
			contours.push_back(vector<Point>());

			Rect bbox = boundingRect(contours[0]);
			//2. bbox�߶Ȳ���̫С; 3. bbox ���ز��ܸ��ڰ�������Ϊ�˽Ų�λ�ýϵ�
			if (bbox.height < 80
				|| bbox.br().y < dmat.rows / 2)
				return false;


			return true;
		}//updateMask

		Scalar getColor(){
			return _humColor;
		}

		Mat getCurrMask(){
			return _currMask;
		}//getCurrMask

		void setCurrMask(Mat newMask){
			_currMask = newMask;
			_currCenter = getContMassCenter(_currMask);
			_currMaskArea = countNonZero(_currMask != 0);
		}//setCurrMask

		void setPrevMask(Mat newMask){
			_prevMask = newMask;
			_prevCenter = getContMassCenter(_prevMask);
		}//setPrevMask

		//---------------


	protected:
	private:
		Mat _dmat;

		Mat _prevMask;
		Point _prevCenter;

		Mat _currMask;
		int _currMaskArea;
		Point _currCenter;

		Scalar _humColor;

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

	};//class HumanFg

	//vector<Mat> bboxFilter(const vector<Mat> &origMasks);
	vector<Mat> bboxFilter(Mat dmat, const vector<Mat> &origMasks);

	//radius: kernel size is (2*radius+1)^2
	//shape: default MORPH_RECT
	Mat getMorphKrnl(int radius = 1, int shape = MORPH_RECT);//getMorphKrnl

	//region-grow ���� 
	// 1. �Ҳ������ӵ㣬��������ʧ�ܵ������ 
	// 2. ����ͻ�����������ǰһ֡����LowPass����δ��ɡ�
	Mat postRegionGrow(const Mat &flagMat, int xyThresh, int zThresh, bool debugDraw = false);

}//namespace zc

using zc::HumanFg;