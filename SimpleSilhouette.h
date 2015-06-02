#include <iostream>
#include <vector>
#include <queue>
#include <opencv2/opencv.hpp>
#include <ctime>

#include "CapgSkeleton.h"
#include "bodyPartRecognizer.h"

namespace zc{
	//BPRecognizer* getBprAndLoadFeature();
	//BPRecognizer* getBprAndLoadFeature(const char *featurePath = nullptr);
	BPRecognizer* getBprAndLoadFeature(const string &featurePath);

	Mat simpleMask(const Mat &curMat, bool debugDraw = false);

	Point simpleSeed(const Mat &dmat, int *outVeryDepth = 0, bool debugDraw = false);

	//Mat simpleRegionGrow(const Mat &dmat, Point seed, int thresh, bool debugDraw = false);
	Mat _simpleRegionGrow(const Mat &dmat, Point seed, int thresh, const Rect roi, bool debugDraw = false);
	Mat _simpleRegionGrow(const Mat &dmat, vector<Point> seeds, int thresh, const Rect roi, bool debugDraw = false);

	Mat _simpleRegionGrow(const Mat &dmat, Point seed, int thresh, const Mat &mask, bool debugDraw = false);
	Mat _simpleRegionGrow(const Mat &dmat, vector<Point> seeds, int thresh, const Mat &mask, bool debugDraw = false);
	vector<Mat> simpleRegionGrow(const Mat &dmat, vector<Point> seeds, int thresh, const Mat &mask, bool getMultiMasks = false, bool debugDraw = false);

	Mat getFloorApartMask( Mat dmat, bool debugDraw = false);

	void drawOneSkeleton(Mat &img, CapgSkeleton &sk);
	void drawSkeletons(Mat &img, const vector<CapgSkeleton> &sklts, int skltIdx);

	void eraseNonHumanContours(vector<vector<Point> > &contours);
	bool isHumanContour(const vector<Point> &cont);
	bool isHumanMask(const Mat &msk, int fgPxCntThresh = 1000);

	Mat distMap2contours(const Mat &dmat, bool debugDraw = false);

	//region-grow ���� 
	// 1. �Ҳ������ӵ㣬��������ʧ�ܵ������ 
	// 2. ����ͻ�����������ǰһ֡����LowPass����δ��ɡ�
	Mat postRegionGrow(const Mat &flagMat, int xyThresh, int zThresh, bool debugDraw = false);

// 	Mat erodeDilateN(Mat &m, int ntimes);
}//namespace zc
