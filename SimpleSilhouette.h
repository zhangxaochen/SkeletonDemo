#include <vector>
#include <queue>
#include <opencv2/opencv.hpp>

#include "CapgSkeleton.h"
#include "bodyPartRecognizer.h"

namespace zc{
#define ZCDEBUG 1
#define ZCDEBUG_WRITE 1

	//BPRecognizer* getBprAndLoadFeature();
	//BPRecognizer* getBprAndLoadFeature(const char *featurePath = nullptr);
	BPRecognizer* getBprAndLoadFeature(const string &featurePath);

	Mat simpleMask(const Mat &curMat, bool debugDraw = false);

	Point simpleSeed(const Mat &dmat, int *outVeryDepth = 0, bool debugDraw = false);

	//Mat simpleRegionGrow(const Mat &dmat, Point seed, int thresh, bool debugDraw = false);
	Mat simpleRegionGrow(const Mat &dmat, Point seed, int thresh, const Rect roi, bool debugDraw = false);
	Mat simpleRegionGrow(const Mat &dmat, vector<Point> seeds, int thresh, const Rect roi, bool debugDraw = false);

	Mat simpleRegionGrow(const Mat &dmat, Point seed, int thresh, const Mat &mask, bool debugDraw = false);
	Mat simpleRegionGrow(const Mat &dmat, vector<Point> seeds, int thresh, const Mat &mask, bool debugDraw = false);

	//region-grow ���� 1. �Ҳ������ӵ㣬��������ʧ�ܵ������ 2. ����ͻ�����������ǰһ֡����LowPass��
	Mat postRegionGrow(const Mat &flagMat, int xyThresh, int zThresh, bool debugDraw = false);

// 	Mat erodeDilateN(Mat &m, int ntimes);
}
