#include <queue>
#include <opencv2/opencv.hpp>

#include "CapgSkeleton.h"
#include "bodyPartRecognizer.h"

namespace zc{
#define ZCDEBUG 1

	BPRecognizer* getBprAndLoadFeature();

	Mat simpleMask(const Mat &curMat, bool debugDraw = false);

	//Mat simpleRegionGrow(const Mat &dmat, Point seed, int thresh, bool debugDraw = false);
	Mat simpleRegionGrow(const Mat &dmat, Point seed, int thresh, const Rect roi, bool debugDraw = false);

	Point simpleSeed(const Mat &dmat, int *outVeryDepth = 0, bool debugDraw = false);

}
