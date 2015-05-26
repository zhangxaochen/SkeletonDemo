#include <iostream>
#include <ctime>

#include <opencv2/opencv.hpp>
#include <XnCppWrapper.h>

#include "../SimpleSilhouette.h"
#include "../sgf_seed/sgf_segment.h"

using namespace std;
using namespace cv;

int g_ImgIndex = 0;

bool checkOpenNIError(XnStatus rc, string status){
	if(rc != XN_STATUS_OK){
		cerr<<status<<" Error: "<<xnGetStatusString(rc)<<endl;
		return false;
	}
	else
		return true;
}//checkOpenNIError

void main(){
	clock_t begt,
		seedSumt = 0,
		sgfSeedSumt = 0,
		rgSumt = 0,
		predAndMergeSumt = 0,
		cannySumt = 0;
	int frameCntr = 0;

	XnStatus rc = XN_STATUS_OK;

	xn::Context ctx;
	rc = ctx.Init();
	if(!checkOpenNIError(rc, "init context"))
		return;

	const char *oniFname = "E:/oni_data/oni132x64/zc-walk-wo-feet-qvga.oni";
	//oniFname = "E:/oni_data/oni132x64/zc-walk-wo-feet-qvga-107x.oni";
	// 	oniFname = "E:/oni_data/oni132_orig/zc_indoor_walk.oni";
	// 	oniFname = "E:/oni_data/oni132_orig/zc_indoor_walk-last.oni";
	// 	oniFname = "E:/oni_data/oni132x64/zc-stand-w-feet.oni";
	//oniFname = "E:/oni_data/oni132x64/zc-stand-wo-feet-qvga.oni";

	xn::Player plyr;
	plyr.SetRepeat(true);
	//Player 这么用？：
	plyr.SetSource(XN_RECORD_MEDIUM_FILE, oniFname);
	rc = ctx.OpenFileRecording(oniFname, plyr);
	if(!checkOpenNIError(rc, "ctx.OpenFileRecording"))
		return;

	xn::DepthGenerator dg;
	rc = dg.Create(ctx);
	if(!checkOpenNIError(rc, "create dg"))
		return;
	xn::DepthMetaData depthMD;

	XnMapOutputMode mapMode;
	mapMode.nXRes = 320;
	mapMode.nYRes = 240;
	mapMode.nFPS = 30;
	rc = dg.SetMapOutputMode(mapMode);
	if(!checkOpenNIError(rc, "dg.SetMapOutputMode"))
		//return;
		;

	rc = ctx.StartGeneratingAll();
	if(!checkOpenNIError(rc, "ctx.StartGeneratingAll"))
		return;

	char key = 0;
	while(key != 27){
		//frameCntr++;
		frameCntr = depthMD.FrameID();

		ctx.WaitAndUpdateAll();

		dg.GetMetaData(depthMD);
		Mat dm(depthMD.FullYRes(), depthMD.FullXRes(), CV_16UC1, (void*)depthMD.Data());
		Mat dm_draw;
		normalize(dm, dm_draw, UCHAR_MAX, 0, NORM_MINMAX, CV_8UC1);

// 		//边缘检测速度 2ms (320*240):
// 		begt = clock();
// 		Mat edge;
// 		Canny(dm_draw, edge, 55, 88);
// 		//cout<<"Canny.t: "<<clock()-begt<<endl;
// 		cannySumt += (clock()-begt);
// 		imshow("canny", edge);

		//时序上前后帧求 abs(sub)
		Mat currPreDiffMsk = zc::simpleMask(dm, ZCDEBUG);

		//种子点寻找：
		begt = clock();
		int veryDepth = -1;
		Point seed = zc::simpleSeed(dm, &veryDepth, ZCDEBUG);
		//cout<<"simpleSeed.t: "<<clock()-begt<<endl;
		seedSumt += (clock()-begt);

		circle(dm_draw, seed, 3, 255, 2);

		if(ZCDEBUG){
			Mat vdMat;
			findNonZero(dm==veryDepth, vdMat);
			printf("vdMat.total(): %d\n", vdMat.total());
			for(int i=0; i<vdMat.total(); i++)
				circle(dm_draw, vdMat.at<Point>(i), 0, 122, 1);

// 			if(vdMat.total()>0){
// 				circle(dm_draw, vdMat.at<Point>(0), 5, 255, 2);
// 			}
		}
		imshow("dm_draw", dm_draw);

				//孙国飞寻找头部种子点
		begt = clock();
		const string sgf_configPath = "../../sgf_seed/config.txt",
			sgf_headTemplatePath = "../../sgf_seed/headtemplate.bmp";
		vector<Point> sgfSeeds = zc::getHeadSeeds(dm, 
			sgf_configPath, sgf_headTemplatePath, ZCDEBUG);
		cout<<"sgfSeeds.size(): "<<sgfSeeds.size()<<endl;
		sgfSeedSumt += (clock()-begt);



		//区域增长 thresh=50cm
		begt = clock();
		//纵向简单去掉屏幕下缘 1/4，即地面,防止脚部与地面连成一片：
		Mat fgMsk = zc::simpleRegionGrow(dm, seed, 333, Rect(0, 0, dm.cols, dm.rows*3/4), ZCDEBUG);
		//cout<<"simpleRegionGrow.t: "<<clock()-begt<<endl;
		rgSumt += (clock()-begt);

		//兼容林驰代码：
		CapgSkeleton tSklt;
		Mat tmpDm;
		dm.convertTo(tmpDm, CV_32SC1);
		tmpDm.setTo(INT_MAX, fgMsk==0);
		imshow("tmpDm", tmpDm);
		
		IplImage depthImg = tmpDm,
			silImg = fgMsk,
			maskImg = currPreDiffMsk;
		bool useDense = false,
			usePre = false;

		string featurePath = "../../feature";
		BPRecognizer *bpr = zc::getBprAndLoadFeature(featurePath);

// 		IplImage *pLabelImage = bpr->predict(&silImg, &maskImg, useDense, usePre);
// 		Mat labelMat(pLabelImage);
// 		imshow("pLabelImage", labelMat);
// 		printf("labelMat.depth(): %d, %d, %d\n", labelMat.depth(), labelMat.channels(), labelMat.type()); //0, 1, 0

		begt = clock();
		//bpr->predictAndMergeJoint(&silImg, tSklt, &maskImg, false, false, true);
		bpr->predictAndMergeJoint(&depthImg, tSklt, &maskImg, false, false, true);
		predAndMergeSumt  += (clock()-begt);


		cout<<"frameCntr: "<<frameCntr<<endl;

		key = waitKey(0);
	}//while
}//main