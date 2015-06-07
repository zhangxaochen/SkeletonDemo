#include <iostream>
#include <ctime>

#include <opencv2/opencv.hpp>
#include <XnCppWrapper.h>

#include "../SimpleSilhouette.h"
#include "../sgf_seed/sgf_segment.h"

using namespace std;
using namespace cv;

#define ZCDEBUG 1
#define ZCDEBUG_WRITE 1

int g_ImgIndex = 0;

clock_t begt,
	seedSumt = 0,
	sgfSeedSumt = 0,
	rgSumt = 0,
	predAndMergeSumt = 0,
	predictSumt = 0,
	cannySumt = 0;

const int QVGA_WIDTH = 320,
	QVGA_HEIGHT = 240,
	MAX_VALID_DEPTH = 10000;

Mat dm_draw;
int rgThresh = 55;

bool checkOpenNIError(XnStatus rc, string status){
	if(rc != XN_STATUS_OK){
		cerr<<status<<" Error: "<<xnGetStatusString(rc)<<endl;
		return false;
	}
	else
		return true;
}//checkOpenNIError

void nmhSilhouAndSklt( Mat &dm, int fid ){
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

	}

	//区域增长 thresh=50cm
	begt = clock();
	//纵向简单去掉屏幕下缘 1/4，即地面,防止脚部与地面连成一片：
	Rect rgRoi(0, 0, dm.cols, dm.rows*3/4);
	Mat fgMsk = zc::simpleRegionGrow(dm, seed, rgThresh, rgRoi, ZCDEBUG);
	//cout<<"simpleRegionGrow.t: "<<clock()-begt<<endl;
	rgSumt += (clock()-begt);
	if(ZCDEBUG_WRITE)
		imwrite("simpFlagMat_"+std::to_string((long long)fid)+".jpg", fgMsk);


	//兼容林驰代码：
	CapgSkeleton tSklt;
	Mat tmpDm;
	dm.convertTo(tmpDm, CV_32SC1);
	//背景填充最大值，所以前景反而黑色：
	tmpDm.setTo(INT_MAX, fgMsk==0);
	imshow("tmpDm", tmpDm);

	IplImage depthImg = tmpDm,
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
	bpr->predictAndMergeJoint(&depthImg, tSklt, &maskImg, false, false, ZCDEBUG);
	predAndMergeSumt  += (clock()-begt);
}//nmhSilhouAndSklt


void main(int argc, char **argv){
	freopen("_debug_info.log", "w", stdout);

	float zoomFactor = 1;

	int frameCntr = 0;

	XnStatus rc = XN_STATUS_OK;
	
	xn::Context ctx;
	rc = ctx.Init();
	if(!checkOpenNIError(rc, "init context"))
		return;

	const char *oniFname = "E:/oni_data/oni132x64/zc-walk-wo-feet-qvga.oni";
	oniFname = "E:/oni_data/oni132_orig/sun_han_short.oni";
	oniFname = "E:/oni_data/oni132x64/sgf_zc_w_feet.oni";
	oniFname = "E:/oni_data/oni132x64/sgf_zc_w_feet-190-3xx.oni";
	oniFname = "E:/oni_data/oni132x64/sgf_zc_w_feet-650-776.oni";
	
	//oniFname = "E:/oni_data/oni132x64/sgf_zc_w_feet-wo-overlap.oni";
	
	//oniFname = "E:/oni_data/oni132x64/zc-walk-wo-feet-qvga-107x.oni";
	// 	oniFname = "E:/oni_data/oni132_orig/zc_indoor_walk.oni";
	// 	oniFname = "E:/oni_data/oni132_orig/zc_indoor_walk-last.oni";
	// 	oniFname = "E:/oni_data/oni132x64/zc-stand-w-feet.oni";
	//oniFname = "E:/oni_data/oni132x64/zc-stand-wo-feet-qvga.oni";

	xn::Player plyr;
	rc = ctx.OpenFileRecording(oniFname, plyr);
	plyr.SeekToFrame("Depth1", std::stoi(argv[1]), XN_PLAYER_SEEK_SET);
	plyr.SetRepeat(string(argv[2])=="true"); //放在 OpenFileRecording 之后才有效
	if(!checkOpenNIError(rc, "ctx.OpenFileRecording"))
		return;

	xn::DepthGenerator dg;
	rc = dg.Create(ctx);
	if(!checkOpenNIError(rc, "create dg"))
		return;
	xn::DepthMetaData depthMD;

	XnMapOutputMode mapMode;
	mapMode.nXRes = QVGA_WIDTH;
	mapMode.nYRes = QVGA_HEIGHT;
	mapMode.nFPS = 30;
	rc = dg.SetMapOutputMode(mapMode);
	if(!checkOpenNIError(rc, "dg.SetMapOutputMode"))
		dg.GetMapOutputMode(mapMode);
	
	zoomFactor = QVGA_WIDTH / mapMode.nXRes;

	rc = ctx.StartGeneratingAll();
	if(!checkOpenNIError(rc, "ctx.StartGeneratingAll"))
		return;

	char key = 0;
	while(key != 27){
		frameCntr++;
		//frameCntr = depthMD.FrameID(); //若计时用，不用 .FrameID
		int fid = depthMD.FrameID();
		cout<<"frameCntr, fid: "<<frameCntr<<", "<<fid<<endl;

		ctx.WaitAndUpdateAll();

		dg.GetMetaData(depthMD);
		Mat dm(depthMD.FullYRes(), depthMD.FullXRes(), CV_16UC1, (void*)depthMD.Data());
		//临时去掉太深的杂乱背景，填充为最深：
		//dm.setTo(MAX_VALID_DEPTH, dm>6000);
		//pyrDown(dm, dm, Size(dm.cols*zoomFactor, dm.rows*zoomFactor));

		//---------------1. NMH 简单种子点，silhouette； skeleton
		//nmhSilhouAndSklt(dm, fid);

		//---------------2. 孙国飞寻找头部种子点：
		normalize(dm, dm_draw, UCHAR_MAX, 0, NORM_MINMAX, CV_8UC1);

		begt = clock();
		const string sgf_configPath = "../../sgf_seed/config.txt",
			sgf_headTemplatePath = "../../sgf_seed/headtemplate.bmp";
		vector<Point> sgfSeeds = zc::getHeadSeeds(dm, 
			sgf_configPath, sgf_headTemplatePath, ZCDEBUG);
		sgfSeedSumt += (clock()-begt);
		cout<<"sgfSeeds.size(): "<<sgfSeeds.size()<<endl;
		

		for(size_t i=0; i<sgfSeeds.size(); i++){
			Point sdi = sgfSeeds[i];
			circle(dm_draw, sdi, 9, 255, 2);
		}
		imshow("1-dm_draw", dm_draw);
		if(ZCDEBUG_WRITE)
			imwrite("1-dm_draw_"+std::to_string((long long)fid)+".jpg", dm_draw);

		Mat flrApartMsk = zc::getFloorApartMask(dm, ZCDEBUG);

		Mat sgfFgMsk = zc::simpleRegionGrow(dm, sgfSeeds, rgThresh, flrApartMsk, ZCDEBUG);
		imshow("2-sgfFgMsk", sgfFgMsk);

// 轮廓开操作，判定哪些是人， 依据： 1. bbox长宽比， 2. bbox.height 物理尺寸。【未完成】
		int anch = std::stoi(argv[3]);
		Mat morphKrnl = getStructuringElement(MORPH_RECT, Size(anch*2+1, anch*2+1), Point(anch, anch) );
		Mat sgfFgMsk_open;
		morphologyEx(sgfFgMsk, sgfFgMsk_open, CV_MOP_OPEN, morphKrnl);
		imshow("3-sgfFgMsk_open", sgfFgMsk_open);
 		
 		if(ZCDEBUG_WRITE){
 			imwrite("2-sgfFgMsk_"+std::to_string((long long)fid)+".jpg", sgfFgMsk);
 			imwrite("3-sgfFgMsk_open_"+std::to_string((long long)fid)+".jpg", sgfFgMsk_open);
 		}

		sgfFgMsk = sgfFgMsk_open;

		//兼容林驰代码：
		Mat tmpDm;
		dm.convertTo(tmpDm, CV_32SC1);
		//背景填充最大值，所以前景反而黑色：
		tmpDm.setTo(INT_MAX, sgfFgMsk==0);
		//imshow("tmpDm", tmpDm);

		IplImage depthImg = tmpDm;
		bool useDense = false,
			useErode = false,
			usePre = true;

		string featurePath = "../../feature";
		BPRecognizer *bpr = zc::getBprAndLoadFeature(featurePath);

		//因为predict是逐像素的，所以一个mask包含多人前景并不影响结果：
		begt = clock();
		Mat labelMat = bpr->predict(&depthImg, nullptr, useDense, usePre);
		predictSumt += (clock()-begt);

		if(ZCDEBUG){
			imshow("labelMat", labelMat);
			Mat rgbLabelMat = label_gray2rgb(labelMat);
			imshow("6-rgbLabelMat", rgbLabelMat);
			if(ZCDEBUG_WRITE)
				imwrite("6-rgbLabelMat_"+std::to_string((long long)fid)+".jpg", rgbLabelMat);
		}

		//将多人轮廓分割为 n*单人 mat：
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(sgfFgMsk, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

		cout<<"contours.size(): "<<contours.size()<<endl;
		const size_t origContSize = contours.size();
		for(size_t i = 0; i < origContSize; i++)
			cout<<"i: "<<i<<", "<<contours[i].size()<<endl;
		
		if(contours.size()>2)
			int dumb = 0;
		zc::eraseNonHumanContours(contours);
		cout<<"eraseNonHumanContours.size(): "<<contours.size()<<endl;

		const size_t contSize = contours.size();
		vector<vector<Point> > contours_poly(contSize);
		vector<Rect> boundRect(contSize);
		vector<Point2f> center(contSize);
		vector<float> radius(contSize);

		vector<CapgSkeleton> sklts;
		
		//调试绘制
		Mat contMat = Mat::zeros(sgfFgMsk.size(), CV_8UC1);

		for(size_t i = 0; i < contSize; i++){
			approxPolyDP((contours[i]), contours_poly[i], 3, true);

			//单人
			Mat soloContMask = Mat::zeros(sgfFgMsk.size(), CV_8UC1);
			drawContours(soloContMask, contours_poly, i, 255, -1);
			Mat soloLabelMat = labelMat.clone();
			soloLabelMat.setTo(BodyLabel_Background, soloContMask==0);

			bool useErode = false,
				usePre = false;

			CapgSkeleton tSklt;
			bpr->mergeJoint(new IplImage(soloLabelMat), &depthImg, tSklt, useErode, usePre);
			sklts.push_back(tSklt);

			if(ZCDEBUG){
				//计算 bbox, enclosingCircle:
				boundRect[i] = boundingRect(contours_poly[i]);
				minEnclosingCircle(contours_poly[i], center[i], radius[i]);
				//调试绘制
				Scalar c(255);
				drawContours(contMat, contours_poly, i, c, -1);
				drawContours(contMat, contours, i, c);
				rectangle(contMat, boundRect[i].tl(), boundRect[i].br(), c);
				circle(contMat, center[i], radius[i], c);

				Mat soloRgbLabelMat = label_gray2rgb(soloLabelMat);
				imshow("5-soloContMask-"+std::to_string((long long)i), soloContMask);
				imshow("7-soloRgbLabelMat-"+std::to_string((long long)i), soloRgbLabelMat);
				cout<<"===tSklt.size(): "<<tSklt.size()<<", "<<contours[i].size()<<", "<<contours_poly[i].size()<<endl;

				if(ZCDEBUG_WRITE){
					imwrite("5-soloContMask-"+std::to_string((long long)i)+"_"+std::to_string((long long)fid)+".jpg", soloContMask);
					imwrite("7-soloRgbLabelMat-"+std::to_string((long long)i)+"_"+std::to_string((long long)fid)+".jpg", soloRgbLabelMat);
				}
			}
		}
		imshow("4-contMat", contMat);
		if(ZCDEBUG_WRITE)
			imwrite("4-contMat_"+std::to_string((long long)fid)+".jpg", contMat);

		cout<<"sklts.size(): "<<sklts.size()<<endl;

		Mat skCanvas = Mat::ones(sgfFgMsk.size(), CV_8UC1)*UCHAR_MAX;
		zc::drawSkeletons(skCanvas, sklts, -1);
		if(ZCDEBUG){
			imshow("8-skCanvas", skCanvas);
			if(ZCDEBUG_WRITE)
				imwrite("8-skCanvas_"+std::to_string((long long)fid)+".jpg", skCanvas);
		}

		key = waitKey(01);
	}//while

	//统计几个平均时间(ms)：
	cout<<"frameCntr, seedSumt, sgfSeedSumt, rgSumt, cannySumt, predAndMergeSumt, predictSumt: "
		<<frameCntr<<":: "
		<<seedSumt*1./frameCntr<<", "<<sgfSeedSumt*1./frameCntr<<", "
		<<rgSumt*1./frameCntr<<", "<<cannySumt*1./frameCntr<<", "<<predAndMergeSumt*1./frameCntr<<", "<<predictSumt*1./frameCntr<<endl;

	destroyAllWindows();
	ctx.StopGeneratingAll();
	ctx.Release();

}//main
