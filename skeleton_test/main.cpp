#include <iostream>
#include <ctime>

#include <opencv2/opencv.hpp>
#include <XnCppWrapper.h>

#include "../SimpleSilhouette.h"
#include "../sgf_seed/sgf_segment.h"

using namespace std;
using namespace cv;

#define ZC_DEBUG_LV1 01	//��ʾ�ؼ����ͼ
#define ZC_DEBUG_LV2 0	//��ʾĳЩ�м���ͼ
#define ZC_WRITE 01		//����ؽڽ����jpg�ļ�
#define ZC_CLEAN 01		//@deprecated, ����� CAPG_SKEL_VERSION_0_9 ��

#define CAPG_SKEL_VERSION_0_1 //��������Ĵ���汾
int g_ImgIndex = 0;

#undef CAPG_SKEL_VERSION_0_1 //���� v0.9. 2015��7��1��00:07:01
#define CAPG_SKEL_VERSION_0_9 //��Ӧ skeleton_test v0.9. 2015��6��30��23:59:18

#undef CAPG_SKEL_VERSION_0_9
#define CAPG_SKEL_VERSION_0_9_1	//���ӷ���������������mask��������һ���Ӻ���ָby����ɣ����ַ��������߶ԱȲ���

//���ո����ֶ����ţ�
bool isManually = false;
bool isExit = false;
//��λ����
int seekId = -1;



clock_t begt,
	seedSumt = 0,
	sgfSeedSumt = 0,
	rg2msksSumt = 0,
	rgSumt = 0,
	predAndMergeSumt = 0,
	predictSumt = 0,
	cannySumt = 0,
	distMap2contoursSumt = 0;


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

#ifdef CAPG_SKEL_VERSION_0_1
void nmhSilhouAndSklt( Mat &dm, int fid ){
	normalize(dm, dm_draw, UCHAR_MAX, 0, NORM_MINMAX, CV_8UC1);

	// 		//��Ե����ٶ� 2ms (320*240):
	// 		begt = clock();
	// 		Mat edge;
	// 		Canny(dm_draw, edge, 55, 88);
	// 		//cout<<"Canny.t: "<<clock()-begt<<endl;
	// 		cannySumt += (clock()-begt);
	// 		imshow("canny", edge);

	//ʱ����ǰ��֡�� abs(sub)
	Mat currPreDiffMsk = zc::simpleMask(dm, ZC_DEBUG_LV1);

	//���ӵ�Ѱ�ң�
	begt = clock();
	int veryDepth = -1;
	Point seed = zc::seedSimple(dm, &veryDepth, ZC_DEBUG_LV1);
	//cout<<"simpleSeed.t: "<<clock()-begt<<endl;
	seedSumt += (clock()-begt);

	circle(dm_draw, seed, 3, 255, 2);

	if(ZC_DEBUG_LV1){
		Mat vdMat;
		findNonZero(dm==veryDepth, vdMat);
		printf("vdMat.total(): %d\n", vdMat.total());
		for(int i=0; i<vdMat.total(); i++)
			circle(dm_draw, vdMat.at<Point>(i), 0, 122, 1);

	}

	//�������� thresh=50cm
	begt = clock();
	//�����ȥ����Ļ��Ե 1/4��������,��ֹ�Ų����������һƬ��
	Rect rgRoi(0, 0, dm.cols, dm.rows*3/4);
	Mat fgMsk = zc::_simpleRegionGrow(dm, seed, rgThresh, rgRoi, ZC_DEBUG_LV1);
	//cout<<"simpleRegionGrow.t: "<<clock()-begt<<endl;
	rgSumt += (clock()-begt);
	if(ZC_WRITE)
		imwrite("simpFlagMat_"+std::to_string((long long)fid)+".jpg", fgMsk);


	//�����ֳ۴��룺
	CapgSkeleton tSklt;
	Mat tmpDm;
	dm.convertTo(tmpDm, CV_32SC1);
	//����������ֵ������ǰ��������ɫ��
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
	bpr->predictAndMergeJoint(&depthImg, tSklt, &maskImg, false, false, ZC_DEBUG_LV1);
	predAndMergeSumt  += (clock()-begt);
}//nmhSilhouAndSklt
#endif // CAPG_SKEL_VERSION_0_1


void main(int argc, char **argv){
	freopen("_debug_info.log", "w", stdout);

	float zoomFactor = 1;

	int frameCnt = 0;

	XnStatus rc = XN_STATUS_OK;
	
	xn::Context ctx;
	rc = ctx.Init();
	if(!checkOpenNIError(rc, "init context"))
		return;

	const char *oniFname = "E:/oni_data/oni132x64/zc-walk-wo-feet-qvga.oni";
	oniFname = "E:/oni_data/oni132_orig/sun_han_short.oni";
// 	oniFname = "E:/oni_data/oni132x64/sgf_zc_w_feet.oni";
// 	oniFname = "E:/oni_data/oni132x64/sgf_zc_w_feet-190-3xx.oni";
// 	oniFname = "E:/oni_data/oni132x64/sgf_zc_w_feet-650-776.oni";
// 	//oniFname = "E:/oni_data/oni132x64/sgf_zc_w_feet-372-591.oni";
// 	
	//oniFname = "E:/oni_data/oni132x64/sgf-zc-sit-front-desk.oni";
// 	oniFname = "E:/oni_data/oni132x64/sgf-zc-sit-low-front-desk-wall.oni";
	oniFname = "E:/oni_data/oni132x64/sgf-zc-sit-no-front-desk-wall.oni";	//������ɳ��
// 	oniFname = "E:/oni_data/oni132_orig/zc-indoor_sit.oni";
//  	oniFname = "E:/oni_data/oni132_orig/zc-indoor_sit2.oni";

// 	oniFname = "E:/oni_data/oni132_orig/sgf-zc_indoor_issue_gesture.oni";
	//oniFname = "E:/oni_data/oni132_orig/sgf-zc_indoor_issue_gesture-102-350.oni";
	//oniFname = "E:/oni_data/oni132_orig/sgf-zc_indoor_issue_gesture-2156-2589.oni";
	
	//oniFname = "E:/oni_data/oni132x64/sgf_zc_w_feet-wo-overlap.oni";
	
	//oniFname = "E:/oni_data/oni132x64/zc-walk-wo-feet-qvga-107x.oni";
		//oniFname = "E:/oni_data/oni132_orig/zc_indoor_walk.oni";
// 	 	oniFname = "E:/oni_data/oni132_orig/zc_indoor_walk-last.oni";
	// 	oniFname = "E:/oni_data/oni132x64/zc-stand-w-feet.oni";
	//oniFname = "E:/oni_data/oni132x64/zc-stand-wo-feet-qvga.oni";
// 	oniFname = "E:/oni_data/oni132_orig/zc_indoor_stand.oni";
	//oniFname = "E:/oni_data/oni132_orig/zc_indoor_touch_panel.oni"; //���ִ�����ƽ���
	//oniFname = "E:/oni_data/oni132_orig/indoor-driver-noise.oni";	//�ԱȲ��� MS/primesense ����
	oniFname = "E:/oni_data/oni@orbbec/��.oni";
	oniFname = "E:/oni_data/oni@orbbec/�ɻ�.oni";

	//oniFname = ""; //����

	xn::Player plyr;
	rc = ctx.OpenFileRecording(oniFname, plyr);
	int frameOffset = std::stoi(argv[1]);
	plyr.SeekToFrame("Depth1", frameOffset, XN_PLAYER_SEEK_SET);
	plyr.SetRepeat(string(argv[2])=="true"); //���� OpenFileRecording ֮�����Ч
	if(!checkOpenNIError(rc, "ctx.OpenFileRecording"))
		return;

	xn::DepthGenerator dg;
	rc = dg.Create(ctx);
	if(!checkOpenNIError(rc, "create dg"))
		return;
	xn::DepthMetaData depthMD;

	XnMapOutputMode mapMode;
// 	mapMode.nXRes = QVGA_WIDTH;
// 	mapMode.nYRes = QVGA_HEIGHT;
// 	mapMode.nFPS = 30;
// 	rc = dg.SetMapOutputMode(mapMode);
// 	if(!checkOpenNIError(rc, "dg.SetMapOutputMode"))
// 		dg.GetMapOutputMode(mapMode);
	dg.GetMapOutputMode(mapMode);
	
	zoomFactor = 1. * QVGA_WIDTH / mapMode.nXRes;

	rc = ctx.StartGeneratingAll();
	if(!checkOpenNIError(rc, "ctx.StartGeneratingAll"))
		return;

	char key = 0;
	while(1){
		frameCnt++;
		//frameCnt = depthMD.FrameID(); //����ʱ�ã����� .FrameID
		int fid = depthMD.FrameID();
		cout<<"frameCnt, fid: "<<frameCnt<<", "<<fid<<endl;

		ctx.WaitAndUpdateAll();

		dg.GetMetaData(depthMD);
		Mat dmat(depthMD.FullYRes(), depthMD.FullXRes(), CV_16UC1, (void*)depthMD.Data());

		//��ʱȥ��̫������ұ��������Ϊ���
		//dm.setTo(MAX_VALID_DEPTH, dm>6000);
		//pyrDown(dm, dm, Size(dm.cols*zoomFactor, dm.rows*zoomFactor));
		//dm.resize(Size(dm.cols*zoomFactor, dm.rows*zoomFactor));	//��, is cv:resize
		Mat dm_resized; //��������ʱ���ڴ治�ɸģ��ʴ�
		resize(dmat, dm_resized, Size(dmat.cols*zoomFactor, dmat.rows*zoomFactor));
		dmat = dm_resized;

		dmat.convertTo(dm_draw, CV_8U, 1.*UCHAR_MAX / MAX_VALID_DEPTH);
		imshow("dm_draw", dm_draw);

#ifdef CAPG_SKEL_VERSION_0_1
#pragma region //���� distMap2contours[Debug], dmat2TopDownView[Debug]
// 		begt = clock();
// 		Mat cont_draw = zc::distMap2contoursDebug(dm, ZCDEBUG_LV1);
// 		distMap2contoursSumt += (clock()-begt);
// 
// 		if(ZCDEBUG_LV1){
// 			imshow("cont_draw", cont_draw);
// 			if(ZCDEBUG_WRITE)
// 				imwrite("cont_draw_"+std::to_string((long long)fid)+".jpg", cont_draw);
// 		}
		
		//���ԶԱ���������distMap2contours�� distMap2contoursDebugʵ���Ƿ�һ�£�
		Mat dist_cont_debug_draw;
		vector<vector<Point> > distMapContGood = zc::distMap2contours(dmat, ZC_DEBUG_LV1, dist_cont_debug_draw);

		Mat dist_cont_draw_good = Mat::zeros(dmat.size(), CV_8UC1);
		drawContours(dist_cont_draw_good, distMapContGood, -1, 255, -1);
		if(ZC_DEBUG_LV1){
			imshow("dist_cont_debug_draw", dist_cont_debug_draw);
			if(ZC_WRITE)
				imwrite("dist_cont_debug_draw_"+std::to_string((long long)fid)+".jpg", dist_cont_debug_draw);

			imshow("dist_cont_draw_good", dist_cont_draw_good);
		}

// 		static int topDownViewSumt = 0;
// 		begt = clock();
// 		Mat topDownView = zc::dmat2TopDownViewDebug(dm, ZCDEBUG_LV1);
// 
// 		int anch = std::stoi(argv[3]);
// 		Mat morphKrnl = getStructuringElement(MORPH_RECT, Size(anch*2+1, anch*2+1), Point(anch, anch) );
// 		morphologyEx(topDownView, topDownView, CV_MOP_CLOSE, morphKrnl);
// 
// 		topDownViewSumt += (clock()-begt);
// 		if(ZCDEBUG_LV1){
// 			cout<<"topDownViewSumt.rate: "<<1.*topDownViewSumt/frameCnt<<endl;
// 			imshow("topDownView", topDownView);
// 			if(ZCDEBUG_WRITE)
// 				imwrite("topDownView_"+std::to_string((long long)fid)+".jpg", topDownView);
// 		}

		//---------------top-down-view �ϻ� conts, ���ߡ�ϸ��bboxs
		Mat tdv_debug_draw;
		vector<vector<Point>> tdv_cont_good = zc::dmat2TopDownView(dmat, 0.0255, ZC_DEBUG_LV1, tdv_debug_draw);
		Mat tdv_cont_good_draw = Mat::zeros(dmat.size(), CV_8UC1);
		drawContours(tdv_cont_good_draw, tdv_cont_good, -1, 255, -1);
		if(ZC_DEBUG_LV1){
			imshow("tdv_cont_good_draw", tdv_cont_good_draw);
			imshow("tdv_debug_draw", tdv_debug_draw);
			if(ZC_WRITE)
				imwrite("tdv_debug_draw_"+std::to_string((long long)fid)+".jpg", tdv_debug_draw);
		}

		//---------------��top-down-view ��ͬʱ������ contours, bbox, �۲��󽻽��
		begt = clock();
		Mat bbox_cross_draw;
		vector<Mat> bboxMsks = zc::findFgMasksUseBbox(dmat, ZC_DEBUG_LV1, bbox_cross_draw);
		cout<<"findFgMasksUseBbox.rate: "<<1.*(clock()-begt)/(fid+1)<<endl;
		if(ZC_DEBUG_LV1)
			imshow("bbox_cross_draw", bbox_cross_draw);

		cout<<"bboxMsks.size: "<<bboxMsks.size()<<endl;
		if(ZC_DEBUG_LV1){
			imshow("bbox_cross_draw", bbox_cross_draw);
			if(ZC_WRITE)
				imwrite("bbox_cross_draw_"+std::to_string((long long)fid)+".jpg", bbox_cross_draw);

// 			if(bboxMsks.size() > 0){
// 				Mat bboxMsksSum = bboxMsks[0];
// 				for(size_t i=1; i<bboxMsks.size(); i++)
// 					bboxMsksSum += bboxMsks[i];
// 				imshow("bboxMsksSum", bboxMsksSum);
// 				if(ZCDEBUG_WRITE)
// 					imwrite("bboxMsksSum_"+std::to_string((long long)fid)+".jpg", bboxMsksSum);
// 			}

			Mat humans_draw = zc::getHumansMask(bboxMsks, dmat.size());
			imshow("humans_draw", humans_draw);
			if(ZC_WRITE)
				imwrite("humans_draw_"+std::to_string((long long)fid)+".jpg", humans_draw);

		}

#pragma endregion //���� distMap2contours[Debug], dmat2TopDownView[Debug]

#pragma region //���ԡ�����������
		//background subtraction:
		Mat bgsMat = zc::simpleMask(dmat, ZC_DEBUG_LV1);

#pragma endregion //���ԡ�����������

		//---------------1. NMH �����ӵ㣬silhouette�� skeleton
		//nmhSilhouAndSklt(dm, fid);

		//---------------2. �����Ѱ��ͷ�����ӵ㣺
		normalize(dmat, dm_draw, UCHAR_MAX, 0, NORM_MINMAX, CV_8UC1);

// 		begt = clock();
// 		const string sgf_configPath = "../../sgf_seed/config.txt",
// 			sgf_headTemplatePath = "../../sgf_seed/headtemplate.bmp";
// 		vector<Point> sgfSeeds = zc::getHeadSeeds(dm, 
// 			sgf_configPath, sgf_headTemplatePath, ZCDEBUG_LV1);
// 		sgfSeedSumt += (clock()-begt);
// 		cout<<"sgfSeeds.size(): "<<sgfSeeds.size()<<endl;
// 		
// 
// 		for(size_t i=0; i<sgfSeeds.size(); i++){
// 			Point sdi = sgfSeeds[i];
// 			circle(dm_draw, sdi, 9, 255, 2);
// 		}
		imshow("1-dm_draw", dm_draw);
		if(ZC_WRITE)
			imwrite("1-dm_draw_"+std::to_string((long long)fid)+".jpg", dm_draw);

		begt = clock();
		Mat flrApartMsk = zc::getFloorApartMask(dmat, false);
		cout<<"getFloorApartMask.rate: "<<1.*(clock()-begt)/(fid+1)<<endl;

		//+++++++++++++++
		begt = clock();
		//vector<Mat> fgMsks = zc::simpleRegionGrow(dm, sgfSeeds, rgThresh, flrApartMsk, true, ZCDEBUG_LV1);
		vector<Mat> fgMsks = zc::findFgMasksUseBbox(dmat, false);

		rg2msksSumt += (clock()-begt);
		int regionCnt = fgMsks.size();
// 		Mat fgMsksSum = fgMsks[0];
// 		for(size_t i=1; i<regionCnt; i++)
// 			fgMsksSum += fgMsks[i];
// 		imshow("fgMsksSum", fgMsksSum);

// 		for(size_t i=0; i<regionCnt; i++){
// 			if(ZCDEBUG)
// 				imshow("fgMsks-"+std::to_string((long long)i), fgMsks[i]);
// 			if(ZCDEBUG_WRITE)
// 				imwrite("fgMsks-"+std::to_string((long long)i)+"_"+std::to_string((long long)fid)+".jpg", fgMsks[i]);
// 		}

		int anch = std::stoi(argv[3]);
		Mat morphKrnl = getStructuringElement(MORPH_RECT, Size(anch*2+1, anch*2+1), Point(anch, anch) );

		vector<CapgSkeleton> sklts;
		for(size_t ir=0; ir<regionCnt; ir++){
			Mat soloFgMsk = fgMsks[ir];

			if(ZC_DEBUG_LV2)
				imshow("soloFgMsk-"+std::to_string((long long)ir), soloFgMsk);

			//open, �����cont, ������ˣ���predict�õ�sklt��
			morphologyEx(soloFgMsk, soloFgMsk, CV_MOP_OPEN, morphKrnl);

			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			findContours(soloFgMsk, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
			size_t contSz = contours.size();

			//���ܿ�����֮��ͼ���ȫ�ڣ�
			if(0==contSz)
				continue;

			int contMaxCnt = 0, idx = 0;
			for(size_t ic = 0; ic < contSz; ic++){
				if(contours[ic].size() > contMaxCnt){
					idx = ic;
					contMaxCnt = contours[ic].size();
				}
			}
// 			if(idx==-1)
// 				int dummy = 0;
			//vector<Point> &theCont = contours[idx];
			if(zc::isHumanContour(contours[idx])){
				
				Mat soloContMask = Mat::zeros(dmat.size(), CV_8UC1);
				drawContours(soloContMask, contours, idx, 255, -1);

				//�����ֳ۴��룺
				Mat dm32s;
				dmat.convertTo(dm32s, CV_32SC1);
				//����������ֵ������ǰ��������ɫ��
				dm32s.setTo(INT_MAX, soloContMask==0);
				//imshow("tmpDm", tmpDm);

				IplImage depthImg = dm32s;
				bool useDense = false,
					useErode = false,
					usePre = true;

				string featurePath = "../../feature";
				BPRecognizer *bpr = zc::getBprAndLoadFeature(featurePath);

				//��Ϊpredict�������صģ�����һ��mask��������ǰ������Ӱ������
				begt = clock();
				Mat soloLabelMat = bpr->predict(&depthImg, nullptr, useDense, usePre);
				predictSumt += (clock()-begt);

				if(ZC_DEBUG_LV2){
					Mat rgbSoloLabelMat = label_gray2rgb(soloLabelMat);
					imshow("rgbSoloLabelMat-"+std::to_string((long long)ir), rgbSoloLabelMat);
					if(ZC_WRITE)
						imwrite("rgbSoloLabelMat-"+std::to_string((long long)ir)+"_"+std::to_string((long long)fid)+".jpg", rgbSoloLabelMat);
				}

				useErode = false;
				usePre = false;

				CapgSkeleton tSklt;
				bpr->mergeJoint(new IplImage(soloLabelMat), &depthImg, tSklt, useErode, usePre);
				sklts.push_back(tSklt);
			}//if(isHumanContour(contours[idx]))
			
		}//for(size_t ir=0; ir<regionCnt; ir++)
		
		//+++++++++++++++


// 		Mat sgfFgMsk = zc::_simpleRegionGrow(dm, sgfSeeds, rgThresh, flrApartMsk, ZCDEBUG);
// 		imshow("2-sgfFgMsk", sgfFgMsk);
// 
// // �������������ж���Щ���ˣ� ���ݣ� 1. bbox����ȣ� 2. bbox.height ����ߴ硣��δ��ɡ�
// // 		int anch = std::stoi(argv[3]);
// // 		Mat morphKrnl = getStructuringElement(MORPH_RECT, Size(anch*2+1, anch*2+1), Point(anch, anch) );
// 		Mat sgfFgMsk_open;
// 		morphologyEx(sgfFgMsk, sgfFgMsk_open, CV_MOP_OPEN, morphKrnl);
// 		imshow("3-sgfFgMsk_open", sgfFgMsk_open);
//  		
//  		if(ZCDEBUG_WRITE){
//  			imwrite("2-sgfFgMsk_"+std::to_string((long long)fid)+".jpg", sgfFgMsk);
//  			imwrite("3-sgfFgMsk_open_"+std::to_string((long long)fid)+".jpg", sgfFgMsk_open);
//  		}
// 
// 		sgfFgMsk = sgfFgMsk_open;
// 
// 		//�����ֳ۴��룺
// 		Mat tmpDm;
// 		dm.convertTo(tmpDm, CV_32SC1);
// 		//����������ֵ������ǰ��������ɫ��
// 		tmpDm.setTo(INT_MAX, sgfFgMsk==0);
// 		//imshow("tmpDm", tmpDm);
// 
// 		IplImage depthImg = tmpDm;
// 		bool useDense = false,
// 			useErode = false,
// 			usePre = true;
// 
// 		string featurePath = "../../feature";
// 		BPRecognizer *bpr = zc::getBprAndLoadFeature(featurePath);
// 
// 		//��Ϊpredict�������صģ�����һ��mask��������ǰ������Ӱ������
// 		begt = clock();
// 		Mat labelMat = bpr->predict(&depthImg, nullptr, useDense, usePre);
// 		predictSumt += (clock()-begt);
// 
// 		if(ZCDEBUG){
// 			imshow("labelMat", labelMat);
// 			Mat rgbLabelMat = label_gray2rgb(labelMat);
// 			imshow("6-rgbLabelMat", rgbLabelMat);
// 			if(ZCDEBUG_WRITE)
// 				imwrite("6-rgbLabelMat_"+std::to_string((long long)fid)+".jpg", rgbLabelMat);
// 		}
// 
// 		//�����������ָ�Ϊ n*���� mat��
// 		vector<vector<Point> > contours;
// 		vector<Vec4i> hierarchy;
// 		findContours(sgfFgMsk, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
// 
// 		cout<<"contours.size(): "<<contours.size()<<endl;
// 		const size_t origContSize = contours.size();
// 		for(size_t i = 0; i < origContSize; i++)
// 			cout<<"i: "<<i<<", "<<contours[i].size()<<endl;
// 		
// 		if(contours.size()>2)
// 			int dummy = 0;
// 		zc::eraseNonHumanContours(contours);
// 		cout<<"eraseNonHumanContours.size(): "<<contours.size()<<endl;
// 
// 		const size_t contSize = contours.size();
// 		vector<vector<Point> > contours_poly(contSize);
// 		vector<Rect> boundRect(contSize);
// 		vector<Point2f> center(contSize);
// 		vector<float> radius(contSize);
// 
// 		vector<CapgSkeleton> sklts;
// 		
// 		//���Ի���
// 		Mat contMat = Mat::zeros(sgfFgMsk.size(), CV_8UC1);
// 
// 		for(size_t i = 0; i < contSize; i++){
// 			approxPolyDP((contours[i]), contours_poly[i], 3, true);
// 
// 			//����
// 			Mat soloContMask = Mat::zeros(sgfFgMsk.size(), CV_8UC1);
// 			drawContours(soloContMask, contours_poly, i, 255, -1);
// 			Mat soloLabelMat = labelMat.clone();
// 			soloLabelMat.setTo(BodyLabel_Background, soloContMask==0);
// 
// 			bool useErode = false,
// 				usePre = false;
// 
// 			CapgSkeleton tSklt;
// 			bpr->mergeJoint(new IplImage(soloLabelMat), &depthImg, tSklt, useErode, usePre);
// 			sklts.push_back(tSklt);
// 
// 			if(ZCDEBUG){
// 				//���� bbox, enclosingCircle:
// 				boundRect[i] = boundingRect(contours_poly[i]);
// 				minEnclosingCircle(contours_poly[i], center[i], radius[i]);
// 				//���Ի���
// 				Scalar c(255);
// 				drawContours(contMat, contours_poly, i, c, -1);
// 				drawContours(contMat, contours, i, c);
// 				rectangle(contMat, boundRect[i].tl(), boundRect[i].br(), c);
// 				circle(contMat, center[i], radius[i], c);
// 
// 				Mat soloRgbLabelMat = label_gray2rgb(soloLabelMat);
// 				imshow("5-soloContMask-"+std::to_string((long long)i), soloContMask);
// 				imshow("7-soloRgbLabelMat-"+std::to_string((long long)i), soloRgbLabelMat);
// 				cout<<"===tSklt.size(): "<<tSklt.size()<<", "<<contours[i].size()<<", "<<contours_poly[i].size()<<endl;
// 
// 				if(ZCDEBUG_WRITE){
// 					imwrite("5-soloContMask-"+std::to_string((long long)i)+"_"+std::to_string((long long)fid)+".jpg", soloContMask);
// 					imwrite("7-soloRgbLabelMat-"+std::to_string((long long)i)+"_"+std::to_string((long long)fid)+".jpg", soloRgbLabelMat);
// 				}
// 			}
// 		}
// 		imshow("4-contMat", contMat);
// 		if(ZCDEBUG_WRITE)
// 			imwrite("4-contMat_"+std::to_string((long long)fid)+".jpg", contMat);

		cout<<"sklts.size(): "<<sklts.size()<<endl;

		Mat skCanvas = Mat::ones(dmat.size(), CV_8UC1)*UCHAR_MAX;
		zc::drawSkeletons(skCanvas, sklts, -1);
		imshow("8-skCanvas", skCanvas);
		if(ZC_DEBUG_LV1){
			imshow("8-skCanvas", skCanvas);
			if(ZC_WRITE)
				imwrite("8-skCanvas_"+std::to_string((long long)fid)+".jpg", skCanvas);
		}
#endif // CAPG_SKEL_VERSION_0_1

#ifdef CAPG_SKEL_VERSION_0_9 //��Ӧ ZC_CLEAN
#pragma region //һЩ����

		//---------------���Ը߶ȼ���
		Mat htMap0 = zc::calcHeightMap0(dmat);
		Mat htMap = zc::calcHeightMap1(dmat);
		if (ZC_DEBUG_LV1){
			Mat htMap0_show, htMap_show;
			int maxHt = 3e3;
			htMap0.convertTo(htMap0_show, CV_8U, 1.* UCHAR_MAX / maxHt);
			htMap.convertTo(htMap_show, CV_8U, 1.* UCHAR_MAX / maxHt);
			imshow("htMap0_show", htMap0_show);
			imshow("htMap_show", htMap_show);
		}

		//---------------����ת�� top-down-view
		{
			Mat flrApartMask = zc::fetchFloorApartMask(dmat, false);
			Mat maskedDmat_no_flr = dmat.clone();
			maskedDmat_no_flr.setTo(0, (flrApartMask == 0));
			Mat tdview = zc::dmat2tdview_core(maskedDmat_no_flr);
			imshow("tdview", tdview);
			tdview.convertTo(tdview, CV_8U);
			imshow("tdview", tdview);
		}

#pragma endregion //һЩ����

		clock_t begttotal = clock();

		//���룺��ʼ��prevDmat
		zc::initPrevDmat(dmat);
		static vector<Mat> prevMaskVec;
		static vector<HumanObj> humVec;

		clock_t begt = clock();
		//A.ȥ������ & ���棺
		Mat bgMsk = zc::fetchBgMskUseWallAndHeight(dmat);
		Mat flrApartMask = zc::fetchFloorApartMask(dmat, false);
		Mat maskedDmat_no_wall_flr = dmat.clone();
		maskedDmat_no_wall_flr.setTo(0, bgMsk | (flrApartMask == 0));
		//ֻȥ���棬��ȥ��ǽ��
		//maskedDmat_no_wall_flr.setTo(0, flrApartMask == 0);
		cout << "aaa.maskedDmat_no_wall_flr.ts: " << clock() - begt << endl;
		if (ZC_DEBUG_LV1){
			Mat maskedDmat_no_wall_flr_show;
			normalize(maskedDmat_no_wall_flr, maskedDmat_no_wall_flr_show, 0, UCHAR_MAX, NORM_MINMAX, CV_8UC1);
			imshow("atmp-maskedDmat_no_wall_flr", maskedDmat_no_wall_flr_show);
		}

		//B.������ǰ������ʼ�����˴��õ�bbox������
		begt = clock();
		//vector<Mat> fgMskVec = zc::findFgMasksUseWallAndHeight(dmat, ZC_DEBUG_LV1);
		Mat tmp;
		//vector<Mat> fgMskVec = zc::findFgMasksUseBbox(maskedDmat, ZC_DEBUG_LV1, tmp);
		int rgThresh = 55;
#if 0 //XY-XZ-bbox �����ж�
		vector<vector<Point>> sdBboxVov = zc::seedUseBboxXyXz(maskedDmat, ZC_DEBUG_LV1, tmp);
		vector<Mat> fgMskVec = zc::simpleRegionGrow(maskedDmat, sdBboxVov, rgThresh, flrApartMask, false);
#elif 0 //ͷ�������ж�
		vector<vector<Point>> sdHeadBodyVov = zc::seedUseHeadAndBodyCont(dmat, ZC_DEBUG_LV1, tmp);
		vector<Mat> fgMskVec = zc::simpleRegionGrow(maskedDmat, sdHeadBodyVov, rgThresh, flrApartMask, false);
#elif 1 //MOG2 �˶���ⷽ��
		{
			Mat dmat8u;
			dmat.convertTo(dmat8u, CV_8U, 1.*UCHAR_MAX / MAX_VALID_DEPTH);

			int noMoveThresh = 100;
			int history = 100;
			double varThresh = 1;
			double learnRate = -0.005;
			//Mat tmp;
// 			Mat testMsk = zc::maskMoveAndNoMove(dmat, prevMaskVec, noMoveThresh, history, varThresh, learnRate, ZC_DEBUG_LV1, tmp);

			bool detectShadows = false;
			static Ptr<BackgroundSubtractorMOG2> pMog2 = createBackgroundSubtractorMOG2(history, varThresh, detectShadows);
			Mat fgMskMog2;
			pMog2->apply(dmat8u, fgMskMog2);

			imshow("test0", fgMskMog2);
		}

		bool isNewFrame = true;
		int erodeRadius = 13;
		Mat sdMoveMat = zc::seedBgsMOG2(dmat, isNewFrame, erodeRadius, ZC_DEBUG_LV1, tmp);
		if (ZC_DEBUG_LV1){
			imshow("sdMoveMat", sdMoveMat);
			//imshow("seedBgsMOG2", tmp); //_debug_mat û�ã���ʱ����
		}

		rgThresh = 55;
		bool getMultiMasks = true;
		vector<Mat> fgMskVec = zc::simpleRegionGrow(maskedDmat, sdMoveMat, rgThresh, flrApartMask, getMultiMasks);
#endif //N�ֳ�����������

		cout << "bbb.findFgMasksUseBbox.ts: " << clock() - begt << endl;
		if (ZC_DEBUG_LV1){
			Mat btmp = zc::getHumansMask(fgMskVec, dmat.size());
			string txt = "fgMskVec.size: " + to_string((long long)fgMskVec.size());
			putText(btmp, txt, Point(0, 50), FONT_HERSHEY_PLAIN, 1, 255);
			imshow("btmp-bbox-init", btmp);
		}

		if (ZC_DEBUG_LV1){
			Mat prevMaskVec2msk = zc::getHumansMask(prevMaskVec, dmat.size());
			string txt = "prevMaskVec.size: " + to_string((long long) prevMaskVec.size());
			putText(prevMaskVec2msk, txt, Point(0, 50), FONT_HERSHEY_PLAIN, 1, 255);
			imshow("prevMaskVec2msk", prevMaskVec2msk);
		}


		//C.��������٣�ʹǰ����ȫ���ȶ���
		begt = clock();
		fgMskVec = zc::trackingNoMove(dmat, prevMaskVec, fgMskVec, ZC_DEBUG_LV1);
		cout << "ccc.trackingNoMove.ts: " << clock() - begt << endl;
		if (ZC_DEBUG_LV1){
			Mat ctmp = zc::getHumansMask(fgMskVec, dmat.size());
			string txt = "fgMskVec.size: " + to_string((long long) fgMskVec.size());
			putText(ctmp, txt, Point(0, 50), FONT_HERSHEY_PLAIN, 1, 255);
			imshow("ctmp-trackingNoMove", ctmp);
		}

		//D.������ͻȻ����ɼ����ֵ�ǰ���������롢���˴���
		begt = clock();

		//---------------����sep-xy��Ϊ�ԱȲ��ԣ�
		//fgMskVec = zc::separateMasksXYview(dmat, fgMskVec, ZC_DEBUG_LV1);

		//---------------Ŀǰʹ��sep-xz��
		fgMskVec = zc::separateMasksXZview(dmat, fgMskVec, ZC_DEBUG_LV1);
		prevMaskVec = fgMskVec;
		cout << "ddd.separateMasksXZview.ts: " << clock() - begt << endl;
		static int tSumt = 0;
		tSumt += (clock() - begttotal);
		cout << "find+tracking.rate: " << 1.*tSumt / (frameCnt + 1) << endl;

		if (ZC_DEBUG_LV1){
			Mat dtmp = zc::getHumansMask(fgMskVec, dmat.size());
			string txt = "fgMskVec.size: " + to_string((long long) fgMskVec.size());
			putText(dtmp, txt, Point(0, 50), FONT_HERSHEY_PLAIN, 1, 255);

			imshow("dtmp-separateMasks", dtmp);
		}

		Mat humMsk = zc::getHumansMask(fgMskVec, dmat.size());
		if (ZC_DEBUG_LV1){
			// 			QtFont font = fontQt("Times");
			// 			cv::addText(humMsk, "some-text", { 55, 55 }, font);
			putText(humMsk, "fid: " + to_string((long long) fid), Point(0, 30), FONT_HERSHEY_PLAIN, 1, 255);
			imshow("humMsk", humMsk);
			//if (ZC_WRITE)
			//	imwrite("humMsk_" + std::to_string((long long)fid) + ".jpg", humMsk);
		}

		zc::getHumanObjVec(dmat, fgMskVec, humVec);
		if (humVec.size())
			int dummy = 0;
		Mat humMsk��ɫ = zc::getHumansMask(dmat, humVec);
		if (ZC_DEBUG_LV1){
			putText(humMsk��ɫ, "fid: " + to_string((long long) fid), Point(0, 30), 
				FONT_HERSHEY_PLAIN, 1, Scalar(255, 255, 255));

			putText(humMsk��ɫ, "humVec.size: " + to_string((long long) humVec.size()), Point(0, 50), 
				FONT_HERSHEY_PLAIN, 1, Scalar(255, 255, 255));

			//���ƶ��˹�����
			Mat skCanvas = Mat::ones(dmat.size(), CV_8UC1)*UCHAR_MAX;
			zc::drawSkeletons(humMsk��ɫ, humVec, -1);
			imshow("humMsk-color", humMsk��ɫ);
			if (ZC_WRITE)
				imwrite("humMsk-color_" + std::to_string((long long)fid) + ".jpg", humMsk��ɫ);

		}

		//���룺 ���� prevDmat
		zc::setPrevDmat(dmat);

#elif defined CAPG_SKEL_VERSION_0_9_1
		//---------------�����ͷ�����ӵ㣬��ʼ�����ã�
		const char *sgfConfigFn = "d:/Users/zhangxaochen/Desktop/SenseKitSDK-0.1.4-20150424T043853Z-win32/samples/plugins/orbbec_skeleton/sgf_seed/config.txt";
		const char *sgfTempl = "D:/Users/zhangxaochen/Desktop/SenseKitSDK-0.1.4-20150424T043853Z-win32/samples/plugins/orbbec_skeleton/sgf_seed/headtemplate.bmp";
		// 	segment my_seg;
		// 	my_seg.read_config(sgfConfigFn);
		// 	my_seg.set_headTemplate2D(sgfTempl);
		sgf::loadSeedHeadConf(sgfConfigFn, sgfTempl);

		clock_t begt = clock();

		vector<Mat> fgMskVec = zc::getFgMaskVec(dmat, fid, ZC_DEBUG_LV1);

		cout << "getFgMaskVec.ts: " << clock() - begt << endl;
#endif //CAPG_SKEL_VERSION_0_9, 0_9_1

		begt = clock();
		static vector<HumanObj> humVec;
		zc::getHumanObjVec(dmat, fgMskVec, humVec);
		cout << "getHumanObjVec.ts: " << clock() - begt << endl;

		if (humVec.size())
			int dummy = 0;
#if 1
		bool debugWrite = ZC_WRITE;
		zc::debugDrawHumVec(dmat, fgMskVec, humVec, fid, debugWrite);
#else
		Mat humMsk��ɫ = zc::getHumansMask(dmat, humVec);
		if (ZC_DEBUG_LV1){
			putText(humMsk��ɫ, "fid: " + to_string((long long) fid), Point(0, 30), 
				FONT_HERSHEY_PLAIN, 1, Scalar(255, 255, 255));

			putText(humMsk��ɫ, "humVec.size: " + to_string((long long) humVec.size()), Point(0, 50), 
				FONT_HERSHEY_PLAIN, 1, Scalar(255, 255, 255));

			//���ƶ��˹�����
			Mat skCanvas = Mat::ones(dmat.size(), CV_8UC1)*UCHAR_MAX;
			zc::drawSkeletons(humMsk��ɫ, humVec, -1);
			imshow("humMsk-color", humMsk��ɫ);
			if (ZC_WRITE)
				imwrite("humMsk-color_" + std::to_string((long long)fid) + ".jpg", humMsk��ɫ);

		}
#endif

		key = waitKey(isManually ? 0 : 1);
		switch (key)
		{
		case 27: //ESC
			isExit = true;
			break;
		case ' ': //��ͣ&��֡
			isManually = !isManually;
			break;
		case 'r': //reset
			plyr.SeekToFrame("Depth1", frameOffset, XN_PLAYER_SEEK_SET);
#ifdef CAPG_SKEL_VERSION_0_9
			prevMaskVec.clear();
			humVec.clear();
#endif //CAPG_SKEL_VERSION_0_9
			break;

		case 'b':
			cout << "Enter a number to set the begin index:" << endl;
			cin >> frameOffset;
			if (cin.fail()){
				cin.clear();
				cin.ignore();

				frameOffset = 0;
			}
			break;
		}//switch

		if(isExit)
			break;
	}//while-1

	//ͳ�Ƽ���ƽ��ʱ��(ms)��
	cout<<"frameCnt:: seedSumt, sgfSeedSumt, rg2msksSumt, rgSumt, cannySumt, predAndMergeSumt, predictSumt, distMap2contoursSumt: "
		<<frameCnt<<":: "
		<<seedSumt*1./frameCnt<<", "<<sgfSeedSumt*1./frameCnt<<", "
		<<rg2msksSumt*1./frameCnt<<", "<<rgSumt*1./frameCnt<<", "<<cannySumt*1./frameCnt<<", "
		<<predAndMergeSumt*1./frameCnt<<", "<<predictSumt*1./frameCnt<<", "
		<<distMap2contoursSumt*1./frameCnt<<", "
		<<endl;

	destroyAllWindows();
	ctx.StopGeneratingAll();
	ctx.Release();

}//main
