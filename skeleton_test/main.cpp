#include <iostream>
#include <ctime>

#include <opencv2/opencv.hpp>
#include <XnCppWrapper.h>

#include "../SimpleSilhouette.h"
#include "../sgf_seed/sgf_segment.h"

using namespace std;
using namespace cv;

#define ZCDEBUG_LV1 1	//��ʾ�ؼ����ͼ
#define ZCDEBUG_LV2 0	//��ʾĳЩ�м���ͼ
#define ZCDEBUG_WRITE 1

int g_ImgIndex = 0;

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
	Mat currPreDiffMsk = zc::simpleMask(dm, ZCDEBUG_LV1);

	//���ӵ�Ѱ�ң�
	begt = clock();
	int veryDepth = -1;
	Point seed = zc::simpleSeed(dm, &veryDepth, ZCDEBUG_LV1);
	//cout<<"simpleSeed.t: "<<clock()-begt<<endl;
	seedSumt += (clock()-begt);

	circle(dm_draw, seed, 3, 255, 2);

	if(ZCDEBUG_LV1){
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
	Mat fgMsk = zc::_simpleRegionGrow(dm, seed, rgThresh, rgRoi, ZCDEBUG_LV1);
	//cout<<"simpleRegionGrow.t: "<<clock()-begt<<endl;
	rgSumt += (clock()-begt);
	if(ZCDEBUG_WRITE)
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
	bpr->predictAndMergeJoint(&depthImg, tSklt, &maskImg, false, false, ZCDEBUG_LV1);
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
	oniFname = "E:/oni_data/oni132x64/sgf_zc_w_feet-372-591.oni";
	
	//oniFname = "E:/oni_data/oni132x64/sgf-zc-sit-front-desk.oni";
	
	//oniFname = "E:/oni_data/oni132x64/sgf_zc_w_feet-wo-overlap.oni";
	
	//oniFname = "E:/oni_data/oni132x64/zc-walk-wo-feet-qvga-107x.oni";
	// 	oniFname = "E:/oni_data/oni132_orig/zc_indoor_walk.oni";
	// 	oniFname = "E:/oni_data/oni132_orig/zc_indoor_walk-last.oni";
	// 	oniFname = "E:/oni_data/oni132x64/zc-stand-w-feet.oni";
	//oniFname = "E:/oni_data/oni132x64/zc-stand-wo-feet-qvga.oni";

	xn::Player plyr;
	rc = ctx.OpenFileRecording(oniFname, plyr);
	plyr.SeekToFrame("Depth1", std::stoi(argv[1]), XN_PLAYER_SEEK_SET);
	plyr.SetRepeat(string(argv[2])=="true"); //���� OpenFileRecording ֮�����Ч
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
		//frameCntr = depthMD.FrameID(); //����ʱ�ã����� .FrameID
		int fid = depthMD.FrameID();
		cout<<"frameCntr, fid: "<<frameCntr<<", "<<fid<<endl;

		ctx.WaitAndUpdateAll();

		dg.GetMetaData(depthMD);
		Mat dm(depthMD.FullYRes(), depthMD.FullXRes(), CV_16UC1, (void*)depthMD.Data());

#pragma region //���� distMap2contours[Debug], dmat2TopDownView[Debug]
		begt = clock();
		Mat cont_draw = zc::distMap2contoursDebug(dm, ZCDEBUG_LV1);
		distMap2contoursSumt += (clock()-begt);

		if(ZCDEBUG_LV1){
			imshow("cont_draw", cont_draw);
			if(ZCDEBUG_WRITE)
				imwrite("cont_draw_"+std::to_string((long long)fid)+".jpg", cont_draw);
		}
		
		//���ԶԱ���������distMap2contours�� distMap2contoursDebugʵ���Ƿ�һ�£�
		vector<vector<Point> > distMapContGood = zc::distMap2contours(dm, ZCDEBUG_LV1);
		Mat dist_cont_draw_good = Mat::zeros(dm.size(), CV_8UC1);
		drawContours(dist_cont_draw_good, distMapContGood, -1, 255, -1);
		if(ZCDEBUG_LV1){
			imshow("dist_cont_draw_good", dist_cont_draw_good);
		}

		static int topDownViewSumt = 0;
		begt = clock();
		Mat topDownView = zc::dmat2TopDownViewDebug(dm, ZCDEBUG_LV1);

		int anch = std::stoi(argv[3]);
		Mat morphKrnl = getStructuringElement(MORPH_RECT, Size(anch*2+1, anch*2+1), Point(anch, anch) );
		morphologyEx(topDownView, topDownView, CV_MOP_CLOSE, morphKrnl);

		topDownViewSumt += (clock()-begt);
		if(ZCDEBUG_LV1){
			cout<<"topDownViewSumt.rate: "<<1.*topDownViewSumt/frameCntr<<endl;
			imshow("topDownView", topDownView);
			if(ZCDEBUG_WRITE)
				imwrite("topDownView_"+std::to_string((long long)fid)+".jpg", topDownView);
		}

		//top-down view ��ͨ�� bbox ������ֵ���ˣ�
		vector<vector<Point> > tdvContours;
		vector<Vec4i> tdvHierarchy;
		findContours(topDownView, tdvContours, tdvHierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		if(ZCDEBUG_LV1){
			Mat tdvCont_draw = Mat::zeros(topDownView.size(), CV_8UC1);
			size_t tdvContSize = tdvContours.size();
			for(size_t i = 0; i < tdvContSize; i++){
				drawContours(tdvCont_draw, tdvContours, i, 255, -1);
				Rect boundRect = boundingRect(tdvContours[i]);
				//Size bsz = boundRect.size();
				cout<<"tdvContours.boundRect: "<<boundRect<<"; "
					<<boundRect.height * MAX_VALID_DEPTH*1. / UCHAR_MAX<<", "<<boundRect.width * MAX_VALID_DEPTH*1. / UCHAR_MAX<<endl;

				//1. Z����ж� >3px(3*10000mm/256=117mm), <26(26*10000mm/256=1016mm)�� 
				//2. X����ж� (200mm~2000mm) MAX_VALID_DEPTH / UCHAR_MAX / XTION_FOCAL_XY = 1/7.68
				if(3 <= boundRect.height && boundRect.height < 26
					//&& 1.*MAX_VALID_DEPTH/UCHAR_MAX * boundRect.y / XTION_FOCAL_XY * boundRect.width < 2000 //y * width < 7.68*2000mm
					&& 7.68*200 < boundRect.y * boundRect.width && boundRect.y * boundRect.width < 7.68*2000)
				{
					rectangle(tdvCont_draw, boundRect, 255, 2);
					cout<<"��"<<endl;
				}
				else
					rectangle(tdvCont_draw, boundRect, 255);
			}
			drawContours(tdvCont_draw, tdvContours, -1, 255, -1);

			imshow("tdvCont_draw", tdvCont_draw);
			if(ZCDEBUG_WRITE)
				imwrite("tdvCont_draw_"+std::to_string((long long)fid)+".jpg", tdvCont_draw);

		}

		vector<vector<Point>> tdv_cont_good = zc::dmat2TopDownView(dm, ZCDEBUG_LV1);
		Mat tdv_cont_good_draw = Mat::zeros(dm.size(), CV_8UC1);
		drawContours(tdv_cont_good_draw, tdv_cont_good, -1, 255, -1);
		if(ZCDEBUG_LV1){
			imshow("tdv_cont_good_draw", tdv_cont_good_draw);
		}

		//---------------��top-down-view ��ͬʱ������ contours, bbox, �۲��󽻽��
		if(ZCDEBUG_LV1){
			Mat tdvCont_draw_cross = Mat::zeros(topDownView.size(), CV_8UC1);

			//tdview ʵ��cont�� bbox�� 128��ɫ��
			size_t tdv_cont_good_size = tdv_cont_good.size();
			Scalar gray_c = 128;
			for(size_t i = 0; i < tdv_cont_good_size; i++){
				drawContours(tdvCont_draw_cross, tdv_cont_good, i, gray_c, -1);
				Rect tmp_rect = boundingRect(tdv_cont_good[i]);
				rectangle(tdvCont_draw_cross, tmp_rect, gray_c, 2);
			}


			//dist-map ת��XZ����bbox
			size_t dtrans_cont_size = distMapContGood.size();
			for(size_t i = 0; i < dtrans_cont_size; i++){
				Mat cont_mask = Mat::zeros(dm.size(), CV_8UC1);
				drawContours(cont_mask, distMapContGood, i, 255, -1);
				//Z�����½磺
				double dmin, dmax;
				minMaxLoc(dm, &dmin, &dmax, nullptr, nullptr, cont_mask);
				//����һ���ж�����: mask ���ֵ���� >1500mm
				if(dmax-dmin > 1500)
					continue;

				//���ű� MAX_VALID_DEPTH / UCHAR_MAX������ top-down-view:
				Rect bbox_dtrans_cont = boundingRect(distMapContGood[i]);
				double ratio = 1. * UCHAR_MAX / MAX_VALID_DEPTH;
				Rect bbox_dtrans_cont_to_tdview(
					bbox_dtrans_cont.x, dmin * ratio,
					bbox_dtrans_cont.width, (dmax-dmin) * ratio);

				cout<<"bbox_dtrans_cont_to_tdview: "<<bbox_dtrans_cont_to_tdview<<"; "<<dmin<<", "<<dmax<<endl;
				rectangle(tdvCont_draw_cross, bbox_dtrans_cont_to_tdview, 255, 2);
			}//for(size_t i = 0; i < dtrans_cont_size; i++)


			imshow("tdvCont_draw_cross", tdvCont_draw_cross);
			if(ZCDEBUG_WRITE)
				imwrite("tdvCont_draw_cross_"+std::to_string((long long)fid)+".jpg", tdvCont_draw_cross);
		}

		begt = clock();
		vector<Mat> bboxMsks = zc::findHumanMasksUseBbox(dm, ZCDEBUG_LV1);
		cout<<"bboxMsks.size: "<<bboxMsks.size()<<endl;
		cout<<"findHumanMasksUseBbox.rate: "<<1.*(clock()-begt)/(fid+1)<<endl;

		if(ZCDEBUG_LV1 && bboxMsks.size() > 0){
			Mat bboxMsksSum = bboxMsks[0];
			for(size_t i=1; i<bboxMsks.size(); i++)
				bboxMsksSum += bboxMsks[i];
			imshow("bboxMsksSum", bboxMsksSum);
			if(ZCDEBUG_WRITE)
				imwrite("bboxMsksSum_"+std::to_string((long long)fid)+".jpg", bboxMsksSum);
		}

#pragma endregion //���� distMap2contours[Debug], dmat2TopDownView[Debug]

		//��ʱȥ��̫������ұ��������Ϊ���
		//dm.setTo(MAX_VALID_DEPTH, dm>6000);
		//pyrDown(dm, dm, Size(dm.cols*zoomFactor, dm.rows*zoomFactor));

		//---------------1. NMH �����ӵ㣬silhouette�� skeleton
		//nmhSilhouAndSklt(dm, fid);

		//---------------2. �����Ѱ��ͷ�����ӵ㣺
		normalize(dm, dm_draw, UCHAR_MAX, 0, NORM_MINMAX, CV_8UC1);

		begt = clock();
		const string sgf_configPath = "../../sgf_seed/config.txt",
			sgf_headTemplatePath = "../../sgf_seed/headtemplate.bmp";
		vector<Point> sgfSeeds = zc::getHeadSeeds(dm, 
			sgf_configPath, sgf_headTemplatePath, ZCDEBUG_LV1);
		sgfSeedSumt += (clock()-begt);
		cout<<"sgfSeeds.size(): "<<sgfSeeds.size()<<endl;
		

		for(size_t i=0; i<sgfSeeds.size(); i++){
			Point sdi = sgfSeeds[i];
			circle(dm_draw, sdi, 9, 255, 2);
		}
		imshow("1-dm_draw", dm_draw);
		if(ZCDEBUG_WRITE)
			imwrite("1-dm_draw_"+std::to_string((long long)fid)+".jpg", dm_draw);

		Mat flrApartMsk = zc::getFloorApartMask(dm, ZCDEBUG_LV1);

		//+++++++++++++++
		begt = clock();
		vector<Mat> fgMsks = zc::simpleRegionGrow(dm, sgfSeeds, rgThresh, flrApartMsk, true, ZCDEBUG_LV1);
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

// 		int anch = std::stoi(argv[3]);
// 		Mat morphKrnl = getStructuringElement(MORPH_RECT, Size(anch*2+1, anch*2+1), Point(anch, anch) );

		vector<CapgSkeleton> sklts;
		for(size_t ir=0; ir<regionCnt; ir++){
			Mat soloFgMsk = fgMsks[ir];

			if(ZCDEBUG_LV2)
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
				
				Mat soloContMask = Mat::zeros(dm.size(), CV_8UC1);
				drawContours(soloContMask, contours, idx, 255, -1);

				//�����ֳ۴��룺
				Mat tmpDm;
				dm.convertTo(tmpDm, CV_32SC1);
				//����������ֵ������ǰ��������ɫ��
				tmpDm.setTo(INT_MAX, soloContMask==0);
				//imshow("tmpDm", tmpDm);

				IplImage depthImg = tmpDm;
				bool useDense = false,
					useErode = false,
					usePre = true;

				string featurePath = "../../feature";
				BPRecognizer *bpr = zc::getBprAndLoadFeature(featurePath);

				//��Ϊpredict�������صģ�����һ��mask��������ǰ������Ӱ������
				begt = clock();
				Mat soloLabelMat = bpr->predict(&depthImg, nullptr, useDense, usePre);
				predictSumt += (clock()-begt);

				if(ZCDEBUG_LV2){
					Mat rgbSoloLabelMat = label_gray2rgb(soloLabelMat);
					imshow("rgbSoloLabelMat-"+std::to_string((long long)ir), rgbSoloLabelMat);
					if(ZCDEBUG_WRITE)
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

		Mat skCanvas = Mat::ones(dm.size(), CV_8UC1)*UCHAR_MAX;
		zc::drawSkeletons(skCanvas, sklts, -1);
		if(ZCDEBUG_LV1){
			imshow("8-skCanvas", skCanvas);
			if(ZCDEBUG_WRITE)
				imwrite("8-skCanvas_"+std::to_string((long long)fid)+".jpg", skCanvas);
		}

// 		if(fid>113)
// 			key = waitKey(0);
		key = waitKey(01);
	}//while

	//ͳ�Ƽ���ƽ��ʱ��(ms)��
	cout<<"frameCntr:: seedSumt, sgfSeedSumt, rg2msksSumt, rgSumt, cannySumt, predAndMergeSumt, predictSumt, distMap2contoursSumt: "
		<<frameCntr<<":: "
		<<seedSumt*1./frameCntr<<", "<<sgfSeedSumt*1./frameCntr<<", "
		<<rg2msksSumt*1./frameCntr<<", "<<rgSumt*1./frameCntr<<", "<<cannySumt*1./frameCntr<<", "
		<<predAndMergeSumt*1./frameCntr<<", "<<predictSumt*1./frameCntr<<", "
		<<distMap2contoursSumt*1./frameCntr<<", "
		<<endl;

	destroyAllWindows();
	ctx.StopGeneratingAll();
	ctx.Release();

}//main
