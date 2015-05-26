#include <iostream>
#include "SimpleSilhouette.h"

namespace zc{

//from CKernal.cpp
#if defined(ANDROID)
#define FEATURE_PATH "/data/data/com.motioninteractive.zte/app_feature/"
#else
	//#define FEATURE_PATH "../Skeleton/feature"
#define FEATURE_PATH "../../../plugins/orbbec_skeleton/feature"
#endif

	static BPRecognizer *bpr = nullptr;
	BPRecognizer* getBprAndLoadFeature(const string &featurePath){
		if (nullptr == bpr){
			bpr = new BPRecognizer();
			//string featurePath(FEATURE_PATH);
			if (!bpr->load(const_cast<string&>(featurePath))){
				printf("body part feature loader fail\n");
				return nullptr;
			}
		}
		return bpr;
	}
	//zhangxaochen: 参见 hist_analyse.m 中我的实现
	Point simpleSeed(const Mat &dmat, int *outVeryDepth, bool debugDraw){
		Size sz = dmat.size();
		int hh = sz.height,
			ww = sz.width;

		//NMH：若找不到， 返回中心点
		Point ptCenter(ww / 2, hh / 2),
			res = ptCenter;

		//ROI 子窗口边界
		int top = hh / 4,
			bottom = hh,
			left = ww / 4,
			right = ww * 3 / 4;

		//深度范围， dfar 可能在后面的 while 中迭代改变
		int dnear = 900,
			dfar = 3000;

		Mat roiMat(dmat, Rect(left, top, right - left, bottom - top));
		//Mat msk = dnear <= roiMat & roiMat <= dfar;
		Mat msk = dnear <= dmat & dmat <= dfar;


		Mat histo;

		int veryDepth;
		//while(1){
		//do-while loop
		bool isLoop = true;
		do
		{
			//人离得太近(<90cm),但远的又是背景, 所以重置种子点为屏幕中心:
			if (dfar <= dnear){
				res = ptCenter;
				veryDepth = dmat.at<ushort>(res);
				break;
			}
			int histSize = dfar - dnear; //每毫米一个 bar
			float range[] = { dnear, dfar };
			const float *histRange = { range };
			//calcHist(&roiMat, 1, 0, msk, histo, 1, &histSize, &histRange);
			//尝试不用 roiMat：
			calcHist(&dmat, 1, 0, msk, histo, 1, &histSize, &histRange);

			int cntUplim = ww*hh / 70;
			int cntMax = -1;
			veryDepth = -1;

			float cnt;
			for (int i = 0; i < histSize; i++){
				cnt = histo.at<float>(i);
				// 		if(cnt < cntUplim && cntMax < cnt){ //若用 while(1), 就不在这里判断 cnt < cntUplim
				if (cntMax < cnt){
					cntMax = cnt;
					veryDepth = i + dnear;
				}
			}
			if (cntMax < cntUplim){
				isLoop = false;
				//printf("---------------isLoop = false\n");
				//break;
			}
			else{
				dfar = veryDepth - 200;
				//printf("new dfar: %d\n", dfar);
			}

			//printf("veryDepth: %d\n", veryDepth);

			//绘制直方图
			if (debugDraw){
				int histw = ww, histh = hh;
				double binw = histw*1. / histSize;
				//cvRound(histw*1./histSize);
				Mat histImage(histh, histw, CV_8UC1, Scalar::all(0));

				normalize(histo, histo, 0, histh - 10, NORM_MINMAX);
				int sft = 5;
				for (int i = 1; i < histSize; i++){
					int x1 = (binw * (i - 1)),
						y1 = (histh - cvRound(histo.at<float>(i - 1))) - sft,
						x2 = (binw * i),
						y2 = (histh - cvRound(histo.at<float>(i))) - sft;
					line(histImage, Point(x1, y1), Point(x2, y2), 111);
					//printf("%d, %d, %d, %d\n", x1, y1, x2, y2);
				}
				int vdInImg = (veryDepth - dnear)*histw / histSize;
				line(histImage, Point(vdInImg, 0), Point(vdInImg, histh), 255, 2);
				imshow("hist", histImage);
			}
			Mat vdPts;
			//findNonZero(roiMat==veryDepth, vdPts);
			//尝试不用 roiMat：
			findNonZero(dmat == veryDepth, vdPts);
			if (vdPts.total()>0){
				//res = vdPts.at<Point>(0)+Point(left, top);
				//尝试不用 roiMat：
				res = vdPts.at<Point>(0);//+Point(left, top);
			}
			else{
				printf("vdPts empty, veryDepth: %d\n", veryDepth);
			}
			//}//while
		} while (isLoop);

		*outVeryDepth = veryDepth;
		return res;
	}//simpleSeed

	Mat simpleMask(const Mat &curMat, bool debugDraw){
		static bool isFirst = true;
		static Mat prevMat;
		Mat mask;
		if (isFirst){
			isFirst = false;
			mask = Mat::zeros(curMat.size(), CV_8U);
		}
		else
			mask = cv::abs(prevMat - curMat) > 50;
		prevMat = curMat.clone();

		if (debugDraw)
			imshow("simpleMask", mask);

		return mask;
	}//simpleMask

	//zhangxaochen: simple region-grow, actually a flood fill method
	// RETURN: a mask mat, foreground is white
	Mat simpleRegionGrow(const Mat &dmat, Point seed, int thresh, const Rect roi, bool debugDraw){
		Size sz = dmat.size();
		int hh = sz.height,
			ww = sz.width;

		//实际进行区域增长的 roi：
		int top = max(0, roi.y),
			left = max(0, roi.x),
			bottom = min(hh, roi.y + roi.height),
			right = min(ww, roi.x + roi.width);

		//1. init
		//存标记：0未查看， 1在queue中， 255已处理过neibor；最终得到的正是 mask
		Mat flagMat = Mat::zeros(sz, CV_8UC1);

		//存满足条件的点
		queue<Point> pts;
		flagMat.at<uchar>(seed) = 1;
		pts.push(seed);
		//目前增长看四邻域：
		const int nnbr = 4;
		int dx[nnbr] = { 0, -1, 0, 1 },
			dy[nnbr] = { 1, 0, -1, 0 };

		//2. loop
		int maxPts = -1;
		while (!pts.empty()){
			int qsz = pts.size();
			if (qsz > maxPts)
				maxPts = qsz;

			Point pt = pts.front();
			const ushort& depPt = dmat.at<ushort>(pt);
			pts.pop();
			for (int i = 0; i < nnbr; i++){
				Point npt = pt + Point(dx[i], dy[i]);
				//纵向简单去掉地面,防止脚部与地面连成一片：
				//if(0<=npt.x && npt.x<ww && 0<=npt.y && npt.y<hh*6/8)
				if (left <= npt.x && npt.x < right && top <= npt.y && npt.y < bottom)
				{
					const ushort& depNpt = dmat.at<ushort>(npt);
					uchar& flgNpt = flagMat.at<uchar>(npt);
					if (flgNpt == 0
						&& abs(depPt - depNpt) <= thresh){
						//printf("val, nval: %d, %d\n", depPt, depNpt);
						flgNpt = 1;
						pts.push(npt);
					}
				}
			}
			flagMat.at<uchar>(pt) = UCHAR_MAX;
		}//while


		//一般前景点个数约 ww*hh/10
		static int prevFgPtCnt = ww*hh*15e-2;
		static Mat prevFlagMat = flagMat.clone();

		int currFgPtCnt = countNonZero(flagMat == UCHAR_MAX);
		// 	//如果新一帧前景点个数突变： 1.增多 50%，可能是背景误入； 2.个数太少，可能区域增长失败。 则放弃之
		// 	if(currFgPtCnt < ww*hh*1e-2 || currFgPtCnt*1./prevFgPtCnt > 1.5){
		// 		printf("currFgPtCnt/prevFgPtCnt > 1.5: %d, %d\n", prevFgPtCnt, currFgPtCnt);
		// 		flagMat = prevFlagMat;
		// 	}
		// 	else{
		// 		printf("~currFgPtCnt/prevFgPtCnt > 1.5: %d, %d\n", prevFgPtCnt, currFgPtCnt);
		// 		prevFgPtCnt = currFgPtCnt;
		// 		prevFlagMat = flagMat;
		// 	}

		if (debugDraw){
			//printf("maxPts: %d\n", maxPts);
			imshow("simpleRegionGrow.flagMat", flagMat);
			//cout<<flagMat(Rect(0,0, 5,5))<<endl;
		}

		return flagMat;
	}//simpleRegionGrow

}//zc
