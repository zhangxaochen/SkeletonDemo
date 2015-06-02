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

	BPRecognizer* getBprAndLoadFeature(const string &featurePath){
		static BPRecognizer *bpr = nullptr;
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
	Mat _simpleRegionGrow(const Mat &dmat, Point seed, int thresh, const Rect roi, bool debugDraw){
		vector<Point> seeds;
		seeds.push_back(seed);

		return _simpleRegionGrow(dmat, seeds, thresh, roi, debugDraw);
	}//simpleRegionGrow

	//预先用 roi 生成 mask
	Mat _simpleRegionGrow( const Mat &dmat, vector<Point> seeds, int thresh, const Rect roi, bool debugDraw /*= false*/ ){
		Mat _mask = Mat::zeros(dmat.size(), CV_8UC1);
		_mask(roi).setTo(UCHAR_MAX);

		return _simpleRegionGrow(dmat, seeds, thresh, _mask, debugDraw);
	}//simpleRegionGrow

	Mat _simpleRegionGrow( const Mat &dmat, Point seed, int thresh, const Mat &mask, bool debugDraw /*= false*/ ){
		vector<Point> seeds;
		seeds.push_back(seed);

		return _simpleRegionGrow(dmat, seeds, thresh, mask, debugDraw);
	}//simpleRegionGrow

	Mat _simpleRegionGrow(const Mat &dmat, vector<Point> seeds, int thresh, const Mat &mask, bool debugDraw /*= false*/){

		Size sz = dmat.size();
		int hh = sz.height,
			ww = sz.width;

// 		//实际进行区域增长的 roi：
// 		int top = max(0, roi.y),
// 			left = max(0, roi.x),
// 			bottom = min(hh, roi.y + roi.height),
// 			right = min(ww, roi.x + roi.width);

		//1. init
		//存标记：0未查看， 1在queue中， 255已处理过neibor；最终得到的正是 mask
		Mat flagMat = Mat::zeros(sz, CV_8UC1);

		//存满足条件的点
		queue<Point> pts;

		//初始种子点入队&标记，需满足条件：mask 有效
		for(size_t i=0; i<seeds.size(); i++){
			Point sd = seeds[i];
			//if(roi.contains(sd)){
			if(mask.at<uchar>(sd) == UCHAR_MAX){
				flagMat.at<uchar>(sd) = 1;
				pts.push(sd);
			}
		}
		//flagMat.at<uchar>(seed) = 1;
		//pts.push(seed);

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
				//roi 判断不要了：
				//if (left <= npt.x && npt.x < right && top <= npt.y && npt.y < bottom)
				if (Rect(Point(), dmat.size()).contains(npt))
				{
					const ushort& depNpt = dmat.at<ushort>(npt);
					uchar& flgNpt = flagMat.at<uchar>(npt);
					if (flgNpt == 0
						&& abs(depPt - depNpt) <= thresh
						//增加mask 判断：
						&& mask.at<uchar>(npt) == UCHAR_MAX){
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

	vector<Mat> simpleRegionGrow(const Mat &dmat, vector<Point> seeds, int thresh, const Mat &mask, bool getMultiMasks /*= false*/, bool debugDraw /*= false*/){
		vector<Mat> res;
		if(!getMultiMasks){
			Mat msk = _simpleRegionGrow(dmat, seeds, thresh, mask, debugDraw);
			res.push_back(msk);
		}
		else{//getMultiMasks
			size_t sdsz = seeds.size();
			for(size_t i = 0; i < sdsz; i++){
				Point sdi = seeds[i];

				bool regionExists = false;
				int regionCnt = res.size();
				for(size_t i = 0; i < regionCnt; i++){
					if(res[i].at<uchar>(sdi)==UCHAR_MAX)
						regionExists = true;
				}

				//若sdi不存在于之前任何一个mask，则新增长一个：
				if(!regionExists)
					res.push_back(_simpleRegionGrow(dmat, sdi, thresh, mask, debugDraw));
			}
		}
		return res;
	}//simpleRegionGrow

	Mat getFloorApartMask( Mat dmat, bool debugDraw /*= false*/){
		//分离地面滤波，做成mask：
		int flrKrnlArr[] = {1,1,1,1,1, -1,-1,-1,-1,-1};
		Mat flrKrnl((sizeof flrKrnlArr)/(sizeof flrKrnlArr[0]), 1, CV_32S, flrKrnlArr);
		//cv::flip(flrKrnl, flrKrnl, 0);

		Mat flrApartMat;
		filter2D(dmat, flrApartMat, CV_32F, flrKrnl);
		Mat flrApartMsk = abs(flrApartMat)<500;
		//上半屏不管，防止手部、肩部被误删过滤：
		flrApartMsk(Rect(0,0, dmat.cols, dmat.rows/2)).setTo(UCHAR_MAX);

		if(debugDraw){
			Rect flroi(60, 200, 10, 10);
			//cout<<flrApartMat(flroi)<<endl;

			//floorApartMat = cv::abs(floorApartMat);
			flrApartMat.setTo(0, abs(flrApartMat)>2000);
			rectangle(flrApartMat, flroi, 0);
			imshow("floorApartMat", flrApartMat);

			Mat flrApartMat_draw;
			normalize(flrApartMat, flrApartMat_draw, 0, UCHAR_MAX, NORM_MINMAX, CV_8UC1);
			imshow("flrApartMat_draw", flrApartMat_draw);
			imshow("flrApartMsk", flrApartMsk);
		}

		return flrApartMsk;
	}//getFloorApartMask

	void drawOneSkeleton(Mat &img, CapgSkeleton &sk){
		line(img, Point(sk[0].x(), sk[0].y()), Point(sk[1].x(), sk[1].y()), Scalar(0), 2);
		line(img, Point(sk[1].x(), sk[1].y()), Point(sk[2].x(), sk[2].y()), Scalar(0), 2);
		line(img, Point(sk[3].x(), sk[3].y()), Point(sk[4].x(), sk[4].y()), Scalar(0), 2);
		line(img, Point(sk[4].x(), sk[4].y()), Point(sk[5].x(), sk[5].y()), Scalar(0), 2);
		line(img, Point(sk[0].x(), sk[0].y()), Point(sk[3].x(), sk[3].y()), Scalar(0), 2);
		line(img, Point(sk[6].x(), sk[6].y()), Point(sk[7].x(), sk[7].y()), Scalar(0), 2);

		circle(img, Point(sk[8].x(), sk[8].y()), 2, Scalar(0), 2);
		circle(img, Point(sk[11].x(), sk[11].y()), 2, Scalar(0), 2);
		if(sk[8].x()!=0&&sk[8].y()!=0&&sk[11].x()!=0&&sk[11].y()!=0)
		{
			line(img, Point(sk[8].x(), sk[8].y()), Point(sk[11].x(), sk[11].y()), Scalar(0), 2);
			line(img, Point(sk[7].x(), sk[7].y()),
				Point((sk[8].x()+sk[11].x())/2, (sk[8].y()+sk[11].y())/2), Scalar(0), 2);
		}
	}//drawOneSkeleton

	void drawSkeletons(Mat &img, const vector<CapgSkeleton> &sklts, int skltIdx){
		if(skltIdx>=0){
			CapgSkeleton sk = sklts[skltIdx];
			drawOneSkeleton(img, sk);
		}
		else{
			for(size_t i = 0; i < sklts.size(); i++){
				CapgSkeleton sk = sklts[i];
				drawOneSkeleton(img, sk);
			}
		}
	}//drawSkeletons

	//contours 引用， 会被修改
	void eraseNonHumanContours(vector<vector<Point> > &contours){
		vector<vector<Point> >::iterator it = contours.begin();
		while(it != contours.end()){
			if(isHumanContour(*it))
				it++;
			else
				it = contours.erase(it);
		}//while
	}

	bool isHumanContour(const vector<Point> &cont){
		return cont.size() > 222;
	}//isHumanContour

	//目前仅通过蒙板前景像素点个数判断
	bool isHumanMask(const Mat &msk, int fgPxCntThresh /*= 1000*/){
		int fgPxCnt = countNonZero(msk==UCHAR_MAX);
		cout<<"fgPxCnt: "<<fgPxCnt<<endl;

		return fgPxCnt > fgPxCntThresh;
	}//isHumanMask

	//1. 对 distMap 二值化得到 contours； 2. 对 contours bbox 判断长宽比，得到人体区域
	Mat distMap2contours(const Mat &dmat, bool debugDraw /*= false*/){
		static int frameCnt = 0;

		Mat dm_draw,
			edge_up,		//脚部与地面相连，上半身
// 			edge_up_inv,	//黑色描边
			edge_ft,		//脚部与地面分离，下半身
// 			edge_ft_inv,	//黑色描边
			edge_whole,		//上身+脚部，地面分离
			edge_whole_inv,
			distMap,
			bwImg;			//宽黑边二值图
		normalize(dmat, dm_draw, 0, UCHAR_MAX, NORM_MINMAX, CV_8UC1);

		Canny(dm_draw, edge_up, 80, 160);
// 		edge_up_inv = (edge_up==0);
// 		if(debugDraw){
// 			imshow("edge_up_inv", edge_up_inv);
// 			imshow("edge_up", edge_up);
// 		}

		Mat flrApartMsk = getFloorApartMask(dmat);
		Canny(flrApartMsk, edge_ft, 64, 128);
// 		edge_ft_inv = (edge_ft==0);
// 		if(debugDraw){
// 			imshow("edge_ft_inv", edge_ft_inv);
// 			imshow("edge_ft", edge_ft);
// 		}

		edge_whole = edge_up + edge_ft;
		if(debugDraw){
			imshow("edge_whole", edge_whole);
			imwrite("edge_whole"+std::to_string((long long)frameCnt)+".jpg", edge_whole);
		}

		edge_whole_inv = (edge_whole==0);

		static int distSumt = 0;
		clock_t begt = clock();

		//distMap 是 CV_32FC1
		distanceTransform(edge_whole_inv, distMap, CV_DIST_L2, 3);
		normalize(distMap, distMap, 0, UCHAR_MAX, NORM_MINMAX, CV_8UC1);
		//对 distMap 二值化
		threshold(distMap, bwImg, 20, 255, THRESH_BINARY);

		distSumt += (clock()-begt);
		std::cout<<"distSumt.rate: "<<1.*distSumt/(frameCnt+1)<<std::endl;

		if(debugDraw){
			imshow("distanceTransform.distMap", distMap);
			imshow("threshold.distMap.bwImg", bwImg);
			imwrite("threshold.distMap.bwImg"+std::to_string((long long)frameCnt)+".jpg", bwImg);
		}

		//open
		static int anch = 4;
		static Mat morphKrnl = getStructuringElement(MORPH_RECT, Size(anch*2+1, anch*2+1), Point(anch, anch) );
		morphologyEx(bwImg, bwImg, CV_MOP_OPEN, morphKrnl);
		if(debugDraw){
			imshow("CV_MOP_OPEN.threshold.bwImg", bwImg);
			imwrite("CV_MOP_OPEN.threshold.bwImg"+std::to_string((long long)frameCnt)+".jpg", bwImg);
		}

		//对 edge erode， 与distMap 二值化会有什么区别？MORPH_ELLIPSE 时几乎完全相同
// 		static int erosion_size = 4,
// 			erosion_type = MORPH_RECT;
// 		static Mat erodeKrnl = getStructuringElement( erosion_type,
// 			Size( 2*erosion_size + 1, 2*erosion_size+1 ),
// 			Point( erosion_size, erosion_size ) );

		static int erodeSumt = 0;
		/*clock_t*/ begt = clock();
		erode(edge_whole_inv, edge_whole_inv, morphKrnl);
		erodeSumt += (clock()-begt);
		std::cout<<"+++++++++++++++erodeSumt.rate: "<<1.*erodeSumt/(frameCnt+1)<<std::endl;

		bwImg = edge_whole_inv;
		if(debugDraw){
			imshow("edge_whole_inv.erode.bwImg", edge_whole_inv);
			imwrite("edge_whole_inv.erode.bwImg"+std::to_string((long long)frameCnt)+".jpg", edge_whole_inv);
		}

		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(bwImg, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

		Mat cont_draw = Mat::zeros(dmat.size(), CV_8UC1);
		//drawContours(cont_draw, contours, -1, 255, -1);

		int contSz = contours.size();
		//vector<Rect> boundRect(contSz);
		for(size_t i = 0; i < contSz; i++){
			Rect boundRect = boundingRect(contours[i]);
			//boundRect[i] = boundingRect(contours[i]);
			Size bsz = boundRect.size();
			if(bsz.height*1./bsz.width > 1.5 && bsz.height > 60){
				drawContours(cont_draw, contours, i, 255, -1);
				if(debugDraw)
					rectangle(cont_draw, boundRect, 255);
			}
		}
		if(debugDraw)
			imshow("cont_draw", cont_draw);

		frameCnt++;
		//return distMap;
		return cont_draw;
	}//distMap2contours



	cv::Mat postRegionGrow( const Mat &flagMat, int xyThresh, int zThresh, bool debugDraw /*= false*/ )
	{
		static Mat prevFlagMat;
		//若第一帧：
		if(prevFlagMat.empty()){
			prevFlagMat = flagMat;
		}
		return flagMat;
	}//postRegionGrow

// 	cv::Mat erodeDilateN( Mat &m, int ntimes )
// 	{
// 		erode
// 
// 	}

}//zc
