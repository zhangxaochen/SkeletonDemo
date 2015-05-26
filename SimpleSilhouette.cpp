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
	//zhangxaochen: �μ� hist_analyse.m ���ҵ�ʵ��
	Point simpleSeed(const Mat &dmat, int *outVeryDepth, bool debugDraw){
		Size sz = dmat.size();
		int hh = sz.height,
			ww = sz.width;

		//NMH�����Ҳ����� �������ĵ�
		Point ptCenter(ww / 2, hh / 2),
			res = ptCenter;

		//ROI �Ӵ��ڱ߽�
		int top = hh / 4,
			bottom = hh,
			left = ww / 4,
			right = ww * 3 / 4;

		//��ȷ�Χ�� dfar �����ں���� while �е����ı�
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
			//�����̫��(<90cm),��Զ�����Ǳ���, �����������ӵ�Ϊ��Ļ����:
			if (dfar <= dnear){
				res = ptCenter;
				veryDepth = dmat.at<ushort>(res);
				break;
			}
			int histSize = dfar - dnear; //ÿ����һ�� bar
			float range[] = { dnear, dfar };
			const float *histRange = { range };
			//calcHist(&roiMat, 1, 0, msk, histo, 1, &histSize, &histRange);
			//���Բ��� roiMat��
			calcHist(&dmat, 1, 0, msk, histo, 1, &histSize, &histRange);

			int cntUplim = ww*hh / 70;
			int cntMax = -1;
			veryDepth = -1;

			float cnt;
			for (int i = 0; i < histSize; i++){
				cnt = histo.at<float>(i);
				// 		if(cnt < cntUplim && cntMax < cnt){ //���� while(1), �Ͳ��������ж� cnt < cntUplim
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

			//����ֱ��ͼ
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
			//���Բ��� roiMat��
			findNonZero(dmat == veryDepth, vdPts);
			if (vdPts.total()>0){
				//res = vdPts.at<Point>(0)+Point(left, top);
				//���Բ��� roiMat��
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

		//ʵ�ʽ������������� roi��
		int top = max(0, roi.y),
			left = max(0, roi.x),
			bottom = min(hh, roi.y + roi.height),
			right = min(ww, roi.x + roi.width);

		//1. init
		//���ǣ�0δ�鿴�� 1��queue�У� 255�Ѵ����neibor�����յõ������� mask
		Mat flagMat = Mat::zeros(sz, CV_8UC1);

		//�����������ĵ�
		queue<Point> pts;
		flagMat.at<uchar>(seed) = 1;
		pts.push(seed);
		//Ŀǰ������������
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
				//�����ȥ������,��ֹ�Ų����������һƬ��
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


		//һ��ǰ�������Լ ww*hh/10
		static int prevFgPtCnt = ww*hh*15e-2;
		static Mat prevFlagMat = flagMat.clone();

		int currFgPtCnt = countNonZero(flagMat == UCHAR_MAX);
		// 	//�����һ֡ǰ�������ͻ�䣺 1.���� 50%�������Ǳ������룻 2.����̫�٣�������������ʧ�ܡ� �����֮
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
