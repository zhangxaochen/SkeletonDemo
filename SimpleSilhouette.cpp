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
	Mat _simpleRegionGrow(const Mat &dmat, Point seed, int thresh, const Rect roi, bool debugDraw){
		vector<Point> seeds;
		seeds.push_back(seed);

		return _simpleRegionGrow(dmat, seeds, thresh, roi, debugDraw);
	}//simpleRegionGrow

	//Ԥ���� roi ���� mask
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

// 		//ʵ�ʽ������������� roi��
// 		int top = max(0, roi.y),
// 			left = max(0, roi.x),
// 			bottom = min(hh, roi.y + roi.height),
// 			right = min(ww, roi.x + roi.width);

		//1. init
		//���ǣ�0δ�鿴�� 1��queue�У� 255�Ѵ����neibor�����յõ������� mask
		Mat flagMat = Mat::zeros(sz, CV_8UC1);

		//�����������ĵ�
		queue<Point> pts;

		//��ʼ���ӵ����&��ǣ�������������mask ��Ч
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
				//roi �жϲ�Ҫ�ˣ�
				//if (left <= npt.x && npt.x < right && top <= npt.y && npt.y < bottom)
				if (Rect(Point(), dmat.size()).contains(npt))
				{
					const ushort& depNpt = dmat.at<ushort>(npt);
					uchar& flgNpt = flagMat.at<uchar>(npt);
					if (flgNpt == 0
						&& abs(depPt - depNpt) <= thresh
						//����mask �жϣ�
						&& mask.at<uchar>(npt) == UCHAR_MAX){
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

				//��sdi��������֮ǰ�κ�һ��mask����������һ����
				if(!regionExists)
					res.push_back(_simpleRegionGrow(dmat, sdi, thresh, mask, debugDraw));
			}
		}
		return res;
	}//simpleRegionGrow

	Mat getFloorApartMask( Mat dmat, bool debugDraw /*= false*/){
		//��������˲�������mask��
		int flrKrnlArr[] = {1,1,1,1,1, -1,-1,-1,-1,-1};
		Mat flrKrnl((sizeof flrKrnlArr)/(sizeof flrKrnlArr[0]), 1, CV_32S, flrKrnlArr);
		//cv::flip(flrKrnl, flrKrnl, 0);

		Mat flrApartMat;
		filter2D(dmat, flrApartMat, CV_32F, flrKrnl);
		Mat flrApartMsk = abs(flrApartMat)<500;
		//�ϰ������ܣ���ֹ�ֲ����粿����ɾ���ˣ�
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

	//contours ���ã� �ᱻ�޸�
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

	//Ŀǰ��ͨ���ɰ�ǰ�����ص�����ж�
	bool isHumanMask(const Mat &msk, int fgPxCntThresh /*= 1000*/){
		int fgPxCnt = countNonZero(msk==UCHAR_MAX);
		cout<<"fgPxCnt: "<<fgPxCnt<<endl;

		return fgPxCnt > fgPxCntThresh;
	}//isHumanMask


	//1. �� distMap ��ֵ���õ� contours�� 2. �� contours bbox �жϣ��õ���������
	vector<vector<Point> > distMap2contours( const Mat &dmat, bool debugDraw /*= false*/ ){
		Mat dm_draw,
			edge_up,		//�Ų�������������ϰ���
// 			edge_up_inv,	//��ɫ���
			edge_ft,		//�Ų��������룬�°���
// 			edge_ft_inv,	//��ɫ���
			edge_whole,		//����+�Ų����������
			edge_whole_inv,
			distMap,
			bwImg;			//��ڱ߶�ֵͼ
		normalize(dmat, dm_draw, 0, UCHAR_MAX, NORM_MINMAX, CV_8UC1);

		int th_low = 80;
		Canny(dm_draw, edge_up, th_low, th_low*2);

		Mat flrApartMsk = getFloorApartMask(dmat);
		Canny(flrApartMsk, edge_ft, 64, 128);

		edge_whole = edge_up + edge_ft;
		edge_whole_inv = (edge_whole==0);

		static int anch = 4;
		static Mat morphKrnl = getStructuringElement(MORPH_RECT, Size(anch*2+1, anch*2+1), Point(anch, anch) );
		erode(edge_whole_inv, edge_whole_inv, morphKrnl);
		
		bwImg = edge_whole_inv;

		vector<vector<Point> > contours, cont_good;
		vector<Vec4i> hierarchy;
		findContours(bwImg, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

		size_t contSz = contours.size();
		for(size_t i = 0; i < contSz; i++){
			//����һ�׾�
			Moments mu = moments(contours[i]);
			//Point mc(mu.m10/mu.m00, mu.m01/mu.m00);
			Point mc;
			if(abs(mu.m00)<1e-8) //area is zero
				mc = contours[i][0];
			else
				mc = Point(mu.m10/mu.m00, mu.m01/mu.m00);

			ushort dep_mc = dmat.at<ushort>(mc);

			Rect boundRect = boundingRect(contours[i]);
			Size bsz = boundRect.size();

			//���Թ��������� 1. bbox �����; 2. bbox �߶�; 
			//3. bbox����߶ȸ߶�; 4. bbox ���ز��ܸ��ڰ�������Ϊ�˽Ų�λ�ýϵ�
			if(bsz.height*1./bsz.width > 1.5 && bsz.height > 80
				&& dep_mc * bsz.height > XTION_FOCAL_XY * 1000 && boundRect.br().y > dmat.rows / 2)
			{
				cont_good.push_back(contours[i]);
			}
		}
			return cont_good;
	}//distMap2contours

	//1. �� distMap ��ֵ���õ� contours�� 2. �� contours bbox �жϳ���ȣ��õ���������
	Mat distMap2contoursDebug(const Mat &dmat, bool debugDraw /*= false*/){
		static int frameCnt = 0;

		Mat dm_draw,
			edge_up,		//�Ų�������������ϰ���
// 			edge_up_inv,	//��ɫ���
			edge_ft,		//�Ų��������룬�°���
// 			edge_ft_inv,	//��ɫ���
			edge_whole,		//����+�Ų����������
			edge_whole_inv,
			distMap,
			bwImg;			//��ڱ߶�ֵͼ
		normalize(dmat, dm_draw, 0, UCHAR_MAX, NORM_MINMAX, CV_8UC1);

		int th_low = 80;
		Canny(dm_draw, edge_up, th_low, th_low*2);
// 		edge_up_inv = (edge_up==0);
		if(debugDraw){
// 			imshow("edge_up_inv", edge_up_inv);
			imshow("edge_up", edge_up);
		}


		Mat flrApartMsk = getFloorApartMask(dmat);
		Canny(flrApartMsk, edge_ft, 64, 128);
// 		edge_ft_inv = (edge_ft==0);
		if(debugDraw){
// 			imshow("edge_ft_inv", edge_ft_inv);
			imshow("edge_ft", edge_ft);
		}

		edge_whole = edge_up + edge_ft;
		//edge_whole = edge_up;
		if(debugDraw){
			imshow("edge_whole", edge_whole);
			imwrite("edge_whole_"+std::to_string((long long)frameCnt)+".jpg", edge_whole);
		}

		edge_whole_inv = (edge_whole==0);

		static int distSumt = 0;
		clock_t begt = clock();

// 		//distMap �� CV_32FC1
// 		distanceTransform(edge_whole_inv, distMap, CV_DIST_L2, 3);
// 		normalize(distMap, distMap, 0, UCHAR_MAX, NORM_MINMAX, CV_8UC1);
// 		//�� distMap ��ֵ��
// 		threshold(distMap, bwImg, 20, 255, THRESH_BINARY);
// 
// 		distSumt += (clock()-begt);
// 		if(debugDraw)
// 			std::cout<<"distSumt.rate: "<<1.*distSumt/(frameCnt+1)<<std::endl;
// 
// 		if(debugDraw){
// 			imshow("distanceTransform.distMap", distMap);
// 			imshow("threshold.distMap.bwImg", bwImg);
// 			imwrite("threshold.distMap.bwImg"+std::to_string((long long)frameCnt)+".jpg", bwImg);
// 		}

		//open
		static int anch = 4;
		static Mat morphKrnl = getStructuringElement(MORPH_RECT, Size(anch*2+1, anch*2+1), Point(anch, anch) );
// 		morphologyEx(bwImg, bwImg, CV_MOP_OPEN, morphKrnl);
// 		if(debugDraw){
// 			imshow("CV_MOP_OPEN.threshold.bwImg", bwImg);
// 			imwrite("CV_MOP_OPEN.threshold.bwImg"+std::to_string((long long)frameCnt)+".jpg", bwImg);
// 		}

		//�� edge erode�� ��distMap ��ֵ������ʲô����MORPH_ELLIPSE ʱ������ȫ��ͬ
		static int erodeSumt = 0;
		/*clock_t*/ begt = clock();
		erode(edge_whole_inv, edge_whole_inv, morphKrnl);
		erodeSumt += (clock()-begt);
		if(debugDraw)
			std::cout<<"+++++++++++++++erodeSumt.rate: "<<1.*erodeSumt/(frameCnt+1)<<std::endl;

		bwImg = edge_whole_inv;
		if(debugDraw){
			imshow("edge_whole_inv.erode.bwImg", edge_whole_inv);
			imwrite("edge_whole_inv.erode.bwImg_"+std::to_string((long long)frameCnt)+".jpg", edge_whole_inv);
		}

		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(bwImg, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

		Mat cont_draw_ok = Mat::zeros(dmat.size(), CV_8UC1),
			cont_draw_whole = cont_draw_ok.clone();
		//drawContours(cont_draw, contours, -1, 255, -1);

		size_t contSz = contours.size();
		//vector<Rect> boundRect(contSz);
		for(size_t i = 0; i < contSz; i++){
			//����һ�׾�
			Moments mu = moments(contours[i]);
			Point mc;
			if(abs(mu.m00)<1e-8) //area is zero
				mc = contours[i][0];
			else
				mc = Point(mu.m10/mu.m00, mu.m01/mu.m00);
			ushort dep_mc = dmat.at<ushort>(mc);

			Rect boundRect = boundingRect(contours[i]);
			//boundRect[i] = boundingRect(contours[i]);
			Size bsz = boundRect.size();
			if(bsz.height*1./bsz.width > 1.5 && bsz.height > 80){
				cout<<"mc; dep_mc, width, height; dep_mc*w, dep_mc*h: "<<mc<<"; "
					<<dep_mc<<", "<<bsz.width<<","<<bsz.height<<"; "<<dep_mc*bsz.width<<", "<<dep_mc*bsz.height<<endl;

				drawContours(cont_draw_ok, contours, i, 255, -1);
				//����:
				circle(cont_draw_ok, mc, 5, 128, 2);

				if(debugDraw){
					//���Թ��������� 1. bbox����߶ȸ߶ȣ� 2. bbox ���ز��ܸ��ڰ�������Ϊ�˽Ų�λ�ýϵ�
					if(dep_mc * bsz.height > XTION_FOCAL_XY * 1000
						&& boundRect.br().y > dmat.rows / 2)
						rectangle(cont_draw_ok, boundRect, 255, 2);
					else
						rectangle(cont_draw_ok, boundRect, 255);
				}
			}
			if(debugDraw){
				drawContours(cont_draw_whole, contours, i, 128, -1);
				rectangle(cont_draw_whole, boundRect, 255);
				imshow("cont_draw_whole", cont_draw_whole);
			}
		}
// 		if(debugDraw){
// 			imshow("cont_draw_ok", cont_draw_ok);
// 			imwrite("cont_draw_ok"+std::to_string((long long)frameCnt)+".jpg", cont_draw_ok);
// 		}

		frameCnt++;
		//return distMap;
		return cont_draw_ok;
	}//distMap2contoursDebug

	//dmat �����°���, תΪ top-down-view, ����, ����bbox�ж�, ��ȡ���ʵ�����
	//ע:
	// 1. Z�����ű�Ϊ��ֵ�� UCHAR_MAX/MAX_VALID_DEPTH, �� top-down-view ͼ�߶�Ϊ 256
	// 2. debugDraw, dummy variable
	vector<vector<Point> > dmat2TopDownView( const Mat &dmat, bool debugDraw /*= false*/ ){
		CV_Assert(dmat.type()==CV_16UC1);
		Mat dm_draw; //gray scale
		dmat.convertTo(dm_draw, CV_8UC1, 1.*UCHAR_MAX/MAX_VALID_DEPTH);

		//dmat �����°���
		dm_draw(Rect(0, dmat.rows/2, dmat.cols, dmat.rows/2)).setTo(0);
		Mat tdview = Mat::zeros(Size(dm_draw.cols, UCHAR_MAX+1), CV_16UC1);

		//����������п�ʼ��(ע���� int, ���� size_t)
		for(int i = dm_draw.rows - 1; i >=0; i--){
			const uchar *row = dm_draw.ptr<uchar>(i);
			for(int j = 0; j < dm_draw.cols; j++){
				uchar z = row[j];
				//tdview.at<ushort>(z, j) = dm_draw.rows - i;
				*(tdview.data + z * tdview.step + j * tdview.elemSize()) = dm_draw.rows - i;
			}
		}
		//Ȼ��ת�� uchar
		normalize(tdview, tdview, 0, UCHAR_MAX, NORM_MINMAX, CV_8UC1);

		//����
		static int anch = 3;
		static Mat morphKrnl = getStructuringElement(MORPH_RECT, Size(anch*2+1, anch*2+1), Point(anch, anch) );
		morphologyEx(tdview, tdview, CV_MOP_CLOSE, morphKrnl);

		//top-down view ��ͨ�� bbox ������ֵ���ˣ�
		vector<vector<Point> > tdvContours, tdv_cont_good;
		vector<Vec4i> tdvHierarchy;
		findContours(tdview, tdvContours, tdvHierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		size_t tdvContSize = tdvContours.size();
		for(size_t i = 0; i < tdvContSize; i++){
			Rect boundRect = boundingRect(tdvContours[i]);
			//1. Z����ж� >3px(3*10000mm/256=117mm), <26(26*10000mm/256=1016mm)�� 
			//2. X����ж� (200mm~2000mm) MAX_VALID_DEPTH / UCHAR_MAX / XTION_FOCAL_XY = 1/7.68
			if(3 <= boundRect.height && boundRect.height < 26
				&& 7.68*200 < boundRect.y * boundRect.width && boundRect.y * boundRect.width < 7.68*2000)
			{
				tdv_cont_good.push_back(tdvContours[i]);
			}
		}

		return tdv_cont_good;
	}//dmat2TopDownView

	//Z�����ű�Ϊ��ֵ�� UCHAR_MAX/MAX_VALID_DEPTH
	Mat dmat2TopDownViewDebug(const Mat &dmat, bool debugDraw /*= false*/){
		CV_Assert(dmat.type()==CV_16UC1);

		Mat dm_draw; //gray scale
		//normalize(dmat, dm_draw, UCHAR_MAX, 0, NORM_MINMAX, CV_8UC1);
		//normalize(dmat, dm_draw, 1.*UCHAR_MAX/MAX_VALID_DEPTH, NORM_L1, CV_8UC1);	//��
		dmat.convertTo(dm_draw, CV_8UC1, 1.*UCHAR_MAX/MAX_VALID_DEPTH);

		//dmat �����°���
		dm_draw(Rect(0, dmat.rows/2, dmat.cols, dmat.rows/2)).setTo(0);
		if(debugDraw)
			imshow("dm_draw.setTo", dm_draw);

		//CV_16UC1, ��Ϊ����y_max==480
		//Mat res = Mat::zeros(Size(dmat.cols, MAX_VALID_DEPTH), CV_16UC1);
// 		double dmax, dmin;
// 		minMaxLoc(dmat, &dmin, &dmax);
		Mat res = Mat::zeros(Size(dm_draw.cols, UCHAR_MAX+1), CV_16UC1);
		
		//����������п�ʼ��
		for(int i = dm_draw.rows - 1; i >=0; i--){
			const uchar *row = dm_draw.ptr<uchar>(i);
			for(int j = 0; j < dm_draw.cols; j++){
				uchar z = row[j];
				//res.at<ushort>(z, j) = dm_draw.rows - i;
				*(res.data + z * res.step + j * res.elemSize()) = dm_draw.rows - i;
			}
		}
		
		//Ȼ��ת�� uchar
		normalize(res, res, 0, UCHAR_MAX, NORM_MINMAX, CV_8UC1);

		//resize(res, res, Size(res.cols, 256));
		//res = res.t();
		return res;
	}//dmat2TopDownViewDebug

	vector<Mat> findHumanMasksUseBbox(Mat &dmat, bool debugDraw /*= false*/){
		vector<Mat> res;

		//���Դ�ѡ�õ��ġ��á�cont��
		vector<vector<Point>> dtrans_cont_good = zc::distMap2contours(dmat, debugDraw);
		vector<vector<Point>> tdv_cont_good = zc::dmat2TopDownView(dmat, debugDraw);
		
		size_t tdv_cont_good_size = tdv_cont_good.size();
		vector<Rect> tdvBboxs(tdv_cont_good_size);

		//�õ� tdv_cont ��Ӧ bboxs
		for(size_t i = 0; i < tdv_cont_good_size; i++){
			tdvBboxs[i] = boundingRect(tdv_cont_good[i]);
		}

		Mat flrApartMsk = getFloorApartMask(dmat, debugDraw);

		//������ͼÿ�� cont��ת������ͼ�� �� bbox�� ��
		size_t dtrans_cont_size = dtrans_cont_good.size();
		for(size_t i = 0; i < dtrans_cont_size; i++){
			Mat cont_mask = Mat::zeros(dmat.size(), CV_8UC1);
			drawContours(cont_mask, dtrans_cont_good, i, 255, -1);
			//Z�����½磺
			double dmin, dmax;
			minMaxLoc(dmat, &dmin, &dmax, nullptr, nullptr, cont_mask & dmat!=0);
			
			//����һ���ж�����: mask ���ֵ���� >1500mm
			if(dmax-dmin > 1500)
				continue;

			//���ű� MAX_VALID_DEPTH / UCHAR_MAX������ top-down-view:
			Rect bbox_dtrans_cont = boundingRect(dtrans_cont_good[i]);
			double ratio = 1. * UCHAR_MAX / MAX_VALID_DEPTH;
			Rect bbox_dtrans_cont_to_tdview(
				bbox_dtrans_cont.x, dmin * ratio,
				bbox_dtrans_cont.width, (dmax-dmin) * ratio);

			bool isIntersect = false;
			for(size_t k = 0; k < tdv_cont_good_size; k++){
				Rect bboxIntersect = bbox_dtrans_cont_to_tdview & tdvBboxs[k];
				//�������� bbox �ཻ�� ok��
				if(bboxIntersect.area()!=0){
					if(debugDraw){
						cout<<"bboxIntersect: "<<bboxIntersect<<"; "
						<<bbox_dtrans_cont_to_tdview<<", "<<tdvBboxs[k]<<endl
						<<"dmin, dmax: "<<dmin<<", "<<dmax<<endl;
						
						cout<<"k < tdv_cont_good_size: "<<k<<", "<<tdv_cont_good_size<<endl;
					}
					isIntersect = true;
					break;
				}
			}
			//�������� bbox �ཻ�� ok��
			if(isIntersect){
				if(debugDraw)
					cout<<"isIntersect: "<<isIntersect<<endl;

				int rgThresh = 55;
				Mat msk = _simpleRegionGrow(dmat, dtrans_cont_good[i], rgThresh,
					flrApartMsk, false);
				res.push_back(msk);
			}
		}//for(size_t i = 0; i < dtrans_cont_size; i++)

		return res;
	}//findHumanUseBbox

	cv::Mat postRegionGrow( const Mat &flagMat, int xyThresh, int zThresh, bool debugDraw /*= false*/ )
	{
		static Mat prevFlagMat;
		//����һ֡��
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
