#include <iostream>
#include "SimpleSilhouette.h"
#include <iterator>
#include <algorithm>

//@deprecated, ����֤�� 0�̣� 1��
//abs-diff ʱѡ�õ�ǰ����Ȼ���ƽ����ȣ�
const int mode = 01; //0��ǰ�� 1ƽ��

//simpleRegionGrowXXX, separateMasks �������������������� thickLimit������Ľӿڣ�����global-var
const int thickLimitDefault = 1500;
int thickLimit = thickLimitDefault; //����

namespace zc{

//from CKernal.cpp
#if defined(ANDROID)
#define FEATURE_PATH "/data/data/com.motioninteractive.zte/app_feature/"
#else
	//#define FEATURE_PATH "../Skeleton/feature"
#define FEATURE_PATH "../../../plugins/orbbec_skeleton/feature"
#endif

#ifdef CV_VERSION_EPOCH
//#if CV_VERSION_MAJOR < 3
//lincccc's code below:
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
#endif //CV_VERSION_EPOCH

	// COCiter == Container of Containers Iterator
	// Oiter == Output Iterator
	template <class COCiter, class Oiter>
	void flatten(COCiter start, COCiter end, Oiter dest) {
		while (start != end) {
			dest = std::copy(start->begin(), start->end(), dest);
			++start;
		}
	}
	//zhangxaochen: �μ� hist_analyse.m ���ҵ�ʵ��
	Point seedSimple(Mat dmat, int *outVeryDepth /*= 0*/, bool debugDraw /*= false*/){
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
	}//seedSimple

	Mat getBgMskUseWallAndHeight(Mat dmat){
		int wallDepth = zc::fetchWallDepth(dmat);
		Mat initBgMsk;
		//���޴�ǽ�棬 �����Ͽտ��� ������߶��ж��޳�(<2500mm)
		//������ wallDepth < 0 �Լ����������̫��(<1500mm)�����屻�ж�Ϊǽ�������
		if (wallDepth < 3000){
			Mat heightMsk = getHeightMask(dmat, 2500);
			initBgMsk = (heightMsk == 0);

			//2015��6��24��15:20:41��
			//���ָ߶� initBgMsk ��Ӧ��������������Щ�������˳�����������ֱ�ӷ��ء�
			//������벻��
			return initBgMsk;
		}
		else{ //wallDepth > 0�� ��ǽ��
			initBgMsk = (dmat >= wallDepth);
		}

		//������һ��������thresh Ҫ <= ǰ���� ��
		int rgThresh = 55;
		//������Ҫȥ�����棬�����
		Mat flrApartMask = zc::fetchFloorApartMask(dmat, false);

		//��������������thickLimit����Ϊ���ֵ��
		int oldThickLimit = thickLimit;
		thickLimit = MAX_VALID_DEPTH;
		Mat bgMsk = zc::simpleRegionGrow(dmat, initBgMsk, rgThresh, flrApartMask, false, false)[0];
		thickLimit = oldThickLimit;

		return bgMsk;
	}//getBgMskUseWallAndHeight

	Mat fetchBgMskUseWallAndHeight(Mat dmat){
		static Mat res;

		//static Mat dmatOld;// = dmat.clone();
		Mat dmatOld = getPrevDmat();
		if (dmatOld.empty() || countNonZero(dmatOld != dmat) > 0)
			res = getBgMskUseWallAndHeight(dmat);// , debugDraw);

		//dmatOld = dmat.clone();

		return res;
	}//fetchBgMskUseWallAndHeight


	//@deprecated, ���岻�������ã���ʵ����������ȥ���� & findFgMasksUseBbox
	//1. �ұ�����ǽ�棻
	//2. ��ûǽ��˵�������տ�������߶��ж��޳�(<2500mm)
	//3. ʣ����ߵ������ӵ�
	vector<Mat> findFgMasksUseWallAndHeight(Mat dmat, /*bool usePre / *= false* /, */bool debugDraw /*= false*/){
		//vector<Mat> res;
		//clock_t begttotal = clock();
		clock_t begt = clock();

		Mat maskedDmat = dmat.clone();

		Mat bgMsk = fetchBgMskUseWallAndHeight(dmat);
		maskedDmat.setTo(0, bgMsk);

		if (debugDraw)
			cout << "findFgMasksUseWallAndHeight.part2.ts: " << clock() - begt << endl;

		if (debugDraw){
			Mat maskedDmat_show;
			normalize(maskedDmat, maskedDmat_show, 0, UCHAR_MAX, NORM_MINMAX, CV_8UC1);
			imshow("maskedDmat", maskedDmat_show);
		}

		Mat tmp;
		// 			debugDraw = true;
		//bool usePre = true;
		return findFgMasksUseBbox(maskedDmat, /*usePre, */debugDraw, tmp);

	}//findFgMasksUseWallAndHeight

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

		//��ʴ��
		int anch = 2;
		Mat morphKrnl = getStructuringElement(MORPH_RECT, Size(anch*2+1, anch*2+1), Point(anch, anch) );
		erode(mask, mask, morphKrnl);

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

		return _simpleRegionGrow_core_vec2mat(dmat, seeds, thresh, _mask, debugDraw);
	}//simpleRegionGrow

	//
	Mat _simpleRegionGrow_core_pt2mat(const Mat &dmat, Point seed, int thresh, const Mat &validMask, bool debugDraw /*= false*/){
		//1. 2015��6��22��21:55:54 ���ǸĻ��� core_vec2mat �汾��
		vector<Point> seeds;
		seeds.push_back(seed);

		return _simpleRegionGrow_core_vec2mat(dmat, seeds, thresh, validMask, debugDraw);

		//2. 2015��6��22��20:17:29�������Ϊ���ĺ���
		//�� mode=0ʱ��Ч��һ�㣻mode=1ʱ����ǽ�泡����Ƶ�����ô˺�����Ч�ʵ�(caller core_mat2mat�� ~30ms)
		Size sz = dmat.size();
		int hh = sz.height,
			ww = sz.width;

		Rect dmRect(Point(), dmat.size());

		//1. init
		//���ǣ�0δ�鿴�� 1��queue�У� 255�Ѵ����neibor�����յõ������� mask
		Mat flagMat = Mat::zeros(sz, CV_8UC1);

		//�����������ĵ�
		queue<Point> pts;

		//abs-diff ʱѡ�õ�ǰ����Ȼ���ƽ����ȣ�
		const int mode = 01; //0��ǰ�� 1ƽ��
		//״̬����
		double depAvg = 0; //ǰ����ƽ�����
		size_t ptCnt = 0; //ǰ�������


		//��ʼ���ӵ����&��ǣ�������������mask ��Ч
// 		for (size_t i = 0; i < sdsVec.size(); i++){
// 			Point sd = sdsVec[i];
// 			//if(roi.contains(sd)){
// 			if (validMask.at<uchar>(sd) == UCHAR_MAX){
// 				flagMat.at<uchar>(sd) = 1;
// 				pts.push(sd);
// 			}
// 		}
		if (validMask.at<uchar>(seed) == UCHAR_MAX){
			flagMat.at<uchar>(seed) = 1;
			pts.push(seed);

			//��ʼ��ƽ����ȣ�
			depAvg = dmat.at<ushort>(seed); 
		}

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

			flagMat.at<uchar>(pt) = UCHAR_MAX;

			//����ƽ����ȣ�
			depAvg = (depAvg*ptCnt + depPt) / (ptCnt + 1);
			ptCnt++;

			for (int i = 0; i < nnbr; i++){
				Point npt = pt + Point(dx[i], dy[i]);
				//roi �жϲ�Ҫ�ˣ�
				//if (left <= npt.x && npt.x < right && top <= npt.y && npt.y < bottom)
				if (dmRect.contains(npt))
				{
					const ushort& depNpt = dmat.at<ushort>(npt);
					uchar& flgNpt = flagMat.at<uchar>(npt);
					if (flgNpt == 0
						//&& abs(depPt - depNpt) <= thresh
						//����mask �жϣ�
						&& validMask.at<uchar>(npt) == UCHAR_MAX){
						//printf("val, nval: %d, %d\n", depPt, depNpt);

// 						bool isOk = false;
// 						switch (mode)
// 						{
// 						case 0:
// 							isOk = abs(depPt - depNpt) <= thresh;
// 							break;
// 						case 1:
// 							isOk = abs(depAvg - depNpt) <= thresh;
// 							break;
// 						default:
// 							break;
// 						}

						bool isOk = abs((mode == 0 ? depPt : (ushort)depAvg) - depNpt) <= thresh;

						if (isOk){
							flgNpt = 1;
							pts.push(npt);
						}
					}
				}
			}//for-i-nnbr

		}//while

// 		if (debugDraw){
// 			//printf("maxPts: %d\n", maxPts);
// 			imshow("simpleRegionGrow.flagMat", flagMat);
// 			//cout<<flagMat(Rect(0,0, 5,5))<<endl;
// 		}

		return flagMat;
	}//_simpleRegionGrow_core_pt2mat

	//_simpleRegionGrow ���桿���ĺ�����Ч�ʸ��� core_vec2mat��
	Mat _simpleRegionGrow_core_mat2mat(const Mat &dmat, Mat sdsMat, int thresh, const Mat &validMask, bool debugDraw /*= false*/){
		//���ڵ���ż�����������
		bool isDebugError = false;
		bool isAnimSlowly = false; //�������̶��� imshow
		Mat errorMat,
			errorSeedMat;
		if (isDebugError){
			errorMat = Mat::zeros(dmat.size(), CV_8UC1);
			errorSeedMat = errorMat.clone();
		}

		//1. core_vec2mat ���ˣ� ���Ե���֮���ԣ�
// 		clock_t begt = clock();
// 		Mat res = _simpleRegionGrow_core_vec2mat(dmat, maskMat2pts(sdsMat),
// 			thresh, validMask, debugDraw);
// 		cout << "_simpleRegionGrow_core_mat2mat.ts: " << clock() - begt << endl;
// 		return res;

		//2. 2015��6��22��20:38:31 ֮ǰ�汾
		Size sz = dmat.size();
		int hh = sz.height,
			ww = sz.width;

		Rect dmRect(Point(), dmat.size());
		
		//Ŀǰ������������
		const int nnbr = 4;
		int dx[nnbr] = { 0, -1, 0, 1 },
			dy[nnbr] = { 1, 0, -1, 0 };

		//1. init
		//���ǣ�0δ�鿴�� 1��queue�У� 255�Ѵ����neibor�����յõ������� mask
		Mat flagMat = Mat::zeros(sz, CV_8UC1);

		//�����������ĵ�
		queue<Point> pts;

		//@deprecated:
		//abs-diff ʱѡ�õ�ǰ����Ȼ���ƽ����ȣ�
		//����֤�� mode==1ʱ��Ч���ܲ ��Ϊ��ֵ�������µ����
		const int mode = 0; //0��ǰ�� 1ƽ��
		//״̬���� ǰ����ƽ�����, & ����
		double depAvg = 0;
		size_t ptCnt = 0;

		//״̬���� mask����ȷ�Χ [dmin, dmax]
		ushort dmin = MAX_VALID_DEPTH,
			dmax = 0;

		//��ʼ���ӵ����&��ǣ�������������validMask ��Ч, ��������δȫ�������
		for (size_t i = 0; i < dmat.rows; i++){
			for (size_t j = 0; j < dmat.cols; j++){
				Point pt(j, i);
				const ushort& depPt = dmat.at<ushort>(pt);

				if (sdsMat.at<uchar>(pt) == UCHAR_MAX
					&& validMask.at<uchar>(pt) == UCHAR_MAX){
					bool isInnerPt = true;
					for (int k = 0; k < nnbr; k++){
						int nx = j + dx[k];
						int ny = i + dy[k];
						Point npt(nx, ny);
						if (dmRect.contains(npt)
							&& sdsMat.at<uchar>(npt) == 0){
							isInnerPt = false;
							break;
						}
					}
					if (!isInnerPt){ //�����㣬 ���
						flagMat.at<uchar>(pt) = 1;
						pts.push(pt);

						if (isDebugError){
							errorMat.at<uchar>(pt) = 128;
							if (isAnimSlowly){
								imshow("errorMat", errorMat);
								imshow("errorMat-seed", errorMat == 128);
								waitKey(1);
							}
						}
					}
					else{ //�ڵ㣬ֱ�ӱ�Ϊ�Ѷ�
						flagMat.at<uchar>(pt) = UCHAR_MAX;

						//����ƽ����ȣ�
						depAvg = (depAvg*ptCnt + depPt) / (ptCnt + 1);
						ptCnt++;

						if (isDebugError){
							errorMat.at<uchar>(pt) = 255;
							if (isAnimSlowly){
								imshow("errorMat", errorMat);
								imshow("errorMat-seed", errorMat == 128);
								waitKey(1);
							}
						}

					}

					//������ȷ�Χ������Ϊ�ڵ���ܸ���ȷ
					if (depPt < dmin)
						dmin = depPt;
					if (depPt > dmax)
						dmax = depPt;

				}
			}//for-j
		}//for-i


		//2. loop
		int maxPts = -1;
		while (!pts.empty()){
			int qsz = pts.size();
			if (qsz > maxPts)
				maxPts = qsz;

			Point pt = pts.front();
			pts.pop();
			const ushort& depPt = dmat.at<ushort>(pt);

			//������������������ƣ�
			if (abs(depPt - dmax) > thickLimit
				|| abs(depPt - dmin) > thickLimit)
				continue;

			//������Ⱥ�ȷ�Χ��
			if (depPt < dmin)
				dmin = depPt;
			if (depPt > dmax)
				dmax = depPt;

			//����ƽ����ȣ�
			depAvg = (depAvg*ptCnt + depPt) / (ptCnt + 1);
			ptCnt++;

			for (int i = 0; i < nnbr; i++){
				Point npt = pt + Point(dx[i], dy[i]);
				//roi �жϲ�Ҫ�ˣ�
				//if (left <= npt.x && npt.x < right && top <= npt.y && npt.y < bottom)
				if (dmRect.contains(npt))
				{
					const ushort& depNpt = dmat.at<ushort>(npt);
					uchar& flgNpt = flagMat.at<uchar>(npt);
					if (flgNpt == 0
						//&& abs(depPt - depNpt) <= thresh //��Ϊ isOk �ж�
						//����mask �жϣ�
						&& validMask.at<uchar>(npt) == UCHAR_MAX){
						//printf("val, nval: %d, %d\n", depPt, depNpt);

						bool isOk = abs((mode == 0 ? depPt : (ushort)depAvg) - depNpt) <= thresh;

						if (isOk){
							flgNpt = 1;
							pts.push(npt);

							if (isDebugError){
								errorMat.at<uchar>(npt) = 128;
								if (isAnimSlowly){
									imshow("errorMat", errorMat);
									imshow("errorMat-seed", errorMat == 128);
									waitKey(1);
								}
							}

						}
					}
				}
			}
			flagMat.at<uchar>(pt) = UCHAR_MAX;

			if (isDebugError){
				errorMat.at<uchar>(pt) = 255;
				if (isAnimSlowly){
					imshow("errorMat", errorMat);
					imshow("errorMat-seed", errorMat == 128);
					waitKey(1);
				}
			}

		}//while

		return flagMat;
	}//_simpleRegionGrow_core

	//_simpleRegionGrow ���ĺ�����
	Mat _simpleRegionGrow_core_vec2mat(const Mat &dmat, vector<Point> sdsVec, int thresh, const Mat &validMask, bool debugDraw /*= false*/){
		//1. 2015��6��22��20:31:38�� ���� core_pt2mat:
// 		Mat res = Mat::zeros(dmat.size(), CV_8UC1);
// 		size_t sdsVecSz = sdsVec.size();
// 		for (size_t i = 0; i < sdsVecSz; i++){
// 			Point sdi = sdsVec[i];
// 
// 			//��sdi��������֮ǰ�κ�һ��mask����������һ����
// 			if (res.at<uchar>(sdi) == 0){
// 				Mat newRgMat = _simpleRegionGrow_core_pt2mat(dmat, sdi, thresh, validMask, debugDraw);
// 				res += newRgMat;
// 			}
// 		}
// 
// 		return res;


		//2. ���Ը�Ϊ���� sdsMat �汾, Ч�� fgMasksWallAndHeightSumt 19ms->16ms��
		return _simpleRegionGrow_core_mat2mat(dmat, pts2maskMat(sdsVec, dmat.size()),
			thresh, validMask, debugDraw);

		//3. 2015��6��22��20:29:31 ֮ǰ�汾�� 
		Size sz = dmat.size();
		int hh = sz.height,
			ww = sz.width;

		Rect dmRect(Point(), dmat.size());

		//1. init
		//���ǣ�0δ�鿴�� 1��queue�У� 255�Ѵ����neibor�����յõ������� mask
		Mat flagMat = Mat::zeros(sz, CV_8UC1);

		//�����������ĵ�
		queue<Point> pts;

		//��ʼ���ӵ����&��ǣ�������������mask ��Ч
		for(size_t i=0; i<sdsVec.size(); i++){
			Point sd = sdsVec[i];
			//if(roi.contains(sd)){
			if(validMask.at<uchar>(sd) == UCHAR_MAX){
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
				if (dmRect.contains(npt))
				{
					const ushort& depNpt = dmat.at<ushort>(npt);
					uchar& flgNpt = flagMat.at<uchar>(npt);
					if (flgNpt == 0
						&& abs(depPt - depNpt) <= thresh
						//����mask �жϣ�
						&& validMask.at<uchar>(npt) == UCHAR_MAX){
							//printf("val, nval: %d, %d\n", depPt, depNpt);
							flgNpt = 1;
							pts.push(npt);
					}
				}
			}
			flagMat.at<uchar>(pt) = UCHAR_MAX;
		}//while

		if (debugDraw){
			//printf("maxPts: %d\n", maxPts);
			imshow("simpleRegionGrow.flagMat", flagMat);
			//cout<<flagMat(Rect(0,0, 5,5))<<endl;
		}

		return flagMat;
	}//_simpleRegionGrow_core_vec2mat


	cv::Mat _simpleRegionGrow(const Mat &dmat, Mat sdsMat, int thresh, const Mat &mask, bool debugDraw /*= false*/){
		clock_t begt = clock();
		//Mat sdPtsMat;
		vector<Point> sdPtsVec;
		findNonZero(sdsMat == UCHAR_MAX, sdPtsVec);
		if (debugDraw){
			cout << "sdPtsVec.size: " << sdPtsVec.size()
				<< ", .ts: " << clock() - begt << endl;
		}

		return _simpleRegionGrow_core_vec2mat(dmat, sdPtsVec, thresh, mask, debugDraw);
	}//_simpleRegionGrow

	vector<Mat> simpleRegionGrow(const Mat &dmat, vector<Point> seedsVec, int thresh, const Mat &mask, bool getMultiMasks /*= false*/, bool debugDraw /*= false*/){
		vector<Mat> res;
		if(!getMultiMasks){
			clock_t begt = clock();

			Mat msk = _simpleRegionGrow_core_vec2mat(dmat, seedsVec, thresh, mask, debugDraw);
			res.push_back(msk);

			//if (debugDraw)
				cout << "_simpleRegionGrow_core.vec->mat:ts: " << clock() - begt << endl;
		}
		else{//getMultiMasks
			size_t sdsz = seedsVec.size();
			for(size_t i = 0; i < sdsz; i++){
				Point sdi = seedsVec[i];

				bool regionExists = false;
				int regionCnt = res.size();
				for(size_t i = 0; i < regionCnt; i++){
					if(res[i].at<uchar>(sdi)==UCHAR_MAX)
						regionExists = true;
				}

				//��sdi��������֮ǰ�κ�һ��mask����������һ����
				if(!regionExists)
					res.push_back(_simpleRegionGrow_core_pt2mat(dmat, sdi, thresh, mask, debugDraw));
			}
		}
		return res;
	}//simpleRegionGrow

	vector<Mat> simpleRegionGrow(const Mat &dmat, Mat sdsMat, int thresh, const Mat &mask, bool getMultiMasks /*= false*/, bool debugDraw /*= false*/){
		if (!getMultiMasks){
			clock_t begt = clock();

			vector<Mat> res;
			Mat msk = _simpleRegionGrow_core_mat2mat(dmat, sdsMat, thresh, mask, debugDraw);

			//����֤�� 5~7ms
// 			vector<Point> sdsVec = maskMat2pts(sdsMat, 5);
// 			Mat msk = _simpleRegionGrow_core_vec2mat(dmat, sdsVec, thresh, mask, debugDraw);

			res.push_back(msk);

			cout << "simpleRegionGrow.mat->mat:ts: " << clock() - begt << endl;
			return res;
		}
		else{

			vector<Point> sdsVec;
			findNonZero(sdsMat == UCHAR_MAX, sdsVec);

			return simpleRegionGrow(dmat, sdsVec, thresh, mask, getMultiMasks, debugDraw);
		}
	}//simpleRegionGrow


	vector<Mat> simpleRegionGrow(const Mat &dmat, vector<vector<Point>> seedsVecOfVec, int thresh, const Mat &mask, bool debugDraw /*= false*/){
		vector<Mat> res;

		size_t sz = seedsVecOfVec.size();
		for (size_t i = 0; i < sz; i++){
			res.push_back(_simpleRegionGrow_core_vec2mat(dmat, seedsVecOfVec[i], thresh, mask, debugDraw));
		}

		return res;
	}//simpleRegionGrow

	vector<Mat> simpleRegionGrow(const Mat &dmat, vector<Mat> sdMats, int thresh, const Mat &mask, bool debugDraw /*= false*/){
		vector<Mat> res;

		size_t sz = sdMats.size();
		for (size_t i = 0; i < sz; i++){
			//2015��6��24��19:14:11�� ������Ч�Լ�⣺ȫ������
			Mat mski = _simpleRegionGrow(dmat, sdMats[i], thresh, mask, debugDraw);
			if (countNonZero(mski != 0) > 0)
				res.push_back(mski);
		}

		return res;
	}//simpleRegionGrow


	cv::Mat pts2maskMat(const vector<Point> pts, Size matSz){
		Mat res = Mat::zeros(matSz, CV_8UC1);

		size_t ptsVecSz = pts.size();
		for (size_t i = 0; i < ptsVecSz; i++){
			res.at<uchar>(pts[i]) = UCHAR_MAX;
		}

		return res;
	}//pts2maskMat


	vector<Point> maskMat2pts(Mat maskMat, int step /*= 1*/){
		vector<Point> res;

		clock_t begt = clock();

		Mat tmp = Mat::zeros(maskMat.size(), maskMat.type());
		for (size_t i = 0; i < tmp.rows; i += step){
			for (size_t j = 0; j < tmp.cols; j += step){
				tmp.at<uchar>(i, j) = maskMat.at<uchar>(i, j);
			}
		}
		findNonZero(tmp == UCHAR_MAX, res);

		//����֤�� <1ms
		cout << "maskMat2pts.ts: " << clock() - begt << endl;

		return res;
	}//maskMat2pts

	//inner cpp, ��������
	//ע:
	//1. ������ seedsMask ��ɫ̫��(>total*0.5)��Ҫ; 
	//2. ����ֵ�� һ���������candidateMsk̫С(<20*30)��Ҫ
	//---------------@deprecated, ����˼·���������ض����������ƣ�
// 	vector<Mat> simpleRegionGrow(Mat dmat, Mat seedsMask, int thresh, Mat mask){
// 		vector<Mat> res;
// 
// 		Mat sdPts;
// 		cv::findNonZero(seedsMask, sdPts);
// 
// 		size_t sdsz = sdPts.total();
// 		
// 		//����ʼ�� seedsMask ����ȫ�ף�˵����֡���ģ�������
// // 		if (sdsz > dmat.total()*.5)
// // 			return res;
// 
// 		for (size_t i = 0; i < sdsz; i++){
// 			Point sdi = sdPts.at<Point>(i);
// 
// 			bool regionExists = false;
// 			int regionCnt = res.size();
// 			for (size_t k = 0; k < regionCnt; k++){
// 				if (res[k].at<uchar>(sdi) == UCHAR_MAX){
// 					regionExists = true;
// 					break;
// 				}
// 			}
// 
// 			//��sdi��������֮ǰ�κ�һ��mask����������һ����
// 			if (!regionExists){
// 				Mat candidateMsk = _simpleRegionGrow(dmat, sdi, thresh, mask);
// // 				if (countNonZero(candidateMsk) > 20 * 30)
// 					res.push_back(candidateMsk);
// 			}
// 		}
// 		return res;
// 	}//simpleRegionGrow

	cv::Mat getFloorApartMask(Mat dmat, bool debugDraw /*= false*/){
		int flrKrnlArr[] = { 1, 1, 1, -1, -1, -1 };
		Mat flrKrnl((sizeof flrKrnlArr) / (sizeof flrKrnlArr[0]), 1, CV_32S, flrKrnlArr);
		//cv::flip(flrKrnl, flrKrnl, 0);

		int mskThresh = 1000;
		int morphRadius = 1;
		return getFloorApartMask(dmat, flrKrnl, mskThresh, morphRadius, debugDraw);
	}//getFloorApartMask

	Mat getFloorApartMask(Mat dmat, Mat flrKrnl, int mskThresh, int morphRadius, bool debugDraw /*= false*/){
		Mat morphKrnl = getMorphKrnl(morphRadius);

		Mat flrApartMat, flrApartMsk;
		Mat tmp;

		filter2D(dmat, flrApartMat, CV_32F, flrKrnl);
		
		//---------------2. 2015��6��23��20:54:15�� ���Գ��Ը߶�ϵ����
		tmp = flrApartMat.clone();
		tmp.setTo(0, abs(tmp) > 500); //ԭʼͼ��Ч������ͻ�䣬ȥ����Щ��
		for (int i = 0; i < tmp.rows; i++){
			Mat row = tmp.row(i);
			row *= (1e-6*i*i*i);
		}
		flrApartMsk = abs(tmp) < mskThresh;
		//��or��ɫ���ͣ�open������
		morphologyEx(flrApartMsk, tmp, MORPH_CLOSE, morphKrnl);
		flrApartMsk = tmp;

		imshow("floor-height-factor", flrApartMsk);
		return flrApartMsk;

		//---------------1. 2015��6��23��20:54:32�� ֮ǰ�汾��1/2, 3/4��һ���У�����
		flrApartMsk = abs(flrApartMat)<mskThresh;
// 		Mat flrApartMsk2 = abs(flrApartMat)<500 | abs(flrApartMat)>1000;
		//�ϰ������ܣ���ֹ�ֲ����粿����ɾ���ˣ�
		Rect upHalfWin(0, 0, dmat.cols, dmat.rows / 2);
		flrApartMsk(upHalfWin).setTo(UCHAR_MAX);

		//flrApartMsk ���������ǽ��䣩����Ч����������ֹ�ֲ����ڵ�ʱ���˵������У��ο�flrApartMsk2Ч����

		//���ͣ����������ذ壬ȥ�����ֲ��γɵı�Ե����ʴ��ϣ���Ų���Χ�պϡ�
		//Ч�����ʴ�����½Ų��ѿ�
// 		dilate(flrApartMsk, flrApartMsk, morphKrnl); //res320*240, 
// 		morphKrnl = getMorphKrnl(12);
// 		erode(flrApartMsk, flrApartMsk, morphKrnl); //res320*240, 

		//�ײ� open������ ��close��������߽�ʱ�Ų�����Ļ��Ե����һƬ
		Mat flrApartMsk_feet;
		morphologyEx(flrApartMsk, flrApartMsk_feet, MORPH_OPEN, morphKrnl);
		Rect up3of4Win(0, 0, dmat.cols, dmat.rows * 3 / 4);
		flrApartMsk_feet(up3of4Win).setTo(UCHAR_MAX);

		//close������-��ʴ������
		morphologyEx(flrApartMsk, tmp, MORPH_CLOSE, morphKrnl); //res320*240, 
		flrApartMsk = tmp;

		//��ɫ�󽻣���ɫ��ͣ�
		tmp = flrApartMsk & flrApartMsk_feet;
		if (debugDraw)
			imshow("tmp", tmp);
		flrApartMsk = tmp;

// 		//ͼ��ײ���Ե��ʴ��ʹȫ�ڣ�
// 		int krnlHt = 30;
// 		Mat morphKrnl2 = getStructuringElement(MORPH_CROSS, Size(1, krnlHt));
// 
// 		erode(flrApartMsk, tmp, morphKrnl2, Point(0, krnlHt - 1));
// 		//flrApartMsk = tmp;
// 		imshow("tmp", tmp);

// 		//���ذ��bbox���ĵ�߶����£�����㣺
// 		vector<vector<Point>> contours;
// 		vector<Vec4i> hie;
// 		Mat flrApartMskInv = (flrApartMsk==0);
// 		findContours(flrApartMskInv.clone(), contours, hie, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
// 
// 		size_t contSz = contours.size();
// 		Rect bbox_whole;
// 		if(contSz){
// 			bbox_whole = boundingRect(contours[0]);
// 			for(size_t i = 1; i < contSz; i++){
// 				bbox_whole |= boundingRect(contours[i]);
// 			}
// 			int pt_y = bbox_whole.y+bbox_whole.height/2;
// 			flrApartMsk(Rect(0, pt_y, dmat.cols, dmat.rows - pt_y)).setTo(0);
// 		}

		if(debugDraw){
			Rect flroi(60, 200, 10, 10);
			//cout<<flrApartMat(flroi)<<endl;

			//floorApartMat = cv::abs(floorApartMat);
			flrApartMat.setTo(0, abs(flrApartMat)>2000);
			rectangle(flrApartMat, flroi, 0);

			imshow("floorApartMat", flrApartMat);

			Mat flrApartMat_draw;
			normalize(flrApartMat, flrApartMat_draw, 0, UCHAR_MAX, NORM_MINMAX, CV_8UC1);

			//rectangle(flrApartMat_draw, bbox_whole, 0, 3);
			//cout<<"bbox_whole: "<<bbox_whole<<endl;

			imshow("flrApartMat_draw", flrApartMat_draw);
			imshow("flrApartMsk", flrApartMsk);
// 			imshow("flrApartMsk2", flrApartMsk2);
		}

		return flrApartMsk;
	}//getFloorApartMask

	//�� getFloorApartMask �������ڣ� ����ÿ���㣬 ֻ�� dmat ���ݱ��ˣ��Ÿ���
	Mat fetchFloorApartMask(Mat dmat, bool debugDraw /*= false*/){
		static Mat res;

		//static Mat dmatOld;// = dmat.clone();
		Mat dmatOld = getPrevDmat();

		if (dmatOld.empty() || countNonZero(dmatOld != dmat) > 0)
			res = getFloorApartMask(dmat, debugDraw);

		//dmatOld = dmat.clone();
		
		return res;
	}//fetchFloorApartMask


	Mat calcWidthMap(Mat dmat, int centerX /*= 0*/, bool debugDraw /*= false*/){
		Mat res = dmat.clone(); //���� cv16uc1 Ӧ��û��

		int ww = res.cols;
		for (size_t j = 0; j < ww; j++){
			Mat col = res.col(j);
			col = col * (j - centerX) / XTION_FOCAL_XY;
		}

		return res;
	}//calcWidthMap

	Mat fetchWidthMap(Mat dmat, int centerX /*= 0*/, bool debugDraw /*= false*/){
		static Mat res;

		static Mat dmatOld;
		if (dmatOld.empty() || countNonZero(dmatOld != dmat) > 0)
			res = calcWidthMap(dmat, centerX, debugDraw);

		dmatOld = dmat.clone();

		return res;
	}//fetchWidthMap

	//֮ǰ�Ĵ���ʵ�֣��ԱȲ��ԣ�
	cv::Mat calcHeightMap0(Mat dmat, bool debugDraw /*= false*/)
	{
		Mat res = dmat.clone(); //���� cv16uc1 Ӧ��û��

		int hh = res.rows;
		CV_Assert(hh > 0);
		//2015��6��25��15:10:53�� �������ԣ��õ�����ÿ�㵽��������Ļ��͵���룬���ǵ�����߶�
		//����ǣ�1.����ǵȸ�, 2.�����߷ǵǸ�
		for (int i = 0; i < hh; i++){
			//const uchar *row = res.ptr<uchar>(i);
			Mat row = res.row(i);
			row = row * (hh - i) / XTION_FOCAL_XY;
		}
		return res;
	}//calcHeightMapWrong

	//��������߶ȸ߶�Mat��
	cv::Mat calcHeightMap1(Mat dmat, bool debugDraw /*= false*/)
	{
		//Mat res = dmat.clone(); //���� cv16uc1 Ӧ��û����
		Mat res;
		dmat.convertTo(res, CV_32S);

		int hh = res.rows;
		CV_Assert(hh > 0);
		//2015��6��25��15:10:53�� �������ԣ��õ�����ÿ�㵽��������Ļ��͵���룬���ǵ�����߶�
		//����ǣ�1.����ǵȸ�, 2.�����߷ǵǸ�
// 		for (int i = 0; i < hh; i++){
// 			//const uchar *row = res.ptr<uchar>(i);
// 			Mat row = res.row(i);
// 			row = row * (hh - i) / XTION_FOCAL_XY;
// 		}

		//��Ϊ��������Ϊ0�߶ȣ����ͳһ����������͵�(offset)
		for (int i = 0; i < hh; i++){
			Mat drow = res.row(i);
			drow = drow * (hh / 2 - i) / XTION_FOCAL_XY;
		}

		//hmin �����㲻�ã�����Ϊ����ԭ�㲻��ȷ���²��ȶ�
		Mat flrMsk = (fetchFloorApartMask(dmat) == 0);
		Mat morphKrnl = getMorphKrnl(3);
		dilate(flrMsk, flrMsk, morphKrnl);

		double hmin, hmax;
		minMaxLoc(res, &hmin, &hmax, 0, 0, flrMsk);
		res -= hmin;

		res.setTo(0, dmat == 0);

		return res;
	}//calcHeightMap

	//�� calcHeightMap �������ڣ� ����ÿ���㣬 ֻ�� dmat ���ݱ��ˣ��Ÿ���
	Mat fetchHeightMap(Mat dmat, bool debugDraw /*= false*/){
		static Mat res;

		//static Mat dmatOld;
		Mat dmatOld = getPrevDmat();

		if (dmatOld.empty() || countNonZero(dmatOld != dmat) > 0)
			res = calcHeightMap0(dmat, debugDraw);

		//dmatOld = dmat.clone();

		return res;
	}//fetchHeightMap

	//��ȡ����߶ȸ߶�<limitMs(����)���ص���mask
	Mat getHeightMask(Mat dmat, int limitMs /*= 2500*/){
		Mat htMap = zc::fetchHeightMap(dmat);
		//imshow("htMap", htMap);

		//Mat htMap_show;
		//htMap.convertTo(htMap_show, CV_8UC1, 1.*UCHAR_MAX / 6e3);//��
		//imshow("htMap_show", htMap_show);

		Mat fgHeightMap = htMap < limitMs & dmat != 0;
		//imshow("fgHeightMap", fgHeightMap);

		return fgHeightMap;
	}//getHeightMask

	int getWallDepth(Mat &dmat){
		double dmin, dmax;
		cv::minMaxLoc(dmat, &dmin, &dmax);
		//��ȫ�ڣ�
		if (dmax == 0)
			return -1;

		int histSize = dmax - dmin;
		float range[] = { dmin, dmax };
		const float *histRange = { range };
		//const float *histRange2 = range ; //��

		Mat histo;
		calcHist(&dmat, 1, 0, Mat(), histo, 1, &histSize, &histRange);
		// 
		// 			Mat histo2;
		// 			calcHist(&dmat, 1, 0, Mat(), histo2, 1, &histSize, &histRange, true, true); //accumulate=true û��

		//��Ȼ�������50cm��
		int winLen = 500;
		//winLen���ڳ�����bar�ۼӸ߶ȣ�
		float maxBarSum = 0;
		int maxIdx = 0;
		//i=1 ��ʼ��
		//for (int i = 1; i < histSize - winLen; i++){
		for (int i = 1; i < histSize - winLen; i += 50){

			float barSumCnt = 0;
			for (int k = i; k < i + winLen; k++){
				float barCnt = histo.at<float>(k);
				barSumCnt += barCnt;
			}
			if (maxBarSum < barSumCnt){
				maxIdx = i;
				maxBarSum = barSumCnt;
			}
		}

		//cout << "maxBarSum: " << maxBarSum << endl;
		if (maxBarSum < dmat.total() * 0.3)
			//---------------���� -1 ��ʾû�ҵ��㹻���ǽ�棺
			return -1;
		else{
			//С��Χ[maxIdx, maxIdx+maxBarSum) �ҷ�ֵ
			float maxBarSum_2 = 0;
			int maxIdx_2 = 0;

			for (int i = maxIdx; i < maxIdx + winLen; i++){
				float barCnt = histo.at<float>(i);
				if (maxBarSum_2 < barCnt){
					maxIdx_2 = i;
					maxBarSum_2 = barCnt;
				}
			}
			//��ǰ20cm��
			//int wallDepth = maxIdx_2 - 200; //��Ҫ���ض������Դ���
			int wallDepth = maxIdx_2;
			return wallDepth;
		}
	}//getWallDepth

	int fetchWallDepth(Mat &dmat){
		static int res;

		//static Mat dmatOld;
		Mat dmatOld = getPrevDmat();

		if (dmatOld.empty() || countNonZero(dmatOld != dmat) > 0)
			res = getWallDepth(dmat);

		//dmatOld = dmat.clone();

		return res;

	}

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

	Mat getDmatGrayscale(const Mat &dmat){
		Mat res;
		dmat.convertTo(res, CV_8UC1, 1. * UCHAR_MAX / MAX_VALID_DEPTH);
		return res;
	}//getDmatGrayscale

	Mat fetchDmatGrayscale(const Mat &dmat){
		static Mat res;

		//static Mat dmatOld;
		Mat dmatOld = getPrevDmat();

		if (dmatOld.empty() || countNonZero(dmatOld != dmat) > 0){
			//dmat.convertTo(res, CV_8UC1, 1. * UCHAR_MAX / MAX_VALID_DEPTH);

			res = getDmatGrayscale(dmat);
		}

		//dmatOld = dmat.clone();

		return res;
	}//fetchDmatGrayscale

	Mat getHumanEdge(const Mat &dmat, bool debugDraw /*= false*/){
		Mat edge_up,
			edge_ft,
			edge_whole;
		Mat dm_draw = fetchDmatGrayscale(dmat);
		
		//�ϰ����Ե��
		int th_low = 40;
		Canny(dm_draw, edge_up, th_low, th_low * 2);
		if (debugDraw){
			imshow("getHumanEdge.edge_up", edge_up);
		}

		//�Ų���Ե��
		Mat flrApartMsk = fetchFloorApartMask(dmat);
		Canny(flrApartMsk, edge_ft, 64, 128);
		if (debugDraw){
			imshow("getHumanEdge.edge_ft", edge_ft);
		}
		edge_whole = edge_up + edge_ft;

		return edge_whole;
	}//getHumanEdge

	vector<vector<Point> > distMap2contours(const Mat &dmat, bool debugDraw /*= false*/, OutputArray _debug_mat /*= noArray()*/){
	//vector<vector<Point> > distMap2contours( const Mat &dmat, bool debugDraw /*= false*/ ){

		Mat debug_mat;
		if(debugDraw){
			_debug_mat.create(dmat.size(), CV_8UC1);
			debug_mat = _debug_mat.getMat();
			debug_mat.setTo(0);
		}

		Mat dm_draw,
			edge_up,		//�Ų�������������ϰ���
// 			edge_up_inv,	//��ɫ���
			edge_ft,		//�Ų��������룬�°���
// 			edge_ft_inv,	//��ɫ���
			edge_whole,		//����+�Ų����������
			edge_whole_inv,
			distMap,
			bwImg;			//��ڱ߶�ֵͼ
// 		normalize(dmat, dm_draw, 0, UCHAR_MAX, NORM_MINMAX, CV_8UC1);
// 
// 		//TODO: Ҫ��Ҫ�������˹������ֱ۰�����ֳ����ε����⣿����������ʱ����
// 
// 		int th_low = 40;
// 		Canny(dm_draw, edge_up, th_low, th_low*2);
// 		if(debugDraw){
// 			imshow("distMap2contours.edge_up", edge_up);
// 		}
// 
// 		//Mat flrApartMsk = getFloorApartMask(dmat);
// 		Mat flrApartMsk = fetchFloorApartMask(dmat);
// 		Canny(flrApartMsk, edge_ft, 64, 128);
// 		if(debugDraw){
// 			imshow("distMap2contours.edge_ft", edge_ft);
// 		}
// 
// 		//edge_whole = edge_up;
// 		edge_whole = edge_up + edge_ft;

		edge_whole = getHumanEdge(dmat, debugDraw);

		edge_whole_inv = (edge_whole==0);
		if(debugDraw){
			imshow("distMap2contours.edge_whole", edge_whole);
		}

		static int anch = 4;
		static Mat morphKrnl = getStructuringElement(MORPH_RECT, Size(anch*2+1, anch*4+1), Point(anch, anch) );
		erode(edge_whole_inv, edge_whole_inv, morphKrnl); //res320*240, costs 0.07ms
		
		bwImg = edge_whole_inv;
		//ȥ����Ч���򣬲�������������������
		bwImg &= (dmat != 0);

		vector<vector<Point> > contours, cont_good;
		vector<Vec4i> hierarchy;
		findContours(bwImg.clone(), contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

// 		//��һ��õ������������ܰ�����Ч�������
// 		Mat roughMask = Mat::zeros(dmat.size(), CV_8UC1);
// 		drawContours(roughMask, contours, -1, 255, -1);
// 		
// 		//�ڶ�������������������� roughMask & (dm_draw!=0), ������ν��
// 		bwImg = (roughMask & dm_draw);
// 		//bwImg = roughMask & (dm_draw != 0);
		if(debugDraw){
			imshow("distMap2contours.bwImg", bwImg);
		}
// 
// 		findContours(bwImg.clone(), contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
// 
// 		if(debugDraw){
// 			Mat tmp_msk = Mat::zeros(dmat.size(), CV_8UC1);
// 			drawContours(tmp_msk, contours, -1, 255, -1);
// 			imshow("2nd.contours", tmp_msk);
// 			//imwrite("edge_whole_inv.erode.bwImg_"+std::to_string((long long)frameCnt)+".jpg", edge_whole_inv);
// 		}

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
			if(bsz.height*1./bsz.width > MIN_VALID_HW_RATIO && bsz.height > 80){
				if(debugDraw){
					cout<<"mc; dep_mc, width, height; dep_mc*w, dep_mc*h: "<<mc<<"; "
						<<dep_mc<<", "<<bsz.width<<","<<bsz.height<<"; "
						<<dep_mc*bsz.width<<", "<<dep_mc*bsz.height<<endl;
					drawContours(debug_mat, contours, i, 255, -1);
					circle(debug_mat, mc, 5, 128, 2);
				}

				//�����ж��� 3. bbox����߶ȸ߶�; 4. bbox ���ز��ܸ��ڰ�������Ϊ�˽Ų�λ�ýϵ�
				if(dep_mc * bsz.height > XTION_FOCAL_XY * MIN_VALID_HEIGHT_PHYSICAL_SCALE
					&& boundRect.br().y > dmat.rows / 2)
				{
					cont_good.push_back(contours[i]);
					if(debugDraw){
						rectangle(debug_mat, boundRect, 255, 2);
					}
				}
				else if(debugDraw)
					rectangle(debug_mat, boundRect, 255);
			}
			else if(debugDraw) //! if(bsz.height*1./bsz.width > 1.5 && bsz.height > 80)
				rectangle(debug_mat, boundRect, 128);
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

		edge_whole = edge_up;
		//edge_whole = edge_up + edge_ft;
		if(debugDraw){
			imshow("edge_whole", edge_whole);
			//imwrite("edge_whole_"+std::to_string((long long)frameCnt)+".jpg", edge_whole);
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
		static Mat morphKrnl = getStructuringElement(MORPH_RECT, Size(anch*2+1, anch*5+1), Point(anch, anch) );
// 		morphologyEx(bwImg, bwImg, CV_MOP_OPEN, morphKrnl);
// 		if(debugDraw){
// 			imshow("CV_MOP_OPEN.threshold.bwImg", bwImg);
// 			imwrite("CV_MOP_OPEN.threshold.bwImg"+std::to_string((long long)frameCnt)+".jpg", bwImg);
// 		}

		//�� edge erode�� ��distMap ��ֵ������ʲô����MORPH_ELLIPSE ʱ������ȫ��ͬ
		static int erodeSumt = 0;
		/*clock_t*/ begt = clock();
// 		Mat tmp_edge_whole_inv = edge_whole_inv.clone();
// 		static Mat morphKrnl2 = getStructuringElement(MORPH_RECT, Size(anch*2+1, anch*2+1), Point(anch, anch) );
// 		erode(tmp_edge_whole_inv, tmp_edge_whole_inv, morphKrnl2);
// 		imshow("edge_whole_inv.anch*2", tmp_edge_whole_inv);

		erode(edge_whole_inv, edge_whole_inv, morphKrnl);
		//imshow("edge_whole_inv.anch*4", edge_whole_inv);

		
		erodeSumt += (clock()-begt);
		if(debugDraw)
			std::cout<<"+++++++++++++++erodeSumt.rate: "<<1.*erodeSumt/(frameCnt+1)<<std::endl;

		bwImg = edge_whole_inv;
		if(debugDraw){
			imshow("distMap2contoursDebug.bwImg", bwImg);
			//imwrite("edge_whole_inv.erode.bwImg_"+std::to_string((long long)frameCnt)+".jpg", edge_whole_inv);
		}

		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(bwImg.clone(), contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

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
			if(bsz.height*1./bsz.width > MIN_VALID_HW_RATIO && bsz.height > 80){
				cout<<"mc; dep_mc, width, height; dep_mc*w, dep_mc*h: "<<mc<<"; "
					<<dep_mc<<", "<<bsz.width<<","<<bsz.height<<"; "<<dep_mc*bsz.width<<", "<<dep_mc*bsz.height<<endl;

				drawContours(cont_draw_ok, contours, i, 255, -1);
				//����:
				circle(cont_draw_ok, mc, 5, 128, 2);

				if(debugDraw){
					//���Թ��������� 1. bbox����߶ȸ߶ȣ� 2. bbox ���ز��ܸ��ڰ�������Ϊ�˽Ų�λ�ýϵ�
					if(dep_mc * bsz.height > XTION_FOCAL_XY * MIN_VALID_HEIGHT_PHYSICAL_SCALE
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

	vector<vector<Point> > dmat2TopDownView(const Mat &dmat, double ratio /*= 1. * UCHAR_MAX / MAX_VALID_DEPTH*/, bool debugDraw /*= false*/, OutputArray _debug_mat /*= noArray()*/){
		CV_Assert(dmat.type()==CV_16UC1);

		Mat dm_draw = dmat.clone(); //gray scale
		//dmat Ԥ���� �����°��� & ȥ�����棻
		dm_draw(Rect(0, dmat.rows/2, dmat.cols, dmat.rows/2)).setTo(0);
		Mat flrApartMsk = zc::fetchFloorApartMask(dmat);
		dm_draw.setTo(0, flrApartMsk==0);

		//dmat.convertTo(dm_draw, CV_8UC1, 1.*UCHAR_MAX / MAX_VALID_DEPTH);
		dm_draw.convertTo(dm_draw, CV_8UC1, ratio);

		if(debugDraw)
			imshow("dmat2TopDownView.dm_draw", dm_draw);


		Mat tdview = Mat::zeros(Size(dm_draw.cols, UCHAR_MAX+1), CV_16UC1);

		Mat debug_mat; //��������۲�ĵ���ͼ
		if(debugDraw){
			_debug_mat.create(tdview.size(), CV_8UC1);
			debug_mat = _debug_mat.getMat();
			//debug_mat = Mat::zeros(tdview.size(), CV_8UC1);
			debug_mat.setTo(0);
		}

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
		Mat tmp;
		normalize(tdview, tmp, 0, UCHAR_MAX, NORM_MINMAX, CV_8UC1);
		tdview = tmp;

		if(debugDraw){
			imshow("tdview0", tdview);
		}

		tdview.setTo(0, tdview<128);

		//����
		int anch = 1;
		Mat morphKrnl = getStructuringElement(MORPH_RECT, Size(anch*2+1, anch*2+1), Point(anch, anch) );
		//morphologyEx(tdview, tdview, CV_MOP_CLOSE, morphKrnl);
		erode(tdview, tdview, morphKrnl);
		if(debugDraw)
			imshow("tdview.erode", tdview);

		anch = 1;
		morphKrnl = getStructuringElement(MORPH_RECT, Size(anch*2+1, anch*2+1), Point(anch, anch) );
		dilate(tdview, tdview, morphKrnl);

		if(debugDraw)
			imshow("tdview", tdview);

		//top-down view ��ͨ�� bbox ������ֵ���ˣ�
		vector<vector<Point> > tdvContours, tdv_cont_good;
		vector<Vec4i> tdvHierarchy;
		findContours(tdview.clone(), tdvContours, tdvHierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		size_t tdvContSize = tdvContours.size();
		for(size_t i = 0; i < tdvContSize; i++){
			Rect boundRect = boundingRect(tdvContours[i]);

			if(debugDraw)
				drawContours(debug_mat, tdvContours, i, 255, -1);

			//1. Z����ж� >3px(3*10000mm/256=117mm), <26(26*10000mm/256=1016mm)�� 
			//2. X����ж� (200mm~2000mm) MAX_VALID_DEPTH / UCHAR_MAX / XTION_FOCAL_XY = 1/7.68
			if(3 <= boundRect.height && boundRect.height < 26 * 1.5
				&& 7.68*200 < boundRect.y * boundRect.width && boundRect.y * boundRect.width < 7.68*2000)
			{
				tdv_cont_good.push_back(tdvContours[i]);
				if(debugDraw)
					rectangle(debug_mat, boundRect, 255, 2);
			}
			else{
				if(debugDraw)
					rectangle(debug_mat, boundRect, 255);
			}
		}

		return tdv_cont_good;
	}//dmat2TopDownView


	Mat dmat2tdview_core(const Mat &dmat, double ratio /*= 1. * UCHAR_MAX / MAX_VALID_DEPTH*/, bool debugDraw /*= false*/){
		Mat dmatSquash;
		//��������ѹ��� MAX_VALID_DEPTH / ratio ���غ��
		//ͨ�ýӿڣ���16u������8u����Ϊδ��ratio��С
		dmat.convertTo(dmatSquash, CV_16U, ratio);

		//����convertTo �� round�� ����Ϊfloat-> CV_16U 
		//Mat test;
		//dmat.convertTo(test, CV_32F, ratio);

		Mat tdview = Mat::zeros(Size(dmatSquash.cols, MAX_VALID_DEPTH * ratio + 1), CV_16UC1);

		//����������п�ʼ��(ע���� int, ���� size_t)
		for (int i = dmatSquash.rows - 1; i >= 0; i--){
			const ushort *row = dmatSquash.ptr<ushort>(i);
			for (int j = 0; j < dmatSquash.cols; j++){
				ushort z = row[j];
				//tdview.at<ushort>(z, j) = dm_draw.rows - i;
				*(tdview.data + z * tdview.step + j * tdview.elemSize()) = dmatSquash.rows - i;
			}
		}

		//����������㣬ȫ�ڡ���ֹ��Ч������ţ�
		tdview.row(0) = 0;

		return tdview;
	}//dmat2tdview_core


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

	vector<vector<Point>> seedUseBbox(Mat dmat, bool debugDraw /*= false*/, OutputArray _debug_mat /*= noArray()*/){
		vector<vector<Point>> res;

		//���Դ�ѡ�õ��ġ��á�cont�� 
		//��Ϊ�ɱ�Ե���õ������Կ��ܰ�����Ч������Ҫ�ٹ���һ��
		vector<vector<Point>> dtrans_cont_good = zc::distMap2contours(dmat, false); //����� dtrans-cont �������Դ�

		Mat tdv_debug_draw;
		vector<vector<Point>> tdv_cont_good = zc::dmat2TopDownView(dmat, 1. * UCHAR_MAX / MAX_VALID_DEPTH, debugDraw, tdv_debug_draw);

		Mat debug_mat; //��������۲�ĵ���ͼ
		if (debugDraw){
			_debug_mat.create(tdv_debug_draw.size(), CV_8UC1);
			debug_mat = _debug_mat.getMat();
			debug_mat.setTo(0);

			//tdview ʵ��cont�� bbox�� 128��ɫ��
			debug_mat.setTo(128, tdv_debug_draw == 255);
		}

		size_t tdv_cont_good_size = tdv_cont_good.size();
		vector<Rect> tdvBboxs(tdv_cont_good_size);

		//�õ� tdv_cont ��Ӧ bboxs
		for (size_t i = 0; i < tdv_cont_good_size; i++){
			tdvBboxs[i] = boundingRect(tdv_cont_good[i]);
		}

		Mat flrApartMsk = getFloorApartMask(dmat, debugDraw);

		//������ͼÿ�� cont��ת������ͼ�� �� bbox�� ��
		size_t dtrans_cont_size = dtrans_cont_good.size();
		for (size_t i = 0; i < dtrans_cont_size; i++){
			Mat cont_mask = Mat::zeros(dmat.size(), CV_8UC1);
			drawContours(cont_mask, dtrans_cont_good, i, 255, -1);
			//Z�����½磺
			double dmin, dmax;
			minMaxLoc(dmat, &dmin, &dmax, nullptr, nullptr, cont_mask & dmat != 0);

			//���ű� MAX_VALID_DEPTH / UCHAR_MAX������ top-down-view:
			Rect bbox_dtrans_cont = boundingRect(dtrans_cont_good[i]);
			double ratio = 1. * UCHAR_MAX / MAX_VALID_DEPTH;
			Rect bbox_dtrans_cont_to_tdview(
				bbox_dtrans_cont.x, dmin * ratio,
				bbox_dtrans_cont.width, (dmax - dmin) * ratio);

			//����һ���ж�����: mask ���ֵ���� >1500mm
			if (dmax - dmin > 1500){
				if (debugDraw){
					rectangle(debug_mat, bbox_dtrans_cont_to_tdview, 255, 1);
				}
				continue;
			}

			if (debugDraw){
				cout << "bbox_dtrans_cont_to_tdview: " << bbox_dtrans_cont_to_tdview << "; " << dmin << ", " << dmax << endl;
				rectangle(debug_mat, bbox_dtrans_cont_to_tdview, 255, 2);
			}

			bool isIntersect = false;
			for (size_t k = 0; k < tdv_cont_good_size; k++){
				Rect bboxIntersect = bbox_dtrans_cont_to_tdview & tdvBboxs[k];
				//�������� bbox �ཻ�� ok��
				if (bboxIntersect.area() != 0){
					if (debugDraw){
						cout << "bboxIntersect: " << bboxIntersect << "; "
							<< bbox_dtrans_cont_to_tdview << ", " << tdvBboxs[k] << endl
							<< "dmin, dmax: " << dmin << ", " << dmax << endl;

						cout << "k < tdv_cont_good_size: " << k << ", " << tdv_cont_good_size << endl;
					}
					isIntersect = true;
					break;
				}
			}
			//�������� bbox �ཻ�� ok������������ȡ����ǰ��msk
			if (isIntersect){
				if (debugDraw)
					cout << "isIntersect: " << isIntersect << endl;

// 				int rgThresh = 55;
// 				Mat msk = _simpleRegionGrow(dmat, dtrans_cont_good[i], rgThresh,
// 					flrApartMsk, false);
// 				res.push_back(msk);

				res.push_back(dtrans_cont_good[i]);
			}
		}//for(size_t i = 0; i < dtrans_cont_size; i++)

		return res;
	}//seedUseBbox

	vector<Mat> findFgMasksUseBbox(Mat &dmat, /*bool usePre / *= false* /, */bool debugDraw /*= false*/, OutputArray _debug_mat /*= noArray()*/){
	//vector<Mat> findHumanMasksUseBbox(Mat &dmat, bool debugDraw /*= false*/){
		
		vector<Mat> resVec;

		clock_t begt = clock();
		vector<vector<Point>> seedVov = seedUseBbox(dmat, debugDraw, _debug_mat);

		Mat dm_draw;// = dmat.clone();
		if (debugDraw){
			cout << "findFgMasksUseBbox.part1.seedUseBbox.ts: " << clock() - begt << endl;
			normalize(dmat, dm_draw, 0, UCHAR_MAX, NORM_MINMAX, CV_8UC1);
			drawContours(dm_draw, seedVov, -1, 0, 3);
			imshow("seedUseBbox", dm_draw);
		}

		//����ȥ�����ȶ��㣬�в����졣Ч��΢������������
// 		seedVov = seedNoMove(dmat, seedVov);
// 
// 		if (debugDraw){
// 			drawContours(dm_draw, seedVov, -1, 255, 1);
// 			imshow("seedUseBbox", dm_draw);
// 		}

		size_t seedVecSize = seedVov.size();
		
		Mat flrApartMsk = fetchFloorApartMask(dmat, debugDraw);

		begt = clock();
		int rgThresh = 55;
		for (size_t i = 0; i < seedVecSize; i++){
			bool isExist = false;
			size_t resVecSz = resVec.size();
			for (size_t k = 0; k < resVecSz; k++){
				if (resVec[k].at<uchar>(seedVov[i][0]) == UCHAR_MAX){
					isExist = true;
					break;
				}
			}

			if (!isExist){
				//bbox����ʱthickLimit��Сһ�㣺
				int oldThickLimit = thickLimit;
				thickLimit = 1000;
				Mat msk = _simpleRegionGrow_core_vec2mat(dmat, seedVov[i], rgThresh,
					flrApartMsk, false);
				thickLimit = oldThickLimit;

				resVec.push_back(msk);
				//seedVec[i];
			}
		}

		if (debugDraw)
			cout << "findFgMasksUseBbox.part2.ts: " << clock() - begt << endl;

// 		begt = clock();
// 		//������ʱ����Ϣ��������ѡ�㡿���������ڲ����֡��©��ǰ����
// 		static vector<Mat> prevMaskVec; //������һ֡������findFgMasksUseWallAndHeight
// 		if (usePre){// && prevMaskVec.size() != 0){
// 			
// 			if (debugDraw){
// 				Mat prevMaskVec2msk = getHumansMask(prevMaskVec, dmat.size());
// 				imshow("prevMaskVec2msk", prevMaskVec2msk);
// 				//imwrite()
// 			}
// 
// 			int noMoveThresh = 55;
// 			vector<Mat> sdsUsePreVec = seedNoMove(dmat, prevMaskVec, noMoveThresh);
// 
// 			size_t prevVecSize = prevMaskVec.size();
// // 			for (size_t i = 0; i < prevVecSize; i++){
// // 				Mat morphKrnl = getMorphKrnl(5);
// // 				//erode(prevMaskVec[i], prevMaskVec[i], morphKrnl);
// // 
// // 				Mat tmp = dmat>3000 & prevMaskVec[i];
// // 				imshow("prevMaskVec" + std::to_string((long long)i) + "-error", tmp);
// // 
// // 				Mat tmp2 = dmat > 3000 & sdsUsePreVec[i];
// // 				imshow("sdsUsePreVec" + std::to_string((long long)i) + "-error", tmp2);
// // 
// // 			}
// 
// 
// 			int rgThresh = 55;
// 			//�ò������������� mask-vec
// 			vector<Mat> noMoveFgMskVec = simpleRegionGrow(dmat, sdsUsePreVec, rgThresh, flrApartMsk, debugDraw);
// 
// 			//prevMaskVec = resVec; //�� Ӧ��������
// 
// 			//---------------fgMskVec �� res ���ӣ��󡰻�
// 			vector<Mat> newResVec = resVec;
// 
// 			//����ѭ��˳���� newResVec ��˭(resVec or noMoveFgMskVec)�й�
// 			size_t noMoveFgMskVecSize = noMoveFgMskVec.size();
// 			for (size_t i = 0; i < noMoveFgMskVecSize; i++){
// 				Mat noMoveMsk_i = noMoveFgMskVec[i];
// 
// 				bool foundIntersect = false;
// 
// 				size_t origResSz = resVec.size();
// 				for (size_t k = 0; k < origResSz; k++){
// 					Mat resMsk_k = resVec[k];
// 					//���ݶ���ֻҪ�ཻ������ͬһ��mask
// 					if (countNonZero(noMoveMsk_i & resMsk_k) != 0){
// 						foundIntersect = true;
// 						//newResVec[k] = (noMoveMsk_i | resMsk_k); //ԭ�� newResVec[k] == resMsk_k
// 						newResVec[k] = noMoveMsk_i;
// 
// 						//break; //��򣬲�Ҫ����
// 					}
// 				}
// 
// 				if (!foundIntersect){//û�����ӣ�������
// 					newResVec.push_back(noMoveMsk_i); //pushback ����Ӱ������� k ������ ��Ϊ k < origResSz
// 				}
// 			}//for
// 
// 			prevMaskVec = newResVec;
// 
// 			if (debugDraw)
// 				cout << "findFgMasksUseBbox.part3.ts: " << clock() - begt << endl;
// 
// 			return newResVec;
// 		}//if (usePre)

		return resVec;
	}//findFgMasksUseBbox

	vector<Mat> trackingNoMove(Mat dmat, const vector<Mat> &prevFgMaskVec, const vector<Mat> &currFgMskVec, bool debugDraw /*= false*/){
		//static vector<Mat> prevFgMaskVec; //������һ֡������

		int noMoveThresh = 55;
		vector<Mat> sdsUsePreVec = seedNoMove(dmat, prevFgMaskVec, noMoveThresh); //��
		//vector<Mat> sdsUsePreVec = seedNoMove(dmat, currFgMskVec, noMoveThresh);

		if (debugDraw){
			Mat tmp = dmat.clone();
			if (sdsUsePreVec.size() > 0)
				tmp.setTo(0, sdsUsePreVec[0]==0);
			imshow("tmp", tmp);
		}

		//����555mm�賤�����У� ���˽���ã����˽���ֻ�����У�
		int rgThresh = 55;
		Mat flrApartMsk = fetchFloorApartMask(dmat, debugDraw);

		//�ò������������� mask-vec
		//�ģ� flrApartMsk -> validMsk
		Mat bgMsk = fetchBgMskUseWallAndHeight(dmat);
		Mat validMsk = flrApartMsk & (bgMsk == 0);
// 		Mat maskedDmat = dmat.clone();
// 		maskedDmat.setTo(0, fetchBgMskUseWallAndHeight(dmat));
		vector<Mat> noMoveFgMskVec = simpleRegionGrow(dmat, sdsUsePreVec, rgThresh, validMsk, debugDraw);

		//---------------2015��6��27��14:03:15	�ĳɣ� �Ը���Ϊ׼�����ٲ�������������
		//�����ٽ����Ϊbase resVec, ��ʼҲҪ���ء�ȥ�أ�
		size_t noMoveFgMskVecSz = noMoveFgMskVec.size();
		vector<Mat> resVec;// (noMoveFgMskVecSz);
		for (size_t i = 0; i < noMoveFgMskVecSz; i++){
			
			bool isExist = false;
			size_t resVecSz = resVec.size();
			for (size_t k = 0; k < resVecSz; k++){
				Mat resMsk_k = resVec[k];

				if (countNonZero(noMoveFgMskVec[i] & resMsk_k) != 0){
					isExist = true;
					break;
				}
			}

			if (!isExist){
				//resVec[i] = noMoveFgMskVec[i].clone();
				resVec.push_back(noMoveFgMskVec[i].clone());
			}
		}

		//��������ص���������
		size_t currFgMskVecSz = currFgMskVec.size();
		for (size_t i = 0; i < currFgMskVecSz; i++){
			Mat currFgMsk_i = currFgMskVec[i];

			bool isExist = false;
			size_t resVecSz = resVec.size();
			for (size_t k = 0; k < resVecSz; k++){
				Mat resMsk_k = resVec[k];

				if (countNonZero(currFgMsk_i & resMsk_k) != 0){
					isExist = true;
					break;
				}
			}

			if (!isExist){
				resVec.push_back(currFgMsk_i);
			}
		}


// 		//---------------fgMskVec �� res ���ӣ��󡰻�
// 		//vector<Mat> newResVec = currFgMskVec; //ǳ���������ܺ�����и����ã���clone
// 		size_t currFgMskVecSz = currFgMskVec.size();
// 		vector<Mat> resVec(currFgMskVecSz);
// 		for (size_t i = 0; i < currFgMskVecSz; i++)
// 			resVec[i] = currFgMskVec[i].clone();
// 
// 		//Ŀǰ�㷨�� ��ζ���--�ϲ�Ը�ϣ��ֲ�Ը�֣�˭����ѭ������Ҫ��
// 		size_t noMoveFgMskVecSz = noMoveFgMskVec.size();
// 		for (size_t i = 0; i < noMoveFgMskVecSz; i++){
// 			Mat noMoveMsk_i = noMoveFgMskVec[i];
// 
// 			size_t resVecSz = resVec.size();
// 			for (size_t k = 0; k < resVecSz; k++){
// 				Mat resMsk_k = resVec[k];
// 
// 			}
// 		}

		//����ѭ��˳���� newResVec ��˭(resVec or noMoveFgMskVec)�й�
// 		size_t noMoveFgMskVecSize = noMoveFgMskVec.size();
// 		for (size_t i = 0; i < noMoveFgMskVecSize; i++){
// 			Mat noMoveMsk_i = noMoveFgMskVec[i];
// 			int noMoveMsk_i_area = countNonZero(noMoveMsk_i);
// 
// 			bool foundIntersect = false;
// 
// 			size_t origResSz = currFgMskVec.size();
// 			for (size_t k = 0; k < origResSz; k++){
// 				Mat resMsk_k = currFgMskVec[k];
// 				int resMsk_k_area = countNonZero(resMsk_k);
// 				int intersect_area = countNonZero(noMoveMsk_i & resMsk_k);
// 				//���ݶ���ֻҪ�ཻ������ͬһ��mask
// 				if (intersect_area != 0){
// 					foundIntersect = true;
// 					//newResVec[k] = (noMoveMsk_i | resMsk_k); //ԭ�� newResVec[k] == resMsk_k
// 					//break; //��򣬲�Ҫ����
// 
// 					//�ٶ� tracking ����ɿ�����ֵ�������
// 					newResVec[k] = noMoveMsk_i;
// 					//break;
// 				}
// 			}
// 
// 			if (!foundIntersect){//û�����ӣ�������
// 				newResVec.push_back(noMoveMsk_i); //pushback ����Ӱ������� k ������ ��Ϊ k < origResSz
// 			}
// 		}//for

		//prevFgMaskVec = newResVec;

// 		if (debugDraw)
// 			cout << "findFgMasksUseBbox.part3.ts: " << clock() - begt << endl;

		return resVec;
	}//trackingNoMove

	//�˰汾ֻ�����XY��ͼ������������ԡ�XZ��ͼ�����벻�������Բ���
	vector<Mat> separateMasksXYview(Mat dmat, vector<Mat> &inMaskVec, bool debugDraw /*= false*/){
		vector<Mat> res;
		size_t mskVecSz = inMaskVec.size();
		//��ÿ�� mask��
		for (size_t i = 0; i < mskVecSz; i++){
			Mat mski = inMaskVec[i];
			vector<vector<Point>> contours;
			findContours(mski.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

			size_t contsSz = contours.size();
			//CV_Assert(contsSz >= 1);
			if (contsSz == 0){
				//���󣡵��ԣ�
				imshow("mski", mski);
				waitKey(0);
			}

			if (contsSz == 1){
				res.push_back(mski);
				continue;
			}
			//else: //contsSz > 1
			for (size_t i = 0; i < contsSz; i++){
				Mat tmp = Mat::zeros(dmat.size(), CV_8UC1);
				drawContours(tmp, contours, i, 255, -1);
				tmp &= mski;

				if (bboxIsHuman(dmat, tmp)){
					bool isExist = false;
					size_t resSz = res.size();
					for (size_t k = 0; k < resSz; k++){
						if (countNonZero(res[k] & tmp) != 0){
							isExist = true;
							break;
						}
					}
					if (!isExist)
						res.push_back(tmp);
				}
			}

// 			vector<Rect> xzBboxVec;
// 			for (size_t i = 0; i < contsSz; i++){
// 				Rect bbox = contour2XZbbox(dmat, contours, i);
// 				xzBboxVec.push_back(bbox);
// 			}
// 
// 			size_t xzBboxVecSz = xzBboxVec.size();//��ʵ ==contsSz
// 			for (size_t i = 0; i < xzBboxVecSz; i++){
// 				for (size_t j = i; j < xzBboxVecSz; j++){
// 					Rect intersectBbox = xzBboxVec[i] & xzBboxVec[j];
// 					//��������XZ�������򣬷��롣��ʱ����Ϊ i, j���������������
// 					if (intersectBbox.area() == 0){
// 						//bboxIsHuman(dmat, )
// 					}
// 				}
// 			}
				
		}//for (size_t i = 0; i < mskVecSz; i++)
		
		return res;
	}//separateMasksXYview


	vector<Mat> separateMasksXZview(Mat dmat, vector<Mat> &inMaskVec, bool debugDraw /*= false*/){
		vector<Mat> res;
		size_t mskVecSz = inMaskVec.size();
		//��XYview ÿ�� mask��
		for (size_t i = 0; i < mskVecSz; i++){
			Mat mskXY_i = inMaskVec[i];

			Mat maskedDmat = dmat.clone();
			maskedDmat.setTo(0, mskXY_i == 0);
			//��֮ǰ��rgThresh����һ�£�
			int rgThresh = 55;
			double ratio = 1. / rgThresh;
			Mat tdview = dmat2tdview_core(maskedDmat, ratio);
			tdview.convertTo(tdview, CV_8U);
			
			if (debugDraw){
				imshow("separateMasksXZview.tdview", tdview);
			}

			vector<vector<Point>> contoursXZ;
			findContours(tdview.clone(), contoursXZ, RETR_EXTERNAL, CHAIN_APPROX_NONE);
			size_t contXZ_size = contoursXZ.size();
			CV_Assert(contXZ_size > 0);
			if (contXZ_size == 1){//XZ����ͼ����һ���������������������·ֿ����龳
				res.push_back(mskXY_i);
			}
			else{ //contXZ_size > 1, �������������ҷֿ���ǰ��ֿ��龳
				//1. ����һ�� �������������ɼ������������㼸����
				//��ȱ�ݡ������ֹܷ�ͷ��Ҫ��merge�ϲ������鷳��Ч����������
// 				vector<vector<Point>> contoursXY;
// 				findContours(mski.clone(), contoursXY, RETR_EXTERNAL, CHAIN_APPROX_NONE);
// 
// 				//a). flatten contours(XY, not XZ) as new seeds:
// 				vector<Point> seedsVec;
// 				//auto it = contours.begin(),
// 				//	it_end = contours.end();
// 				//while (it != it_end){
// 				//	std::copy(it->begin(), it->end(), back_inserter(seedsVec));
// 				//}
// 				flatten(contoursXY.begin(), contoursXY.end(), back_inserter(seedsVec));
// 
// 				Mat rgMask = Mat(dmat.size(), CV_8UC1, UCHAR_MAX);
// 				bool getMultiMask = true;
// 				vector<Mat> subMskVec = simpleRegionGrow(maskedDmat, seedsVec, rgThresh, rgMask, getMultiMask);
// 
// 				//����ȫpush��Ҫ����
// 				//res.insert(res.end(), subMskVec.begin(), subMskVec.end());
// 
// 				size_t subMskVecSz = subMskVec.size();
// 				for (size_t i = 0; i < subMskVecSz; i++){
// 					if (bboxIsHuman(dmat, subMskVec[i]))
// 						res.push_back(subMskVec[i]);
// 				}

				//2. �������� contoursXZ��ͶӰ��ÿ��cont������õ�һ��humMsk��2015��6��27��23:30:01
				//��XZviewÿ��cont��

				bool debugError = false;

				for (size_t i = 0; i < contXZ_size; i++){
					//�����Ľ����
					Mat newMskXY = Mat::zeros(dmat.size(), CV_8UC1);

					vector<Point> &contXZi = contoursXZ[i];
					//cont-mask, �ڲ�ȫ�ף�δȥ���׶�������������(dmin, dmax)�㹻��
					Mat cmskXZ_i = Mat::zeros(tdview.size(), CV_8UC1);
					drawContours(cmskXZ_i, contoursXZ, i, 255, -1);

					//���ұ߽磺
					Rect bboxXZi = boundingRect(contXZi);
					int left = bboxXZi.x,
						right = bboxXZi.x + bboxXZi.width - 1;//��ע�⡿ -1

					//��contXZi������ÿһcol����ͶӰ����(dmin, dmax)
					for (int k = left; k <= right; k++){//��ע�⡿ <=
						//tdview �ϣ�
						//Mat colXZ_k = tdview.col(k);//��
						Mat colXZ_k = cmskXZ_i.col(k);//��

						vector<Point> nonZeroPts;
						findNonZero(colXZ_k, nonZeroPts);
						Rect bboxCol_k = boundingRect(nonZeroPts);
						int dmin = (bboxCol_k.y - 0.5) * rgThresh,
							dmax = (bboxCol_k.br().y + 0.5) * rgThresh - 1; //��ע�⡿ +0.5, -1
						
						//��Ӧ XYview �ϣ�
						Mat colXY_k = maskedDmat.col(k);

						newMskXY.col(k) = (dmin <= colXY_k & colXY_k <= dmax) * UCHAR_MAX;

						if (debugError){
							imshow("newMskXY", newMskXY);
							waitKey(k < right ? 30 : 0);
						}
					}
					
					if (bboxIsHuman(dmat, newMskXY))
						res.push_back(newMskXY);
				}
			}//else-- contXZ_size > 1
		}//for- i < mskVecSz

		return res;
	}//separateMasksXZview


	Rect contour2XZbbox(Mat dmat, vector<vector<Point>> &contours, int contIdx)
	{
		Mat contMsk = Mat::zeros(dmat.size(), CV_8UC1);
		drawContours(contMsk, contours, contIdx, 255, -1);

		double dmin, dmax;
		minMaxLoc(dmat, &dmin, &dmax, nullptr, nullptr, contMsk);

		const vector<Point> &conti = contours[contIdx];
// 		size_t contSz = conti.size();
// 		int xmin = dmat.cols + 1,
// 			xmax = -1;
// 		for (size_t i = 0; i < contSz; i++){
// 			int ptx = conti[i].x;
// 			if (ptx > xmax)
// 				xmax = ptx;
// 			if (ptx < xmin)
// 				xmin = ptx;
// 		}
		Rect xyBbox = boundingRect(conti);
		
		return Rect(xyBbox.x, dmin, xyBbox.width, dmax - dmin);
	}//contour2XZbbox

#ifdef CV_VERSION_EPOCH
// #if CV_VERSION_EPOCH >= 2
// #elif
// 
// #else
// 	//TODO
// #if CV_VERSION_MAJOR >= 3
// #endif // CV_VERSION_EPOCH
#elif CV_VERSION_MAJOR >= 3
	cv::Mat seedUseBGS(Mat &dmat, bool isNewFrame /*= true*/, bool usePre /*= false*/, bool debugDraw /*= false*/){
		Mat dm_show, tmp;
		dmat.convertTo(dm_show, CV_8UC1, 1.*UCHAR_MAX / MAX_VALID_DEPTH);

		int history = 500;
		double varThresh = 1;
		bool detectShadows = false;
		static Ptr<BackgroundSubtractorMOG2> pMOG2 =
			//createBackgroundSubtractorMOG2();// 500, 5, false);
			createBackgroundSubtractorMOG2(history, varThresh, detectShadows);

		Mat bgImg, fgMskMOG2;
		static Mat prevFgMskMOG2;
		static Mat prevDmat;
		static bool isFirstTime = true;

		if (isFirstTime || isNewFrame){

			double learningRate = .1281;
			//���� mat-type ����Ϊ 8uc1, or 8uc3:
			pMOG2->apply(dm_show, fgMskMOG2, learningRate);// , .83);

			fgMskMOG2 &= (dmat != 0);
			if (debugDraw)
				imshow("fgMskMOG2", fgMskMOG2);

			int anch = 2;
			Mat morphKrnl = getStructuringElement(MORPH_RECT, Size(2 * anch + 1, 2 * anch + 1), Point(anch, anch));
			erode(fgMskMOG2, tmp, morphKrnl);
			fgMskMOG2 = tmp;

			//---------------
			prevFgMskMOG2 = fgMskMOG2;
			isFirstTime = false;
			prevDmat = dmat.clone();
		}
		else{
			CV_Assert(countNonZero(prevDmat != dmat) == 0);
			fgMskMOG2 = prevFgMskMOG2;
		}

		pMOG2->getBackgroundImage(bgImg);
		if (debugDraw)
			imshow("bgImg", bgImg);

		return fgMskMOG2;
	}//seedUseBGS

	vector<Mat> findFgMasksUseBGS(Mat &dmat, bool usePre /*= false*/, bool debugDraw /*= false*/, OutputArray _debug_mat /*= noArray()*/){
		vector<Mat> res;
		if (usePre){
			static vector<Mat> prevHumMasks;
			static vector<Point> humCenters;
		}
		Mat debug_mat; //��������۲�ĵ���ͼ
		if (debugDraw){
			_debug_mat.create(dmat.size(), CV_8UC1);
			debug_mat = _debug_mat.getMat();
			debug_mat.setTo(0);
		}

		bool isNewFrame = true;
		Mat fgMskMOG2 = seedUseBGS(dmat, isNewFrame, usePre, debugDraw);
		if (debugDraw){
			imshow("fgMskMOG2-erode", fgMskMOG2);
			//debug_mat = fgMskMOG2; //shallow-copy, ��
			fgMskMOG2.copyTo(debug_mat);
		}

		Mat flrApartMsk = getFloorApartMask(dmat, debugDraw);
		int rgThresh = 55;
		vector<Mat> roughFgMsks = simpleRegionGrow(dmat, fgMskMOG2, rgThresh, flrApartMsk);

		res = roughFgMsks;
		return res;
	}//findFgMasksUseBGS
#endif


	Mat seedNoMove(Mat dmat, /*Mat prevDmat, */Mat mask, int thresh /*= 50*/){
	//Mat seedNoMove(Mat dmat, Mat mask, int thresh /*= 50*/){
		Mat res = mask.clone();

		//��һ�ε���ʱ prevDmat ȫ�ڣ����diff > thresh�����ص� mask Ӧ��Ҳȫ�ڣ�
// 		static Mat prevDmat = Mat::zeros(dmat.size(), dmat.type());
		Mat prevDmat = getPrevDmat();

		res &= (cv::abs(dmat - prevDmat) < thresh);

		//��ʴһ�£���ֹ dmax-dmin>thickLimit ������ܶž�����
		Mat morphKrnl = getMorphKrnl(3);
		erode(res, res, morphKrnl);

		//prevDmat = dmat.clone();

		return res;
	}//seedNoMove

	vector<Mat> seedNoMove(Mat dmat, vector<Mat> masks, int thresh /*= 50*/){
		vector<Mat> res;
		size_t mskVecSize = masks.size();
		for (size_t i = 0; i < mskVecSize; i++){
			Mat newMask = seedNoMove(dmat, /*prevDmat, */masks[i], thresh);
			res.push_back(newMask);
		}

		return res;
	}//seedNoMove

	vector<Point> seedNoMove(Mat dmat, vector<Point> sdVec, int thresh /*= 50*/)
	{
		//Mat res = Mat::zeros(dmat.size(), CV_8UC1);
		vector<Point> newSdVec;

		Mat prevDmat = getPrevDmat();
		//��ǰһ֡ȫ�ڣ���Ϊ�ǵ�һ֡��ԭ������
		if (countNonZero(prevDmat != 0) == 0)
			return sdVec;

		Mat mskWhole = abs(dmat - getPrevDmat()) < thresh;
		size_t sdVecSz = sdVec.size();
		for (size_t i = 0; i < sdVecSz; i++){
			Point sdi = sdVec[i];
			//res.at<uchar>(sdi) = mskWhole.at<uchar>(sdi) != 0;
			if (mskWhole.at<uchar>(sdi) != 0)
				newSdVec.push_back(sdi);
		}

		//return res;
		return newSdVec;
	}//seedNoMove

	vector<vector<Point>> seedNoMove(Mat dmat, vector<vector<Point>> sdVov, int thresh /*= 50*/)
	{
		vector<vector<Point>> res;

		size_t sdVovSz = sdVov.size();
		for (size_t i = 0; i < sdVovSz; i++){
			res.push_back(seedNoMove(dmat, sdVov[i], thresh));
		}

		return res;
	}//seedNoMove

	Mat getHumansMask(vector<Mat> masks, Size sz){
	//Mat getHumansMask( vector<Mat> masks ){
		Mat res = Mat::zeros(sz, CV_8UC1);

		//ǰ����ɫ�����
		//RNG rng( 0xFFFFFFFF );
		
		//������� N���ˣ� ǰ�����˻������� ����ȫ��
		const int colors[100] = {50, 100, 150, 200, 250, 250};

		size_t msz = masks.size();
		for(size_t i = 0; i < msz; i++){
			//res += masks[i];
			//int color = rng.uniform(128, 255);
			int color = colors[i];
			res.setTo(color, masks[i]);

			Moments mo = moments(masks[i]);
			Point mc(mo.m10 / mo.m00, mo.m01 / mo.m00);
			circle(res, mc, 3, 255, 2);
		}

		return res;
	}//getHumansMask

	//��ȫ�ֵ� vector<HumanFg> ��һ����ɫ mask mat:
	Mat getHumansMask(Mat dmat, const vector<HumanFg> &humVec){
		Mat res;
		//8uc3, ��ɫ��
		Mat dm_show;
		dmat.convertTo(dm_show, CV_8U, 1.*UCHAR_MAX / MAX_VALID_DEPTH); //channel ���ɱ䣬 ��
		cvtColor(dm_show, res, CV_GRAY2BGRA);

		size_t humSz = humVec.size();
		for (size_t i = 0; i < humSz; i++){
			HumanFg hum = humVec[i];
			Mat humMask = hum.getCurrMask();
			Scalar c = hum.getColor();
			//res.setTo(c, humMask); //ʵ���ѷ�����ȫ�ڵ����󣬸Ļ�����
			vector<vector<Point>> contours;
			findContours(humMask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
			drawContours(res, contours, -1, c, 2);

			//���� �ڵ��ص�ʱ�����������������ģ�
// 			size_t contSz = contours.size();
// 			for (size_t i = 0; i < contSz; i++){
// 				//����һ�׾�
// 				Moments mu = moments(contours[i]);

			vector<Point> flatConts;
			flatten(contours.begin(), contours.end(), back_inserter(flatConts));
			Moments mu = moments(flatConts);
			Point mc;
			if (abs(mu.m00) < 1e-8) //area is zero
				mc = contours[i][0];
			else
				mc = Point(mu.m10 / mu.m00, mu.m01 / mu.m00);

			circle(res, mc, 5, c, 2);
// 			}
			
		}

		return res;
	}//getHumansMask

	void getHumanObjVec(Mat &dmat, vector<Mat> fgMasks, vector<HumanFg> &outHumVec){
		//ȫ�ֶ��У�
		//static vector<HumanFg> outHumVec;

		size_t fgMskSize = fgMasks.size(),
			humVecSize = outHumVec.size();

		Mat dmatClone = dmat.clone();

		//����δ��⵽���ˣ��ҵ�֡ fgMasks �����ݣ���ʵ����Ҫ��:
		if (humVecSize == 0 && fgMskSize > 0){
			cout << "+++++++++++++++humVecSize == 0" << endl;
			for (size_t i = 0; i < fgMskSize; i++){
				//outHumVec.push_back({ dmatClone, fgMasks[i] });
				outHumVec.push_back(HumanFg(dmatClone, fgMasks[i]));
			}
		}
		//���Ѽ�⵽ HumanFg, �ҵ�֡ fgMasks �����������ݿɸ��£�
		else if (humVecSize > 0){//&& fgMskSize > 0){

			vector<bool> fgMsksUsedFlagVec(fgMskSize);

			vector<HumanFg>::iterator it = outHumVec.begin();
			while (it != outHumVec.end()){
				bool isUpdated = it->updateMask(dmatClone, fgMasks, fgMsksUsedFlagVec);
				if (isUpdated)
					it++;
				else{
					it = outHumVec.erase(it);
					cout << "------------------------------humVec.erase" << endl;
				}
			}//while

			//��� fgMsksUsedFlagVec�� ���������壺
			for (size_t i = 0; i < fgMskSize; i++){
				if (fgMsksUsedFlagVec[i]==true)
					continue;

				outHumVec.push_back(HumanFg(dmatClone, fgMasks[i]));
			}
		}

		//return outHumVec;
	}//getHumanObjVec

	vector<Mat> bboxFilter(Mat dmat, const vector<Mat> &origMasks){
		vector<Mat> res;
		size_t origMskSz = origMasks.size();
		for (size_t i = 0; i < origMskSz; i++){
			Mat mski = origMasks[i];

// 			double dmin, dmax;
// 			minMaxLoc(dmat, &dmin, &dmax, nullptr, nullptr, mski);
// 			
// 			vector<vector<Point> > contours;
// 			findContours(mski.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
// 
// 			CV_Assert(contours.size() > 0);
// 			Rect bbox = boundingRect(contours[0]);
// 
// 			//���� distMap2contours ��bbox �ж����ˣ�
// 			//1. ����̫��; 2. bbox�߶Ȳ���̫С; 3. bbox ���ز��ܸ��ڰ�������Ϊ�˽Ų�λ�ýϵ�
// 			bool notTooThick = dmax - dmin < 1500,
// 				pxHeightEnough = bbox.height >80,
// 				feetLowEnough = bbox.br().y > dmat.rows / 2;
// 
// 			cout << "notTooThick, pxHeightEnough, feetLowEnough: "
// 				<< notTooThick << ", "
// 				<< pxHeightEnough << ", "
// 				<< feetLowEnough << endl;
				

			//if (notTooThick && pxHeightEnough&& feetLowEnough)
			if (bboxIsHuman(dmat, mski))
			{
				res.push_back(mski);
			}
		}
		return res;
	}//bboxFilter

	bool bboxIsHuman(Mat dmat, Mat mask){
		double dmin, dmax;
		minMaxLoc(dmat, &dmin, &dmax, nullptr, nullptr, mask);

		vector<vector<Point> > contours;
		findContours(mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

		CV_Assert(contours.size() > 0);
// 		return bboxIsHuman(dmat, contours[0]);

		Rect bbox = boundingRect(contours[0]);

		//���� distMap2contours ��bbox �ж����ˣ�
		//1. ����̫��; 2. bbox�߶Ȳ���̫С; 3. bbox ���ز��ܸ��ڰ�������Ϊ�˽Ų�λ�ýϵ�
		bool notTooThick = dmax - dmin < thickLimit,
			pxHeightEnough = bbox.height >80,
			feetLowEnough = bbox.br().y > dmat.rows / 2;

		//return notTooThick && pxHeightEnough && feetLowEnough;
		//2015��6��24��16:09:26�� 
		//��ʱ���� notTooThick �ж��� ������ܵ��²��ȶ�
		return /*notTooThick && */pxHeightEnough && feetLowEnough;
	}//bboxIsHuman

// 	bool bboxIsHuman(Mat dmat, vector<Point> cont){
// 		Rect bbox = boundingRect(cont);
// 
// 		//���� distMap2contours ��bbox �ж����ˣ�
// 		//1. ����̫��; 2. bbox�߶Ȳ���̫С; 3. bbox ���ز��ܸ��ڰ�������Ϊ�˽Ų�λ�ýϵ�
// 		bool notTooThick = dmax - dmin < 1500,
// 			pxHeightEnough = bbox.height >80,
// 			feetLowEnough = bbox.br().y > dmat.rows / 2;
// 
// 		return notTooThick && pxHeightEnough && feetLowEnough;
// 
// 	}//bboxIsHuman

	cv::Mat getMorphKrnl(int radius /*= 1*/, int shape /*= MORPH_RECT*/)
	{
		return getStructuringElement(shape, Size(2 * radius + 1, 2 * radius + 1), Point(radius, radius));
	}

	static Mat prevDmat;
	void setPrevDmat(Mat currDmat){
		prevDmat = currDmat.clone();
	}//setPrevDmat
	
	void initPrevDmat(Mat currDmat){
		if (prevDmat.empty())
			prevDmat = Mat::zeros(currDmat.size(), currDmat.type());
	}//initPrevDmat

	Mat getPrevDmat(){
		return prevDmat;
	}//getPrevDmat





	cv::Mat postRegionGrow( const Mat &flagMat, int xyThresh, int zThresh, bool debugDraw /*= false*/ )
	{
		static Mat prevFlagMat;
		//����һ֡��
		if(prevFlagMat.empty()){
			prevFlagMat = flagMat;
		}
		return flagMat;
	}//postRegionGrow



}//zc
