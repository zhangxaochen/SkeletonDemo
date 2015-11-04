#include <iostream>
#include "SimpleSilhouette.h"
#include <iterator>
#include <algorithm>
#include <functional>

//2015��8��15��16:12:27
#define DBG_STD_COUT 00

//2015��7��22��13:33:53�� Ŀǰ���������� 1. ����������+SGF���ֺ��� 2. ��������mask
#define SOLUTION_1 0	//��0��Ŀǰ����m2
#define M1_E1 0	//��0��Ŀǰ���� m1-e2

//@deprecated, ����֤�� 0�̣� 1��
//abs-diff ʱѡ�õ�ǰ����Ȼ���ƽ����ȣ�
const int mode = 01; //0��ǰ�� 1ƽ��

//simpleRegionGrowXXX, separateMasks �������������������� thickLimit������Ľӿڣ�����global-var
const int thickLimitDefault = 1500;
int thickLimit = thickLimitDefault; //����

const int gRgThresh = 155;
const int gNoMoveThresh = 100;

const int distType = CV_DIST_L1;//CV_DIST_L2~1.0ms; L1~0.5ms
const int maskSize = CV_DIST_MASK_PRECISE; //CV_DIST_MASK_3 ���

//һЩ������ɫ��
Scalar cwhite(255, 255, 255);
Scalar cred(0, 0, 255);
Scalar cgreen(0, 255, 0);
Scalar cblue(255, 0, 0);
Scalar cyellow(0, 255, 255);

//ȫ�֣������ɳ�Ա��
vector<HumanObj> humVec;

//�ԱȲ��� MOG & KNN
static Ptr<BackgroundSubtractor> pBgSub;

//�� MOG or KNN ���ƿ���
#define USE_MOG2 01

namespace zc{

//from CKernal.cpp
#if defined(ANDROID)
#define FEATURE_PATH "/data/data/com.motioninteractive.zte/app_feature/"
#else
	//#define FEATURE_PATH "../Skeleton/feature"
#define FEATURE_PATH "../../../plugins/orbbec_skeleton/feature"
#endif

//#if 1
#ifdef CV_VERSION_EPOCH //if opencv2.x
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

	//const char *matNodeName = "mat";
	//const char *matVecName = "mat_vec";


	deque<Mat> loadVideo(const char *fname){
		cout << "loadVideo(): " << endl;
		//vector<Mat> res;
		deque <Mat> res;
		//�ļ����ж��ͬ��key��д���ԣ�������CV_PARSE_ERROR( "Duplicated key" );
		//ʹ��SEQ "[]"�� vector��
		FileStorage fstorage(fname, FileStorage::READ);
		cout << "loadVideo.fstorage" << endl;
		FileNode matVecNode = fstorage[matVecName];
		cout << "matVecNode" << endl;
		if (matVecNode.type() != FileNode::SEQ){
			cerr << "NODE: '" << matVecName << "' is not a sequence! FAIL" << endl;
			return res;
		}

		FileNodeIterator it = matVecNode.begin(),
			it_end = matVecNode.end();

		//for (; it != it_end; it++){
		while (it != it_end){
			//res.push_back((Mat)(*it)); //û������ת��
			Mat m;
			it >> m; //>>�Դ�++it, ���Բ�Ҫ�ֶ�++
			res.push_back(m);
		}

		// 	while (1){
		// 		Mat m;
		// 		fstorage[matNodeName] >> m;
		// 	}
		return res;
	}//loadVideo

	//@brief �Ѷ��vovתΪ����vec
	//@code flatten(contours.begin(), contours.end(), back_inserter(flatConts));
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
			if(countNonZero(dmat == veryDepth))
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
			Mat heightMsk = getFakeHeightMask(dmat, 2500);
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
		int rgThresh = 25;
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

		static Mat dmatOld;// = dmat.clone();
		//Mat dmatOld = getPrevDmat(); //���� ����һ֡ѭ����ֻ��һ��
		if (dmatOld.empty() || countNonZero(dmatOld != dmat) > 0){
			res = getBgMskUseWallAndHeight(dmat);// , debugDraw);
			dmatOld = dmat.clone();
		}

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
		if(countNonZero(sdsMat == UCHAR_MAX))
			findNonZero(sdsMat == UCHAR_MAX, sdPtsVec);
		if (debugDraw){
			if (DBG_STD_COUT)
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
			if(countNonZero(sdsMat == UCHAR_MAX))
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

	vector<Mat> simpleRegionGrow(const Mat &dmat, vector<Mat> &sdMats, int thresh, const Mat &mask, bool debugDraw /*= false*/){
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

	vector<Mat> simpleRegionGrow(const Mat &dmat, vector<Mat> &sdMatVec, int thresh, const vector<Mat> &rgMaskVec, bool debugDraw /*= false*/){
		CV_Assert(sdMatVec.size() == rgMaskVec.size());

		vector<Mat> res;
		size_t sz = sdMatVec.size();
		for (size_t i = 0; i < sz; i++){
			Mat validMsk_i = rgMaskVec[i];
			Mat res_i = _simpleRegionGrow(dmat, sdMatVec[i], thresh, validMsk_i, debugDraw);

			//��Ч�Լ�⣺ȫ��mask����
			if (countNonZero(res_i != 0) > 0)
				res.push_back(res_i);
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
		if(countNonZero(tmp == UCHAR_MAX))
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
		int morphRadius = 2;
		return getFloorApartMask(dmat, flrKrnl, mskThresh, morphRadius, debugDraw);
	}//getFloorApartMask

	Mat getFloorApartMask(Mat dmat, Mat flrKrnl, int mskThresh, int morphRadius, bool debugDraw /*= false*/){
		Mat morphKrnl = getMorphKrnl(morphRadius);

		Mat flrApartMat, flrApartMsk;
		Mat tmp;

		filter2D(dmat, flrApartMat, CV_32F, flrKrnl);
		
		//---------------2. 2015��6��23��20:54:15�� ���Գ��Ը߶�ϵ����
		tmp = flrApartMat.clone();
		//tmp.setTo(0, abs(tmp) > 500); //ԭʼͼ��Ч������ͻ�䣬ȥ����Щ�� //�Զ���ʱ�������ڴ�й¶��
		Mat tmp_diff_msk = abs(tmp) > 500;
		tmp.setTo(0, tmp_diff_msk); //ԭʼͼ��Ч������ͻ�䣬ȥ����Щ��
		for (int i = 0; i < tmp.rows; i++){
			Mat row = tmp.row(i);
			row *= (1e-6*i*i*i);
		}
		flrApartMsk = abs(tmp) < mskThresh;
		//��ɫ���ͣ�close������
		//imshow("floor-height-factor-no-close", flrApartMsk);
		//tmp.data �ڴ����·����ˣ�û����ԭ�ڴ渲д��
		morphologyEx(flrApartMsk, tmp, MORPH_CLOSE, morphKrnl); 
		flrApartMsk = tmp;

		if (debugDraw)
			imshow("flrApartMsk", flrApartMsk);

		//---------------3. 2015��7��5��01:56:01�� 
		//�߶�ͼ�ضϷ���
		int ww = dmat.cols,
			hh = dmat.rows;
		Rect bottomBorder(0, 9. / 10 * hh, ww, hh / 10);
		//����Ļ�ױ�Ե���º�ɫ(>50%)��˵����Ұǰ��ƽ�������
		if (countNonZero(flrApartMsk(bottomBorder) == 0) > ww*hh / 10 * 0.5){
			Mat hmap1 = calcHeightMap1(dmat, false);
			//��ֵ����10cm��
			int flrHeight = cv::mean(hmap1, flrApartMsk == 0)[0];
			flrHeight += 100;

			//max ���ܻᵼ�¹��߽ض�
// 			double hmax; 
// 			minMaxLoc(hmap1, 0, &hmax, 0, 0, flrApartMsk == 0);
// 			int flrHeight = hmax;
			flrApartMsk = (hmap1 > flrHeight);

			if (debugDraw)
				imshow("flrApartMsk.height-cut", flrApartMsk);
		}

		return flrApartMsk;

#if 0	//---------------1. 2015��6��23��20:54:32�� ֮ǰ�汾��1/2, 3/4��һ���У�����
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
// 			bbox_whole = zc::boundingRect(contours[0]);
// 			for(size_t i = 1; i < contSz; i++){
// 				bbox_whole |= zc::boundingRect(contours[i]);
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
#endif
	}//getFloorApartMask

	//�� getFloorApartMask �������ڣ� ����ÿ���㣬 ֻ�� dmat ���ݱ��ˣ��Ÿ���
	Mat fetchFloorApartMask(Mat dmat, bool debugDraw /*= false*/){
		static Mat res;

		static Mat dmatOld;// = dmat.clone();
		//Mat dmatOld = getPrevDmat();

		if (dmatOld.empty() || countNonZero(dmatOld != dmat) > 0){
			res = getFloorApartMask(dmat, debugDraw);
			dmatOld = dmat.clone();
		}
		
		return res;

	}//fetchFloorApartMask


	Mat calcWidthMap(Mat dmat, int centerX /*= 0*/, bool debugDraw /*= false*/){
		//Mat res = dmat.clone(); //���� cv16uc1 Ӧ��û�����и�����
		Mat res;
		dmat.convertTo(res, CV_32S);

		int ww = res.cols;
		//for (size_t j = 0; j < ww; j++){
		for (int j = 0; j < ww; j++){//������ int, �� uint, 
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

	Mat calcHeightMap1(Mat dmat, bool debugDraw /*= false*/){
		//Mat res = dmat.clone(); //���� cv16uc1 Ӧ��û������Ϊ�и���
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

		//hmin �����㲻�ã�����Ϊ����ԭ�㲻��ȷ���²��ȶ����Ҷ���˸
		//2015��7��5��02:29:51�� ���ȶ���û��ϵ
		double hmin, hmax;
		minMaxLoc(res, &hmin, &hmax);
		res -= hmin;

// 		Mat flrMsk = (fetchFloorApartMask(dmat) == 0);
// 		Mat morphKrnl = getMorphKrnl(3);
// 		dilate(flrMsk, flrMsk, morphKrnl);
// 
// 		double hmin, hmax;
// 		minMaxLoc(res, &hmin, &hmax, 0, 0, flrMsk);

		res.setTo(0, dmat == 0);

		return res;
	}//calcHeightMap1

	//�� calcHeightMap �������ڣ� ����ÿ���㣬 ֻ�� dmat ���ݱ��ˣ��Ÿ���
	Mat fetchHeightMap0(Mat dmat, bool debugDraw /*= false*/){
		static Mat res;

		static Mat dmatOld;
		//Mat dmatOld = getPrevDmat();

		if (dmatOld.empty() || countNonZero(dmatOld != dmat) > 0){
			res = calcHeightMap0(dmat, debugDraw);
			dmatOld = dmat.clone();
		}

		return res;
	}//fetchHeightMap0

	//��ȡ����߶ȸ߶�<limitMs(����)���ص���mask
	Mat getFakeHeightMask(Mat dmat, int limitMs /*= 2500*/){
		Mat htMap = zc::fetchHeightMap0(dmat); //�õ� v0 �ٸ߶ȼ��㷽��
		//imshow("htMap", htMap);

		//Mat htMap_show;
		//htMap.convertTo(htMap_show, CV_8UC1, 1.*UCHAR_MAX / 6e3);//��
		//imshow("htMap_show", htMap_show);

		Mat fgHeightMap = htMap < limitMs & dmat != 0;
		//imshow("fgHeightMap", fgHeightMap);

		return fgHeightMap;
	}//getFakeHeightMask

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

		static Mat dmatOld;
		//Mat dmatOld = getPrevDmat();

		if (dmatOld.empty() || countNonZero(dmatOld != dmat) > 0){
			res = getWallDepth(dmat);
			dmatOld = dmat.clone();
		}

		return res;
	}

	void drawOneSkeleton(Mat &img, CapgSkeleton &sk){
		if (sk.size() == 0)
			return;

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

		//zhangxaochen: ������֫�ؽڻ��� //2015-11-2 21:00:43
		line(img, Point(sk[8].x(), sk[8].y()), Point(sk[9].x(), sk[9].y()), Scalar(0), 2);
		line(img, Point(sk[9].x(), sk[9].y()), Point(sk[10].x(), sk[10].y()), Scalar(0), 2);
		line(img, Point(sk[11].x(), sk[11].y()), Point(sk[12].x(), sk[12].y()), Scalar(0), 2);
		line(img, Point(sk[12].x(), sk[12].y()), Point(sk[13].x(), sk[13].y()), Scalar(0), 2);
		
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

	void drawSkeletons( Mat &img, vector<HumanObj> &humObjVec, int skltIdx ){
		vector<CapgSkeleton> sklts;
		size_t humObjVecSz = humObjVec.size();
		for(size_t i = 0; i < humObjVecSz; i++){
			sklts.push_back(humObjVec[i].getSkeleton());
		}

		drawSkeletons(img, sklts, skltIdx);
	}//drawSkeletons

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

		static Mat dmatOld;
		//Mat dmatOld = getPrevDmat();

		if (dmatOld.empty() || countNonZero(dmatOld != dmat) > 0){
			//dmat.convertTo(res, CV_8UC1, 1. * UCHAR_MAX / MAX_VALID_DEPTH);
			res = getDmatGrayscale(dmat);

			dmatOld = dmat.clone();
		}

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
		if (debugDraw){
			imshow("getHumanEdge.edge_whole", edge_whole);
		}

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

		//clock_t begt = clock(); //1~2ms
		edge_whole = getHumanEdge(dmat, debugDraw);
		//cout << "getHumanEdge.ts: " << clock() - begt << endl;

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

			Rect boundRect = zc::boundingRect(contours[i]);
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

	vector<vector<Point> >(*getHumanContoursXY)(const Mat &, bool, OutputArray) = distMap2contours_new;

	vector<vector<Point> > distMap2contours_new(const Mat &dmat, bool debugDraw /*= false*/, OutputArray _debug_mat /*= noArray()*/){
		Mat debug_mat;
		if (debugDraw){
			//����mat���ò�ɫ���ƣ�
			_debug_mat.create(dmat.size(), CV_8UC3);
			debug_mat = _debug_mat.getMat();
			//debug_mat.setTo(0);
			//dmat.convertTo(debug_mat, CV_8U, 1.*UCHAR_MAX / MAX_VALID_DEPTH);
			Mat dmat_gray;
			dmat.convertTo(dmat_gray, CV_8U, 1.*UCHAR_MAX / MAX_VALID_DEPTH);
			vector<Mat> cn3(3, dmat_gray);
			cv::merge(cn3, debug_mat);
		}

		Mat edge_whole, //����+�Ų����������
			edge_whole_inv, //��ɫ���
			bwImg //��ֵͼ�����dist-map��ֵ����ʵ����canny+erode���
			;
		edge_whole = getHumanEdge(dmat, debugDraw);
		edge_whole_inv = (edge_whole == 0);
		static int anch = 4;
		static Mat morphKrnl = getStructuringElement(MORPH_RECT, Size(anch * 2 + 1, anch * 4 + 1), Point(anch, anch));
		erode(edge_whole_inv, bwImg, morphKrnl); //res320*240, costs 0.07ms

		//ȥ����Ч����
		bwImg &= (dmat != 0);

		if (debugDraw){
			debug_mat.setTo(cblue, bwImg == 0); //�Ȼ�����
			debug_mat.setTo(0, edge_whole); //ϸ��
		}

		vector<vector<Point>> contours;
		vector<vector<Point>> res;
		
		findContours(bwImg.clone(), contours, RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		size_t contSz = contours.size();
		for (size_t i = 0; i < contSz; i++){
			vector<Point> conti = contours[i];
			Rect bbox = zc::boundingRect(conti);

			//if (contIsHuman(dmat.size(), conti)){
			if (bboxIsHuman(dmat.size(), bbox)){
				res.push_back(conti);

				if (debugDraw){
					rectangle(debug_mat, bbox, cwhite, 2);
					drawContours(debug_mat, contours, i, cred, 2);
					Moments mu = moments(conti);
					Point mc(mu.m10 / mu.m00, mu.m01 / mu.m00);
					circle(debug_mat, mc, 5, cred, 2);
				}
			}
			else if (debugDraw){
				//rectangle(debug_mat, bbox, 255, 1);
				rectangle(debug_mat, bbox, cwhite, 1);
			}
		}

		return res;
	}//distMap2contours_new

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

			Rect boundRect = zc::boundingRect(contours[i]);
			//boundRect[i] = zc::boundingRect(contours[i]);
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
			Rect boundRect = zc::boundingRect(tdvContours[i]);

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

	vector<vector<Point>> seedUseBboxXyXz(Mat dmat, bool debugDraw /*= false*/, OutputArray _debug_mat /*= noArray()*/){
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
			tdvBboxs[i] = zc::boundingRect(tdv_cont_good[i]);
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
			Rect bbox_dtrans_cont = zc::boundingRect(dtrans_cont_good[i]);
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
	}//seedUseBboxXyXz

	vector<Mat> findFgMasksUseBbox(Mat &dmat, /*bool usePre / *= false* /, */bool debugDraw /*= false*/, OutputArray _debug_mat /*= noArray()*/){
	//vector<Mat> findHumanMasksUseBbox(Mat &dmat, bool debugDraw /*= false*/){
		
		vector<Mat> resVec;

		clock_t begt = clock();
		vector<vector<Point>> seedVov = seedUseBboxXyXz(dmat, debugDraw, _debug_mat);

		Mat dm_draw;// = dmat.clone();
		if (debugDraw){
			cout << "findFgMasksUseBbox.part1.seedUseBboxXyXz.ts: " << clock() - begt << endl;
			normalize(dmat, dm_draw, 0, UCHAR_MAX, NORM_MINMAX, CV_8UC1);
			drawContours(dm_draw, seedVov, -1, 0, 3);
			imshow("seedUseBboxXyXz", dm_draw);
		}

		//����ȥ�����ȶ��㣬�в����졣Ч��΢������������
// 		seedVov = seedNoMove(dmat, seedVov);
// 
// 		if (debugDraw){
// 			drawContours(dm_draw, seedVov, -1, 255, 1);
// 			imshow("seedUseBboxXyXz", dm_draw);
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

		return resVec;
	}//findFgMasksUseBbox


	vector<vector<Point>> seedUseHeadAndBodyCont(Mat dmat, bool debugDraw /*= false*/, OutputArray _debug_mat /*= noArray()*/){
		Mat debug_mat;
		if (debugDraw){
			_debug_mat.create(dmat.size(), CV_8UC1);
			debug_mat = _debug_mat.getMat();
			//debug_mat.setTo(0);
			dmat.convertTo(debug_mat, CV_8UC1, 1.*UCHAR_MAX / MAX_VALID_DEPTH);
		}

		vector<vector<Point>> res;
		//��ѡ����������
		vector<vector<Point>> humContours = getHumanContoursXY(dmat, false, noArray());
		size_t humContoursSz = humContours.size();

		//ͷ�����ӵ㣨Բ��ģ��ƥ�䣩��
		vector<Point> sdHeadVec = sgf::seedHeadTempMatch(dmat, false);
		size_t sdHeadVecSz = sdHeadVec.size();
		if (debugDraw){
			for (size_t k = 0; k < sdHeadVecSz; k++){
				Point sdHead_k = sdHeadVec[k];
				//����ͷ��
				circle(debug_mat, sdHead_k, 11, 188, 2);
			}
		}

		for (size_t i = 0; i < humContoursSz; i++){
			if (debugDraw){
				drawContours(debug_mat, humContours, i, 255, 1);
			}

			//1. ����һ���������� & ͷ�����ӵ� XZ����. 2015��7��5��00:26:06
			//������������������������ʱ�н�
			vector<Point> conti = humContours[i];
			Rect bbox = zc::boundingRect(conti);
			Moments mu = moments(conti);
			Point mc_i(mu.m10 / mu.m00, mu.m01 / mu.m00);
			ushort dep_mci = dmat.at<ushort>(mc_i);

			for (size_t k = 0; k < sdHeadVecSz; k++){
				Point sdHead_k = sdHeadVec[k];
				ushort dep_sdHeadk = dmat.at<ushort>(sdHead_k);

				if (
					bbox.x < sdHead_k.x && sdHead_k.x < bbox.br().x //ͷ��bbox���ұ߽�֮��
					//abs(mc_i.x - sdHead_k.x) < 50
					&& abs(dep_mci - dep_sdHeadk) < 500 //dZ < 50cm
					&& mc_i.y > sdHead_k.y //ͷ������, yС
					){

					if (debugDraw){
						drawContours(debug_mat, humContours, i, 255, 2);
						circle(debug_mat, mc_i, 5, 255, 2);

						//����ͷ��
						circle(debug_mat, sdHead_k, 5, 188, 3);
					}
					res.push_back(conti);
					break;
				}
			}

		}
		return res;
	}//seedUseHeadAndBodyCont


	vector<Point> seedUseMovingHead(Mat dmat, bool isNewFrame /*= true*/, bool debugDraw /*= false*/, OutputArray _debug_mat /*= noArray()*/){
		Mat debug_mat;
		if (debugDraw){
			_debug_mat.create(dmat.size(), CV_8UC1);
			debug_mat = _debug_mat.getMat();
			dmat.convertTo(debug_mat, CV_8UC1, 1.*UCHAR_MAX / MAX_VALID_DEPTH);
		}

		//---------------MOG2:
		Mat dm_show;
		dmat.convertTo(dm_show, CV_8UC1, 1.*UCHAR_MAX / MAX_VALID_DEPTH);
		
		int history = 100;
		double varThresh = 16;
		bool detectShadows = false;

#ifdef CV_VERSION_EPOCH //if opencv2.x
		static Ptr<BackgroundSubtractorMOG2> pMOG2 = new BackgroundSubtractorMOG2(history, varThresh, detectShadows);

#elif CV_VERSION_MAJOR >= 3 //if opencv3
		static Ptr<BackgroundSubtractorMOG2> pMOG2 =
			createBackgroundSubtractorMOG2(history, varThresh, detectShadows);
#endif //CV_VERSION_EPOCH
		Mat bgImg, fgMskMOG2;

		//�� isNewFrame=false����ֱ����prevFgMskMOG2��Ϊ��ǰ֡ǰ��
		static Mat prevFgMskMOG2;
		static bool isFirstTime = true;

		if (isFirstTime || isNewFrame){

			double learningRate = -0.005;
			//���� mat-type ����Ϊ 8uc1, or 8uc3:
#ifdef CV_VERSION_EPOCH //if opencv2.x
			(*pMOG2)(dm_show, fgMskMOG2, learningRate);
#elif CV_VERSION_MAJOR >= 3 //if opencv3
			pMOG2->apply(dm_show, fgMskMOG2, learningRate);
#endif //CV_VERSION_EPOCH

		}


		//---------------seed-Head, ������fgMskMOG2�����ظ����ﵽ��ֵ����OK
		vector<Point> res;

		//ͷ�����ӵ㣨Բ��ģ��ƥ�䣩��
		vector<Point> sdHeadVec = sgf::seedHeadTempMatch(dmat, false);
		size_t sdHeadVecSz = sdHeadVec.size();


		return res;
	}


	vector<Mat> findFgMasksUseHeadAndBodyCont(Mat &dmat, bool debugDraw /*= false*/){
		vector<Mat> resVec;

		return resVec;
	}//findFgMasksUseHeadAndBodyCont

	//@note 2015��7��9��23:32:37�� currInitFgMskVec����Ӧ����ͣ����� mergeFgMaskVec��δ�����
	vector<Mat> trackingNoMove(Mat dmat, const vector<Mat> &prevFgMaskVec, const vector<Mat> &currInitFgMskVec, int noMoveThresh /*= 55*/, Mat moveMaskMat /*= Mat()*/, bool debugDraw /*= false*/){
		//seedNoMove ���˴����� o(�s���t)o
		vector<Mat> sdsUsePreVec = seedNoMove(dmat, prevFgMaskVec, noMoveThresh); //��

		if (debugDraw){
			Mat sdsUsePreVec2show = getHumansMask(sdsUsePreVec, dmat.size());
			imshow("sdsUsePreVec2show", sdsUsePreVec2show);
		}

		//����555mm�賤�����У� ���˽���ã����˽���ֻ�����У�
#if SOLUTION_1
		int rgThresh = 55;
#else
		int rgThresh = gRgThresh;
#endif

#if 0	//��ʵ�ֵ� validMsk ����
		Mat validMsk = zc::calcPotentialMask(dmat, moveMaskMat, prevFgMaskVec, noMoveThresh, debugDraw);
#elif 1	//�����ʵ�ֵ� validMsk ����
		clock_t begt;
		begt = clock();

		Mat validMsk = sgf::calcPotentialMask(dmat, getPrevDmat());

		static int fcnt = 0;
		static float sumt = 0;
		fcnt++;
		sumt += clock() - begt;
		if(DBG_STD_COUT) cout << "calcPotentialMask.ts: " << sumt / fcnt << endl;	//1.21ms

#endif

		if (debugDraw)
			imshow("trackingNoMove.validMsk-final", validMsk);

#if 0	//v1, ����һ������ǰ��Ԥ�ȷ�����������mask���������⣺Ĩ����ʵǰ������ image-seq-reset-big-area-error.yaml��ԭ�� ����validMsk�����ָδ���������Ϣ
#if 0	//������������mask����	2015��7��29��12:47:55	good��
		{
			clock_t begt;
			begt = clock();

			vector<Mat> tmp = getRgMaskVec(dmat, prevFgMaskVec, validMsk);

			static int fcnt = 0;
			static float sumt = 0;
			fcnt++;
			sumt += clock() - begt;
			cout << "getRgMaskVec.ts: " << sumt / fcnt << endl;	//1.21ms

			Mat tmp_show = getHumansMask(tmp, dmat.size());
			imshow("getRgMaskVec", tmp_show);
		}
#endif
		//֮ǰ�ĵ�һmask���ĳ���mask-vec: 2015��7��29��12:47:27
		//vector<Mat> noMoveFgMskVec = simpleRegionGrow(dmat, sdsUsePreVec, rgThresh, validMsk, debugDraw);
		Mat getRgMaskVec_dbg;
		vector<Mat> rgMskVec = getRgMaskVec(dmat, prevFgMaskVec, validMsk, debugDraw, getRgMaskVec_dbg);

		//������������mask����	2015��7��29��12:47:55	good��
		if (debugDraw){
			Mat rgMskVec_show = getHumansMask(rgMskVec, dmat.size());
			imshow("getRgMaskVec", rgMskVec_show);
			//if (!getRgMaskVec_dbg.empty()) //Ӧ�ò��أ�
				imshow("getRgMaskVec_dbg", getRgMaskVec_dbg);
		}

		vector<Mat> noMoveFgMskVec = simpleRegionGrow(dmat, sdsUsePreVec, rgThresh, rgMskVec, debugDraw);
#elif 1	//v2, ���Զ�������������mask A,B �н�������Ȼ��ȫ�غϣ���������⡾�������򡿣��������������޽�������fgMskֱ����
		//��������
		vector<Mat> noMoveFgMskVec = simpleRegionGrow(dmat, sdsUsePreVec, rgThresh, validMsk, debugDraw);
		size_t noMoveFgMskVecSz = noMoveFgMskVec.size();
		if (noMoveFgMskVecSz > 1)
			int dummy = 0;

#if 01	//�������������������ж� getRealFgMaskVec
		//if (prevFgMaskVec.size() > 0 && noMoveFgMskVec.size() > 0){
			//�Ƿ�����vec��Ȼ�ȳ��� ��
			//CV_Assert(prevFgMaskVec.size() == noMoveFgMskVec.size());
		if (sdsUsePreVec.size() > 0 && sdsUsePreVec.size() == noMoveFgMskVec.size()){ //��

			clock_t begt = clock();

			Mat getRealFgMaskVec_dbg;
			noMoveFgMskVec = getRealFgMaskVec(dmat, prevFgMaskVec, noMoveFgMskVec, debugDraw, getRealFgMaskVec_dbg);
			//static int fcnt = 0;
			//static float sumt = 0;
			//fcnt++;
			//sumt += clock() - begt;
			//cout << "getRealFgMaskVec.ts: " << sumt / fcnt << endl;	//1.21ms

			if (debugDraw){
				//���� debugDraw ���� 0.3ms, ����
				static int fcnt = 0;
				static float sumt = 0;
				fcnt++;
				sumt += clock() - begt;
				if (DBG_STD_COUT)
					cout << "getRealFgMaskVec.ts: " << sumt / fcnt << endl;	//1.21ms

				Mat tmpFgMsk = getHumansMask(noMoveFgMskVec, dmat.size());
				imshow("getRealFgMaskVec", tmpFgMsk);
				imshow("getRealFgMaskVec_dbg", getRealFgMaskVec_dbg);
			}
		}
#endif	//�������������������ж� getRealFgMaskVec

#endif	//��������������ʽ

#if 0	//---------------2015��6��27��14:03:15	�ĳɣ� �Ը���Ϊ׼�����ٲ�������������
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

#elif 1	//trackingNoMove�ع���merge�ӿڣ�
		vector<Mat> resVec = mergeFgMaskVec(noMoveFgMskVec, currInitFgMskVec);
#endif

		return resVec;
	}//trackingNoMove

	cv::Mat calcPotentialMask(const Mat &dmat, const Mat moveMaskMat, const vector<Mat> &prevFgMaskVec, int noMoveThresh, bool debugDraw /*= false*/){
		Mat validMsk;
		//�� SOLUTION_1 ���ã�	2015��7��28��21:04:19
		//Mat flrApartMsk = fetchFloorApartMask(dmat, debugDraw);

		//---------------������ν��potentialMask. @ǰ��������������ɰ����ɷ���V0.1.docx
#if SOLUTION_1	//v1. 2015��7��10��21:44:17	֮ǰ����һ, ȥ��(����+ǽ��)����������������
		//�������⣺ ɳ���ȡ���ͨ�������޷����뱳��

		//�ò������������� mask-vec
		//�ģ� flrApartMsk -> validMsk
		Mat bgMsk = fetchBgMskUseWallAndHeight(dmat);
		validMsk = flrApartMsk & (bgMsk == 0);

#elif 01	//v2. 2015��7��10��21:45:47	�˶�����+�������� && ȥ������
#if 10	//v2 + v3, ��һ����
		// 		int history = 100;
		// 		double varThresh = 1;
		// 		double learnRate = -1;
		// 		//Mat tmp;
		// 		validMsk = zc::maskMoveAndNoMove(dmat, prevFgMaskVec, noMoveThresh, history, varThresh, learnRate, false);
		Mat tmp_mman;
		validMsk = zc::maskMoveAndNoMove(dmat, moveMaskMat, prevFgMaskVec, noMoveThresh, debugDraw, tmp_mman);
		if (debugDraw){
			imshow("maskMoveAndNoMove-debug", tmp_mman);
			//imshow("moveMaskMat", moveMaskMat);
		}

		//validMsk &= flrApartMsk;
#elif 0	//v3. 2015��7��12��14:18:57	�˶�����+ǰһ֡�ɰ�, 
		//ʵ����Ӧ��ÿ��mask��������һ��new-mask, �˴��򻯲���, N������һ��:
		Mat prevFgMask_whole = Mat::zeros(dmat.size(), CV_8UC1);
		size_t prevFgMskVecSz = prevFgMaskVec.size();
		for (size_t i = 0; i < prevFgMskVecSz; i++){
			prevFgMask_whole += prevFgMaskVec[i];
		}
		validMsk = (prevFgMask_whole | moveMaskMat) & flrApartMsk;
#endif	//v2 + v3,

		if (debugDraw)
			imshow("trackingNoMove.validMsk", validMsk);

#if	10	//---------------potentialMask ������
		Mat maxDepBgMask = zc::getMaxDepthBgMask(dmat, true, debugDraw);
		//Ϊʲô�ղ����������ˡ��� 2015��7��22��14:04:57
		//cv::morphologyEx(maxDepBgMask, maxDepBgMask, MORPH_CLOSE, getMorphKrnl());

		if (debugDraw){
			imshow("trackingNoMove.maxDepBgMask", maxDepBgMask);
			imshow("trackingNoMove.~maxDepBgMask", maxDepBgMask == 0);
		}

		//validMsk.setTo(0, maxDepBgMask);
		validMsk &= maxDepBgMask == 0;
#endif

#endif	//����һvs���� v1 + v2

		// 		if (debugDraw)
		// 			imshow("trackingNoMove.validMsk-final", validMsk);

#if 10	//������ν����˲�
		validMsk = largeContPassFilter(validMsk, CONT_AREA, 10);
		if (debugDraw)
			imshow("trackingNoMove.validMsk-final-LCPF", validMsk);
#endif

#if 0	//��������˲���������
		{
			clock_t begt;
			begt = clock();
			Mat tmp = largeContPassFilter(validMsk, CONT_LENGTH, 10);

			static int fcnt = 0;
			static float sumt = 0;
			fcnt++;
			sumt += clock() - begt;
			cout << "largeContPassFilter-CONT_LENGTH.ts: " << sumt / fcnt << endl;	//1.21ms

			imshow("largeContPassFilter-CONT_LENGTH", tmp);
		}
		{
			clock_t begt;
			begt = clock();
			Mat tmp = largeContPassFilter(validMsk, CONT_AREA, 10);

			static int fcnt = 0;
			static float sumt = 0;
			fcnt++;
			sumt += clock() - begt;
			cout << "largeContPassFilter-CONT_AREA.ts: " << sumt / fcnt << endl; //1.29ms

			imshow("largeContPassFilter-CONT_AREA", tmp);
		}
#endif	//��������˲���������

		return validMsk;
	}//calcPotentialMask

	vector<Mat> getRgMaskVec(const Mat &dmat, const vector<Mat> &prevFgMaskVec, Mat currPotentialMask, bool debugDraw /*= false*/, OutputArray _debug_mat /*= noArray()*/){
		Mat debug_mat; //��������۲�ĵ���ͼ
		if (debugDraw){
			_debug_mat.create(dmat.size(), CV_8UC3);
			debug_mat = _debug_mat.getMat(); //��ȡ����ָ��
			debug_mat = Mat::zeros(dmat.size(), CV_8UC3);

			//��
// 			Mat dmat8u;
// 			dmat.convertTo(dmat8u, CV_8U, 1.*UCHAR_MAX / MAX_VALID_DEPTH);
// 			vector<Mat> cn3(3, dmat8u);
// 			cv::merge(cn3, debug_mat);
		}

		vector<Mat> res;

		//CV_Assert(prevFgMaskVec.size() > 0);	//��Ϊ�յ�ʱ����? ��
		size_t prevFgMaskVecSz = prevFgMaskVec.size();
		if (prevFgMaskVecSz == 0)
			return res;

#if 0	//v1, ��һ֡ȫ����Ϊ��ȷ�����򡿡�����������һ���ٷ���ʱ�޷���ȷ����
		Mat prevFgMaskWhole = maskVec2mask(dmat.size(), prevFgMaskVec);
#elif 1	//v2, �ĳ�����Ԥ����������������ȷ�����򣬹���С�����㣺
		//Ҳ������ж���³����һ��(m1)�ֱ۰���һ��(m2)�ֳ����飬��m2����һ���ַǳ�Сʱ����ΪС�����ӻء��������򡿣������ж�ʱ������ dist-map ��2D���㣬����ʵ��m2�����򱻴��󻮷ֵ�m1��
		//Mat prevFgMaskWhole = Mat::zeros(dmat.size(), CV_8UC1);

		//��ǰһ֡ÿ��mask������������
		vector<Mat> prevFgMskLargeAreaVec;
		for (size_t i = 0; i < prevFgMaskVecSz; i++){
			Mat largeArea = largeContPassFilter(prevFgMaskVec[i], CONT_AREA, 1200);
			prevFgMskLargeAreaVec.push_back(largeArea);

			//prevFgMaskWhole.setTo(UCHAR_MAX, largeArea);
		}
		Mat prevFgMaskLargeAreaWhole = maskVec2mask(dmat.size(), prevFgMskLargeAreaVec);
#endif	//��ȷ�������ж�����

		//��ǰ֡��ǰһ֡����Ǳ������(��������)��û�õ������Թ۲��ã� 2015��7��29��13:03:46
		Mat currUnsureMask = currPotentialMask - prevFgMaskLargeAreaWhole;
		
		//�� prev-vec��һ�� dist-map:
		vector<Mat> dtransVec;
		for (size_t i = 0; i < prevFgMaskVecSz; i++){
			//��
			//Mat prevFg_i = prevFgMaskVec[i];
			//�ĳ��ô����vec��
			Mat prevFg_i = prevFgMskLargeAreaVec[i];

			//ȷ������
			Mat prevFg_i_sure = prevFg_i & currPotentialMask;
			Mat  dtrans_i;
			distanceTransform(prevFg_i_sure == 0, dtrans_i, distType, maskSize);

			dtransVec.push_back(dtrans_i);
		}

		//���Ի��ơ�ȷ�����򡿣�K�׻�ɫ
		vector<Mat> prevFgSureVec;
		if (debugDraw){
			for (size_t i = 0; i < prevFgMaskVecSz; i++){
				//��
				//Mat prevFg_i = prevFgMaskVec[i];
				//�ĳ��ô����vec��
				Mat prevFg_i = prevFgMskLargeAreaVec[i];

				//ȷ������
				Mat prevFg_i_sure = prevFg_i & currPotentialMask;
				prevFgSureVec.push_back(prevFg_i_sure);
			}
			Mat prevFgSureWhole = getHumansMask(prevFgSureVec, dmat.size());
			vector<Mat> cn3(3, prevFgSureWhole);
			cv::merge(cn3, debug_mat);
		}

		//��ÿ�����أ��õ���С��dtransֵ
		size_t dtransVecSz = dtransVec.size();//==prevFgMaskVecSz
#if 0	//v1, ѭ���� dtransMin
		Mat dtransMin = dtransVec[0].clone();
		for (size_t i = 1; i < dtransVecSz; i++){
			Mat dtrans_i = dtransVec[i];
			dtransMin = cv::min(dtrans_i, dtransMin);
		}
#elif 1	//v2, �ĳ�ֱ����: Ч�ʻ�ߣ���δ���ԡ�
		Mat dtransMin;
		//distanceTransform((prevFgMaskWhole & currPotentialMask) == 0, dtransMin, distType, maskSize);
		distanceTransform((prevFgMaskLargeAreaWhole & currPotentialMask) == 0, dtransMin, distType, maskSize);
#endif	//v1,v2, dtransMin ������

#if 0	//���� dtransPrevWhole & dtransMin �Ƿ�ȼۣ��ǣ�
		{
			Mat dtransPrevWhole;
			//��ע�⡿�� �� prevFgMaskWhole & currPotentialMask, ��ֻ�� prevFgMaskWhole
			distanceTransform((prevFgMaskWhole & currPotentialMask) == 0, dtransPrevWhole, CV_DIST_L2, DIST_MASK_PRECISE);
			CV_Assert(countNonZero(abs(dtransMin - dtransPrevWhole) > 1e-5) == 0);
			if (countNonZero(abs(dtransMin - dtransPrevWhole) > 1e-5) != 0)
				int dummy = 0;
		}
#endif	//���� dtransPrevWhole & dtransMin �Ƿ�ȼۣ��ǣ�

		//dtransMin �� N��dtrans�Ƚϣ�
		//���������������ԣ�
		Mat mutexMat = Mat::zeros(dmat.size(), CV_8UC1);
		for (size_t i = 0; i < dtransVecSz; i++){
			Mat msk_i = abs(dtransMin - dtransVec[i]) < 1e-5 
				& currPotentialMask 
				& (mutexMat == 0);

			mutexMat += msk_i;

			res.push_back(msk_i);
		}

		//���Ի��ơ��������򡿣�K�ס���ɫ��
		if (debugDraw){
			vector<Mat> currFgUnsureVec;

			size_t resSz = res.size();//==dtransVecSz
			for (size_t i = 0; i < resSz; i++){
				Mat res_i = res[i],
					prevFgSure = prevFgSureVec[i];

				//��������
				Mat currFgUnsure = res_i - prevFgSure;
				currFgUnsureVec.push_back(currFgUnsure);
			}

			//Mat prevFgSureWhole = getHumansMask(prevFgSureVec, dmat.size());
			Mat currFgUnsureWhole = getHumansMask(currFgUnsureVec, dmat.size());
			Mat blackMat = Mat::zeros(dmat.size(), CV_8UC1);
			vector<Mat> cn3;
			//BGR-order:
			cn3.push_back(blackMat);
			cn3.push_back(blackMat);
			cn3.push_back(currFgUnsureWhole);

			Mat tmp;
			cv::merge(cn3, tmp);
			debug_mat += tmp;
		}

		return res;
	}//getRgMaskVec

	vector<Mat> getRealFgMaskVec(const Mat &dmat, const vector<Mat> &prevFgMaskVec, const vector<Mat> &currFgMaskVec, bool debugDraw /*= false*/, OutputArray _debug_mat /*= noArray()*/){
		Mat debug_mat; //��������۲�ĵ���ͼ
		if (debugDraw){
			_debug_mat.create(dmat.size(), CV_8UC3);
			debug_mat = _debug_mat.getMat(); //��ȡ����ָ��
			debug_mat = Mat::zeros(dmat.size(), CV_8UC3);
		}

		//����ֵ��
		vector<Mat> res;

		size_t prevFgMaskVecSz = prevFgMaskVec.size();
		//��Ϊ�յ�ʱ����? Ҳ�У�
		CV_Assert(prevFgMaskVecSz > 0);	
		//�Ƿ��Ȼ�ȳ�����δ���ԡ�
		CV_Assert(prevFgMaskVecSz == currFgMaskVec.size()); 

		if (prevFgMaskVecSz == 0)
			return res;

		//�ص���������ǣ���Ϊ�鷺�����ص���Ȼ�д����ԣ�����
		size_t currFgMaskVecSz = currFgMaskVec.size();
		vector<bool> overlapFlagVec(currFgMaskVecSz, false);
		for (size_t i = 0; i < currFgMaskVecSz; i++){
			if (overlapFlagVec[i])
				continue;

			Mat currFg_i = currFgMaskVec[i];
			for (size_t k = i + 1; k < currFgMaskVecSz; k++){
				//������ÿ������ overlapFlagVec[k] ��Ȼδ������ //��
				//CV_Assert(overlapFlagVec[k] != true); //��, e.g., 1��3, 2����
				if (overlapFlagVec[k]){
					//if (debugDraw){
					//	Mat zeroMat = Mat::zeros(dmat.size(), CV_8UC1);
					//	imshow("CV_Assert(overlapFlagVec[k] != true)", zeroMat);
					//}

					continue;
				}

				Mat currFg_k = currFgMaskVec[k];
				//���ص���
				if (countNonZero(currFg_i & currFg_k) > 0){
					overlapFlagVec[i] = true;
					overlapFlagVec[k] = true;
					//break; //��Ҫ����
				}
			}
		}//for-�ص�

		Mat currFgMask_whole = maskVec2mask(dmat.size(), currFgMaskVec);

		//���ɡ�ȷ������vec:
		vector<Mat> fgMskSureVec;
		vector<Mat> noMoveMatVec = seedNoMove(dmat, prevFgMaskVec, gNoMoveThresh);
		//����ÿ��maskSure��һ�� dist-map:
		vector<Mat> dtransVec;

		size_t noMoveMatVecSz = noMoveMatVec.size();
		CV_Assert(noMoveMatVecSz == prevFgMaskVecSz);
		for (size_t i = 0; i < prevFgMaskVecSz; i++){
			//���������ص�����������ȷ������
			if (!overlapFlagVec[i]){
				fgMskSureVec.push_back(currFgMaskVec[i]);
			}
			//�����ص�(�غ�),������һ֡û������:
			else{
				Mat fgMskSure_i = noMoveMatVec[i] & currFgMask_whole; //���� currPotentialMask
				fgMskSureVec.push_back(fgMskSure_i);
			}

			Mat  dtrans_i;
			distanceTransform(fgMskSureVec[i] == 0, dtrans_i, distType, maskSize);
			dtransVec.push_back(dtrans_i);
		}

		//���Ի��ơ�ȷ�����򡿣�K�׻�ɫ
		if (debugDraw){
			Mat fgMskSure_whole = getHumansMask(fgMskSureVec, dmat.size());
			vector<Mat> cn3(3, fgMskSure_whole);
			cv::merge(cn3, debug_mat);
		}

		//��ÿ�����أ��õ���С��dtransֵ
		size_t dtransVecSz = dtransVec.size();//==prevFgMaskVecSz
#if 01	//v1, ѭ���� dtransMin
		Mat dtransMin = dtransVec[0].clone();
		for (size_t i = 1; i < dtransVecSz; i++){
			Mat dtrans_i = dtransVec[i];
			dtransMin = cv::min(dtrans_i, dtransMin);
		}
#endif

		//dtransMin �� N��dtrans�Ƚϣ�
		//���������������ԣ�
		Mat mutexMat = Mat::zeros(dmat.size(), CV_8UC1);
		for (size_t i = 0; i < dtransVecSz; i++){
			Mat msk_i = abs(dtransMin - dtransVec[i]) < 1e-5
				& currFgMask_whole
				& (mutexMat == 0);

			mutexMat += msk_i;

			res.push_back(msk_i);
		}

		//���Ի��ơ��������򡿣�K�ס���ɫ��
		if (debugDraw){
			vector<Mat> currFgUnsureVec;

			size_t resSz = res.size();//==dtransVecSz
			for (size_t i = 0; i < resSz; i++){
				Mat res_i = res[i],
					fgMskSure = fgMskSureVec[i];

				//��������
				Mat currFgUnsure = res_i - fgMskSure;
				currFgUnsureVec.push_back(currFgUnsure);
			}

			//Mat prevFgSureWhole = getHumansMask(prevFgSureVec, dmat.size());
			Mat currFgUnsureWhole = getHumansMask(currFgUnsureVec, dmat.size());
			Mat blackMat = Mat::zeros(dmat.size(), CV_8UC1);
			vector<Mat> cn3;
			//BGR-order:
			cn3.push_back(blackMat);
			cn3.push_back(blackMat);
			cn3.push_back(currFgUnsureWhole);

			Mat tmp;
			cv::merge(cn3, tmp);
			debug_mat += tmp;
		}

		return res;
	}//getRealFgMaskVec


	vector<Mat> trackFgMaskVec(Mat &dmat, const vector<Mat> &prevFgMaskVec, vector<Mat> rgMaskVec, int noMoveThresh /*= 55*/, bool debugDraw /*= false*/){
		//seedNoMove ���˴����� o(�s���t)o
		vector<Mat> sdsUsePreVec = seedNoMove(dmat, prevFgMaskVec, noMoveThresh); //��
		if (debugDraw){
			Mat sdsUsePreVec2show = getHumansMask(sdsUsePreVec, dmat.size());
			imshow("sdsUsePreVec2show", sdsUsePreVec2show);
		}

		int rgThresh = 355;
		vector<Mat> noMoveFgMskVec = simpleRegionGrow(dmat, sdsUsePreVec, rgThresh, rgMaskVec, debugDraw);
		return noMoveFgMskVec;
	}//trackFgMaskVec


	vector<Mat> mergeFgMaskVec(const vector<Mat> &baseVec, const vector<Mat> &newVec, bool debugDraw /*= false*/){
		vector<Mat> resVec;
		size_t baseVecSz = baseVec.size();
#if 0	//2015��7��30��19:19:08	merge֮ǰ���뱣֤�����ԡ����ظ������������ⲻȥ��
		//��ʼ�� baseVec push resVec, ҲҪ����ȥ��
		for (size_t i = 0; i < baseVecSz; i++){
			bool isExist = false;
			size_t resVecSz = resVec.size();
			for (size_t k = 0; k < resVecSz; k++){
				Mat resMsk_k = resVec[k];
				if (countNonZero(baseVec[i] & resMsk_k) != 0){
					isExist = true;
					break;
				}
			}
			if (!isExist){
				resVec.push_back(baseVec[i].clone());
			}
		}
#elif 1	//ֱ�Ӹ��ƣ���ȥ�أ�
		resVec = baseVec;
#endif

		//�����newVec �� resVec ���ص���������
		size_t newVecSz = newVec.size();
		for (size_t i = 0; i < newVecSz; i++){
			Mat newMsk_i = newVec[i];

			bool isExist = false;
			size_t resVecSz = resVec.size();
			for (size_t k = 0; k < resVecSz; k++){
				Mat resMsk_k = resVec[k];

				if (countNonZero(newMsk_i & resMsk_k) != 0){
					isExist = true;
					break;
				}
			}

			if (!isExist){
				resVec.push_back(newMsk_i);
			}
		}

		return resVec;
	}//mergeFgMaskVec

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

				if (fgMskIsHuman(dmat, tmp)){
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
// 						//fgMskIsHuman(dmat, )
// 					}
// 				}
// 			}
				
		}//for (size_t i = 0; i < mskVecSz; i++)
		
		return res;
	}//separateMasksXYview


	vector<Mat> separateMasksXZview(Mat dmat, vector<Mat> &inMaskVec, int zSepThresh /*= 300*/, bool debugDraw /*= false*/){
		vector<Mat> res;
		size_t mskVecSz = inMaskVec.size();
		//��XYview ÿ�� mask��
		for (size_t i = 0; i < mskVecSz; i++){
			Mat mskXY_i = inMaskVec[i];

			Mat maskedDmat = dmat.clone();
			maskedDmat.setTo(0, mskXY_i == 0);
			//��֮ǰ��rgThresh����һ�£�
			//int rgThresh = gRgThresh;
			int rgThresh = zSepThresh;
			double ratio = 1. / rgThresh;
			Mat tdview = dmat2tdview_core(maskedDmat, ratio);
			tdview.convertTo(tdview, CV_8U);
			
			if (debugDraw){
				imshow("separateMasksXZview.tdview", tdview);
			}

			vector<vector<Point>> contoursXZ;
			findContours(tdview.clone(), contoursXZ, RETR_EXTERNAL, CHAIN_APPROX_NONE);
			size_t contXZ_size = contoursXZ.size();
			//CV_Assert(contXZ_size > 0);//��, �޳�ǽ�汳�����ܵ���ǰ����ɾ��
			if (contXZ_size == 0){
				//do-nothing?
			}
			else if (contXZ_size == 1){//XZ����ͼ����һ���������������������·ֿ����龳
				res.push_back(mskXY_i);
			}
			else{ //contXZ_size > 1, �������������ҷֿ���ǰ��ֿ��龳
				
#if 0			//1. ����һ�� �������������ɼ������������㼸����
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
// 					if (fgMskIsHuman(dmat, subMskVec[i]))
// 						res.push_back(subMskVec[i]);
// 				}

#elif 1			//2. �������� contoursXZ��ͶӰ��ÿ��cont������õ�һ��humMsk��2015��6��27��23:30:01
				//��XZviewÿ��cont��

				bool debugError = false;

				//ͳ�Ƶ�ǰcont sepXZ֮���м�����������ˣ�
				int cnt = 0;
				//���ֿ��ļ������������С, ����ѡ����������, �Ա�֤����ͻȻ��ʧ
				int maxArea = 0;
				Mat maxAreaMat;
				for (size_t i = 0; i < contXZ_size; i++){
					//�����Ľ����
					Mat newMskXY = Mat::zeros(dmat.size(), CV_8UC1);

					vector<Point> &contXZi = contoursXZ[i];
					//cont-mask, �ڲ�ȫ�ף�δȥ���׶�������������(dmin, dmax)�㹻��
					Mat cmskXZ_i = Mat::zeros(tdview.size(), CV_8UC1);
					drawContours(cmskXZ_i, contoursXZ, i, 255, -1);

					//���ұ߽磺
					Rect bboxXZi = zc::boundingRect(contXZi);
					int left = bboxXZi.x,
						right = bboxXZi.x + bboxXZi.width - 1;//��ע�⡿ -1

					//��contXZi������ÿһcol����ͶӰ����(dmin, dmax)
					for (int k = left; k <= right; k++){//��ע�⡿ <=
						//tdview �ϣ�
						//Mat colXZ_k = tdview.col(k);//��
						Mat colXZ_k = cmskXZ_i.col(k);//��

						vector<Point> nonZeroPts;
						if(countNonZero(colXZ_k))
							findNonZero(colXZ_k, nonZeroPts);
						Rect bboxCol_k = zc::boundingRect(nonZeroPts);
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
					
					//1. ��ǰ���ж�
					//if (fgMskIsHuman(dmat, newMskXY))
					//2. fgMskIsHuman ���岻��, �ĳ� bboxWscale: 2015��8��2��23:58:41
					//if (bboxIsHumanWscale(dmat, newMskXY))
					//3. bboxIsHumanWscale ������, ����wscale����� 2015��8��3��00:00:26
					if (maskedCvSum(dmat, newMskXY)[0] > 100 * 100 * 1000){
						res.push_back(newMskXY);
						cnt++;
					}

					int currArea = countNonZero(newMskXY);
					if (currArea > maxArea){
						maxArea = currArea;
						maxAreaMat = newMskXY;
					}
				}//for-contXZ_size

				//��֮ǰ�ֿ��ļ�������Ϊ���̫С��û�����, �����������:
				if (cnt == 0)
					res.push_back(maxAreaMat);

#endif	//�������� vs. contoursXZ��ͶӰ

			}//else-- contXZ_size > 1
		}//for- i < mskVecSz

		return res;
	}//separateMasksXZview


	vector<Mat> separateMasksXXview(Mat dmat, vector<Mat> &inMaskVec, bool debugDraw /*= false*/){
		vector<Mat> res;

		size_t mskVecSz = inMaskVec.size();
		for (size_t i = 0; i < mskVecSz; i++){
			Mat mskXY_i = inMaskVec[i];

			Mat maskedDmat = dmat.clone();
			maskedDmat.setTo(0, mskXY_i == 0);

			vector<vector<Point>> contoursXY;
			findContours(mskXY_i.clone(), contoursXY, RETR_EXTERNAL, CHAIN_APPROX_NONE);

			size_t contXYsz = contoursXY.size();
			for (size_t k = 0; k < contXYsz; k++){
				//�����ָ�/�������Ľ����
				Mat tmp = Mat::zeros(dmat.size(), CV_8UC1);
				drawContours(tmp, contXYsz, i, 255, -1);
				tmp &= mskXY_i;

			}
		}

		return res;
	}//separateMasksXXview


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
		Rect xyBbox = zc::boundingRect(conti);
		
		return Rect(xyBbox.x, dmin, xyBbox.width, dmax - dmin);
	}//contour2XZbbox


	vector<Mat> separateMasksMoving(Mat dmat, vector<Mat> &inMaskVec, bool debugDraw /*= false*/){
		vector<Mat> res;

		Mat dmat8u;
		dmat.convertTo(dmat8u, CV_8U, 1.*UCHAR_MAX / MAX_VALID_DEPTH);
		
		int history = 10;
		int diffThresh = 25;
		static MyBGSubtractor myBgsub(history, diffThresh);


		size_t mskVecSz = inMaskVec.size();
		//��XYview ÿ�� mask��
		for (size_t i = 0; i < mskVecSz; i++){
			Mat mskXY_i = inMaskVec[i];

			Mat maskedDmat = dmat.clone();
			maskedDmat.setTo(0, mskXY_i == 0);

		}

		return res;
	}

#ifdef CV_VERSION_EPOCH //if opencv2.x
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

	cv::Mat seedBgsMOG2(const Mat &dmat, bool isNewFrame /*= true*/, int erodeRadius /*= 2*/, bool debugDraw /*= false*/, OutputArray _debug_mat /*= noArray()*/)
{
		Mat res;
		Mat dm_show;
		dmat.convertTo(dm_show, CV_8UC1, 1.*UCHAR_MAX / MAX_VALID_DEPTH);
		int history = 100;
		double varThresh = 1;// 6;
		bool detectShadows = false;
		static Ptr<BackgroundSubtractorMOG2> pMOG2 =
			//createBackgroundSubtractorMOG2();// 500, 5, false);
			createBackgroundSubtractorMOG2(history, varThresh, detectShadows);

		Mat bgImg, fgMskMOG2;

		//�� isNewFrame=false����ֱ����prevFgMskMOG2��Ϊ��ǰ֡ǰ��
		static Mat prevFgMskMOG2; 
		static bool isFirstTime = true;

		if (isFirstTime || isNewFrame){

			double learningRate = -0.005;
			//���� mat-type ����Ϊ 8uc1, or 8uc3:
			pMOG2->apply(dm_show, fgMskMOG2, learningRate);
			
			if (debugDraw)
				imshow("seedBgsMOG2.fgMskMOG2", fgMskMOG2);

			//��һ֡fgMskMOG2����û��history����ȫ�ף��ʲ���apply�����
			if (isFirstTime)
				fgMskMOG2 = Mat::zeros(dmat.size(), CV_8UC1);

			if (erodeRadius > 0){
				int anch = erodeRadius;
				Mat morphKrnl = getStructuringElement(MORPH_RECT, Size(2 * anch + 1, 2 * anch + 1), Point(anch, anch));
				erode(fgMskMOG2, fgMskMOG2, morphKrnl);
			}

			//ȥ����Ч����
			fgMskMOG2 &= (dmat != 0);

			//---------------
			prevFgMskMOG2 = fgMskMOG2;// .clone(); //���� clone��
			isFirstTime = false;
		}
		else{
			fgMskMOG2 = prevFgMskMOG2;
		}
		pMOG2->getBackgroundImage(bgImg);
		if (debugDraw)
			imshow("seedBgsMOG2.bgImg", bgImg);

		return fgMskMOG2;
	}

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
#endif //CV_VERSION_MAJOR >= 3


	Mat seedNoMove(Mat dmat, /*Mat prevDmat, */Mat mask, int thresh /*= 50*/){
	//Mat seedNoMove(Mat dmat, Mat mask, int thresh /*= 50*/){
		Mat res = mask.clone();

		//��һ�ε���ʱ prevDmat ȫ�ڣ����diff > thresh�����ص� mask Ӧ��Ҳȫ�ڣ�
// 		static Mat prevDmat = Mat::zeros(dmat.size(), dmat.type());
		Mat prevDmat = getPrevDmat();

		res &= (cv::abs(dmat - prevDmat) < thresh);

		//��ʴһ�£���ֹ dmax-dmin>thickLimit ������ܶž�����
// 		Mat morphKrnl = getMorphKrnl(3);
// 		erode(res, res, morphKrnl);

		//prevDmat = dmat.clone();

		return res;
	}//seedNoMove

	vector<Mat> seedNoMove(Mat dmat, vector<Mat> maskVec, int thresh /*= 50*/){
		vector<Mat> res;
		size_t mskVecSize = maskVec.size();
		for (size_t i = 0; i < mskVecSize; i++){
			Mat newMask = seedNoMove(dmat, /*prevDmat, */maskVec[i], thresh);
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

		Mat mskWhole = abs(dmat - prevDmat) < thresh;
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

	//@test ���û���⣬pass
	cv::Mat testBgFgMOG2(const Mat &dmat, int history /*= 100*/, double varThresh /*= 1*/, double learnRate /*= -1*/)
{
		Mat dmat8u;
		dmat.convertTo(dmat8u, CV_8UC1, 1.*UCHAR_MAX / MAX_VALID_DEPTH);

		bool detectShadows = false;
#ifdef CV_VERSION_EPOCH //if opencv2.x
		static Ptr<BackgroundSubtractorMOG2> pMog2 = new BackgroundSubtractorMOG2(history, varThresh, detectShadows);
#elif CV_VERSION_MAJOR >= 3 //if opencv3
		static Ptr<BackgroundSubtractorMOG2> pMog2 = createBackgroundSubtractorMOG2(history, varThresh, detectShadows);
#endif //CV_VERSION_EPOCH

		Mat fgMskMog2;

#ifdef CV_VERSION_EPOCH //if opencv2.x
		(*pMog2)(dmat8u, fgMskMog2);
#elif CV_VERSION_MAJOR >= 3 //if opencv3
		pMog2->apply(dmat8u, fgMskMog2, learnRate);
#endif //CV_VERSION_EPOCH

		return fgMskMog2;
	}//testBgFgMOG2

	Mat testBgFgKNN(const Mat &dmat, int history /*= 500*/, double dist2Threshold /*= 400.0*/, double learnRate /*= -1*/ ){
		Mat dmat8u;
		dmat.convertTo(dmat8u, CV_8UC1, 1.*UCHAR_MAX / MAX_VALID_DEPTH);

		Mat fgMskKnn;

#ifdef CV_VERSION_EPOCH //if opencv2.x
		//cv2 û��ʵ�� KNN������
#elif CV_VERSION_MAJOR >= 3 //if opencv3

		bool detectShadows = false;
		static Ptr<BackgroundSubtractorKNN> pKnn = createBackgroundSubtractorKNN(history, dist2Threshold, detectShadows);

		pKnn->apply(dmat8u, fgMskKnn, learnRate);
#endif //CV_VERSION_EPOCH

		return fgMskKnn;
	}//testBgFgKNN


	cv::Mat maskMoveAndNoMove(Mat dmat, vector<Mat> prevFgMaskVec, int noMoveThresh /*= 50*/, int history /*= 100*/, double varThresh /*= 1*/, double learnRate /*= -1*/, bool debugDraw /*= false*/, OutputArray _debug_mat /*= noArray()*/){
 
		Mat dmat8u;
		dmat.convertTo(dmat8u, CV_8UC1, 1.*UCHAR_MAX / MAX_VALID_DEPTH);

		Mat debug_mat;
		if (debugDraw){
			//����mat���ò�ɫ���ƣ�
			//_debug_mat.create(dmat.size(), CV_8UC4);
			_debug_mat.create(dmat.size(), CV_8UC3);
			debug_mat = _debug_mat.getMat();
			debug_mat.setTo(0);
		}
 
		Mat res;
 
		bool detectShadows = false;
#ifdef CV_VERSION_EPOCH //if opencv2.x
#elif CV_VERSION_MAJOR >= 3 //if opencv3
		static Ptr<BackgroundSubtractorMOG2> pMog2 = createBackgroundSubtractorMOG2(history, varThresh, detectShadows);
#endif //CV_VERSION_EPOCH

		Mat fgMskMog2;

#ifdef CV_VERSION_EPOCH //if opencv2.x
#elif CV_VERSION_MAJOR >= 3 //if opencv3
		pMog2->apply(dmat8u, fgMskMog2);
#endif //CV_VERSION_EPOCH

		if (debugDraw)
			imshow("maskMoveAndNoMove.fgMskMog2", fgMskMog2);

		res = fgMskMog2;

		if (debugDraw){
// 			Mat fgMskMog2cn4;
// 			cv::merge(vector<Mat>(4, fgMskMog2), fgMskMog2cn4);
// 			debug_mat.setTo(cwhite, fgMskMog2cn4);
			debug_mat.setTo(cwhite, fgMskMog2);
		}
 
		vector<Mat> sdNoMoveVec = seedNoMove(dmat, prevFgMaskVec, noMoveThresh);
		size_t sdNoMoveVecSz = sdNoMoveVec.size();
		for (size_t i = 0; i < sdNoMoveVecSz; i++){
			Mat sdNoMoveMsk_i = sdNoMoveVec[i];
			res |= sdNoMoveMsk_i;
			
			if (debugDraw){
				debug_mat.setTo(cgreen, sdNoMoveMsk_i);
				debug_mat.setTo(cred, sdNoMoveMsk_i & fgMskMog2);
			}
		}
		
		return res;
	}//maskMoveAndNoMove

	cv::Mat maskMoveAndNoMove(Mat dmat, Mat moveMask, vector<Mat> prevFgMaskVec, int noMoveThresh /*= 50*/, bool debugDraw /*= false*/, OutputArray _debug_mat /*= noArray()*/){
		Mat res;
		Mat debug_mat;
		if (debugDraw){
			//����mat���ò�ɫ���ƣ�
			//_debug_mat.create(dmat.size(), CV_8UC4);
			_debug_mat.create(dmat.size(), CV_8UC3);
			debug_mat = _debug_mat.getMat();
			debug_mat.setTo(0);
			debug_mat.setTo(cwhite, moveMask);
		}

#if 0	//v1�汾�� �� seedNoMove
		res = moveMask.clone();

		vector<Mat> sdNoMoveVec = seedNoMove(dmat, prevFgMaskVec, noMoveThresh);
		size_t sdNoMoveVecSz = sdNoMoveVec.size();
		for (size_t i = 0; i < sdNoMoveVecSz; i++){
			Mat sdNoMoveMsk_i = sdNoMoveVec[i];
			res |= sdNoMoveMsk_i;

			if (debugDraw){
				debug_mat.setTo(cgreen, sdNoMoveMsk_i);
				debug_mat.setTo(cred, sdNoMoveMsk_i & moveMask);
			}
		}
#elif 1	//v2�汾�� �� moveMask ȡ���� no-move-mask
		//��ʵ��v3��ֻ��v2���Դ��ڿ���������
		res = moveMask.clone();

		//����mat�ı�����
		Mat bgMask = (moveMask == 0);

		size_t prevFgMskVecSz = prevFgMaskVec.size();
		for (size_t i = 0; i < prevFgMskVecSz; i++){
			Mat prevFgMsk_i = prevFgMaskVec[i];
			//��������ǰһ֡msk�뵱ǰMOG����msk�󽻣�
			Mat noMoveMsk_i = prevFgMsk_i & bgMask;
			res |= noMoveMsk_i;

			if (debugDraw){
				debug_mat.setTo(cgreen, noMoveMsk_i);
				debug_mat.setTo(cred, noMoveMsk_i & moveMask); //��Ȼ������֣���Ϊ����
			}
		}

#elif 1 //v3. 2015��7��12��14:18:57	�˶�����+ǰһ֡�ɰ�, ʵ�ʵȼ���v2��v3Ч�ʻ���𣿡�δ���ԡ�
		Mat prevFgMask_whole = Mat::zeros(dmat.size(), CV_8UC1);
		size_t prevFgMskVecSz = prevFgMaskVec.size();
		for (size_t i = 0; i < prevFgMskVecSz; i++){
			prevFgMask_whole += prevFgMaskVec[i];
		}
		res = (prevFgMask_whole | moveMaskMat) & flrApartMsk;

#endif

#if 0	//�������ͣ����ܲ��ȶ����ó�������ûʲô��
		int krnlRadius = 6;
		dilate(res, res, getMorphKrnl(krnlRadius));
#endif
		//��ע�⡿һ���۳���Ч����
		res.setTo(0, dmat == 0);

		return res;
	}//maskMoveAndNoMove


	cv::Mat maskMoveAndStable(Mat moveMask, Mat prevFgMask)
	{
		Mat res;

		return res;
	}

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

	//@brief ��ȫ�ֵ� vector<HumanObj> ��һ����ɫ mask mat:
	Mat getHumansMask(Mat dmat, const vector<HumanObj> &humVec){
		Mat res;
		//8uc3 (or c4?), ��ɫ��
		Mat dmat8u;
		//һ�»ҶȻ�:
		//dmat.convertTo(dmat8u, CV_8U, 1.*UCHAR_MAX / MAX_VALID_DEPTH);
		//���÷�ɫ�� ��������:
		dmat.convertTo(dmat8u, CV_8U, -1.*UCHAR_MAX / MAX_VALID_DEPTH, UCHAR_MAX);	
		dmat8u.setTo(0, dmat == 0);
		
		//��Ϊ�Աȶ����(��������������˸):
		//normalize(dmat, dmat8u, 0, UCHAR_MAX, NORM_MINMAX, CV_8UC1);

		//cvtColor(dmat8u, res, CV_GRAY2BGRA);	//��Ҫ͸��A
		cvtColor(dmat8u, res, CV_GRAY2BGR);

		size_t humSz = humVec.size();
		for (size_t i = 0; i < humSz; i++){
			HumanObj hum_i = humVec[i];
			Mat humMask_i = hum_i.getCurrMask();
			Scalar c = hum_i.getColor();
#if 01		//v1, ��ʵ�ġ������ѷ�����ȫ�ڵ����󣬸Ļ�����
			res.setTo(0, humMask_i);	//���ڿ�
			
			//Ҫ+=�Ĳ�ɫmask
			Mat fgAreaColorMat = Mat::zeros(dmat.size(), CV_8UC3);
			
			fgAreaColorMat.setTo(c, humMask_i);
#if 0		//BGR->Lab, �Աȶ����normalizeǰ������L��
			cvtColor(fgAreaColorMat, fgAreaColorMat, COLOR_BGR2Lab);
			vector<Mat> cn3;
			cv::split(fgAreaColorMat, cn3);
			cn3[0] = dmat8u.clone();
			normalize(cn3[0], cn3[0], 0, UCHAR_MAX, NORM_MINMAX, -1, humMask_i);

			cv::merge(cn3, fgAreaColorMat);
			cvtColor(fgAreaColorMat, fgAreaColorMat, COLOR_Lab2BGR);
#elif 1		//BGR->HSV, ���Ҷ����á�V��, ��ת��BGR:
			cvtColor(fgAreaColorMat, fgAreaColorMat, COLOR_BGR2HSV);
			vector<Mat> cn3;
			cv::split(fgAreaColorMat, cn3);
			Mat cn_v = dmat8u.clone();
			cn_v.setTo(0, humMask_i == 0);
			normalize(cn_v, cn_v, UCHAR_MAX / 2, UCHAR_MAX, NORM_MINMAX, -1, humMask_i);

			cn3[2] = cn_v;
			cn3[1].setTo(255);//��S�����Ͷ����

			cv::merge(cn3, fgAreaColorMat);
			cvtColor(fgAreaColorMat, fgAreaColorMat, COLOR_HSV2BGR);
#endif

			res += fgAreaColorMat;
#elif 1		//v2, ������
			vector<vector<Point>> contours;
			//findContours(humMask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
			findContours(humMask_i.clone(), contours, RETR_TREE, CHAIN_APPROX_NONE);

			//����һ��mask �ж��cont(e.g., �ڵ��龳)��Ҫ����һ��ɫ��
			drawContours(res, contours, -1, c, 2);

			//���� �ڵ��ص�ʱ�����������������ģ�
// 			size_t contSz = contours.size();
// 			for (size_t i = 0; i < contSz; i++){
// 				//����һ�׾�
// 				Moments mu = moments(contours[i]);

			vector<Point> flatConts;
			flatten(contours.begin(), contours.end(), back_inserter(flatConts));
#if 0	//����flatten-moments�����ģ��Զ����ɢ���������Ǵ��
			Moments mu = moments(flatConts);
			Point mc;
			if (abs(mu.m00) < 1e-8) //area is zero
				//mc = contours[i][0];
				mc = flatConts[0];
			else
				mc = Point(mu.m10 / mu.m00, mu.m01 / mu.m00);

#else	//�ݸ�Ϊ��flatten-mean��Ϊ���ģ�
			Point2f mc(0, 0);

			size_t flatContsSz = flatConts.size();
			for (int i = 0; i < flatContsSz; i++){
				Point2f pt = flatConts[i];
				mc = (mc * i + pt) / (i + 1);
			}
#endif
			circle(res, mc, 5, c, 2);
#endif		//������

		}

		return res;
	}//getHumansMask


	cv::Mat getHumansMask2tdview(Mat dmat, const vector<HumanObj> &humVec){
		//�Լ�ʵ�ֵ�ǰ����⣬�õ�mask��
		Mat tdview = zc::dmat2tdview_core(dmat);
		Mat tdview_show;
		tdview.convertTo(tdview_show, CV_8U);

		int history = 10;
		int diffThresh = 25;
		static MyBGSubtractor myBgsub_tdview(history, diffThresh);
		Mat myFgMsk_tdview = myBgsub_tdview.apply(tdview_show);


		//8uc3, ��ɫ��
		double ratio = 1. * UCHAR_MAX / MAX_VALID_DEPTH;
// 		Mat dmatSquash;
// 		dmat.convertTo(dmatSquash, CV_16U, ratio);
		
		Mat res = 
			//Mat::zeros(Size(dmat.cols, MAX_VALID_DEPTH * ratio + 1), CV_8UC1);
			Mat::zeros(Size(dmat.cols, MAX_VALID_DEPTH * ratio + 1), CV_8UC4);

// 		Mat dmat8u;
// 		dmat.convertTo(dmat8u, CV_8U, 1.*UCHAR_MAX / MAX_VALID_DEPTH);
// 		cvtColor(dmat8u, res, CV_GRAY2BGRA);

		size_t humSz = humVec.size();
		for (size_t i = 0; i < humSz; i++){
			HumanObj hum = humVec[i];
			Mat humMask = hum.getCurrMask();
			Scalar c = hum.getColor();

			//ʵ�ʻҶ�ͼ��
			Mat dmat_hum_masked = dmat.clone();
			dmat_hum_masked.setTo(0, humMask == 0);

			Mat tdv_i = dmat2tdview_core(dmat_hum_masked, ratio);
			tdv_i.convertTo(tdv_i, CV_8U);

			Mat tdv_i_cn4;
			cvtColor(tdv_i, tdv_i_cn4, CV_GRAY2BGRA);
			res += tdv_i_cn4;

			//��ɫbbox����ɫ�� humObj һ�£�
			Rect bbox = zc::boundingRect(tdv_i);
			rectangle(res, bbox, c, 1);

			//��ɫ���� moving ����
			res.setTo(cyellow, myFgMsk_tdview & (tdv_i != 0));

		}

		return res;
	}//getHumansMask2tdview

	vector<Mat> humVec2tdviewVec(Mat dmat, const vector<HumanObj> &humVec){
		vector<Mat> res;

		double ratio = 1. * UCHAR_MAX / MAX_VALID_DEPTH;

		size_t humSz = humVec.size();
		for (size_t i = 0; i < humSz; i++){
			HumanObj hum = humVec[i];
			Mat humMask = hum.getCurrMask();
			Mat dmat_hum_masked = dmat.clone();
			dmat_hum_masked.setTo(0, humMask == 0);

			Mat tdv_i = dmat2tdview_core(dmat_hum_masked, ratio);
			tdv_i.convertTo(tdv_i, CV_8U);
			res.push_back(tdv_i);
		}

		return res;
	}//humVec2tdviewVec

	//vec-heap ��id��Դ��
	static vector<int> idResPool;
	std::greater<int> fnGt = std::greater<int>();

	//Ҫ����һ�� HumanObj ʱ�� id�Ż���Դ��
	void pushPool(int id){
		static bool isFirstTime = true;

		if (isFirstTime){
			make_heap(idResPool.begin(), idResPool.end(), fnGt);
		}

		idResPool.push_back(id);
		push_heap(idResPool.begin(), idResPool.end(), fnGt);

	}//pushPool

	//������С����id�����ޣ�����-1
	int getIdFromPool(){
		if (idResPool.empty())
			return -1;

		int resId = idResPool.front();
		pop_heap(idResPool.begin(), idResPool.end(), fnGt);
		idResPool.pop_back();

		return resId;
	}

	void getHumanObjVec(Mat &dmat, vector<Mat> fgMasks, vector<HumanObj> &outHumVec){
		//ȫ�ֶ��У�
		//static vector<HumanObj> outHumVec;

		size_t fgMskSize = fgMasks.size(),
			humVecSize = outHumVec.size();

		Mat dmatClone = dmat.clone();

		//����δ��⵽���ˣ��ҵ�֡ fgMasks �����ݣ���ʵ����Ҫ��:
		if (humVecSize == 0 && fgMskSize > 0){
			cout << "+++++++++++++++humVecSize == 0" << endl;
			for (size_t i = 0; i < fgMskSize; i++){
				//outHumVec.push_back({ dmatClone, fgMasks[i] });

				int id = getIdFromPool();
				if (id < 0)
					id = outHumVec.size() + 1;
				HumanObj humObj(dmatClone, fgMasks[i], id);
				outHumVec.push_back(humObj);
			}
		}
		//���Ѽ�⵽ HumanObj, �ҵ�֡ fgMasks �����������ݿɸ��£�
		else if (humVecSize > 0){//&& fgMskSize > 0){

			vector<bool> fgMsksUsedFlagVec(fgMskSize);

			vector<HumanObj>::iterator it = outHumVec.begin();
			while (it != outHumVec.end()){
				bool isUpdated = it->updateDmatAndMask(dmatClone, fgMasks, fgMsksUsedFlagVec);
				if (isUpdated)
					it++;
				else{
					//�ͷ�Ψһid����Դ��
					pushPool(it->getHumId());

					it = outHumVec.erase(it);
					cout << "------------------------------humVec.erase" << endl;
				}
			}//while

			//��� fgMsksUsedFlagVec�� ���������壺
			for (size_t i = 0; i < fgMskSize; i++){
				if (fgMsksUsedFlagVec[i]==true)
					continue;

// 				outHumVec.push_back(HumanObj(dmatClone, fgMasks[i]));

				int id = getIdFromPool();
				if (id < 0)
					id = outHumVec.size() + 1;
				HumanObj humObj(dmatClone, fgMasks[i], id);
				outHumVec.push_back(humObj);
			}
		}

		//return outHumVec;
	}//getHumanObjVec


	//@brief ���뵥֡ԭʼ���ͼ�� �õ����� vec-mat-mask. process/run-a-frame
	vector<Mat> getFgMaskVec(Mat &dmat, int fid, bool debugDraw /*= false*/){
#define ZC_WRITE 0

		Mat dm_show
			, dmat8u
			, dmat8uc3
			;
		dmat.convertTo(dm_show, CV_8UC1, 1.*UCHAR_MAX / MAX_VALID_DEPTH);
		dmat8u = dm_show;

		//@deprecated û�õ�
		static bool isFirstTime = true;
		static int oldFid = fid;
		static int frameCnt = -1;//fake frameCnt!!
		frameCnt++;


//#if ZC_CLEAN //��Ӧ CAPG_SKEL_VERSION_0_9, ������Ҫ�ĸɾ��Ĵ���
		//---------------1. Ԥ����
		//���룺��ʼ��prevDmat
		zc::initPrevDmat(dmat);

		clock_t begttotal = clock();
		static vector<Mat> prevMaskVec;
		//static vector<HumanObj> humVec;

		//����Ƶ֡ѭ�����ص���ʼ��
		if (fid <= oldFid){
			prevMaskVec.clear();
			//humVec.clear();
			isFirstTime = true;
		}

#if 0	//����һ�ײ�֣�ͻ��
		static int tmpFcnt = -1;
		tmpFcnt++;
		if (tmpFcnt > 10){
			Mat prevMaskWhole = maskVec2mask(dmat.size(), prevMaskVec);
			Mat prevDmat = getPrevDmat();
			Mat maxDmat = getMaxDmat(dmat);
			Mat currDeeper = dmat - maxDmat > 0;
			imshow("currDeeper", currDeeper);
			Mat diffMat = dmat - prevDmat > 50;
			imshow("diffMat", diffMat);

			static Mat prevDiffFgMsk = Mat::zeros(dmat.size(), CV_8UC1);
			double myVarThresh = 0.035;
			Mat diffFgMsk = (
				(currDeeper == 0 & (dmat < maxDmat * (1 - myVarThresh))) //*0.3
				//+ ((currDeeper + prevMaskWhole) & (dmat - maxDmat < 55)//maxDmat * myVarThresh)
				+ (currDeeper & (dmat < maxDmat + 55/** (1 + myVarThresh)*/))//* 0.5
				+ (prevDiffFgMsk & (abs(dmat - maxDmat) < 10/*maxDmat * myVarThresh*/))
				);
			Mat prevNoMoveMask = (prevDiffFgMsk & (abs(dmat - maxDmat) < 10/*maxDmat * myVarThresh*/));
			imshow("prevNoMoveMask", prevNoMoveMask);

			prevDiffFgMsk = diffFgMsk.clone();
			imshow("diffFgMsk", diffFgMsk);
		}

#endif

		clock_t begt = clock();
#if 0	//A.ȥ������ & ���� & MOG2(or else?)����������
		Mat bgMsk = zc::fetchBgMskUseWallAndHeight(dmat);
		Mat flrApartMask = zc::fetchFloorApartMask(dmat, false);
		Mat no_flr_wall_mask = flrApartMask & (bgMsk == 0);

		Mat maskedDmat = dmat.clone();
		maskedDmat.setTo(0, no_flr_wall_mask == 0);
		//ֻȥ���棬��ȥ��ǽ��
		//maskedDmat.setTo(0, flrApartMask == 0);
		cout << "aaa.maskedDmat.ts: " << clock() - begt << endl;
		if (debugDraw){
			Mat maskedDmat_show;
			normalize(maskedDmat, maskedDmat_show, 0, UCHAR_MAX, NORM_MINMAX, CV_8UC1);
			imshow("atmp-maskedDmat", maskedDmat_show);
		}
#endif	//A.ȥ������ & ���� & MOG2(or else?)��������

		Mat fgMskMotion;

		//Ԥ���������������(MOG2):
		int noMoveThresh = 100;
		int history = 20;// 100;
		//int history = 10;
		double varThresh = 0.3;// 1;
		double learnRate = -1;
		bool detectShadows = false;

		//static Ptr<BackgroundSubtractorMOG2> pMog2;// = createBackgroundSubtractorMOG2(history, varThresh, detectShadows);

		//����Ƶ֡ѭ�����ص���ʼ��
		if (fid <= oldFid){
#ifdef CV_VERSION_EPOCH //if opencv2.x
			pBgSub = new BackgroundSubtractorMOG2(history, varThresh, detectShadows);
#elif CV_VERSION_MAJOR >= 3 //if opencv3
#if USE_MOG2	//---------------MOG2
			clock_t begt = clock();
			//pMog2->clear(); //ûЧ��
			pBgSub = createBackgroundSubtractorMOG2(history, varThresh, detectShadows);

			cout << "createBackgroundSubtractorMOG2.ts: " << clock() - begt << endl;
#elif 1	//---------------KNN
			history = 20;
			double dist2Threshold = 0.8;
			pBgSub = createBackgroundSubtractorKNN(history, dist2Threshold, detectShadows);
#elif 1	//---------------GMG, 2012
			
#endif	//MOG2, KNN

#endif //CV_VERSION_EPOCH
		}//if (fid <= oldFid)

#ifdef CV_VERSION_EPOCH //if opencv2.x
		//(*pMog2)(dmat8u, fgMskMotion);
		(*pBgSub)(dmat8u, fgMskMotion, learnRate);
#elif CV_VERSION_MAJOR >= 3 //if opencv3
		//MOG, KNN ͨ�ã�
		pBgSub->apply(dmat8u, fgMskMotion, learnRate);
#endif //CV_VERSION_EPOCH

		//ǰʮ֡��ȫ��mat
		if (fid < 10)
			fgMskMotion = Mat::zeros(dmat.size(), CV_8UC1);

		//��ע�⡿һ���۳���Ч����
		fgMskMotion.setTo(0, dmat == 0);

		if (debugDraw){
			imshow("fgMskMog2", fgMskMotion);
		}

#if 0	//smooth-mask. @code: D:\opencv300\sources\samples\cpp\bgfg_segm.cpp
		//��Ҫ��ȷ���������ͻ�ģ��
		int ksize = 11;
		double sigma = 3.5;
		GaussianBlur(fgMskMog2, fgMskMog2, Size(ksize, ksize), sigma);
		threshold(fgMskMog2, fgMskMog2, 10, 255, THRESH_BINARY);
		if (debugDraw){
			imshow("fgMskMog2-smooth", fgMskMog2);
		}
#endif


		//B.������ǰ������ʼ�����˴��õ�bbox������
		begt = clock();

		vector<Mat> fgMskVec;
		//fgMskVec = zc::findFgMasksUseWallAndHeight(dmat, debugDraw);
		Mat tmp;
		// 		fgMskVec = zc::findFgMasksUseBbox(maskedDmat, debugDraw, tmp);

#if 0 //XY-XZ-bbox �����ж�
		int rgThresh = 55;
		vector<vector<Point>> sdBboxVov = zc::seedUseBboxXyXz(maskedDmat, debugDraw, tmp);
		fgMskVec = zc::simpleRegionGrow(maskedDmat, sdBboxVov, rgThresh, flrApartMask, false);
#elif 0 //ͷ�������ж�
		int rgThresh = 55;
		vector<vector<Point>> sdHeadBodyVov = zc::seedUseHeadAndBodyCont(dmat, debugDraw, tmp);
		fgMskVec = zc::simpleRegionGrow(maskedDmat, sdHeadBodyVov, rgThresh, flrApartMask, false);
#elif 01 //MOG2 �˶���ⷽ���� ȥ��ʼ֡����ʴ����������
		//"fid>1"�ж���ʽ�����������ϲ���:
		//Mat sdMoveMat = isFirstTime ? Mat::zeros(dmat.size(), CV_8UC1) : fgMskMotion.clone();
		//Mat sdMoveMat = fid < 10 ? Mat::zeros(dmat.size(), CV_8UC1) : fgMskMotion.clone();
		Mat sdMoveMat = fgMskMotion.clone();
		int erodeRadius = 13;
		erode(sdMoveMat, sdMoveMat, zc::getMorphKrnl(erodeRadius));
		if (debugDraw){
			imshow("sdMoveMat", sdMoveMat);
		}

		isFirstTime = false; //������������ﻹ��Ҫ��

		int rgThresh = gRgThresh;
		bool getMultiMasks = true;
		//fgMskVec = zc::simpleRegionGrow(maskedDmat, sdMoveMat, rgThresh, flrApartMask, getMultiMasks);
		fgMskVec = zc::simpleRegionGrow(dmat, sdMoveMat, rgThresh, fgMskMotion, getMultiMasks);

#elif 1	//�����ȵ�maskֱ������ʼ���� 2015��7��29��00:26:02
		Mat sdMoveMat = sgf::calcPotentialMask(dmat, getPrevDmat());
		//�����Ῠ������δ�����
// 		int erodeRadius = 13;
// 		erode(sdMoveMat, sdMoveMat, zc::getMorphKrnl(erodeRadius));
// 
// 		rgThresh = 55;
// 		bool getMultiMasks = true;
// 		fgMskVec = zc::simpleRegionGrow(dmat, sdMoveMat, rgThresh, fgMskMotion, getMultiMasks);

		//ֱ����mask����������
		//fgMskVec.push_back(sdMoveMat);
		if (countNonZero(sdMoveMat) > 0)
			fgMskVec = separateMasksXYview(dmat, vector<Mat>({ sdMoveMat }));

#endif //N�ֳ�����������
		fgMskVec = zc::bboxBatchFilter(dmat, fgMskVec);

		if (DBG_STD_COUT) cout << "bbb.findFgMasksUseBbox.ts: " << clock() - begt << endl;
		if (debugDraw){
			Mat btmp = zc::getHumansMask(fgMskVec, dmat.size());
			string txt = "fgMskVec.size: " + to_string((long long)fgMskVec.size());
			putText(btmp, txt, Point( 0, 50 ), FONT_HERSHEY_PLAIN, 1, 255);
			imshow("btmp-bbox-init", btmp);
		}

		if (debugDraw){
			Mat prevMaskVec2msk = zc::getHumansMask(prevMaskVec, dmat.size());
			string txt = "prevMaskVec.size: " + to_string((long long)prevMaskVec.size());
			putText(prevMaskVec2msk, txt, Point( 0, 50 ), FONT_HERSHEY_PLAIN, 1, 255);
			imshow("prevMaskVec2msk", prevMaskVec2msk);
		}

#if 0	//B2. ���� maskMoveAndNoMove��
#if 0	//v1�汾������
		{
			Mat dmat8u;
			dmat.convertTo(dmat8u, CV_8U, 1.*UCHAR_MAX / MAX_VALID_DEPTH);

			int noMoveThresh = 100;
			int history = 100;
			double varThresh = 1;
			double learnRate = -0.005;
			//Mat tmp;
			// 			Mat testMsk = zc::maskMoveAndNoMove(dmat, prevMaskVec, noMoveThresh, history, varThresh, learnRate, debugDraw, tmp);

			bool detectShadows = false;
			static Ptr<BackgroundSubtractorMOG2> pMog2 = createBackgroundSubtractorMOG2(history, varThresh, detectShadows);
			Mat fgMskMog2;
			pMog2->apply(dmat8u, fgMskMog2);

			imshow("test1", fgMskMog2);
		}

		int noMoveThresh = 100;
		int history = 100;
		double varThresh = 1;
		double learnRate = -0.005;
		//Mat tmp;
		Mat testMsk = zc::maskMoveAndNoMove(dmat, prevMaskVec, noMoveThresh, history, varThresh, learnRate, debugDraw, tmp);
#elif 1	//v2�汾 overload
		Mat tmp_mman;
		Mat testMsk = zc::maskMoveAndNoMove(dmat, fgMskMotion, prevMaskVec, noMoveThresh, debugDraw, tmp_mman);
		imshow("maskMoveAndNoMove", testMsk);
		if (!tmp_mman.empty())
			imshow("maskMoveAndNoMove-debug", tmp_mman);
		//imwrite("maskMoveAndNoMove_" + std::to_string((long long)fid) + ".jpg", tmp);

#endif
		Mat testMOG2mat = zc::testBgFgMOG2(dmat, history, varThresh, learnRate);
		imshow("testMOG2mat", testMOG2mat);
#endif	//B2. ���� maskMoveAndNoMove

#if 1	//post-B: ������ζ�, ������, �����ͻ������Ҳ�ᵼ������ good��
		//Mat fgMskWhole = maskVec2mask(dmat.size(), fgMskVec);
		Mat prevFgMskWhole = maskVec2mask(dmat.size(), prevMaskVec);

		//����: !(ǰ�� | ��Чֵ) 
		//Mat validBgMsk = (fgMskWhole == 0) & (dmat != 0);
		Mat validBgMsk = (prevFgMskWhole == 0) & (dmat != 0);

		//bg���˶����(e.g., MOG2)����:
		Mat bgMoveMask = fgMskMotion & validBgMsk;

		double bgMoveRatioLow = countNonZero(bgMoveMask)*1. / countNonZero(validBgMsk);

		//---------------2015��8��2��22:45:34
		//���� bgMoveRatioLow > bgMoveRatioThresh ����, �޷���������ζ� �� ������ק������(������MOG�������)����, �������� dilate < High �ж�:
		//�����ǲ��á�����
		Mat bgMoveMaskDilate;
		dilate(bgMoveMask, bgMoveMaskDilate, getMorphKrnl(2));
		double bgMoveRatioHigh = countNonZero(bgMoveMaskDilate)*1. / countNonZero(validBgMsk);

		//---------------2015��8��2��23:22:26 ����"����߶����" �ж�, �����ֵ���:
		double bgMoveRatioWscale = 1. * maskedCvSum(dmat, bgMoveMask)[0] / maskedCvSum(dmat, validBgMsk)[0];
		
		if (debugDraw){
			//���Ը�׼ȷ�� ^2 ���, bgMoveRatioWscale ��ʵ�� ^1, ���������������:
			//���� bgMoveRatioWscale�� _real ��� <0.1, ˵�� ^2 ����Ҫ. 2015��8��3��02:38:53
			clock_t begt = clock();
#if 01	//ֱ�� 16u, 0.1ms
			Mat wsAreaDmat = dmat / XTION_FOCAL_XY;
#elif 1	//��ת�� 32f, 0.4ms
			Mat wsAreaDmat;
			dmat.convertTo(wsAreaDmat, CV_32FC1);
			wsAreaDmat /= XTION_FOCAL_XY;
#endif	//16u vs. 32f
			wsAreaDmat = wsAreaDmat.mul(wsAreaDmat);
			double bgMoveRatioWscale_real = 1. * maskedCvSum(wsAreaDmat, bgMoveMask)[0] / maskedCvSum(wsAreaDmat, validBgMsk)[0];

			static int fcnt = 0;
			static float sumt = 0;
			fcnt++;
			sumt += clock() - begt;
			if (DBG_STD_COUT) 
				cout << "bgMoveRatioWscale_real.ts: " << sumt / fcnt << endl;	//0.1ms

			Mat validBgMsk_dbg = validBgMsk.clone();
			cvtColor(validBgMsk_dbg, validBgMsk_dbg, COLOR_GRAY2BGR);
			validBgMsk_dbg.setTo(Scalar(0, 255, 0), bgMoveMask);
			validBgMsk_dbg.setTo(Scalar(255, 0, 0), bgMoveMaskDilate - bgMoveMask);

			putText(validBgMsk_dbg, "ratioLow: " + to_string((long double)bgMoveRatioLow), Point(0, 30), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 2);
			putText(validBgMsk_dbg, "ratioHigh: " + to_string((long double)bgMoveRatioHigh), Point(0, 60), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 2);
			putText(validBgMsk_dbg, "ratioWs: " + to_string((long double)bgMoveRatioWscale), Point(0, 90), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 2);
			putText(validBgMsk_dbg, "ratioWs_R: " + to_string((long double)bgMoveRatioWscale_real), Point(0, 120), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 2);

			imshow("validBgMsk_dbg", validBgMsk_dbg);
		}

		double bgMoveRatioThreshLow = 0.35;
		double bgMoveRatioThreshHigh = 0.85;
		double bgMoveRatioWscaleThresh = 0.5;
		if (01
			//&& bgMoveRatioLow > bgMoveRatioThreshLow 
			//&& bgMoveRatioHigh > bgMoveRatioThreshHigh
			&& bgMoveRatioWscale > bgMoveRatioWscaleThresh
			){
			//��ʱ fgMskVec Ϊ (bbb) ������, ��������, Ҫ����, ��, ǰ���⵽���Ը���, �����Ĳ��ϲ�:
			//fgMskVec.clear(); 
			//����ո�Ϊ���ò����㣺
			fgMskVec = seedNoMove(dmat, prevMaskVec, noMoveThresh);

			//MOG2-mask ����:
			fgMskMotion = Mat::zeros(dmat.size(), CV_8UC1);

			//������mat, mask ������ã�
			//sgf::releasePotentialMask();
			//��Ҫ release, ��Ϊ�����÷� fgMskVec ����:
			sgf::setPotentialMask(dmat, validBgMsk);
		}


#endif	//post-B: ������ζ�, ������, �����ͻ������Ҳ�ᵼ������ good��

		//C.��������٣�ʹǰ����ȫ���ȶ���
		begt = clock();
		fgMskVec = zc::trackingNoMove(dmat, prevMaskVec, fgMskVec, noMoveThresh, fgMskMotion, debugDraw);
		if(DBG_STD_COUT) cout << "ccc.trackingNoMove.ts: " << clock() - begt << endl;
		if (debugDraw){
			Mat ctmp = zc::getHumansMask(fgMskVec, dmat.size());
			string txt = "fgMskVec.size: " + to_string((long long)fgMskVec.size());
			putText(ctmp, txt, Point( 0, 50 ), FONT_HERSHEY_PLAIN, 1, 255);
			imshow("ctmp-trackingNoMove", ctmp);
		}

		//����ַ���һ,������Ҫ����:
#if 01 //D.������ͻȻ����ɼ����ֵ�ǰ���������롢���˴���
		begt = clock();

		//---------------����sep-xy��Ϊ�ԱȲ��ԣ�
		//fgMskVec = zc::separateMasksXYview(dmat, fgMskVec, debugDraw);

		//---------------Ŀǰʹ��sep-xz��
		int armLength = 800; //����ǰ����ֱ��������
		fgMskVec = zc::separateMasksXZview(dmat, fgMskVec, armLength, debugDraw);
		if (DBG_STD_COUT) 
			cout << "ddd.separateMasksXZview.ts: " << clock() - begt << endl;

		static int tSumt = 0;
		tSumt += (clock() - begttotal);
		if (DBG_STD_COUT) 
			cout << "find+tracking.rate: " << 1.*tSumt / (frameCnt + 1) << ", " << frameCnt << endl;

		if (debugDraw){
			Mat dtmp = zc::getHumansMask(fgMskVec, dmat.size());
			string txt = "fgMskVec.size: " + to_string((long long)fgMskVec.size());
			putText(dtmp, txt, Point( 0, 50 ), FONT_HERSHEY_PLAIN, 1, 255);

			imshow("dtmp-separateMasks", dtmp);
		}
#endif	//D.����

#if 10 && !SOLUTION_1	//��������, ������������� potMsk, (ע����!SOLUTION_1)
		if (fgMskVec.size() > 0){
#if 0		//֮ǰʵ��
#if 0		//�����߼�v1: region2reset = _potFgMask - newFgMask; 
			Mat currFgMskWhole = maskVec2mask(dmat.size(), fgMskVec);
			sgf::setPotentialMask(dmat, currFgMskWhole, debugDraw);
#elif 01	//�߼�v2: region2reset = prevFgMskWhole - currFgMskWhole;
			Mat currFgMskWhole = maskVec2mask(dmat.size(), fgMskVec);
			Mat prevFgMskWhole = maskVec2mask(dmat.size(), prevMaskVec);
			sgf::setPotentialMask(dmat, currFgMskWhole, prevFgMskWhole, debugDraw);
#endif
#elif 1		//�ع� setPotentialMask, ������֮ǰ���߼�v2
			Mat currFgMskWhole = maskVec2mask(dmat.size(), fgMskVec);
			Mat prevFgMskWhole = maskVec2mask(dmat.size(), prevMaskVec);
			Mat region2reset = prevFgMskWhole - currFgMskWhole;
			region2reset -= (dmat == 0);

			if (debugDraw)
				imshow("region2reset", region2reset);

			//�����ô��������, e.g.:
			//1. �뿪�˵��α�, �ϴ��ֲ���������ô�� ��
			//2. ǰ����С��Ƭ��������
			region2reset = zc::largeContPassFilter(region2reset, zc::CONT_AREA, 20 * 40);
			if (debugDraw)
				imshow("region2reset-large", region2reset);
			sgf::setPotentialMask(dmat, region2reset);
#endif

		}
		if (debugDraw){
			imshow("sgf::_fgMask", sgf::_potFgMask);
			//imshow("sgf::_max_dmat", sgf::_max_dmat);
			Mat maxDmat8u;
			sgf::_max_dmat.convertTo(maxDmat8u, CV_8UC1, 1.*UCHAR_MAX / MAX_VALID_DEPTH);
			imshow("maxDmat8u", maxDmat8u);

			imshow("sgf::_max_dmat_mask", sgf::_max_dmat_mask);
		}

#endif	//�������� potMsk


#if SOLUTION_1	//E. ��������ɡ����ֺ�������
#if M1_E1 //�ٺ���V�μ��ָ
		fgMskVec = zc::separateMasksContValley(dmat, fgMskVec, debugDraw);
#else	//�ں���ͷ�����ӵ� + �˶�ֱ��ͼ��ֵ���������ж���
		fgMskVec = zc::separateMasksMovingHead(dmat, fgMskVec, fgMskMotion, debugDraw);
#endif

		if (debugDraw){
			Mat etmp = zc::getHumansMask(fgMskVec, dmat.size());
			string txt = "fgMskVec.size: " + to_string((long long)fgMskVec.size());
			putText(etmp, txt, Point(0, 50), FONT_HERSHEY_PLAIN, 1, 255);
			imshow("etmp-separateMasks-sgf", etmp);
		}
#endif	//��������ɡ����ֺ�������

		oldFid = fid;
		prevMaskVec = fgMskVec;
		//���룺 ���� prevDmat
		zc::setPrevDmat(dmat);
//#endif //ZC_CLEAN

		return fgMskVec;
	}//getFgMaskVec


	void debugDrawHumVec(const Mat &dmat, const vector<Mat> &fgMskVec, vector<HumanObj> &humVec, int fid /*= -1*/, bool debugWrite /*= false*/){
		Mat humMsk = zc::getHumansMask(fgMskVec, dmat.size());
		// 			QtFont font = fontQt("Times");
		// 			cv::addText(humMsk, "some-text", { 55, 55 ), font);
		putText(humMsk, "fid: " + to_string((long long)fid), Point(0, 30), FONT_HERSHEY_PLAIN, 1, 255);
		imshow("humMsk", humMsk);
		if (debugWrite)
			imwrite("humMsk_" + std::to_string((long long)fid) + ".jpg", humMsk);

		Mat humMsk��ɫ = zc::getHumansMask(dmat, humVec);
		putText(humMsk��ɫ, "fid: " + to_string((long long)fid), Point(0, 30),
			FONT_HERSHEY_PLAIN, 1, Scalar(255, 255, 255));

		putText(humMsk��ɫ, "humVec.size: " + to_string((long long)humVec.size()), Point(0, 50),
			FONT_HERSHEY_PLAIN, 1, Scalar(255, 255, 255));

		zc::drawSkeletons(humMsk��ɫ, humVec, -1);

		imshow("humMsk-color", humMsk��ɫ);
		if (debugWrite)
			imwrite("humMsk-color_" + std::to_string((long long)fid) + ".jpg", humMsk��ɫ);
	}//debugDrawHumVec

	vector<Mat> bboxBatchFilter(Mat dmat, const vector<Mat> &origMasks){
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
// 			Rect bbox = zc::boundingRect(contours[0]);
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
			if (bboxIsHuman(dmat.size(), mski))
			{
				res.push_back(mski);
			}
		}
		return res;
	}//bboxBatchFilter

	bool fgMskIsHuman(Mat dmat, Mat mask){
		double dmin, dmax;
		minMaxLoc(dmat, &dmin, &dmax, nullptr, nullptr, mask);

// 		vector<vector<Point> > contours;
// 		findContours(mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
// 
// 		CV_Assert(contours.size() > 0);
// // 		return fgMskIsHuman(dmat, contours[0]);
// 
// 		Rect bbox = zc::boundingRect(contours[0]);
		Rect bbox = zc::boundingRect(mask);

// 		//���� distMap2contours ��bbox �ж����ˣ�
// 		//1. ����̫��; 2. bbox�߶Ȳ���̫С; 3. bbox ���ز��ܸ��ڰ�������Ϊ�˽Ų�λ�ýϵ�
// 		bool notTooThick = dmax - dmin < thickLimit,
// 			pxHeightEnough = bbox.height >80,
// 			feetLowEnough = bbox.br().y > dmat.rows / 2;
// 
// 		//return notTooThick && pxHeightEnough && feetLowEnough;
// 		//2015��6��24��16:09:26�� 
// 		//��ʱ���� notTooThick �ж��� ������ܵ��²��ȶ�
// 		return /*notTooThick && */pxHeightEnough && feetLowEnough;

		return bboxIsHuman(dmat.size(), bbox);
	}//fgMskIsHuman

// 	bool fgMskIsHuman(Mat dmat, vector<Point> cont){
// 		Rect bbox = zc::boundingRect(cont);
// 
// 		//���� distMap2contours ��bbox �ж����ˣ�
// 		//1. ����̫��; 2. bbox�߶Ȳ���̫С; 3. bbox ���ز��ܸ��ڰ�������Ϊ�˽Ų�λ�ýϵ�
// 		bool notTooThick = dmax - dmin < 1500,
// 			pxHeightEnough = bbox.height >80,
// 			feetLowEnough = bbox.br().y > dmat.rows / 2;
// 
// 		return notTooThick && pxHeightEnough && feetLowEnough;
// 
// 	}//fgMskIsHuman

	bool contIsHuman(Size matSize, vector<Point> cont){
		Rect bbox = zc::boundingRect(cont);
		return bboxIsHuman(matSize, bbox);
	}//contIsHuman


	bool bboxIsHuman(Size matSize, Rect bbox){
		bool pxHeightEnough = bbox.height > 80,
			narrowEnough = bbox.height*1. / bbox.width > MIN_VALID_HW_RATIO,
			feetLowEnough = bbox.br().y > matSize.height / 2;

		//return pxHeightEnough && narrowEnough && feetLowEnough;
		return pxHeightEnough && feetLowEnough;
	}//bboxIsHuman

	bool bboxIsHuman(Size matSize, const Mat mask){
		Rect bbox = zc::boundingRect(mask);
		return bboxIsHuman(matSize, bbox);
	}//bboxIsHuman


	bool bboxIsHumanWscale(const Mat &dmat, const Mat mask){
		Mat hhMap = calcHeightMap1(dmat);
		Mat wwMap = calcWidthMap(dmat, dmat.cols / 2);

		Rect bbox = zc::boundingRect(mask);
		//����߶�, �߶�����45cm, �������20cm:
		//�߶����±�С���ϼ��£�
		//bool isHighEnough = hhMap.at<int>(bbox.y) - hhMap.at<int>(bbox.br().y) > 450;
		//��������С���Ҽ���
		//bool isWideEnough = wwMap.at<int>(bbox.br().x) - wwMap.at<int>(bbox.x) > 200;

		//bbox �Ķ����ж����ԣ��ĳ� minMaxLoc:
		double hMin, hMax;
		minMaxLoc(hhMap, &hMin, &hMax, NULL, NULL, mask);
		bool isHighEnough = hMax - hMin > 450;

		double wMin, wMax;
		minMaxLoc(wwMap, &wMin, &wMax, NULL, NULL, mask);
		bool isWideEnough = wMax - wMin > 200;

		return isHighEnough && isWideEnough && bboxIsHuman(dmat.size(), bbox);
	}//bboxIsHumanWscale


	cv::Mat getMorphKrnl(int radius /*= 1*/, int shape /*= MORPH_RECT*/)
	{
		return getStructuringElement(shape, Size(2 * radius + 1, 2 * radius + 1), Point(radius, radius));
	}

	static Mat _prevDmat;
	void setPrevDmat(Mat currDmat){
		_prevDmat = currDmat.clone();
	}//setPrevDmat
	
	void initPrevDmat(Mat currDmat){
		if (_prevDmat.empty())
			_prevDmat = Mat::zeros(currDmat.size(), currDmat.type());
	}//initPrevDmat

	Mat getPrevDmat(){
		return _prevDmat;
	}//getPrevDmat


#ifdef CV_VERSION_EPOCH
	const string featurePath = "./feature";
	static BPRecognizer *bpr = nullptr;
	CapgSkeleton calcSkeleton( const Mat &dmat, const Mat &fgMsk ){
		Mat dm32s;
		dmat.convertTo(dm32s, CV_32SC1);
		//����������ֵ������ǰ��������������ɫ��
		dm32s.setTo(INT_MAX, fgMsk==0);

		IplImage depthImg = dm32s;
		bool useDense = false,
			useErode = false, 
			usePre = true;

		if(bpr == nullptr)
			bpr = getBprAndLoadFeature(featurePath);

		CapgSkeleton sklt;
		Mat labelMat = bpr->predict(&depthImg, nullptr, useDense, usePre);
		IplImage cLabelMat = labelMat;

		useErode = false;
		usePre = false;
		bpr->mergeJoint(&cLabelMat, &depthImg, sklt, useErode, usePre);

		return sklt;
	}//calcSkeleton
#endif //CV_VERSION_EPOCH

#pragma region //---------------HumanObj ��Ա�����ǣ�

	HumanObj::HumanObj( const Mat &dmat_, Mat currMask_, int humId ) 
		:_dmat(dmat_), _humId(humId)
		//,_currMask(currMask_)
	{
		//_currCenter = getContMassCenter(_currMask);
		setCurrMask(currMask_);

		if (_prevMask.empty()){
			// 				_currMask.copyTo(_prevMask);
			// 				_prevCenter = _currCenter;
			setPrevMask(_currMask);
		}

		uint64 seed = (clock() % (1 << 16)) << 16;
		RNG rng(seed);
		for (int i = 0; i < 3; i++)
			_humColor[i] = rng.uniform(UCHAR_MAX/2, UCHAR_MAX);
		//_humColor[3] = 100; //��͸��
	}//HumanObj-ctor

#ifdef CV_VERSION_EPOCH
	void HumanObj::calcSkeleton(){
		_sklt = zc::calcSkeleton(_dmat, _currMask);
	}//calcSkeleton
#endif //CV_VERSION_EPOCH

	CapgSkeleton HumanObj::getSkeleton(){
		return _sklt;
	}//getSkeleton

	bool HumanObj::updateDmatAndMask( const Mat &dmat, const vector<Mat> &fgMaskVec, vector<bool> &mskUsedFlags ){
		//���ͼ���£�
		_dmat = dmat;

		//����ǰ��֡��ɫ�����󽻣�
		size_t fgMskVecSz = fgMaskVec.size();
		bool foundNewMask = false;
		for (size_t i = 0; i < fgMskVecSz; i++){
			Mat fgMsk = fgMaskVec[i];
			//�󽻣�
			Mat currNewIntersect = _currMask & fgMsk;
			int intersectArea = countNonZero(currNewIntersect != 0),
				fgMskArea = countNonZero(fgMsk != 0);
			double percent = 0.5;

			//2015��6��27��22:10:18

			if (mskUsedFlags[i] == false
				//���Ĳ��ɿ�
				//&& fgMsk.at<uchar>(_currCenter) == UCHAR_MAX){
				//mask����ռ�ȣ�Ҳ���ɿ�
				//&& (intersectArea > _currMaskArea * percent 
				//	|| intersectArea > fgMskArea * percent)

				//2015��6��27��22:08:20�� mask����������Ȳ��ֵ
				//��ע����intersectArea>0��Ҫ������ mean=0��������
				&& intersectArea > 0 && mean(abs(_dmat - dmat), currNewIntersect)[0] < 55
				){

					// 					_prevMask = _currMask;
					// 					_prevCenter = _currCenter;
					// 
					// 					_currMask = fgMsk;
					// 					_currCenter = getContMassCenter(_currMask);

					setPrevMask(_currMask);
					setCurrMask(fgMsk);

					foundNewMask = true;
					//���ĳmask�ѱ��ù���
					mskUsedFlags[i] = true;

					break;
			}
		}

		//���� fgMasks_�ص���̫��δ�ҵ�������Ϊ������
		if (!foundNewMask){
			return false;
		}

		return true;
	}//updateDmatAndMask

	cv::Mat HumanObj::getCurrMask(){
		return _currMask;
	}//getCurrMask

	void HumanObj::setCurrMask( Mat newMask ){
		_currMask = newMask;
		_currCenter = getContMassCenter(_currMask);
		_currMaskArea = countNonZero(_currMask != 0);

#ifdef CV_VERSION_EPOCH
		//����˳�����Ǽܣ�
		this->calcSkeleton();
#endif //CV_VERSION_EPOCH

	}//setCurrMask

	void HumanObj::setPrevMask( Mat newMask ){
		_prevMask = newMask;
		_prevCenter = getContMassCenter(_prevMask);
	}//setPrevMask

	cv::Scalar HumanObj::getColor(){
		return _humColor;
	}//getColor

	int HumanObj::getHumId(){
		return _humId;
	}//getHumId

#pragma endregion //---------------HumanObj ��Ա������
}//zc

//�� opencv300 ���� boundingRect 
namespace zc{
	//cv300 zc::boundingRect �ɽ���mask-mat
	//D:\opencv300\sources\modules\imgproc\src\shapedescr.cpp L479
#define  CV_TOGGLE_FLT(x) ((x)^((int)(x) < 0 ? 0x7fffffff : 0))


	// Calculates bounding rectagnle of a point set or retrieves already calculated
	static Rect pointSetBoundingRect(const Mat& points)
	{
		int npoints = points.checkVector(2);
		int depth = points.depth();
		CV_Assert(npoints >= 0 && (depth == CV_32F || depth == CV_32S));

		int  xmin = 0, ymin = 0, xmax = -1, ymax = -1, i;
		bool is_float = depth == CV_32F;

		if (npoints == 0)
			return Rect();

		const Point* pts = points.ptr<Point>();
		Point pt = pts[0];

#if CV_SSE4_2
		if (cv::checkHardwareSupport(CV_CPU_SSE4_2))
		{
			if (!is_float)
			{
				__m128i minval, maxval;
				minval = maxval = _mm_loadl_epi64((const __m128i*)(&pt)); //min[0]=pt.x, min[1]=pt.y

				for (i = 1; i < npoints; i++)
				{
					__m128i ptXY = _mm_loadl_epi64((const __m128i*)&pts[i]);
					minval = _mm_min_epi32(ptXY, minval);
					maxval = _mm_max_epi32(ptXY, maxval);
				}
				xmin = _mm_cvtsi128_si32(minval);
				ymin = _mm_cvtsi128_si32(_mm_srli_si128(minval, 4));
				xmax = _mm_cvtsi128_si32(maxval);
				ymax = _mm_cvtsi128_si32(_mm_srli_si128(maxval, 4));
			}
			else
			{
				__m128 minvalf, maxvalf, z = _mm_setzero_ps(), ptXY = _mm_setzero_ps();
				minvalf = maxvalf = _mm_loadl_pi(z, (const __m64*)(&pt));

				for (i = 1; i < npoints; i++)
				{
					ptXY = _mm_loadl_pi(ptXY, (const __m64*)&pts[i]);

					minvalf = _mm_min_ps(minvalf, ptXY);
					maxvalf = _mm_max_ps(maxvalf, ptXY);
				}

				float xyminf[2], xymaxf[2];
				_mm_storel_pi((__m64*)xyminf, minvalf);
				_mm_storel_pi((__m64*)xymaxf, maxvalf);
				xmin = cvFloor(xyminf[0]);
				ymin = cvFloor(xyminf[1]);
				xmax = cvFloor(xymaxf[0]);
				ymax = cvFloor(xymaxf[1]);
			}
		}
		else
#endif
		{
			if (!is_float)
			{
				xmin = xmax = pt.x;
				ymin = ymax = pt.y;

				for (i = 1; i < npoints; i++)
				{
					pt = pts[i];

					if (xmin > pt.x)
						xmin = pt.x;

					if (xmax < pt.x)
						xmax = pt.x;

					if (ymin > pt.y)
						ymin = pt.y;

					if (ymax < pt.y)
						ymax = pt.y;
				}
			}
			else
			{
				Cv32suf v;
				// init values
				xmin = xmax = CV_TOGGLE_FLT(pt.x);
				ymin = ymax = CV_TOGGLE_FLT(pt.y);

				for (i = 1; i < npoints; i++)
				{
					pt = pts[i];
					pt.x = CV_TOGGLE_FLT(pt.x);
					pt.y = CV_TOGGLE_FLT(pt.y);

					if (xmin > pt.x)
						xmin = pt.x;

					if (xmax < pt.x)
						xmax = pt.x;

					if (ymin > pt.y)
						ymin = pt.y;

					if (ymax < pt.y)
						ymax = pt.y;
				}

				v.i = CV_TOGGLE_FLT(xmin); xmin = cvFloor(v.f);
				v.i = CV_TOGGLE_FLT(ymin); ymin = cvFloor(v.f);
				// because right and bottom sides of the bounding rectangle are not inclusive
				// (note +1 in width and height calculation below), cvFloor is used here instead of cvCeil
				v.i = CV_TOGGLE_FLT(xmax); xmax = cvFloor(v.f);
				v.i = CV_TOGGLE_FLT(ymax); ymax = cvFloor(v.f);
			}
		}

		return Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
	}


	static Rect maskBoundingRect(const Mat& img)
	{
		CV_Assert(img.depth() <= CV_8S && img.channels() == 1);

		Size size = img.size();
		int xmin = size.width, ymin = -1, xmax = -1, ymax = -1, i, j, k;

		for (i = 0; i < size.height; i++)
		{
			const uchar* _ptr = img.ptr(i);
			const uchar* ptr = (const uchar*)alignPtr(_ptr, 4);
			int have_nz = 0, k_min, offset = (int)(ptr - _ptr);
			j = 0;
			offset = MIN(offset, size.width);
			for (; j < offset; j++)
				if (_ptr[j])
				{
				have_nz = 1;
				break;
				}
			if (j < offset)
			{
				if (j < xmin)
					xmin = j;
				if (j > xmax)
					xmax = j;
			}
			if (offset < size.width)
			{
				xmin -= offset;
				xmax -= offset;
				size.width -= offset;
				j = 0;
				for (; j <= xmin - 4; j += 4)
					if (*((int*)(ptr + j)))
						break;
				for (; j < xmin; j++)
					if (ptr[j])
					{
					xmin = j;
					if (j > xmax)
						xmax = j;
					have_nz = 1;
					break;
					}
				k_min = MAX(j - 1, xmax);
				k = size.width - 1;
				for (; k > k_min && (k & 3) != 3; k--)
					if (ptr[k])
						break;
				if (k > k_min && (k & 3) == 3)
				{
					for (; k > k_min + 3; k -= 4)
						if (*((int*)(ptr + k - 3)))
							break;
				}
				for (; k > k_min; k--)
					if (ptr[k])
					{
					xmax = k;
					have_nz = 1;
					break;
					}
				if (!have_nz)
				{
					j &= ~3;
					for (; j <= k - 3; j += 4)
						if (*((int*)(ptr + j)))
							break;
					for (; j <= k; j++)
						if (ptr[j])
						{
						have_nz = 1;
						break;
						}
				}
				xmin += offset;
				xmax += offset;
				size.width += offset;
			}
			if (have_nz)
			{
				if (ymin < 0)
					ymin = i;
				ymax = i;
			}
		}

		if (xmin >= size.width)
			xmin = ymin = 0;
		return Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
	}

	cv::Rect boundingRect(InputArray array)
	{
		Mat m = array.getMat();
		return m.depth() <= CV_8U ? maskBoundingRect(m) : pointSetBoundingRect(m);
	}

}//zc--zc::boundingRect 

//---------------���Դ��������
namespace zc{
	Mat getLaplaceEdgeKrnl(size_t krnlSz /*= 5*/){
		CV_Assert(krnlSz % 2 == 1);
		
		Mat krnlMat = Mat::zeros(krnlSz, krnlSz, CV_16SC1);

		krnlMat.row(0) = -1;
		krnlMat.row(krnlMat.rows - 1) = -1;
		krnlMat.col(0) = -1;
		krnlMat.col(krnlMat.cols - 1) = -1;

		int factor = (krnlSz - 1) * 4;
		int center = krnlSz / 2;
		krnlMat.at<short>(center, center) = factor;

		return krnlMat;
	}//getLaplaceEdgeKrnl

	Mat getLaplaceEdgeFilter2D(const Mat &dmat, size_t krnlSz /*= 5*/){
		Mat res;
		Mat krnl = getLaplaceEdgeKrnl(krnlSz);
		filter2D(dmat, res, CV_32F, krnl);

		return res;
	}//getLaplaceEdgeFilter2D

	cv::Mat holeFillNbor(const Mat &dmat, bool hasSideEffect /*= false*/, size_t krnlSz /*= 5*/, int countThresh /*= 3*/)
{
		CV_Assert(krnlSz % 2 == 1);

		Mat res = dmat.clone();

		int radius = krnlSz / 2;
		Mat dmat_with_border;
		cv::copyMakeBorder(dmat, dmat_with_border, radius, radius, radius, radius, BORDER_REPLICATE);
		for (size_t i = 0; i < dmat.rows; i++){
			for (size_t j = 0; j < dmat.cols; j++){
				ushort z = dmat.at<ushort>(i, j);
				//��Ч����
				if (z == 0){
// 					Point pt_w_bdr()
// 					int offset=

					//�����Ǳ߽�����˵� dmat_with_border��
					Rect nbRect(j, i, krnlSz, krnlSz);
					Mat nbMat = dmat_with_border(nbRect);
					//��������Чֵ�����ﵽ��ֵ��
					if (countNonZero(nbMat) > countThresh){
						ushort newz = cv::mean(nbMat, nbMat != 0)[0];
						res.at<ushort>(i, j) = newz;
						if (hasSideEffect)
							dmat_with_border.at<ushort>(i + radius, j + radius) = newz;
					}

				}
			}
		}

		return res;
	}//holeFillNbor


#pragma region //��Ҫ��, �Լ�ʵ��֡���������MyBGSubtractor:
	MyBGSubtractor::MyBGSubtractor(){

	}

	MyBGSubtractor::MyBGSubtractor(int history, int diffThresh)
		:_history(history), _diffThresh(diffThresh)
		, _bgMat32f(0,0, CV_32F)
	
	{

		CV_Assert(_history > 0);
		CV_Assert(_diffThresh > 0);
	}

	MyBGSubtractor::~MyBGSubtractor()
	{

	}

	void MyBGSubtractor::addToHistory(const Mat &frame32f){
		//v1: ��д����Ч�棬ÿ֡���¼�������queue��ֵ. 2015��7��8��14:16:24

		//�� 32f �����㣬 ��ֹ���ȶ�ʧ��
		//Mat _bgMat32f;
		//_bgMat.convertTo(_bgMat32f, CV_32F);

		int histSz = _historyFrames.size();
		if (histSz >= _history){
			Mat front32f = _historyFrames.front();
			if (histSz > 1)
				_bgMat32f = 1. * (_bgMat32f * histSz - front32f) / (histSz - 1);
			else
				_bgMat32f = Mat::zeros(_bgMat32f.size(), CV_32F);

			_historyFrames.pop();
		}
		
		histSz = _historyFrames.size();
		if (histSz > 0){
			_bgMat32f = 1. * (_bgMat32f * histSz + frame32f) / (histSz + 1);

		}
		else
			_bgMat32f = frame32f.clone();

		
		//��ת�أ�
		//_bgMat32f.convertTo(_bgMat, _bgMat.type());

		CV_Assert(frame32f.depth() == CV_32F);
		_historyFrames.push(frame32f.clone());
	}

	cv::Mat MyBGSubtractor::apply(const Mat &currFrame)
{
		//Mat fgMsk = abs(currFrame - _bgMat) > _diffThresh;

		Mat currFrame32f;
		currFrame.convertTo(currFrame32f, CV_32F);
		Mat fgMsk = abs(currFrame32f - _bgMat32f) > _diffThresh;
		addToHistory(currFrame32f);

		return fgMsk;
	}

	cv::Mat MyBGSubtractor::getBgMat()
	{
		//return _bgMat;
		
		//����������ʾ�� bgMat
		Mat bgMat_show;
		_bgMat32f.convertTo(bgMat_show, CV_8U);
		return bgMat_show;
	}
#pragma endregion //��Ҫ��, �Լ�ʵ��֡���������MyBGSubtractor

	Mat getHumVecMaskHisto(Mat dmat, vector<HumanObj> humVec, vector<Mat> moveMaskVec, bool debugDraw /*= false*/){
		//���ڻ���ֱ��ͼ, rgb��ɫ, N��ֱ��ͼ����ͬһ��mat�ϣ�������ɫ���ӦHumanObjһ�£�
		//�ص�������ɫ���ǣ� or ������ or ��ɫ�ںϣ� //��ʱ���ǣ�
		Mat res = Mat::zeros(dmat.size(), CV_8UC4);

		size_t maskVecSz = moveMaskVec.size();
		for (size_t i = 0; i < maskVecSz; i++){
			Scalar co_i = humVec[i].getColor();
			Mat msk_i = moveMaskVec[i];

			Mat histoMat_i = getMaskXyHisto(dmat, msk_i, co_i, debugDraw);
			res += histoMat_i;
		}

		return res;
	}//getHumVecMaskHisto

	//@brief �� maskMat ͳ��X������Yֵͳ��ֱ��ͼ, ��color����
	//@return һ����ɫ histo-mat
	Mat getMaskXyHisto(Mat dmat, Mat maskMat, Scalar color, bool debugDraw /*= false*/){
		
		Mat res=Mat::zeros(maskMat.size(), CV_8UC4);

		int ww = maskMat.cols,
			hh = maskMat.rows;
		for (size_t x = 0; x < ww; x++){
			Mat colx = maskMat.col(x);
			int validPxCnt = countNonZero(colx);
			cv::line(res, Point(x, hh), Point(x, hh - validPxCnt), color, 1);
		}

		return res;
	}//getMaskXyHisto

	//@brief private-global-var, �������������ʷ���ֵ
	static Mat maxDmat;
	//@brief ��ʷ������dmat���У� ���ڶ��׶�βdiff��ȷ����һֱ�󳷣�����������ӣ��龳�У��������򲻱���Ϊ����
	//���к����Ȼ��ǰ�����ֵ��
	static queue<Mat> maxDmatQueue;
	const size_t maxDmatQueueCapacity = 30;
	
	Mat& getMaxDmat(Mat &dmat, bool debugDraw /*= false*/){
		if (maxDmat.empty())
			maxDmat = dmat.clone();

		return maxDmat;
	}//getMaxDmat

	Mat updateMaxDmat(Mat &dmat, bool debugDraw /*= false*/){
		//static Mat maxDmat = dmat.clone();

		//if (maxDmat.empty())
		//	maxDmat = dmat.clone();
		maxDmat = getMaxDmat(dmat, debugDraw);

		//�����µ������Ⱦ���
		maxDmat = cv::max(dmat, maxDmat);

		//��ӣ�
		if (maxDmatQueue.size() >= maxDmatQueueCapacity)
			maxDmatQueue.pop();
		maxDmatQueue.push(maxDmat.clone());

		if (debugDraw){
			Mat maxDepMat8u;
			normalize(maxDmat, maxDepMat8u, 0, UCHAR_MAX, NORM_MINMAX, CV_8UC1);
			imshow("updateMaxDmat.maxDepMat", maxDepMat8u);
		}

		return maxDmat;
	}//updateMaxDmat

	void resetMaxDmat(){
		maxDmat.release();
		maxDmatQueue = queue<Mat>();
	}//resetMaxDmat

	cv::Mat getMaxDepthBgMask(Mat dmat, bool reInit /*= true*/, bool debugDraw /*= false*/)
{
		//Mat maxDmat = updateMaxDmat(dmat, debugDraw);
		maxDmat = getMaxDmat(dmat, debugDraw);
		//ȥ��αǰ����������Ӱ�������� ��ʹMOG2���Ϊ�˶�ǰ�������������ֵ����(diff<10cm)��ʷ�����ȣ���۳�
		//Mat maxDepBgMask = abs(maxDmat - dmat) < 10; //10mm
		Mat maxDepBgMask = abs(maxDmat - dmat) < maxDmat / 30; //10mm
		updateMaxDmat(dmat, debugDraw);

		if (debugDraw)
			imshow("getMDBM.maxDepBgMask", maxDepBgMask);

		//����һֱ���ˣ���ȳ������ӵ����壬�������ᵼ�¿۳��������Թ�������ʵ�˶�����
#if 10	//�����۳�α��Ӱ

		Mat maxDmatMoveAreaMask;
#if 0	//��maxDmatQueueCapacity֡�ڡ���ʷ���ֵ���仯>50mm����Ϊ�˶��ˣ��Ǳ�����
		Mat qFront = maxDmatQueue.front(),
			qBack = maxDmatQueue.back();
		maxDmatMoveAreaMask = qBack - qFront > 20;

#elif 1	//����dmat���б�����ģ
		//int history = 30;
		int history = 20;
		double varThresh = 0.3;
		bool detectShadows = false;
		double bgRatio = 0.29;

		Mat maxDmatFgMOG2;
		Mat maxDmat8u;
		maxDmat.convertTo(maxDmat8u, CV_8UC1, 1.*UCHAR_MAX / MAX_VALID_DEPTH);
#ifdef CV_VERSION_EPOCH //if opencv2.x
		static Ptr<BackgroundSubtractorMOG2> pBgSub_maxDmat = new BackgroundSubtractorMOG2(history, varThresh, detectShadows);
		(*pBgSub_maxDmat)(maxDmat8u, maxDmatFgMOG2);
#elif CV_VERSION_MAJOR >= 3 //if opencv3

		//��δʵ����ɣ� reInit�ȷ��Ų��ã�2015��7��23��23:33:51
// 		static Ptr<BackgroundSubtractorMOG2> pMOG2_maxDmat;
// 		if (reInit){
// 			pMOG2_maxDmat = createBackgroundSubtractorMOG2(history, varThresh, detectShadows);
// 		}

#if USE_MOG2	//---------------MOG2
		static Ptr<BackgroundSubtractor> pBgSub_maxDmat = createBackgroundSubtractorMOG2(history, varThresh, detectShadows);


		//32F���У������˵���ˣ���δ��ʵ��
		//Mat maxDmat32f;
		//maxDmat.convertTo(maxDmat32f, CV_32FC1);
		//maxDmatPMOG2->apply(maxDmat32f, maxDmatFgMOG2);

		//pBgSub_maxDmat->setBackgroundRatio(bgRatio);
#else	//---------------KNN
		history = 20;
		double dist2Threshold = 0.8;
		static Ptr<BackgroundSubtractor> pBgSub_maxDmat = createBackgroundSubtractorKNN(history, dist2Threshold, detectShadows);
#endif	//MOG2, KNN

		//MOG, KNN ͨ�ã�
		pBgSub_maxDmat->apply(maxDmat8u, maxDmatFgMOG2);
		//imshow("maxDmatFgMOG2", maxDmatFgMOG2);

#endif //CV_VERSION_EPOCH

// 		int morphRadius = 2;
// 		morphologyEx(maxDmatFgMOG2, maxDmatFgMOG2, MORPH_DILATE, getMorphKrnl(morphRadius));

		maxDmatMoveAreaMask = maxDmatFgMOG2;
#endif

#if 0	//���� -= vs. &=(x==0)
		Mat tmp = maxDepBgMask.clone();
		tmp &= (maxDmatMoveAreaMask == 0);
		imshow("getMDBM.tmp", tmp);

		Mat tmp2 = maxDepBgMask.clone();
		tmp2 -= maxDmatMoveAreaMask;
		//����cv8uc1 mask, ��Ϊ����ض�, -= �ȼ��� &=(xxx==0)
		CV_Assert(countNonZero(tmp != tmp2) == 0);
#endif

		maxDepBgMask -= maxDmatMoveAreaMask;
		if (debugDraw){
			imshow("getMDBM.moveAreaMask", maxDmatMoveAreaMask);
			imshow("getMDBM.maxDepBgMask-final", maxDepBgMask);
		}

#endif

		return maxDepBgMask;
	}//getMaxDepthBgMask

	cv::Mat maskVec2mask(cv::Size maskSize, const vector<Mat> &maskVec){
		//CV_Assert(maskVec.size() > 0); //����

		Mat res = Mat::zeros(maskSize, CV_8UC1);

		size_t maskVecSz = maskVec.size();
		for (size_t i = 0; i < maskVecSz; i++){
			res += maskVec[i];
		}

		return res;
	}


	cv::Mat largeContPassFilter(const Mat &mask, LCPF_MODE mode /*= CONT_LENGTH*/, int contSizeThresh /*= 10*/){
#if 0	//debug-only
		if (countNonZero(mask)){
			int dummy = 0;
			//���� findContours �� orbbec_skeleton �����³���:
			vector<vector<Point>> contours;
			findContours(mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
			dummy = 0;
		}
#endif	//debug-only

		Mat res = mask.clone();

		vector<vector<Point>> contours;
		findContours(mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

		size_t contSz = contours.size();

#if 0	//������(ȡ����), qvga~1.2ms
		//��������Ҫ����������Ȼ�� mask -= maskRemove
		Mat maskRemove = Mat::zeros(mask.size(), CV_8UC1);
		for (size_t i = 0; i < contSz; i++){
			if (mode == CONT_LENGTH && contours[i].size() < contSizeThresh	//1.21ms
				|| mode == CONT_AREA && contourArea(contours[i]) < contSizeThresh) //1.29ms
				drawContours(maskRemove, contours, i, UCHAR_MAX, -1);
		}

		res -= maskRemove;
#elif 1	//���ӷ�(��), ����Ч, qvga~0.22ms
		Mat maskIntersect = Mat::zeros(mask.size(), CV_8UC1);
		for (size_t i = 0; i < contSz; i++){
			if (mode == CONT_LENGTH && contours[i].size() >= contSizeThresh
				|| mode == CONT_AREA && contourArea(contours[i]) >= contSizeThresh)
				drawContours(maskIntersect, contours, i, UCHAR_MAX, -1);
		}
		
		res &= maskIntersect;
#endif	//�������ӷ��Ա�

		return res;
	}//largeContPassFilter


	cv::Scalar maskedCvSum( const Mat &src, const Mat &mask ){
		Mat tmp = src.clone();
		tmp.setTo(0, mask == 0);
		return cv::sum(tmp);
	}

}//zc-���Դ��������

namespace sgf{
#if 01	//�����ͷ�����ӵ�-wrapper
	static const char *sgfConfigFname = nullptr;
	static const char *sgfTemplFname = nullptr;
	static sgf::segment *my_seg = nullptr;
	sgf::segment* loadSeedHeadConf(const char *confFn /*= "./sgf_seed/config.txt"*/, const char *templFn /*= "./sgf_seed/headtemplate.bmp"*/)
	{
		// 		sgfConfigFname = confFn;
		// 		sgfTemplFname = templFn;
		if (my_seg == nullptr){
			my_seg = new sgf::segment();
			my_seg->read_config(confFn);
			my_seg->set_headTemplate2D(templFn);
		}

		return my_seg;
	}

	vector<Point> seedHeadTempMatch(const Mat &dmat, bool debugDraw /*= false*/){
		CV_Assert(my_seg != nullptr);

		//return my_seg->seedSGF(dmat, debugDraw);
		return my_seg->seedHeadTempMatch(dmat, debugDraw);
	}//seedHeadTempMatch

	vector<double> getHeadSizes(){
		CV_Assert(my_seg != nullptr);

		return my_seg->get_headSize();
	}//getHeadSizes

	vector<Mat> findfgMasksMovingHead(const Mat &dmat, const Mat& mog_fg, int range /*= 2*/, int cntThresh /*= 100*/, bool debugDraw /*= false*/){
		CV_Assert(my_seg != nullptr);
		vector<Point> sdHeadVec = seedHeadTempMatch(dmat, debugDraw);
		if (sdHeadVec.size() > 0)
			int dummy = 0;

		vector<double> headSizeVec = getHeadSizes();

		return my_seg->findfgMasksMovingHead(mog_fg, sdHeadVec, headSizeVec, range, cntThresh, debugDraw);
	}//findfgMasksMovingHead
#endif	//�����ͷ�����ӵ�-wrapper

#if 01	//����ɷ���һ���ֺ�����
	vector<Mat> separateMasksContValley(Mat dmat, vector<Mat> &inMaskVec, bool debugDraw /*= false*/){
		// 		vector<Point> sdHeadVec = seedHead(dmat, debugDraw);
		// 		vector<double> headSizeVec = getHeadSizes();
		//return my_seg->get_seperate_masks(dmat); 

		CV_Assert(my_seg != nullptr);

		vector<Mat> res;
		size_t inMaskVecSz = inMaskVec.size();
		for (size_t i = 0; i < inMaskVecSz; i++){
			Mat msk_i = inMaskVec[i];
			vector<Mat> res_i = my_seg->get_seperate_masks(msk_i, debugDraw);
			res.insert(res.end(), res_i.begin(), res_i.end());
		}

		return res;
	}//separateMasksContValley

	vector<Mat> separateMasksMovingHead(Mat dmat, vector<Mat> &inMaskVec, Mat &mogMask, bool debugDraw /*= false*/){
		CV_Assert(my_seg != nullptr);

		vector<Point> sdHeadVec = seedHeadTempMatch(dmat, debugDraw);
		vector<double> headSizeVec = getHeadSizes();

		vector<Mat> res;
		size_t inMaskVecSz = inMaskVec.size();
		for (size_t i = 0; i < inMaskVecSz; i++){
			Mat msk_i = inMaskVec[i];
			vector<Mat> res_i = my_seg->get_seperate_masks(msk_i, mogMask, sdHeadVec, headSizeVec, debugDraw, debugDraw);
			res.insert(res.end(), res_i.begin(), res_i.end());
		}

		return res;
	}//separateMasksMovingHead
#endif	//����ɷ���һ���ֺ�����

#if 01	//����ɻ�ȡ��������mask����

	static Mat _max_dmat,	//������ͼ
		_potFgMask,	//���յ�"��������mask"
		_max_dmat_mask;	//������ͼ��Ӧ��flag-mat

	Mat calcPotentialMask(const cv::Mat& dmat, const cv::Mat& dmat_old){
		//CV_Assert(my_seg != nullptr);
		//my_seg->buildMaxDepth(dmat, dmat_old, max_dmat, max_dmat, fgMask, fgMask, max_dmat_mask, max_dmat_mask);
		sgf::buildMaxDepth(dmat, dmat_old, _max_dmat, _max_dmat, _potFgMask, _potFgMask, _max_dmat_mask, _max_dmat_mask);

		return _potFgMask;
	}//calcPotentialMask

	void releasePotentialMask(){
		_max_dmat.release();
		_potFgMask.release();
		_max_dmat_mask.release();
	}//releasePotentialMask

	void setPotentialMask(const Mat &dmat, Mat region2reset, bool debugDraw /*= false*/){
		//1. ����ͬʱ���� max-dmat-fg-flag:
		_max_dmat_mask.setTo(0, region2reset);

		//2. ���滹������, ���� _max_dmat ��fg����, ��һ֡�Զ�����
		_max_dmat.setTo(0, region2reset);
		cv::add(_max_dmat, dmat, _max_dmat, region2reset);

		//3. �������������potential-mask���������
		//_potFgMask = newFgMask.clone(); //����, ���䵼��potMaskԽ��ԽС
		_potFgMask.setTo(0, region2reset);
	}//setPotentialMask

	void setPotentialMask(const Mat &dmat, const Mat &currFgMskWhole, const Mat &prevFgMskWhole, bool debugDraw /*= false*/){
		Mat region2reset = prevFgMskWhole - currFgMskWhole;
		region2reset -= (dmat == 0);

		if (debugDraw)
			imshow("region2reset", region2reset);

		//�����ô��������, e.g.:
		//1. �뿪�˵��α�, �ϴ��ֲ���������ô�� ��
		//2. ǰ����С��Ƭ��������
		region2reset = zc::largeContPassFilter(region2reset, zc::CONT_AREA, 20 * 40);
		if (debugDraw)
			imshow("region2reset-large", region2reset);

		//1. ����ͬʱ���� max-dmat-fg-flag:
		_max_dmat_mask.setTo(0, region2reset);

		//2. ���滹������, ���� _max_dmat ��fg����, ��һ֡�Զ�����
		_max_dmat.setTo(0, region2reset);
		cv::add(_max_dmat, dmat, _max_dmat, region2reset);

		//3. �������������potential-mask���������
		_potFgMask.setTo(0, region2reset);
	}//setPotentialMask

	cv::Mat getPotentialMask(){
		return _potFgMask;
	}//getPotentialMask


#endif	//����ɻ�ȡ��������mask����
}//namespace sgf
