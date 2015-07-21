#include <iostream>
#include "SimpleSilhouette.h"
#include <iterator>
#include <algorithm>
#include <functional>


//@deprecated, 已验证： 0√， 1×
//abs-diff 时选用当前点深度还是平均深度：
const int mode = 01; //0当前， 1平均

//simpleRegionGrowXXX, separateMasks 关联变量，增长不许超过 thickLimit，不想改接口，暂用global-var
const int thickLimitDefault = 1500;
int thickLimit = thickLimitDefault; //毫米

//一些调试颜色：
Scalar cwhite(255, 255, 255);
Scalar cred(0, 0, 255);
Scalar cgreen(0, 255, 0);
Scalar cblue(255, 0, 0);
Scalar cyellow(0, 255, 255);

namespace zc{

//from CKernal.cpp
#if defined(ANDROID)
#define FEATURE_PATH "/data/data/com.motioninteractive.zte/app_feature/"
#else
	//#define FEATURE_PATH "../Skeleton/feature"
#define FEATURE_PATH "../../../plugins/orbbec_skeleton/feature"
#endif

//#if 1
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

	const char *matNodeName = "mat";
	const char *matVecName = "mat_vec";

	void saveVideo(const vector<Mat> &matVec, const char *fname){
		FileStorage fstorage(fname, FileStorage::WRITE);

		fstorage << matVecName << "[";

		size_t mvecSz = matVec.size();
		for (size_t i = 0; i < mvecSz; i++){
			//fstorage << matNodeName << matVec[i];
			fstorage << matVec[i];
		}
		fstorage << "]";

		fstorage.release();
	}//saveVideo

	vector<Mat> loadVideo(const char *fname){
		cout << "loadVideo(): " << endl;
		vector<Mat> res;
		//文件中有多个同名key，写可以，读报错：CV_PARSE_ERROR( "Duplicated key" );
		//使用SEQ "[]"存 vector：
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
			//res.push_back((Mat)(*it)); //没法类型转换
			Mat m;
			it >> m; //>>自带++it, 所以不要手动++
			res.push_back(m);
		}

		// 	while (1){
		// 		Mat m;
		// 		fstorage[matNodeName] >> m;
		// 	}
		return res;
	}//loadVideo

	//@brief 把多层vov转为单层vec
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

	//zhangxaochen: 参见 hist_analyse.m 中我的实现
	Point seedSimple(Mat dmat, int *outVeryDepth /*= 0*/, bool debugDraw /*= false*/){
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
			if(countNonZero(dmat == veryDepth))
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
	}//seedSimple

	Mat getBgMskUseWallAndHeight(Mat dmat){
		int wallDepth = zc::fetchWallDepth(dmat);
		Mat initBgMsk;
		//若无大墙面， 背景较空旷， 用物理高度判定剔除(<2500mm)
		//包括： wallDepth < 0 以及因人离相机太近(<1500mm)，身体被判定为墙面的情形
		if (wallDepth < 3000){
			Mat heightMsk = getFakeHeightMask(dmat, 2500);
			initBgMsk = (heightMsk == 0);

			//2015年6月24日15:20:41：
			//发现高度 initBgMsk 不应再区域增长，有些情况会把人长进背景，故直接返回。
			//其余代码不变
			return initBgMsk;
		}
		else{ //wallDepth > 0； 有墙面
			initBgMsk = (dmat >= wallDepth);
		}

		//增长出一个背景，thresh 要 <= 前景的 ！
		int rgThresh = 25;
		//还是需要去除地面，否则错！
		Mat flrApartMask = zc::fetchFloorApartMask(dmat, false);

		//增长背景不设置thickLimit（置为最大值）
		int oldThickLimit = thickLimit;
		thickLimit = MAX_VALID_DEPTH;
		Mat bgMsk = zc::simpleRegionGrow(dmat, initBgMsk, rgThresh, flrApartMask, false, false)[0];
		thickLimit = oldThickLimit;

		return bgMsk;
	}//getBgMskUseWallAndHeight

	Mat fetchBgMskUseWallAndHeight(Mat dmat){
		static Mat res;

		static Mat dmatOld;// = dmat.clone();
		//Mat dmatOld = getPrevDmat(); //×， 不是一帧循环中只用一次
		if (dmatOld.empty() || countNonZero(dmatOld != dmat) > 0){
			res = getBgMskUseWallAndHeight(dmat);// , debugDraw);
			dmatOld = dmat.clone();
		}

		return res;
	}//fetchBgMskUseWallAndHeight


	//@deprecated, 语义不明，弃用；其实包含两步：去背景 & findFgMasksUseBbox
	//1. 找背景大墙面；
	//2. 若没墙，说明背景空旷，物理高度判定剔除(<2500mm)
	//3. 剩余最高点做种子点
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

		//腐蚀：
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

	//预先用 roi 生成 mask
	Mat _simpleRegionGrow( const Mat &dmat, vector<Point> seeds, int thresh, const Rect roi, bool debugDraw /*= false*/ ){
		Mat _mask = Mat::zeros(dmat.size(), CV_8UC1);
		_mask(roi).setTo(UCHAR_MAX);

		return _simpleRegionGrow_core_vec2mat(dmat, seeds, thresh, _mask, debugDraw);
	}//simpleRegionGrow

	//
	Mat _simpleRegionGrow_core_pt2mat(const Mat &dmat, Point seed, int thresh, const Mat &validMask, bool debugDraw /*= false*/){
		//1. 2015年6月22日21:55:54 还是改回用 core_vec2mat 版本：
		vector<Point> seeds;
		seeds.push_back(seed);

		return _simpleRegionGrow_core_vec2mat(dmat, seeds, thresh, validMask, debugDraw);

		//2. 2015年6月22日20:17:29：这个定为核心函数
		//在 mode=0时，效率一般；mode=1时，大墙面场景因频繁调用此函数，效率低(caller core_mat2mat： ~30ms)
		Size sz = dmat.size();
		int hh = sz.height,
			ww = sz.width;

		Rect dmRect(Point(), dmat.size());

		//1. init
		//存标记：0未查看， 1在queue中， 255已处理过neibor；最终得到的正是 mask
		Mat flagMat = Mat::zeros(sz, CV_8UC1);

		//存满足条件的点
		queue<Point> pts;

		//abs-diff 时选用当前点深度还是平均深度：
		const int mode = 01; //0当前， 1平均
		//状态量：
		double depAvg = 0; //前景点平均深度
		size_t ptCnt = 0; //前景点个数


		//初始种子点入队&标记，需满足条件：mask 有效
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

			//初始化平均深度：
			depAvg = dmat.at<ushort>(seed); 
		}

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

			flagMat.at<uchar>(pt) = UCHAR_MAX;

			//更新平均深度：
			depAvg = (depAvg*ptCnt + depPt) / (ptCnt + 1);
			ptCnt++;

			for (int i = 0; i < nnbr; i++){
				Point npt = pt + Point(dx[i], dy[i]);
				//roi 判断不要了：
				//if (left <= npt.x && npt.x < right && top <= npt.y && npt.y < bottom)
				if (dmRect.contains(npt))
				{
					const ushort& depNpt = dmat.at<ushort>(npt);
					uchar& flgNpt = flagMat.at<uchar>(npt);
					if (flgNpt == 0
						//&& abs(depPt - depNpt) <= thresh
						//增加mask 判断：
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

	//_simpleRegionGrow 【真】核心函数，效率高于 core_vec2mat：
	Mat _simpleRegionGrow_core_mat2mat(const Mat &dmat, Mat sdsMat, int thresh, const Mat &validMask, bool debugDraw /*= false*/){
		//用于调试偶尔出错情况：
		bool isDebugError = false;
		bool isAnimSlowly = false; //增长过程动画 imshow
		Mat errorMat,
			errorSeedMat;
		if (isDebugError){
			errorMat = Mat::zeros(dmat.size(), CV_8UC1);
			errorSeedMat = errorMat.clone();
		}

		//1. core_vec2mat 改了， 所以调用之试试：
// 		clock_t begt = clock();
// 		Mat res = _simpleRegionGrow_core_vec2mat(dmat, maskMat2pts(sdsMat),
// 			thresh, validMask, debugDraw);
// 		cout << "_simpleRegionGrow_core_mat2mat.ts: " << clock() - begt << endl;
// 		return res;

		//2. 2015年6月22日20:38:31 之前版本
		Size sz = dmat.size();
		int hh = sz.height,
			ww = sz.width;

		Rect dmRect(Point(), dmat.size());
		
		//目前增长看四邻域：
		const int nnbr = 4;
		int dx[nnbr] = { 0, -1, 0, 1 },
			dy[nnbr] = { 1, 0, -1, 0 };

		//1. init
		//存标记：0未查看， 1在queue中， 255已处理过neibor；最终得到的正是 mask
		Mat flagMat = Mat::zeros(sz, CV_8UC1);

		//存满足条件的点
		queue<Point> pts;

		//@deprecated:
		//abs-diff 时选用当前点深度还是平均深度：
		//经验证， mode==1时，效果很差， 因为均值妨碍了新点加入
		const int mode = 0; //0当前， 1平均
		//状态量： 前景点平均深度, & 个数
		double depAvg = 0;
		size_t ptCnt = 0;

		//状态量： mask内深度范围 [dmin, dmax]
		ushort dmin = MAX_VALID_DEPTH,
			dmax = 0;

		//初始种子点入队&标记，需满足条件：validMask 有效, 且四邻域【未全部】标记
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
					if (!isInnerPt){ //轮廓点， 入队
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
					else{ //内点，直接标为已读
						flagMat.at<uchar>(pt) = UCHAR_MAX;

						//更新平均深度：
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

					//更新深度范围，因认为内点可能更正确
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

			//若超过了增长厚度限制：
			if (abs(depPt - dmax) > thickLimit
				|| abs(depPt - dmin) > thickLimit)
				continue;

			//更新深度厚度范围：
			if (depPt < dmin)
				dmin = depPt;
			if (depPt > dmax)
				dmax = depPt;

			//更新平均深度：
			depAvg = (depAvg*ptCnt + depPt) / (ptCnt + 1);
			ptCnt++;

			for (int i = 0; i < nnbr; i++){
				Point npt = pt + Point(dx[i], dy[i]);
				//roi 判断不要了：
				//if (left <= npt.x && npt.x < right && top <= npt.y && npt.y < bottom)
				if (dmRect.contains(npt))
				{
					const ushort& depNpt = dmat.at<ushort>(npt);
					uchar& flgNpt = flagMat.at<uchar>(npt);
					if (flgNpt == 0
						//&& abs(depPt - depNpt) <= thresh //改为 isOk 判断
						//增加mask 判断：
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

	//_simpleRegionGrow 核心函数：
	Mat _simpleRegionGrow_core_vec2mat(const Mat &dmat, vector<Point> sdsVec, int thresh, const Mat &validMask, bool debugDraw /*= false*/){
		//1. 2015年6月22日20:31:38， 调用 core_pt2mat:
// 		Mat res = Mat::zeros(dmat.size(), CV_8UC1);
// 		size_t sdsVecSz = sdsVec.size();
// 		for (size_t i = 0; i < sdsVecSz; i++){
// 			Point sdi = sdsVec[i];
// 
// 			//若sdi不存在于之前任何一个mask，则新增长一个：
// 			if (res.at<uchar>(sdi) == 0){
// 				Mat newRgMat = _simpleRegionGrow_core_pt2mat(dmat, sdi, thresh, validMask, debugDraw);
// 				res += newRgMat;
// 			}
// 		}
// 
// 		return res;


		//2. 尝试改为调用 sdsMat 版本, 效率 fgMasksWallAndHeightSumt 19ms->16ms：
		return _simpleRegionGrow_core_mat2mat(dmat, pts2maskMat(sdsVec, dmat.size()),
			thresh, validMask, debugDraw);

		//3. 2015年6月22日20:29:31 之前版本： 
		Size sz = dmat.size();
		int hh = sz.height,
			ww = sz.width;

		Rect dmRect(Point(), dmat.size());

		//1. init
		//存标记：0未查看， 1在queue中， 255已处理过neibor；最终得到的正是 mask
		Mat flagMat = Mat::zeros(sz, CV_8UC1);

		//存满足条件的点
		queue<Point> pts;

		//初始种子点入队&标记，需满足条件：mask 有效
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
				if (dmRect.contains(npt))
				{
					const ushort& depNpt = dmat.at<ushort>(npt);
					uchar& flgNpt = flagMat.at<uchar>(npt);
					if (flgNpt == 0
						&& abs(depPt - depNpt) <= thresh
						//增加mask 判断：
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

				//若sdi不存在于之前任何一个mask，则新增长一个：
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

			//已验证， 5~7ms
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

	vector<Mat> simpleRegionGrow(const Mat &dmat, vector<Mat> sdMats, int thresh, const Mat &mask, bool debugDraw /*= false*/){
		vector<Mat> res;

		size_t sz = sdMats.size();
		for (size_t i = 0; i < sz; i++){
			//2015年6月24日19:14:11， 增加有效性检测：全黑舍弃
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
		if(countNonZero(tmp == UCHAR_MAX))
			findNonZero(tmp == UCHAR_MAX, res);

		//已验证， <1ms
		cout << "maskMat2pts.ts: " << clock() - begt << endl;

		return res;
	}//maskMat2pts

	//inner cpp, 暂无声明
	//注:
	//1. 参数： seedsMask 白色太多(>total*0.5)不要; 
	//2. 返回值： 一次增长结果candidateMsk太小(<20*30)不要
	//---------------@deprecated, 错误思路，不该有特定经验性限制！
// 	vector<Mat> simpleRegionGrow(Mat dmat, Mat seedsMask, int thresh, Mat mask){
// 		vector<Mat> res;
// 
// 		Mat sdPts;
// 		cv::findNonZero(seedsMask, sdPts);
// 
// 		size_t sdsz = sdPts.total();
// 		
// 		//若初始的 seedsMask 几乎全白，说明此帧坏的，舍弃：
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
// 			//若sdi不存在于之前任何一个mask，则新增长一个：
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
		
		//---------------2. 2015年6月23日20:54:15， 测试乘以高度系数：
		tmp = flrApartMat.clone();
		tmp.setTo(0, abs(tmp) > 500); //原始图无效区域导致突变，去掉这些点
		for (int i = 0; i < tmp.rows; i++){
			Mat row = tmp.row(i);
			row *= (1e-6*i*i*i);
		}
		flrApartMsk = abs(tmp) < mskThresh;
		//白色膨胀？close操作：
		//imshow("floor-height-factor-no-close", flrApartMsk);
		morphologyEx(flrApartMsk, tmp, MORPH_CLOSE, morphKrnl);
		flrApartMsk = tmp;

		if (debugDraw)
			imshow("flrApartMsk", flrApartMsk);

		//---------------3. 2015年7月5日01:56:01， 
		//高度图截断方法
		int ww = dmat.cols,
			hh = dmat.rows;
		Rect bottomBorder(0, 9. / 10 * hh, ww, hh / 10);
		//若屏幕底边缘大致黑色(>50%)，说明视野前方平地面积大：
		if (countNonZero(flrApartMsk(bottomBorder) == 0) > ww*hh / 10 * 0.5){
			Mat hmap1 = calcHeightMap1(dmat, false);
			//均值上移10cm：
			int flrHeight = cv::mean(hmap1, flrApartMsk == 0)[0];
			flrHeight += 100;

			//max 可能会导致过高截断
// 			double hmax; 
// 			minMaxLoc(hmap1, 0, &hmax, 0, 0, flrApartMsk == 0);
// 			int flrHeight = hmax;
			flrApartMsk = (hmap1 > flrHeight);

			if (debugDraw)
				imshow("flrApartMsk.height-cut", flrApartMsk);
		}

		return flrApartMsk;

#if 0	//---------------1. 2015年6月23日20:54:32， 之前版本，1/2, 3/4屏一刀切，不好
		flrApartMsk = abs(flrApartMat)<mskThresh;
// 		Mat flrApartMsk2 = abs(flrApartMat)<500 | abs(flrApartMat)>1000;
		//上半屏不管，防止手部、肩部被误删过滤：
		Rect upHalfWin(0, 0, dmat.cols, dmat.rows / 2);
		flrApartMsk(upHalfWin).setTo(UCHAR_MAX);

		//flrApartMsk 不连续（非渐变）的无效区域保留，防止手部自遮挡时被滤掉【不行，参考flrApartMsk2效果】

		//膨胀：仅保留大块地板，去掉对手部形成的边缘；腐蚀：希望脚部周围闭合。
		//效果差，腐蚀过大导致脚部难看
// 		dilate(flrApartMsk, flrApartMsk, morphKrnl); //res320*240, 
// 		morphKrnl = getMorphKrnl(12);
// 		erode(flrApartMsk, flrApartMsk, morphKrnl); //res320*240, 

		//底部 open操作， 非close，针对人走近时脚部与屏幕边缘连成一片
		Mat flrApartMsk_feet;
		morphologyEx(flrApartMsk, flrApartMsk_feet, MORPH_OPEN, morphKrnl);
		Rect up3of4Win(0, 0, dmat.cols, dmat.rows * 3 / 4);
		flrApartMsk_feet(up3of4Win).setTo(UCHAR_MAX);

		//close，膨胀-腐蚀等量：
		morphologyEx(flrApartMsk, tmp, MORPH_CLOSE, morphKrnl); //res320*240, 
		flrApartMsk = tmp;

		//白色求交，黑色求和：
		tmp = flrApartMsk & flrApartMsk_feet;
		if (debugDraw)
			imshow("tmp", tmp);
		flrApartMsk = tmp;

// 		//图像底部边缘腐蚀，使全黑：
// 		int krnlHt = 30;
// 		Mat morphKrnl2 = getStructuringElement(MORPH_CROSS, Size(1, krnlHt));
// 
// 		erode(flrApartMsk, tmp, morphKrnl2, Point(0, krnlHt - 1));
// 		//flrApartMsk = tmp;
// 		imshow("tmp", tmp);

// 		//大块地板的bbox中心点高度往下，填充零：
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

	//与 getFloorApartMask 区别在于： 不是每次算， 只有 dmat 内容变了，才更新
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
		Mat res = dmat.clone(); //保持 cv16uc1 应该没错；

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

	cv::Mat calcHeightMap0(Mat dmat, bool debugDraw /*= false*/)
	{
		Mat res = dmat.clone(); //保持 cv16uc1 应该没错；

		int hh = res.rows;
		CV_Assert(hh > 0);
		//2015年6月25日15:10:53， 这样不对！得到的是每点到等深面屏幕最低点距离，不是到地面高度
		//结果是：1.地面非等高, 2.中轴线非登高
		for (int i = 0; i < hh; i++){
			//const uchar *row = res.ptr<uchar>(i);
			Mat row = res.row(i);
			row = row * (hh - i) / XTION_FOCAL_XY;
		}
		return res;
	}//calcHeightMapWrong

	Mat calcHeightMap1(Mat dmat, bool debugDraw /*= false*/){
		//Mat res = dmat.clone(); //保持 cv16uc1 应该没错；错！
		Mat res;
		dmat.convertTo(res, CV_32S);

		int hh = res.rows;
		CV_Assert(hh > 0);
		//2015年6月25日15:10:53， 这样不对！得到的是每点到等深面屏幕最低点距离，不是到地面高度
		//结果是：1.地面非等高, 2.中轴线非登高
// 		for (int i = 0; i < hh; i++){
// 			//const uchar *row = res.ptr<uchar>(i);
// 			Mat row = res.row(i);
// 			row = row * (hh - i) / XTION_FOCAL_XY;
// 		}

		//改为：中轴线为0高度，最后统一减场景中最低点(offset)
		for (int i = 0; i < hh; i++){
			Mat drow = res.row(i);
			drow = drow * (hh / 2 - i) / XTION_FOCAL_XY;
		}

		//hmin 这样算不好，会因为地面原点不精确导致不稳定，灰度闪烁
		//2015年7月5日02:29:51， 不稳定暂没关系
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

	//与 calcHeightMap 区别在于： 不是每次算， 只有 dmat 内容变了，才更新
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

	//截取物理尺度高度<limitMs(毫米)像素点做mask
	Mat getFakeHeightMask(Mat dmat, int limitMs /*= 2500*/){
		Mat htMap = zc::fetchHeightMap0(dmat); //用的 v0 假高度计算方法
		//imshow("htMap", htMap);

		//Mat htMap_show;
		//htMap.convertTo(htMap_show, CV_8UC1, 1.*UCHAR_MAX / 6e3);//×
		//imshow("htMap_show", htMap_show);

		Mat fgHeightMap = htMap < limitMs & dmat != 0;
		//imshow("fgHeightMap", fgHeightMap);

		return fgHeightMap;
	}//getFakeHeightMask

	int getWallDepth(Mat &dmat){
		double dmin, dmax;
		cv::minMaxLoc(dmat, &dmin, &dmax);
		//若全黑：
		if (dmax == 0)
			return -1;

		int histSize = dmax - dmin;
		float range[] = { dmin, dmax };
		const float *histRange = { range };
		//const float *histRange2 = range ; //√

		Mat histo;
		calcHist(&dmat, 1, 0, Mat(), histo, 1, &histSize, &histRange);
		// 
		// 			Mat histo2;
		// 			calcHist(&dmat, 1, 0, Mat(), histo2, 1, &histSize, &histRange, true, true); //accumulate=true 没用

		//深度滑窗长度50cm：
		int winLen = 500;
		//winLen窗口长度内bar累加高度：
		float maxBarSum = 0;
		int maxIdx = 0;
		//i=1 开始：
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
			//---------------返回 -1 表示没找到足够大的墙面：
			return -1;
		else{
			//小范围[maxIdx, maxIdx+maxBarSum) 找峰值
			float maxBarSum_2 = 0;
			int maxIdx_2 = 0;

			for (int i = maxIdx; i < maxIdx + winLen; i++){
				float barCnt = histo.at<float>(i);
				if (maxBarSum_2 < barCnt){
					maxIdx_2 = i;
					maxBarSum_2 = barCnt;
				}
			}
			//往前20cm：
			//int wallDepth = maxIdx_2 - 200; //不要做特定经验性处理！
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
		
		//上半身边缘：
		int th_low = 40;
		Canny(dm_draw, edge_up, th_low, th_low * 2);
		if (debugDraw){
			imshow("getHumanEdge.edge_up", edge_up);
		}

		//脚部边缘：
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
			edge_up,		//脚部与地面相连，上半身
// 			edge_up_inv,	//黑色描边
			edge_ft,		//脚部与地面分离，下半身
// 			edge_ft_inv,	//黑色描边
			edge_whole,		//上身+脚部，地面分离
			edge_whole_inv,
			distMap,
			bwImg;			//宽黑边二值图
// 		normalize(dmat, dm_draw, 0, UCHAR_MAX, NORM_MINMAX, CV_8UC1);
// 
// 		//TODO: 要不要做纵向高斯，解决手臂把身体分成两段的问题？不根本，暂时放弃
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
		//去掉无效区域，不必如下两次求轮廓：
		bwImg &= (dmat != 0);

		vector<vector<Point> > contours, cont_good;
		vector<Vec4i> hierarchy;
		findContours(bwImg.clone(), contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

// 		//第一遍得到的轮廓，可能包含无效深度区域：
// 		Mat roughMask = Mat::zeros(dmat.size(), CV_8UC1);
// 		drawContours(roughMask, contours, -1, 255, -1);
// 		
// 		//第二遍求轮廓：（这里该用 roughMask & (dm_draw!=0), 但无所谓）
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
			//质心一阶矩
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

			//测试过滤条件： 1. bbox 长宽比; 2. bbox 高度; 
			//3. bbox物理尺度高度; 4. bbox 下沿不能高于半屏，因为人脚部位置较低
			if(bsz.height*1./bsz.width > MIN_VALID_HW_RATIO && bsz.height > 80){
				if(debugDraw){
					cout<<"mc; dep_mc, width, height; dep_mc*w, dep_mc*h: "<<mc<<"; "
						<<dep_mc<<", "<<bsz.width<<","<<bsz.height<<"; "
						<<dep_mc*bsz.width<<", "<<dep_mc*bsz.height<<endl;
					drawContours(debug_mat, contours, i, 255, -1);
					circle(debug_mat, mc, 5, 128, 2);
				}

				//二次判定： 3. bbox物理尺度高度; 4. bbox 下沿不能高于半屏，因为人脚部位置较低
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
			//调试mat改用彩色绘制：
			_debug_mat.create(dmat.size(), CV_8UC3);
			debug_mat = _debug_mat.getMat();
			//debug_mat.setTo(0);
			//dmat.convertTo(debug_mat, CV_8U, 1.*UCHAR_MAX / MAX_VALID_DEPTH);
			Mat dmat_gray;
			dmat.convertTo(dmat_gray, CV_8U, 1.*UCHAR_MAX / MAX_VALID_DEPTH);
			vector<Mat> cn3(3, dmat_gray);
			cv::merge(cn3, debug_mat);
		}

		Mat edge_whole, //上身+脚部，地面分离
			edge_whole_inv, //黑色描边
			bwImg //二值图，类比dist-map二值化，实际是canny+erode结果
			;
		edge_whole = getHumanEdge(dmat, debugDraw);
		edge_whole_inv = (edge_whole == 0);
		static int anch = 4;
		static Mat morphKrnl = getStructuringElement(MORPH_RECT, Size(anch * 2 + 1, anch * 4 + 1), Point(anch, anch));
		erode(edge_whole_inv, bwImg, morphKrnl); //res320*240, costs 0.07ms

		//去掉无效区域：
		bwImg &= (dmat != 0);

		if (debugDraw){
			debug_mat.setTo(cblue, bwImg == 0); //先画粗线
			debug_mat.setTo(0, edge_whole); //细线
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

	//1. 对 distMap 二值化得到 contours； 2. 对 contours bbox 判断长宽比，得到人体区域
	Mat distMap2contoursDebug(const Mat &dmat, bool debugDraw /*= false*/){
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

// 		//distMap 是 CV_32FC1
// 		distanceTransform(edge_whole_inv, distMap, CV_DIST_L2, 3);
// 		normalize(distMap, distMap, 0, UCHAR_MAX, NORM_MINMAX, CV_8UC1);
// 		//对 distMap 二值化
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

		//对 edge erode， 与distMap 二值化会有什么区别？MORPH_ELLIPSE 时几乎完全相同
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
			//质心一阶矩
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
				//质心:
				circle(cont_draw_ok, mc, 5, 128, 2);

				if(debugDraw){
					//测试过滤条件： 1. bbox物理尺度高度； 2. bbox 下沿不能高于半屏，因为人脚部位置较低
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
		//dmat 预处理： 砍掉下半屏 & 去掉地面；
		dm_draw(Rect(0, dmat.rows/2, dmat.cols, dmat.rows/2)).setTo(0);
		Mat flrApartMsk = zc::fetchFloorApartMask(dmat);
		dm_draw.setTo(0, flrApartMsk==0);

		//dmat.convertTo(dm_draw, CV_8UC1, 1.*UCHAR_MAX / MAX_VALID_DEPTH);
		dm_draw.convertTo(dm_draw, CV_8UC1, ratio);

		if(debugDraw)
			imshow("dmat2TopDownView.dm_draw", dm_draw);


		Mat tdview = Mat::zeros(Size(dm_draw.cols, UCHAR_MAX+1), CV_16UC1);

		Mat debug_mat; //用于输出观察的调试图
		if(debugDraw){
			_debug_mat.create(tdview.size(), CV_8UC1);
			debug_mat = _debug_mat.getMat();
			//debug_mat = Mat::zeros(tdview.size(), CV_8UC1);
			debug_mat.setTo(0);
		}

		//迭代从最低行开始：(注意用 int, 不用 size_t)
		for(int i = dm_draw.rows - 1; i >=0; i--){
			const uchar *row = dm_draw.ptr<uchar>(i);
			for(int j = 0; j < dm_draw.cols; j++){
				uchar z = row[j];
				//tdview.at<ushort>(z, j) = dm_draw.rows - i;
				*(tdview.data + z * tdview.step + j * tdview.elemSize()) = dm_draw.rows - i;
			}
		}
		//然后转成 uchar
		Mat tmp;
		normalize(tdview, tmp, 0, UCHAR_MAX, NORM_MINMAX, CV_8UC1);
		tdview = tmp;

		if(debugDraw){
			imshow("tdview0", tdview);
		}

		tdview.setTo(0, tdview<128);

		//膨胀
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

		//top-down view 上通过 bbox 经验阈值找人：
		vector<vector<Point> > tdvContours, tdv_cont_good;
		vector<Vec4i> tdvHierarchy;
		findContours(tdview.clone(), tdvContours, tdvHierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		size_t tdvContSize = tdvContours.size();
		for(size_t i = 0; i < tdvContSize; i++){
			Rect boundRect = zc::boundingRect(tdvContours[i]);

			if(debugDraw)
				drawContours(debug_mat, tdvContours, i, 255, -1);

			//1. Z厚度判定 >3px(3*10000mm/256=117mm), <26(26*10000mm/256=1016mm)； 
			//2. X宽度判定 (200mm~2000mm) MAX_VALID_DEPTH / UCHAR_MAX / XTION_FOCAL_XY = 1/7.68
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
		//整个场景压扁成 MAX_VALID_DEPTH / ratio 像素厚度
		//通用接口，用16u，不用8u，因为未必ratio很小
		dmat.convertTo(dmatSquash, CV_16U, ratio);

		//测试convertTo 是 round？ 是因为float-> CV_16U 
		//Mat test;
		//dmat.convertTo(test, CV_32F, ratio);

		Mat tdview = Mat::zeros(Size(dmatSquash.cols, MAX_VALID_DEPTH * ratio + 1), CV_16UC1);

		//迭代从最低行开始：(注意用 int, 不用 size_t)
		for (int i = dmatSquash.rows - 1; i >= 0; i--){
			const ushort *row = dmatSquash.ptr<ushort>(i);
			for (int j = 0; j < dmatSquash.cols; j++){
				ushort z = row[j];
				//tdview.at<ushort>(z, j) = dm_draw.rows - i;
				*(tdview.data + z * tdview.step + j * tdview.elemSize()) = dmatSquash.rows - i;
			}
		}

		//第零行填充零，全黑。防止无效区域干扰：
		tdview.row(0) = 0;

		return tdview;
	}//dmat2tdview_core


	//Z轴缩放比为定值： UCHAR_MAX/MAX_VALID_DEPTH
	Mat dmat2TopDownViewDebug(const Mat &dmat, bool debugDraw /*= false*/){
		CV_Assert(dmat.type()==CV_16UC1);

		Mat dm_draw; //gray scale
		//normalize(dmat, dm_draw, UCHAR_MAX, 0, NORM_MINMAX, CV_8UC1);
		//normalize(dmat, dm_draw, 1.*UCHAR_MAX/MAX_VALID_DEPTH, NORM_L1, CV_8UC1);	//×
		dmat.convertTo(dm_draw, CV_8UC1, 1.*UCHAR_MAX/MAX_VALID_DEPTH);

		//dmat 砍掉下半屏
		dm_draw(Rect(0, dmat.rows/2, dmat.cols, dmat.rows/2)).setTo(0);
		if(debugDraw)
			imshow("dm_draw.setTo", dm_draw);

		//CV_16UC1, 因为可能y_max==480
		//Mat res = Mat::zeros(Size(dmat.cols, MAX_VALID_DEPTH), CV_16UC1);
// 		double dmax, dmin;
// 		minMaxLoc(dmat, &dmin, &dmax);
		Mat res = Mat::zeros(Size(dm_draw.cols, UCHAR_MAX+1), CV_16UC1);
		
		//迭代从最低行开始：
		for(int i = dm_draw.rows - 1; i >=0; i--){
			const uchar *row = dm_draw.ptr<uchar>(i);
			for(int j = 0; j < dm_draw.cols; j++){
				uchar z = row[j];
				//res.at<ushort>(z, j) = dm_draw.rows - i;
				*(res.data + z * res.step + j * res.elemSize()) = dm_draw.rows - i;
			}
		}
		
		//然后转成 uchar
		normalize(res, res, 0, UCHAR_MAX, NORM_MINMAX, CV_8UC1);

		//resize(res, res, Size(res.cols, 256));
		//res = res.t();
		return res;
	}//dmat2TopDownViewDebug

	vector<vector<Point>> seedUseBboxXyXz(Mat dmat, bool debugDraw /*= false*/, OutputArray _debug_mat /*= noArray()*/){
		vector<vector<Point>> res;

		//各自粗选得到的“好”cont： 
		//因为由边缘检测得到，所以可能包含无效区域，需要再过滤一遍
		vector<vector<Point>> dtrans_cont_good = zc::distMap2contours(dmat, false); //这里对 dtrans-cont 不画调试窗

		Mat tdv_debug_draw;
		vector<vector<Point>> tdv_cont_good = zc::dmat2TopDownView(dmat, 1. * UCHAR_MAX / MAX_VALID_DEPTH, debugDraw, tdv_debug_draw);

		Mat debug_mat; //用于输出观察的调试图
		if (debugDraw){
			_debug_mat.create(tdv_debug_draw.size(), CV_8UC1);
			debug_mat = _debug_mat.getMat();
			debug_mat.setTo(0);

			//tdview 实心cont， bbox， 128灰色画
			debug_mat.setTo(128, tdv_debug_draw == 255);
		}

		size_t tdv_cont_good_size = tdv_cont_good.size();
		vector<Rect> tdvBboxs(tdv_cont_good_size);

		//得到 tdv_cont 对应 bboxs
		for (size_t i = 0; i < tdv_cont_good_size; i++){
			tdvBboxs[i] = zc::boundingRect(tdv_cont_good[i]);
		}

		Mat flrApartMsk = getFloorApartMask(dmat, debugDraw);

		//对正视图每个 cont，转到俯视图， 求 bbox， 求交
		size_t dtrans_cont_size = dtrans_cont_good.size();
		for (size_t i = 0; i < dtrans_cont_size; i++){
			Mat cont_mask = Mat::zeros(dmat.size(), CV_8UC1);
			drawContours(cont_mask, dtrans_cont_good, i, 255, -1);
			//Z轴上下界：
			double dmin, dmax;
			minMaxLoc(dmat, &dmin, &dmax, nullptr, nullptr, cont_mask & dmat != 0);

			//缩放比 MAX_VALID_DEPTH / UCHAR_MAX，画到 top-down-view:
			Rect bbox_dtrans_cont = zc::boundingRect(dtrans_cont_good[i]);
			double ratio = 1. * UCHAR_MAX / MAX_VALID_DEPTH;
			Rect bbox_dtrans_cont_to_tdview(
				bbox_dtrans_cont.x, dmin * ratio,
				bbox_dtrans_cont.width, (dmax - dmin) * ratio);

			//补充一个判断条件: mask 深度值域不能 >1500mm
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
				//若存在两 bbox 相交， ok！
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
			//若存在两 bbox 相交， ok！区域增长获取整个前景msk
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

		//测试去除不稳定点，尚不成熟。效果微弱，【舍弃】
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
				//bbox增长时thickLimit设小一点：
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
		//候选人体轮廓：
		vector<vector<Point>> humContours = getHumanContoursXY(dmat, false, noArray());
		size_t humContoursSz = humContours.size();

		//头部种子点（圆形模板匹配）：
		vector<Point> sdHeadVec = seedHead(dmat, false);
		size_t sdHeadVecSz = sdHeadVec.size();
		if (debugDraw){
			for (size_t k = 0; k < sdHeadVecSz; k++){
				Point sdHead_k = sdHeadVec[k];
				//画人头：
				circle(debug_mat, sdHead_k, 11, 188, 2);
			}
		}

		for (size_t i = 0; i < humContoursSz; i++){
			if (debugDraw){
				drawContours(debug_mat, humContours, i, 255, 1);
			}

			//1. 方案一，轮廓质心 & 头部种子点 XZ距离. 2015年7月5日00:26:06
			//这样基本仅在质心在身体上时有解
			vector<Point> conti = humContours[i];
			Rect bbox = zc::boundingRect(conti);
			Moments mu = moments(conti);
			Point mc_i(mu.m10 / mu.m00, mu.m01 / mu.m00);
			ushort dep_mci = dmat.at<ushort>(mc_i);

			for (size_t k = 0; k < sdHeadVecSz; k++){
				Point sdHead_k = sdHeadVec[k];
				ushort dep_sdHeadk = dmat.at<ushort>(sdHead_k);

				if (
					bbox.x < sdHead_k.x && sdHead_k.x < bbox.br().x //头在bbox左右边界之间
					//abs(mc_i.x - sdHead_k.x) < 50
					&& abs(dep_mci - dep_sdHeadk) < 500 //dZ < 50cm
					&& mc_i.y > sdHead_k.y //头在上面, y小
					){

					if (debugDraw){
						drawContours(debug_mat, humContours, i, 255, 2);
						circle(debug_mat, mc_i, 5, 255, 2);

						//画人头：
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
		static Ptr<BackgroundSubtractorMOG2> pMOG2 =
			createBackgroundSubtractorMOG2(history, varThresh, detectShadows);
		Mat bgImg, fgMskMOG2;

		//若 isNewFrame=false，则直接用prevFgMskMOG2作为当前帧前景
		static Mat prevFgMskMOG2;
		static bool isFirstTime = true;

		if (isFirstTime || isNewFrame){

			double learningRate = -0.005;
			//接收 mat-type 必须为 8uc1, or 8uc3:
			pMOG2->apply(dm_show, fgMskMOG2, learningRate);

		}


		//---------------seed-Head, 若邻域fgMskMOG2白像素个数达到阈值，则OK
		vector<Point> res;

		//头部种子点（圆形模板匹配）：
		vector<Point> sdHeadVec = seedHead(dmat, false);
		size_t sdHeadVecSz = sdHeadVec.size();


		return res;
	}


	vector<Mat> findFgMasksUseHeadAndBodyCont(Mat &dmat, bool debugDraw /*= false*/){
		vector<Mat> resVec;

		return resVec;
	}//findFgMasksUseHeadAndBodyCont

	//@note 2015年7月9日23:32:37， currFgMskVec部分应解耦和，命名 mergeFgMaskVec【未解决】
	vector<Mat> trackingNoMove(Mat dmat, const vector<Mat> &prevFgMaskVec, const vector<Mat> &currFgMskVec, int noMoveThresh /*= 55*/, Mat moveMaskMat /*= Mat()*/, bool debugDraw /*= false*/)
{
		//static vector<Mat> prevFgMaskVec; //缓存上一帧抠像结果

		//int noMoveThresh = 55; //mm
		//seedNoMove 仅此处调用 o(╯□╰)o
		vector<Mat> sdsUsePreVec = seedNoMove(dmat, prevFgMaskVec, noMoveThresh); //√
		//vector<Mat> sdsUsePreVec = seedNoMove(dmat, currFgMskVec, noMoveThresh);

		if (debugDraw){
			//Mat tmp = dmat.clone();
			//if (sdsUsePreVec.size() > 0)
			//	tmp.setTo(0, sdsUsePreVec[0]==0);
			//imshow("tmp", tmp);
			Mat sdsUsePreVec2show = getHumansMask(sdsUsePreVec, dmat.size());
			imshow("sdsUsePreVec2show", sdsUsePreVec2show);
		}

		//尝试555mm疯长：不行！ 单人结果好，多人结果差，只能折中：
		int rgThresh = 355;
		Mat validMsk;
		Mat flrApartMsk = fetchFloorApartMask(dmat, debugDraw);

		//---------------计算所谓的potentialMask. @前景检测区域增长蒙板生成方案V0.1.docx
#if 0	//v1. 2015年7月10日21:44:17	之前第一版实现, 去除(地面+墙面)
		//存在问题： 沙发等“连通场景”无法分离背景

		//用不动点增长出的 mask-vec
		//改： flrApartMsk -> validMsk
		Mat bgMsk = fetchBgMskUseWallAndHeight(dmat);
		validMsk = flrApartMsk & (bgMsk == 0);

#elif 01	//v2. 2015年7月10日21:45:47	运动像素+不动像素 && 去除地面
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
#elif 1	//v3. 2015年7月12日14:18:57	运动像素+前一帧蒙板, 
		//实际上应该每个mask单独长成一个new-mask, 此处简化测试, N个长成一个:
		Mat prevFgMask_whole = Mat::zeros(dmat.size(), CV_8UC1);
		size_t prevFgMskVecSz = prevFgMaskVec.size();
		for (size_t i = 0; i < prevFgMskVecSz; i++){
			prevFgMask_whole += prevFgMaskVec[i];
		}
		validMsk = (prevFgMask_whole | moveMaskMat) & flrApartMsk;
#endif
		if (debugDraw)
			imshow("trackingNoMove.validMsk", validMsk);

#if	01	//---------------potentialMask 做减法
		Mat maxDepBgMask = zc::getMaxDepthBgMask(dmat, debugDraw);
		cv::morphologyEx(maxDepBgMask, maxDepBgMask, MORPH_CLOSE, getMorphKrnl());
		validMsk.setTo(0, maxDepBgMask);

		if (debugDraw){
			imshow("trackingNoMove.maxDepBgMask", maxDepBgMask);
			imshow("trackingNoMove.validMsk-final", validMsk);
		}
#endif

// 		Mat maskedDmat = dmat.clone();
// 		maskedDmat.setTo(0, fetchBgMskUseWallAndHeight(dmat));
		vector<Mat> noMoveFgMskVec = simpleRegionGrow(dmat, sdsUsePreVec, rgThresh, validMsk, debugDraw);



		//---------------2015年6月27日14:03:15	改成： 以跟踪为准，跟踪不到的算作新增
		//将跟踪结果作为base resVec, 初始也要查重、去重：
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

		//检查若无重叠，新增：
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


// 		//---------------fgMskVec 与 res 叠加，求“或”
// 		//vector<Mat> newResVec = currFgMskVec; //浅拷贝，可能后面会有副作用，改clone
// 		size_t currFgMskVecSz = currFgMskVec.size();
// 		vector<Mat> resVec(currFgMskVecSz);
// 		for (size_t i = 0; i < currFgMskVecSz; i++)
// 			resVec[i] = currFgMskVec[i].clone();
// 
// 		//目前算法： 楞次定律--合不愿合，分不愿分；谁是外循环不重要：
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

		//内外循环顺序与 newResVec 是谁(resVec or noMoveFgMskVec)有关
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
// 				//【暂定】只要相交，算作同一个mask
// 				if (intersect_area != 0){
// 					foundIntersect = true;
// 					//newResVec[k] = (noMoveMsk_i | resMsk_k); //原本 newResVec[k] == resMsk_k
// 					//break; //求或，不要跳出
// 
// 					//假定 tracking 结果可靠，赋值，不求或：
// 					newResVec[k] = noMoveMsk_i;
// 					//break;
// 				}
// 			}
// 
// 			if (!foundIntersect){//没法叠加，则新增
// 				newResVec.push_back(noMoveMsk_i); //pushback 不会影响上面的 k 迭代， 因为 k < origResSz
// 			}
// 		}//for

		//prevFgMaskVec = newResVec;

// 		if (debugDraw)
// 			cout << "findFgMasksUseBbox.part3.ts: " << clock() - begt << endl;

		return resVec;
	}//trackingNoMove

	//此版本只解决【XY视图】分离情况，对【XZ视图】分离不处理，所以不好
	vector<Mat> separateMasksXYview(Mat dmat, vector<Mat> &inMaskVec, bool debugDraw /*= false*/){
		vector<Mat> res;
		size_t mskVecSz = inMaskVec.size();
		//对每个 mask：
		for (size_t i = 0; i < mskVecSz; i++){
			Mat mski = inMaskVec[i];
			vector<vector<Point>> contours;
			findContours(mski.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

			size_t contsSz = contours.size();
			//CV_Assert(contsSz >= 1);
			if (contsSz == 0){
				//错误！调试：
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
// 			size_t xzBboxVecSz = xzBboxVec.size();//其实 ==contsSz
// 			for (size_t i = 0; i < xzBboxVecSz; i++){
// 				for (size_t j = i; j < xzBboxVecSz; j++){
// 					Rect intersectBbox = xzBboxVec[i] & xzBboxVec[j];
// 					//若存在两XZ孤立区域，分离。暂时简单认为 i, j就是两个最大轮廓
// 					if (intersectBbox.area() == 0){
// 						//fgMskIsHuman(dmat, )
// 					}
// 				}
// 			}
				
		}//for (size_t i = 0; i < mskVecSz; i++)
		
		return res;
	}//separateMasksXYview


	vector<Mat> separateMasksXZview(Mat dmat, vector<Mat> &inMaskVec, bool debugDraw /*= false*/){
		vector<Mat> res;
		size_t mskVecSz = inMaskVec.size();
		//对XYview 每个 mask：
		for (size_t i = 0; i < mskVecSz; i++){
			Mat mskXY_i = inMaskVec[i];

			Mat maskedDmat = dmat.clone();
			maskedDmat.setTo(0, mskXY_i == 0);
			//与之前的rgThresh保持一致：
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
			//CV_Assert(contXZ_size > 0);//×, 剔除墙面背景可能导致前景误删！
			if (contXZ_size == 0){
				//do-nothing?
			}
			else if (contXZ_size == 1){//XZ俯视图仅有一个轮廓，包括两部分上下分开的情境
				res.push_back(mskXY_i);
			}
			else{ //contXZ_size > 1, 包括两部分左右分开，前后分开情境
				//1. 方案一： 重新增长，长成几个独立部分算几个：
				//【缺陷】：可能分过头，要再merge合并，更麻烦低效，【舍弃】
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
// 				//不能全push，要过滤
// 				//res.insert(res.end(), subMskVec.begin(), subMskVec.end());
// 
// 				size_t subMskVecSz = subMskVec.size();
// 				for (size_t i = 0; i < subMskVecSz; i++){
// 					if (fgMskIsHuman(dmat, subMskVec[i]))
// 						res.push_back(subMskVec[i]);
// 				}

				//2. 方案二： contoursXZ反投影，每个cont必须仅得到一个humMsk：2015年6月27日23:30:01
				//对XZview每个cont：

				bool debugError = false;

				for (size_t i = 0; i < contXZ_size; i++){
					//待填充的结果：
					Mat newMskXY = Mat::zeros(dmat.size(), CV_8UC1);

					vector<Point> &contXZi = contoursXZ[i];
					//cont-mask, 内部全白，未去掉孔洞，但对下面求(dmin, dmax)足够了
					Mat cmskXZ_i = Mat::zeros(tdview.size(), CV_8UC1);
					drawContours(cmskXZ_i, contoursXZ, i, 255, -1);

					//左右边界：
					Rect bboxXZi = zc::boundingRect(contXZi);
					int left = bboxXZi.x,
						right = bboxXZi.x + bboxXZi.width - 1;//【注意】 -1

					//对contXZi从左到右每一col，反投影计算(dmin, dmax)
					for (int k = left; k <= right; k++){//【注意】 <=
						//tdview 上：
						//Mat colXZ_k = tdview.col(k);//×
						Mat colXZ_k = cmskXZ_i.col(k);//×

						vector<Point> nonZeroPts;
						if(countNonZero(colXZ_k))
							findNonZero(colXZ_k, nonZeroPts);
						Rect bboxCol_k = zc::boundingRect(nonZeroPts);
						int dmin = (bboxCol_k.y - 0.5) * rgThresh,
							dmax = (bboxCol_k.br().y + 0.5) * rgThresh - 1; //【注意】 +0.5, -1
						
						//对应 XYview 上：
						Mat colXY_k = maskedDmat.col(k);

						newMskXY.col(k) = (dmin <= colXY_k & colXY_k <= dmax) * UCHAR_MAX;

						if (debugError){
							imshow("newMskXY", newMskXY);
							waitKey(k < right ? 30 : 0);
						}
					}
					
					if (fgMskIsHuman(dmat, newMskXY))
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
		//对XYview 每个 mask：
		for (size_t i = 0; i < mskVecSz; i++){
			Mat mskXY_i = inMaskVec[i];

			Mat maskedDmat = dmat.clone();
			maskedDmat.setTo(0, mskXY_i == 0);

		}

		return res;
	}


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
			//接收 mat-type 必须为 8uc1, or 8uc3:
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

		//若 isNewFrame=false，则直接用prevFgMskMOG2作为当前帧前景
		static Mat prevFgMskMOG2; 
		static bool isFirstTime = true;

		if (isFirstTime || isNewFrame){

			double learningRate = -0.005;
			//接收 mat-type 必须为 8uc1, or 8uc3:
			pMOG2->apply(dm_show, fgMskMOG2, learningRate);
			
			if (debugDraw)
				imshow("seedBgsMOG2.fgMskMOG2", fgMskMOG2);

			//第一帧fgMskMOG2会因没有history所以全白，故不用apply结果：
			if (isFirstTime)
				fgMskMOG2 = Mat::zeros(dmat.size(), CV_8UC1);

			if (erodeRadius > 0){
				int anch = erodeRadius;
				Mat morphKrnl = getStructuringElement(MORPH_RECT, Size(2 * anch + 1, 2 * anch + 1), Point(anch, anch));
				erode(fgMskMOG2, fgMskMOG2, morphKrnl);
			}

			//去掉无效区域
			fgMskMOG2 &= (dmat != 0);

			//---------------
			prevFgMskMOG2 = fgMskMOG2;// .clone(); //不用 clone？
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
		Mat debug_mat; //用于输出观察的调试图
		if (debugDraw){
			_debug_mat.create(dmat.size(), CV_8UC1);
			debug_mat = _debug_mat.getMat();
			debug_mat.setTo(0);
		}

		bool isNewFrame = true;
		Mat fgMskMOG2 = seedUseBGS(dmat, isNewFrame, usePre, debugDraw);
		if (debugDraw){
			imshow("fgMskMOG2-erode", fgMskMOG2);
			//debug_mat = fgMskMOG2; //shallow-copy, ×
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

		//第一次调用时 prevDmat 全黑，因而diff > thresh，返回的 mask 应该也全黑：
// 		static Mat prevDmat = Mat::zeros(dmat.size(), dmat.type());
		Mat prevDmat = getPrevDmat();

		res &= (cv::abs(dmat - prevDmat) < thresh);

		//腐蚀一下，防止 dmax-dmin>thickLimit 情况（能杜绝？）
		Mat morphKrnl = getMorphKrnl(3);
		erode(res, res, morphKrnl);

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
		//若前一帧全黑，认为是第一帧，原样返回
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

	//@test 结果没问题，pass
	cv::Mat testMOG2Func(Mat dmat, int history /*= 100*/, double varThresh /*= 1*/, double learnRate /*= -1*/){
		Mat dmat8u;
		dmat.convertTo(dmat8u, CV_8UC1, 1.*UCHAR_MAX / MAX_VALID_DEPTH);

		bool detectShadows = false;
		static Ptr<BackgroundSubtractorMOG2> pMog2 = createBackgroundSubtractorMOG2(history, varThresh, detectShadows);
		Mat fgMskMog2;
		pMog2->apply(dmat8u, fgMskMog2);

		return fgMskMog2;
	}

	cv::Mat maskMoveAndNoMove(Mat dmat, vector<Mat> prevFgMaskVec, int noMoveThresh /*= 50*/, int history /*= 100*/, double varThresh /*= 1*/, double learnRate /*= -1*/, bool debugDraw /*= false*/, OutputArray _debug_mat /*= noArray()*/){
 
		Mat dmat8u;
		dmat.convertTo(dmat8u, CV_8UC1, 1.*UCHAR_MAX / MAX_VALID_DEPTH);

		Mat debug_mat;
		if (debugDraw){
			//调试mat改用彩色绘制：
			//_debug_mat.create(dmat.size(), CV_8UC4);
			_debug_mat.create(dmat.size(), CV_8UC3);
			debug_mat = _debug_mat.getMat();
			debug_mat.setTo(0);
		}
 
		Mat res;
 
		bool detectShadows = false;
		static Ptr<BackgroundSubtractorMOG2> pMog2 = createBackgroundSubtractorMOG2(history, varThresh, detectShadows);
		Mat fgMskMog2;
		pMog2->apply(dmat8u, fgMskMog2);
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
			//调试mat改用彩色绘制：
			//_debug_mat.create(dmat.size(), CV_8UC4);
			_debug_mat.create(dmat.size(), CV_8UC3);
			debug_mat = _debug_mat.getMat();
			debug_mat.setTo(0);
			debug_mat.setTo(cwhite, moveMask);
		}

#if 0	//v1版本， 用 seedNoMove
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
#elif 1	//v2版本， 用 moveMask 取反做 no-move-mask
		//其实≌v3，只是v2调试窗口看起来方便
		res = moveMask.clone();

		//整个mat的背景：
		Mat bgMask = (moveMask == 0);

		size_t prevFgMskVecSz = prevFgMaskVec.size();
		for (size_t i = 0; i < prevFgMskVecSz; i++){
			Mat prevFgMsk_i = prevFgMaskVec[i];
			//不动区域：前一帧msk与当前MOG背景msk求交：
			Mat noMoveMsk_i = prevFgMsk_i & bgMask;
			res |= noMoveMsk_i;

			if (debugDraw){
				debug_mat.setTo(cgreen, noMoveMsk_i);
				debug_mat.setTo(cred, noMoveMsk_i & moveMask); //必然不会出现，因为互斥
			}
		}

#elif 1 //v3. 2015年7月12日14:18:57	运动像素+前一帧蒙板, 实际等价于v2。v3效率会高吗？【未测试】
		Mat prevFgMask_whole = Mat::zeros(dmat.size(), CV_8UC1);
		size_t prevFgMskVecSz = prevFgMaskVec.size();
		for (size_t i = 0; i < prevFgMskVecSz; i++){
			prevFgMask_whole += prevFgMaskVec[i];
		}
		res = (prevFgMask_whole | moveMaskMat) & flrApartMsk;

#endif

#if 0	//尝试膨胀，可能不稳定，该长不长，没什么用
		int krnlRadius = 6;
		dilate(res, res, getMorphKrnl(krnlRadius));
#endif
		//【注意】一定扣除无效区域：
		res.setTo(0, dmat == 0);

		return res;
	}//maskMoveAndNoMove


	Mat getHumansMask(vector<Mat> masks, Size sz){
	//Mat getHumansMask( vector<Mat> masks ){
		Mat res = Mat::zeros(sz, CV_8UC1);

		//前景颜色随机：
		//RNG rng( 0xFFFFFFFF );
		
		//不随机， N个人， 前六个人画出来， 后面全黑
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

	//@brief 用全局的 vector<HumanObj> 画一个彩色 mask mat:
	Mat getHumansMask(Mat dmat, const vector<HumanObj> &humVec){
		Mat res;
		//8uc3 (or c4?), 彩色：
		Mat dmat8u;
		//不要一致灰度化，改为对比度最大化，
		//dmat.convertTo(dmat8u, CV_8U, 1.*UCHAR_MAX / MAX_VALID_DEPTH);
		normalize(dmat, dmat8u, 0, UCHAR_MAX, NORM_MINMAX, CV_8UC1);

		cvtColor(dmat8u, res, CV_GRAY2BGRA);

		size_t humSz = humVec.size();
		for (size_t i = 0; i < humSz; i++){
			HumanObj hum = humVec[i];
			Mat humMask = hum.getCurrMask();
			Scalar c = hum.getColor();
			//res.setTo(c, humMask); //实心难发现完全遮挡错误，改画轮廓
			vector<vector<Point>> contours;
			//findContours(humMask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
			findContours(humMask.clone(), contours, RETR_TREE, CHAIN_APPROX_NONE);

			//可能一个mask 有多段cont(e.g., 遮挡情境)，要画成一个色：
			drawContours(res, contours, -1, c, 1);

			//轮廓 遮挡重叠时看不出来，增加质心：
// 			size_t contSz = contours.size();
// 			for (size_t i = 0; i < contSz; i++){
// 				//质心一阶矩
// 				Moments mu = moments(contours[i]);

			vector<Point> flatConts;
			flatten(contours.begin(), contours.end(), back_inserter(flatConts));
			Moments mu = moments(flatConts);
			Point mc;
			if (abs(mu.m00) < 1e-8) //area is zero
				//mc = contours[i][0];
				mc = flatConts[0];
			else
				mc = Point(mu.m10 / mu.m00, mu.m01 / mu.m00);

			circle(res, mc, 5, c, 2);
// 			}
			
		}

		return res;
	}//getHumansMask


	cv::Mat getHumansMask2tdview(Mat dmat, const vector<HumanObj> &humVec){
		//自己实现的前景检测，得到mask：
		Mat tdview = zc::dmat2tdview_core(dmat);
		Mat tdview_show;
		tdview.convertTo(tdview_show, CV_8U);

		int history = 10;
		int diffThresh = 25;
		static MyBGSubtractor myBgsub_tdview(history, diffThresh);
		Mat myFgMsk_tdview = myBgsub_tdview.apply(tdview_show);


		//8uc3, 彩色：
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

			//实际灰度图：
			Mat dmat_hum_masked = dmat.clone();
			dmat_hum_masked.setTo(0, humMask == 0);

			Mat tdv_i = dmat2tdview_core(dmat_hum_masked, ratio);
			tdv_i.convertTo(tdv_i, CV_8U);

			Mat tdv_i_cn4;
			cvtColor(tdv_i, tdv_i_cn4, CV_GRAY2BGRA);
			res += tdv_i_cn4;

			//彩色bbox，颜色与 humObj 一致：
			Rect bbox = zc::boundingRect(tdv_i);
			rectangle(res, bbox, c, 1);

			//黄色高亮 moving 区域：
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

	//vec-heap 做id资源池
	static vector<int> idResPool;
	std::greater<int> fnGt = std::greater<int>();

	//要销毁一个 HumanObj 时， id放回资源池
	void pushPool(int id){
		static bool isFirstTime = true;

		if (isFirstTime){
			make_heap(idResPool.begin(), idResPool.end(), fnGt);
		}

		idResPool.push_back(id);
		push_heap(idResPool.begin(), idResPool.end(), fnGt);

	}//pushPool

	//返回最小可用id；若无，返回-1
	int getIdFromPool(){
		if (idResPool.empty())
			return -1;

		int resId = idResPool.front();
		pop_heap(idResPool.begin(), idResPool.end(), fnGt);
		idResPool.pop_back();

		return resId;
	}

	void getHumanObjVec(Mat &dmat, vector<Mat> fgMasks, vector<HumanObj> &outHumVec){
		//全局队列：
		//static vector<HumanObj> outHumVec;

		size_t fgMskSize = fgMasks.size(),
			humVecSize = outHumVec.size();

		Mat dmatClone = dmat.clone();

		//若尚未检测到过人，且单帧 fgMasks 有内容（其实不必要）:
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
		//若已检测到 HumanObj, 且单帧 fgMasks 无论有无内容可更新：
		else if (humVecSize > 0){//&& fgMskSize > 0){

			vector<bool> fgMsksUsedFlagVec(fgMskSize);

			vector<HumanObj>::iterator it = outHumVec.begin();
			while (it != outHumVec.end()){
				bool isUpdated = it->updateDmatAndMask(dmatClone, fgMasks, fgMsksUsedFlagVec);
				if (isUpdated)
					it++;
				else{
					//释放唯一id到资源池
					pushPool(it->getHumId());

					it = outHumVec.erase(it);
					cout << "------------------------------humVec.erase" << endl;
				}
			}//while

			//检查 fgMsksUsedFlagVec， 加入新人体：
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


	//@brief 输入单帧原始深度图， 得到最终 vec-mat-mask. process/run-a-frame
	vector<Mat> getFgMaskVec(Mat &dmat, int fid, bool debugDraw /*= false*/){
#define ZC_WRITE 0

		Mat dm_show
			, dmat8u
			, dmat8uc3
			;
		dmat.convertTo(dm_show, CV_8UC1, 1.*UCHAR_MAX / MAX_VALID_DEPTH);
		dmat8u = dm_show;

		static bool isFirstTime = true;
		static int oldFid = fid;
		static int frameCnt = -1;//fake frameCnt!!
		frameCnt++;


//#if ZC_CLEAN //对应 CAPG_SKEL_VERSION_0_9, 最终想要的干净的代码
		//---------------1. 预处理：
		//必须：初始化prevDmat
		zc::initPrevDmat(dmat);

		clock_t begttotal = clock();
		static vector<Mat> prevMaskVec;
		//static vector<HumanObj> humVec;

		//若视频帧循环，回到起始：
		if (fid <= oldFid){
			prevMaskVec.clear();
			//humVec.clear();
		}

		clock_t begt = clock();
		//A.去除背景 & 地面 & MOG2(or else?)背景减除：
		Mat bgMsk = zc::fetchBgMskUseWallAndHeight(dmat);
		Mat flrApartMask = zc::fetchFloorApartMask(dmat, false);
		Mat no_flr_wall_mask = flrApartMask & (bgMsk == 0);

		Mat maskedDmat = dmat.clone();
		maskedDmat.setTo(0, no_flr_wall_mask == 0);
		//只去地面，不去掉墙：
		//maskedDmat.setTo(0, flrApartMask == 0);
		cout << "aaa.maskedDmat.ts: " << clock() - begt << endl;
		if (debugDraw){
			Mat maskedDmat_show;
			normalize(maskedDmat, maskedDmat_show, 0, UCHAR_MAX, NORM_MINMAX, CV_8UC1);
			imshow("atmp-maskedDmat", maskedDmat_show);
		}

		//预处理就做MOG2背景减除:
		int noMoveThresh = 100;
		int history = 30;// 100;
		double varThresh = 0.3;// 1;
		double learnRate = -1;
		bool detectShadows = false;

		static Ptr<BackgroundSubtractorMOG2> pMog2;// = createBackgroundSubtractorMOG2(history, varThresh, detectShadows);
		if (fid <= oldFid){
#ifdef CV_VERSION_EPOCH //if opencv2.x

#elif CV_VERSION_MAJOR >= 3 //if opencv3

			//pMog2->clear(); //没效果
			pMog2 = createBackgroundSubtractorMOG2(history, varThresh, detectShadows);
#endif //CV_VERSION_EPOCH
		}
		
		Mat fgMskMog2;
		pMog2->apply(dmat8u, fgMskMog2);
		//【注意】一定扣除无效区域：
		fgMskMog2.setTo(0, dmat == 0);

		if (debugDraw){
			imshow("fgMskMog2", fgMskMog2);
		}
#if 0	//smooth-mask. @code: D:\opencv300\sources\samples\cpp\bgfg_segm.cpp
		int ksize = 11;
		double sigma = 3.5;
		GaussianBlur(fgMskMog2, fgMskMog2, Size(ksize, ksize), sigma);
		threshold(fgMskMog2, fgMskMog2, 10, 255, THRESH_BINARY);
		if (debugDraw){
			imshow("fgMskMog2-smooth", fgMskMog2);
		}
#endif


		//B.初步找前景，初始化，此处用的bbox方法：
		begt = clock();

		vector<Mat> fgMskVec;
		//fgMskVec = zc::findFgMasksUseWallAndHeight(dmat, debugDraw);
		Mat tmp;
		// 		fgMskVec = zc::findFgMasksUseBbox(maskedDmat, debugDraw, tmp);

		int rgThresh = 55;
#if 0 //XY-XZ-bbox 联合判定
		vector<vector<Point>> sdBboxVov = zc::seedUseBboxXyXz(maskedDmat, debugDraw, tmp);
		fgMskVec = zc::simpleRegionGrow(maskedDmat, sdBboxVov, rgThresh, flrApartMask, false);
#elif 0 //头身联合判定
		vector<vector<Point>> sdHeadBodyVov = zc::seedUseHeadAndBodyCont(dmat, debugDraw, tmp);
		fgMskVec = zc::simpleRegionGrow(maskedDmat, sdHeadBodyVov, rgThresh, flrApartMask, false);
#elif 1 //MOG2 运动检测方法， 去起始帧、大腐蚀，防噪声。
		int erodeRadius = 13;
		//"fid>1"判定方式在在线数据上不对:
		//Mat sdMoveMat = fid > 1 ? fgMskMog2.clone() : Mat::zeros(dmat.size(), CV_8UC1);
		Mat sdMoveMat = isFirstTime ? Mat::zeros(dmat.size(), CV_8UC1) : fgMskMog2.clone();
		erode(sdMoveMat, sdMoveMat, zc::getMorphKrnl(erodeRadius));

		isFirstTime = false; //放这里对吗？哪里还需要？

		rgThresh = 55;
		bool getMultiMasks = true;
		fgMskVec = zc::simpleRegionGrow(maskedDmat, sdMoveMat, rgThresh, flrApartMask, getMultiMasks);
#endif //N种初步增长方法
		fgMskVec = zc::bboxFilter(dmat, fgMskVec);

		cout << "bbb.findFgMasksUseBbox.ts: " << clock() - begt << endl;
		if (debugDraw){
			Mat btmp = zc::getHumansMask(fgMskVec, dmat.size());
			string txt = "fgMskVec.size: " + to_string(fgMskVec.size());
			putText(btmp, txt, Point( 0, 50 ), FONT_HERSHEY_PLAIN, 1, 255);
			imshow("btmp-bbox-init", btmp);
		}

		if (debugDraw){
			Mat prevMaskVec2msk = zc::getHumansMask(prevMaskVec, dmat.size());
			string txt = "prevMaskVec.size: " + to_string(prevMaskVec.size());
			putText(prevMaskVec2msk, txt, Point( 0, 50 ), FONT_HERSHEY_PLAIN, 1, 255);
			imshow("prevMaskVec2msk", prevMaskVec2msk);
		}

#if 0	//B2. 测试 maskMoveAndNoMove：
#if 0	//v1版本，舍弃
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
#elif 1	//v2版本 overload
		Mat tmp_mman;
		Mat testMsk = zc::maskMoveAndNoMove(dmat, fgMskMog2, prevMaskVec, noMoveThresh, debugDraw, tmp_mman);
		imshow("maskMoveAndNoMove", testMsk);
		if (!tmp_mman.empty())
			imshow("maskMoveAndNoMove-debug", tmp_mman);
		//imwrite("maskMoveAndNoMove_" + std::to_string((long long)fid) + ".jpg", tmp);

#endif
		Mat testMOG2mat = zc::testMOG2Func(dmat, history, varThresh, learnRate);
		imshow("testMOG2mat", testMOG2mat);
#endif	//B2. 测试 maskMoveAndNoMove

		//C.不动点跟踪，使前景补全、稳定：
		begt = clock();
		fgMskVec = zc::trackingNoMove(dmat, prevMaskVec, fgMskVec, noMoveThresh, fgMskMog2, debugDraw);
		cout << "ccc.trackingNoMove.ts: " << clock() - begt << endl;
		if (debugDraw){
			Mat ctmp = zc::getHumansMask(fgMskVec, dmat.size());
			string txt = "fgMskVec.size: " + to_string(fgMskVec.size());
			putText(ctmp, txt, Point( 0, 50 ), FONT_HERSHEY_PLAIN, 1, 255);
			imshow("ctmp-trackingNoMove", ctmp);
		}

#if 0	//D.后处理，对突然分离成几部分的前景，做分离、过滤处理
		begt = clock();

		//---------------保留sep-xy作为对比测试：
		//fgMskVec = zc::separateMasksXYview(dmat, fgMskVec, debugDraw);

		//---------------目前使用sep-xz：
		fgMskVec = zc::separateMasksXZview(dmat, fgMskVec, debugDraw);
		cout << "ddd.separateMasksXZview.ts: " << clock() - begt << endl;
		static int tSumt = 0;
		tSumt += (clock() - begttotal);
		cout << "find+tracking.rate: " << 1.*tSumt / (frameCnt + 1) << ", " << frameCnt << endl;

		if (debugDraw){
			Mat dtmp = zc::getHumansMask(fgMskVec, dmat.size());
			string txt = "fgMskVec.size: " + to_string(fgMskVec.size());
			putText(dtmp, txt, Point( 0, 50 ), FONT_HERSHEY_PLAIN, 1, 255);

			imshow("dtmp-separateMasks", dtmp);
		}
#endif	//D.后处理

// 		if (debugDraw){
// 			Mat humMsk = zc::getHumansMask(fgMskVec, dmat.size());
// 			// 			QtFont font = fontQt("Times");
// 			// 			cv::addText(humMsk, "some-text", { 55, 55 ), font);
// 			putText(humMsk, "fid: " + to_string(fid), Point( 0, 30 ), FONT_HERSHEY_PLAIN, 1, 255);
// 			imshow("humMsk", humMsk);
// 			if (ZC_WRITE)
// 				imwrite("humMsk_" + std::to_string((long long)fid) + ".jpg", humMsk);
// 		}

// 		zc::getHumanObjVec(dmat, fgMskVec, humVec);
// 		if (humVec.size() > 2)
// 			int dummy = 0;
// 		Mat humMsk彩色 = zc::getHumansMask(dmat, humVec);
// 		if (debugDraw){
// 			putText(humMsk彩色, "fid: " + to_string(fid), Point( 0, 30 ),
// 				FONT_HERSHEY_PLAIN, 1, { 255, 255, 255 });
// 
// 			putText(humMsk彩色, "humVec.size: " + to_string(humVec.size()), Point( 0, 50 ),
// 				FONT_HERSHEY_PLAIN, 1, { 255, 255, 255 });
// 
// 			imshow("humMsk-color", humMsk彩色);
// 			if (ZC_WRITE)
// 				imwrite("humMsk-color_" + std::to_string((long long)fid) + ".jpg", humMsk彩色);
// 		}

		oldFid = fid;
		prevMaskVec = fgMskVec;
		//必须： 更新 prevDmat
		zc::setPrevDmat(dmat);
//#endif //ZC_CLEAN

		return fgMskVec;
	}//getFgMaskVec

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
// 			Rect bbox = zc::boundingRect(contours[0]);
// 
// 			//类似 distMap2contours 的bbox 判定过滤：
// 			//1. 不能太厚; 2. bbox高度不能太小; 3. bbox 下沿不能高于半屏，因为人脚部位置较低
// 			bool notTooThick = dmax - dmin < 1500,
// 				pxHeightEnough = bbox.height >80,
// 				feetLowEnough = bbox.br().y > dmat.rows / 2;
// 
// 			cout << "notTooThick, pxHeightEnough, feetLowEnough: "
// 				<< notTooThick << ", "
// 				<< pxHeightEnough << ", "
// 				<< feetLowEnough << endl;
				

			//if (notTooThick && pxHeightEnough&& feetLowEnough)
			if (fgMskIsHuman(dmat, mski))
			{
				res.push_back(mski);
			}
		}
		return res;
	}//bboxFilter

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

// 		//类似 distMap2contours 的bbox 判定过滤：
// 		//1. 不能太厚; 2. bbox高度不能太小; 3. bbox 下沿不能高于半屏，因为人脚部位置较低
// 		bool notTooThick = dmax - dmin < thickLimit,
// 			pxHeightEnough = bbox.height >80,
// 			feetLowEnough = bbox.br().y > dmat.rows / 2;
// 
// 		//return notTooThick && pxHeightEnough && feetLowEnough;
// 		//2015年6月24日16:09:26： 
// 		//暂时放弃 notTooThick 判定， 因其可能导致不稳定
// 		return /*notTooThick && */pxHeightEnough && feetLowEnough;

		//即，暂时：2015年7月4日22:48:08
		return bboxIsHuman(dmat.size(), bbox);
	}//fgMskIsHuman

// 	bool fgMskIsHuman(Mat dmat, vector<Point> cont){
// 		Rect bbox = zc::boundingRect(cont);
// 
// 		//类似 distMap2contours 的bbox 判定过滤：
// 		//1. 不能太厚; 2. bbox高度不能太小; 3. bbox 下沿不能高于半屏，因为人脚部位置较低
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


#ifdef CV_VERSION_EPOCH
	const string featurePath = "../../feature";
	static BPRecognizer *bpr = nullptr;
	CapgSkeleton calcSkeleton( const Mat &dmat, const Mat &fgMsk ){
		Mat dm32s;
		dmat.convertTo(dm32s, CV_32SC1);
		//背景填充最大值，所以前景反而看起来黑色：
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

#pragma region //孙国飞头部种子点-wrapper
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

	vector<Point> seedHead(const Mat &dmat, bool debugDraw /*= false*/){
		CV_Assert(my_seg != nullptr);
		
		//return my_seg->seedSGF(dmat, debugDraw);
		return my_seg->head_location_method1(dmat, debugDraw);
	}

#pragma endregion //孙国飞头部种子点-wrapper

#pragma region //---------------HumanObj 成员函数们：

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

		for (int i = 0; i < 3; i++)
			_humColor[i] = rng.uniform(UCHAR_MAX/2, UCHAR_MAX);
		//_humColor[3] = 100; //半透明
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
		//深度图更新：
		_dmat = dmat;

		//改用前后帧白色区域求交：
		size_t fgMskVecSz = fgMaskVec.size();
		bool foundNewMask = false;
		for (size_t i = 0; i < fgMskVecSz; i++){
			Mat fgMsk = fgMaskVec[i];
			//求交：
			Mat currNewIntersect = _currMask & fgMsk;
			int intersectArea = countNonZero(currNewIntersect != 0),
				fgMskArea = countNonZero(fgMsk != 0);
			double percent = 0.5;

			//2015年6月27日22:10:18

			if (mskUsedFlags[i] == false
				//质心不可靠
				//&& fgMsk.at<uchar>(_currCenter) == UCHAR_MAX){
				//mask交集占比，也不可靠
				//&& (intersectArea > _currMaskArea * percent 
				//	|| intersectArea > fgMskArea * percent)

				//2015年6月27日22:08:20， mask交集区域深度差均值
				//【注】：intersectArea>0必要，否则 mean=0导致误判
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
					//标记某mask已被用过：
					mskUsedFlags[i] = true;

					break;
			}
		}

		//若因 fgMasks_重叠度太低未找到，则认为跟丢：
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
		//这里顺便计算骨架：
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

#pragma endregion //---------------HumanObj 成员函数们
}//zc

//从 opencv300 拷贝 zc::boundingRect 
namespace zc{
	//cv300 zc::boundingRect 可接受mask-mat
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

//---------------测试代码放这里
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
				//无效区域：
				if (z == 0){
// 					Point pt_w_bdr()
// 					int offset=

					//看的是边界填充了的 dmat_with_border：
					Rect nbRect(j, i, krnlSz, krnlSz);
					Mat nbMat = dmat_with_border(nbRect);
					//邻域内有效值数量达到阈值：
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


#pragma region //接要求, 自己实现帧差法背景减除MyBGSubtractor:
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
		//v1: 先写个低效版，每帧从新计算整个queue均值. 2015年7月8日14:16:24

		//用 32f 做计算， 防止精度丢失：
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
			_bgMat32f = frame32f;

		
		//再转回：
		//_bgMat32f.convertTo(_bgMat, _bgMat.type());

		CV_Assert(frame32f.depth() == CV_32F);
		_historyFrames.push(frame32f);
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
		
		//这是用来显示的 bgMat
		Mat bgMat_show;
		_bgMat32f.convertTo(bgMat_show, CV_8U);
		return bgMat_show;
	}
#pragma endregion //接要求, 自己实现帧差法背景减除MyBGSubtractor

	Mat getHumVecMaskHisto(Mat dmat, vector<HumanObj> humVec, vector<Mat> moveMaskVec, bool debugDraw /*= false*/){
		//用于绘制直方图, rgb彩色, N个直方图画在同一个mat上，各自颜色与对应HumanObj一致，
		//重叠区域颜色覆盖？ or 错开画？ or 颜色融合？ //暂时覆盖！
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

	//@brief 对 maskMat 统计X轴向上Y值统计直方图, 用color绘制
	//@return 一个彩色 histo-mat
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

	//@brief private-global-var, 各个像素最大历史深度值
	static Mat maxDmat;
	//@brief 历史最大深度dmat队列， 用于队首队尾diff，确保人一直后撤（区域深度增加）情境中，人体区域不被判为背景
	//队列后面必然比前面深度值大
	static queue<Mat> maxDmatQueue;
	const size_t maxDmatQueueCapacity = 30;

	Mat& getMaxDmat(Mat &dmat, bool debugDraw /*= false*/){
		if (maxDmat.empty())
			maxDmat = dmat.clone();

		return maxDmat;
	}

	Mat updateMaxDmat(Mat &dmat, bool debugDraw /*= false*/){
		//static Mat maxDmat = dmat.clone();

		//if (maxDmat.empty())
		//	maxDmat = dmat.clone();
		maxDmat = getMaxDmat(dmat, debugDraw);

		//计算新的最大深度矩阵：
		maxDmat = cv::max(dmat, maxDmat);

		//入队：
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


	Mat  getMaxDepthBgMask(Mat dmat, bool debugDraw /*= false*/){
		//Mat maxDmat = updateMaxDmat(dmat, debugDraw);
		maxDmat = getMaxDmat(dmat, debugDraw);
		//去掉伪前景，即“鬼影”。规则： 即使MOG2检测为运动前景，但因发现深度值等于(diff<10cm)历史最大深度，则扣除
		Mat maxDepBgMask = abs(maxDmat - dmat) < 10; //10mm
		updateMaxDmat(dmat, debugDraw);

		if (debugDraw)
			imshow("getMDBM.maxDepBgMask", maxDepBgMask);

		//对于一直后退，深度持续增加的人体，上面规则会导致扣除错误，所以构建“真实运动区域”
#if 10	//即，扣除伪鬼影

		Mat maxDmatMoveAreaMask;
#if 0	//若maxDmatQueueCapacity帧内【历史最大值】变化>50mm，认为运动了，非背景：
		Mat qFront = maxDmatQueue.front(),
			qBack = maxDmatQueue.back();
		maxDmatMoveAreaMask = qBack - qFront > 20;

#elif 1	//最深dmat序列背景建模
		int history = 30;
		double varThresh = 0.3;
		bool detectShadows = false;
		double bgRatio = 0.29;

#ifdef CV_VERSION_EPOCH //if opencv2.x
#elif CV_VERSION_MAJOR >= 3 //if opencv3

		static Ptr<BackgroundSubtractorMOG2> pMOG2_maxDmat = createBackgroundSubtractorMOG2(history, varThresh, detectShadows);
		Mat maxDmatFgMOG2;
		//32F不行，孙国飞说错了？【未核实】
		//Mat maxDmat32f;
		//maxDmat.convertTo(maxDmat32f, CV_32FC1);
		//maxDmatPMOG2->apply(maxDmat32f, maxDmatFgMOG2);
		Mat maxDmat8u;
		maxDmat.convertTo(maxDmat8u, CV_8UC1, 1.*UCHAR_MAX / MAX_VALID_DEPTH);
		pMOG2_maxDmat->setBackgroundRatio(bgRatio);
		pMOG2_maxDmat->apply(maxDmat8u, maxDmatFgMOG2);
		//imshow("maxDmatFgMOG2", maxDmatFgMOG2);
#endif //CV_VERSION_EPOCH

		int morphRadius = 2;
		morphologyEx(maxDmatFgMOG2, maxDmatFgMOG2, MORPH_DILATE, getMorphKrnl(morphRadius));

		maxDmatMoveAreaMask = maxDmatFgMOG2;
#endif

		maxDepBgMask &= (maxDmatMoveAreaMask == 0);
		if (debugDraw){
			imshow("getMDBM.moveAreaMask", maxDmatMoveAreaMask);
			imshow("getMDBM.maxDepBgMask-final", maxDepBgMask);
		}
#endif

		return maxDepBgMask;
	}//getMaxDepthBgMask

}//zc-测试代码放这里

