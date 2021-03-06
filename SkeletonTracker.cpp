#include "SkeletonTracker.h"

//zhangxaochen:
#include "SimpleSilhouette.h"
#include "./sgf_seed/sgf_segment.h"

using namespace std;
//using namespace zc;

#define ZCDEBUG 00
#define CAPG_SKEL_VERSION_0_1 //最初交付的代码版本

#undef CAPG_SKEL_VERSION_0_1 //改用 v0.9. 2015年7月1日00:07:01
#define CAPG_SKEL_VERSION_0_9 //对应 skeleton_test v0.9. 2015年6月30日23:59:18

#undef CAPG_SKEL_VERSION_0_9
#define CAPG_SKEL_VERSION_0_9_1	//增加方案二（限制增长mask），方案一增加后处理分割（by孙国飞）两种方法，三者对比测试

//#undef CAPG_SKEL_VERSION_0_9_1

int g_ImgIndex = 0;

#ifdef CAPG_SKEL_VERSION_0_1
void nmhSilhouAndSklt(Mat &dm){
	Mat dm_draw;
	normalize(dm, dm_draw, UCHAR_MAX, 0, NORM_MINMAX, CV_8UC1);
	//imshow("dm_draw", dm_draw);

	//时序上前后帧求 abs(sub)
	Mat currPreDiffMsk = zc::simpleMask(dm, ZCDEBUG);

	int veryDepth = -1;
	Point seed = zc::simpleSeed(dm, &veryDepth, ZCDEBUG);

	if (ZCDEBUG){
		circle(dm_draw, seed, 3, 255, 2);

		Mat vdMat;
		findNonZero(dm == veryDepth, vdMat);
		//printf("vdMat.total(): %d\n", vdMat.total());
		for (int i = 0; i < vdMat.total(); i++)
			circle(dm_draw, vdMat.at<Point>(i), 0, 122, 1);

		imshow("dm_draw", dm_draw);
	}

	//纵向简单去掉屏幕下缘 1/4，即地面,防止脚部与地面连成一片：
	Mat fgMsk = zc::simpleRegionGrow(dm, seed, 333, Rect(0, 0, dm.cols, dm.rows * 3 / 4), ZCDEBUG);
	//cout<<"simpleRegionGrow.t: "<<clock()-begt<<endl;

	//兼容林驰代码：
	CapgSkeleton tSklt;
	Mat tmpDm;
	dm.convertTo(tmpDm, CV_32SC1);
	//背景填充大数
	tmpDm.setTo(INT_MAX, fgMsk == 0);
	if (ZCDEBUG)
		imshow("tmpDm", tmpDm);

	IplImage depthImg = tmpDm,
		maskImg = currPreDiffMsk;
	bool useDense = false,
		useErode = false,
		usePre = false;

	const char *featurePath = "../../../plugins/orbbec_skeleton/feature";
	BPRecognizer *bpr = zc::getBprAndLoadFeature(featurePath);
	bpr->predictAndMergeJoint(&depthImg, tSklt, &maskImg, usePre, useErode, ZCDEBUG);
}//nmhSilhouAndSklt

#endif // CAPG_SKEL_VERSION_0_1


namespace sensekit { namespace plugins { namespace skeleton {

	const size_t SkeletonTracker::MAX_SKELETONS = 6;

	void SkeletonTracker::on_frame_ready(StreamReader& reader, Frame& frame)
	{
		cout << "SkeletonTracker::on_frame_ready" << endl;

		if (!m_skeletonStream->has_connections())
			return; // don't waste cycles if no one is listening

		sensekit::DepthFrame depthFrame = frame.get<DepthFrame>();

		if (!depthFrame.is_valid())
			return;

		// do something cool
		int ww = depthFrame.resolutionX(),
			hh = depthFrame.resolutionY();
		float scaleFactor = ww * 1. / QVGA_WIDTH;

		Mat dmat(hh, ww, CV_16UC1, (void*)depthFrame.data());
		resize(dmat, dmat, Size(ww / scaleFactor, hh / scaleFactor));
			
#ifdef CAPG_SKEL_VERSION_0_1

		//---------------1. NMH 简单种子点，silhouette； skeleton
		//nmhSilhouAndSklt(dm);


		//---------------2. 从多个种子点， 区域增长， 到提取多个骨架：
		Mat dm_draw;
		normalize(dmat, dm_draw, UCHAR_MAX, 0, NORM_MINMAX, CV_8UC1);

		const string sgf_configPath = "../../../plugins/orbbec_skeleton/sgf_seed/config.txt",
			sgf_headTemplatePath = "../../../plugins/orbbec_skeleton/sgf_seed/headtemplate.bmp";

		vector<Point> sgfSeeds = zc::getHeadSeeds(dmat, sgf_configPath, sgf_headTemplatePath, ZCDEBUG);
		if (ZCDEBUG){
			for (size_t i = 0; i < sgfSeeds.size(); i++){
				Point sdi = sgfSeeds[i];
				circle(dm_draw, sdi, 9, 255, 2);
			}
			imshow("1-dm_draw", dm_draw);
		}

		Mat flrApartMsk = zc::getFloorApartMask(dmat, ZCDEBUG);
		Mat sgfFgMsk = zc::simpleRegionGrow(dmat, sgfSeeds, 100, flrApartMsk, ZCDEBUG);
		if (ZCDEBUG)
			imshow("2-sgfFgMsk", sgfFgMsk);

		// 轮廓开操作，判定哪些是人，
		int anch = 3;
		Mat morphKrnl = getStructuringElement(MORPH_RECT, Size(anch * 2 + 1, anch * 2 + 1), Point(anch, anch));
		Mat sgfFgMsk_open;
		morphologyEx(sgfFgMsk, sgfFgMsk_open, CV_MOP_OPEN, morphKrnl);
		imshow("3-sgfFgMsk_open", sgfFgMsk_open);

		sgfFgMsk = sgfFgMsk_open;

		//兼容林驰代码：
		Mat tmpDm;
		dmat.convertTo(tmpDm, CV_32SC1);
		//背景填充最大值，所以前景反而黑色：
		tmpDm.setTo(INT_MAX, sgfFgMsk == 0);

		IplImage depthImg = tmpDm;
		bool useDense = false,
			useErode = false,
			usePre = true;

		const char *featurePath = "../../../plugins/orbbec_skeleton/feature";
		BPRecognizer *bpr = zc::getBprAndLoadFeature(featurePath);

		//因为predict是逐像素的，所以一个mask包含多人前景并不影响结果：
		Mat labelMat = bpr->predict(&depthImg, nullptr, false, false);

		if (ZCDEBUG){
			imshow("labelMat", labelMat);
			Mat rgbLabelMat = label_gray2rgb(labelMat);
			imshow("6-rgbLabelMat", rgbLabelMat);
		}

		//将多人轮廓分割为 n*单人 mat：
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(sgfFgMsk, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		zc::eraseNonHumanContours(contours);

		size_t contSize = contours.size();
		vector<vector<Point> > contours_poly(contSize);
		vector<Rect> boundRect(contSize);
		vector<Point2f> center(contSize);
		vector<float> radius(contSize);

		vector<CapgSkeleton> sklts;

		//调试绘制
		Mat contMat = Mat::zeros(sgfFgMsk.size(), CV_8UC1);

		for (size_t i = 0; i < contSize; i++){
			approxPolyDP((contours[i]), contours_poly[i], 3, true);

			//单人
			Mat soloContMask = Mat::zeros(sgfFgMsk.size(), CV_8UC1);
			drawContours(soloContMask, contours_poly, i, 255, -1);
			Mat soloLabelMat = labelMat.clone();
			soloLabelMat.setTo(BodyLabel_Background, soloContMask == 0);

			bool useErode = false,
				usePre = false;

			CapgSkeleton tSklt;
			bpr->mergeJoint(new IplImage(soloLabelMat), &depthImg, tSklt, useErode, usePre);
			sklts.push_back(tSklt);

			if (ZCDEBUG){
				boundRect[i] = boundingRect(contours_poly[i]);
				minEnclosingCircle(contours_poly[i], center[i], radius[i]);
				//调试绘制
				Scalar c(255);
				drawContours(contMat, contours_poly, i, c, -1);
				drawContours(contMat, contours, i, c);
				rectangle(contMat, boundRect[i].tl(), boundRect[i].br(), c);
				circle(contMat, center[i], radius[i], c);

				Mat soloRgbLabelMat = label_gray2rgb(soloLabelMat);
				imshow("5-soloContMask-" + std::to_string((long long)i), soloContMask);
				imshow("7-soloRgbLabelMat-" + std::to_string((long long)i), soloRgbLabelMat);
				//cout << "===tSklt.size(): " << tSklt.size() << ", " << contours[i].size() << ", " << contours_poly[i].size() << endl;
			}
		}
		cout << "skltVec.size(): " << sklts.size() << endl;

		Mat skCanvas = Mat::ones(sgfFgMsk.size(), CV_8UC1)*UCHAR_MAX;
		zc::drawSkeletons(skCanvas, sklts, -1);
		if (ZCDEBUG){
			imshow("4-contMat", contMat);
			imshow("8-skCanvas", skCanvas);
		}
		//+++++++++++++++
#endif // CAPG_SKEL_VERSION_0_1

#ifdef CAPG_SKEL_VERSION_0_9_1
		sensekit_frame_index_t fid = depthFrame.frameIndex();
		vector<Mat> fgMskVec = zc::getFgMaskVec(dmat, fid, ZCDEBUG);
		zc::getHumanObjVec(dmat, fgMskVec, humVec);

		bool debugWrite = false;
		zc::debugDrawHumVec(dmat, fgMskVec, humVec, fid, debugWrite);
#endif // CAPG_SKEL_VERSION_0_9_1

		sensekit_skeletonframe_wrapper_t* skeletonFrame = m_skeletonStream->begin_write(depthFrame.frameIndex());

		if (skeletonFrame != nullptr)
		{
			skeletonFrame->frame.skeletons = reinterpret_cast<sensekit_skeleton_t*>(&(skeletonFrame->frame_data));
			skeletonFrame->frame.skeletonCount = SkeletonTracker::MAX_SKELETONS;

#if 0	//original-demo
			sensekit_skeleton_t& skeleton = skeletonFrame->frame.skeletons[0];
			skeleton.trackingId = 1;
			skeleton.status = SENSEKIT_SKELETON_STATUS_TRACKED;
			skeleton.jointCount = SENSEKIT_MAX_JOINTS;
#endif	//original-demo

			//zhangxaochen:
			static CoordinateMapper mapper = reader.stream<sensekit::DepthStream>().coordinateMapper();

#ifdef CAPG_SKEL_VERSION_0_1

			//---------------单人时，nmhSilhouAndSklt
// 			for (size_t i = 0; i < 8; i++){
// 				skeleton.joints[i].status = SENSEKIT_JOINT_STATUS_TRACKED;
// 				skeleton.joints[i].jointType = static_cast<sensekit_joint_type>(i + 1); //e.g, SENSEKIT_JOINT_TYPE_LEFT_SHOULDER
// 
// // 				skeleton.joints[i].position.x = 555;// tSklt[i].x() * scaleFactor;
// // 				skeleton.joints[i].position.y = 555;// tSklt[i].y() * scaleFactor;
// // 				skeleton.joints[i].position.z = tSklt[i].z();
// 				skeleton.joints[i].position = *reinterpret_cast<sensekit_vector3f_t*>(
// 				//skeleton.joints[i].position = *(sensekit_vector3f_t*)(
// 					&mapper.convert_depth_to_world({ tSklt[i].x()*scaleFactor, tSklt[i].y()*scaleFactor, tSklt[i].z()*1.f }));
// // 				float wx, wy, wz;
// // 				mapper.convert_depth_to_world(tSklt[i].x() * scaleFactor, tSklt[i].y() * scaleFactor, tSklt[i].z(),
// // 					&wx, &wy, &wz);
// // 				skeleton.joints[i].position.x = wx;
// // 				skeleton.joints[i].position.y = wy;
// // 				skeleton.joints[i].position.z = wz;
// 
// 			}
// 
// 			for (int i = 1; i < MAX_SKELETONS; i++)
// 			{
// 				sensekit_skeleton_t& skeleton = skeletonFrame->frame.skeletons[i];
// 				skeleton.trackingId = i + 1;
// 				skeleton.status = SENSEKIT_SKELETON_STATUS_NOT_TRACKED;
// 			}


			//---------------多人时：
			for (int i = 0; i < min(sklts.size(), MAX_SKELETONS); i++){
				CapgSkeleton tSklt = sklts[i];

				sensekit_skeleton_t& skeleton = skeletonFrame->frame.skeletons[i];
				skeleton.trackingId = i + 1;
				skeleton.status = SENSEKIT_SKELETON_STATUS_TRACKED;
				skeleton.jointCount = SENSEKIT_MAX_JOINTS;

				static CoordinateMapper mapper = reader.stream<sensekit::DepthStream>().coordinateMapper();

				for (size_t i = 0; i < 8; i++){
					skeleton.joints[i].status = SENSEKIT_JOINT_STATUS_TRACKED;
					skeleton.joints[i].jointType = static_cast<sensekit_joint_type>(i + 1); //e.g, SENSEKIT_JOINT_TYPE_LEFT_SHOULDER
					skeleton.joints[i].position = *reinterpret_cast<sensekit_vector3f_t*>(
						//skeleton.joints[i].position = *(sensekit_vector3f_t*)(
						&mapper.convert_depth_to_world({ tSklt[i].x()*scaleFactor, tSklt[i].y()*scaleFactor, tSklt[i].z()*1.f }));

				}

			}

#endif // CAPG_SKEL_VERSION_0_1

#ifdef CAPG_SKEL_VERSION_0_9_1_BUGGY
			size_t humVecSz = humVec.size();
			for (int i = 0; i < min(humVecSz, MAX_SKELETONS); i++){
				sensekit_skeleton_t& skeleton = skeletonFrame->frame.skeletons[i];

				bool isIdFound = false;
				for (int k = 0; k < humVecSz; k++){
					HumanObj humObj = humVec[k];
					CapgSkeleton sklt = humObj.getSkeleton();
					int humId = humObj.getHumId();

					if (i == humId){
						isIdFound = true;

						skeleton.trackingId = humId;
						skeleton.status = SENSEKIT_SKELETON_STATUS_TRACKED;
						skeleton.jointCount = SENSEKIT_MAX_JOINTS;

						cout << "==================" << humId << endl;
						for (size_t j = 0; j < 8; j++){
							skeleton.joints[j].status = SENSEKIT_JOINT_STATUS_TRACKED;
							skeleton.joints[j].jointType = static_cast<sensekit_joint_type>(j + 1); //e.g, SENSEKIT_JOINT_TYPE_LEFT_SHOULDER
							//Vector3f pt = { sklt[i].x()*scaleFactor, sklt[i].y()*scaleFactor, sklt[i].z()*1.f };
							Vector3f pt(sklt[j].x() * scaleFactor, sklt[j].y() * scaleFactor, sklt[j].z() * 1.f);
							cout << "[ " << pt.x << ", " << pt.y << ", " << pt.z << " ]" << endl;

							//pt in world-scale-coord
							Vector3f ptw = mapper.convert_depth_to_world(pt);
							cout << "ws: [ " << ptw.x << ", " << ptw.y << ", " << ptw.z << " ]" << endl;

							//skeleton.joints[i].position = *(sensekit_vector3f_t*)(
							skeleton.joints[j].position = *reinterpret_cast<sensekit_vector3f_t*>(&ptw);
						}

						break;
					}
				}//for, k < humVecSz;
				if (!isIdFound)
					skeleton.status = SENSEKIT_SKELETON_STATUS_NOT_TRACKED;
			}
#endif // CAPG_SKEL_VERSION_0_9_1_BUGGY

#ifdef CAPG_SKEL_VERSION_0_9_1
			size_t humVecSz = humVec.size();
			for (int i = 0; i < min(humVecSz, MAX_SKELETONS); i++){
				sensekit_skeleton_t& skeleton = skeletonFrame->frame.skeletons[i];
				HumanObj &humObj = humVec[i];
				CapgSkeleton sklt = humObj.getSkeleton();
				int humId = humObj.getHumId();

				skeleton.status = SENSEKIT_SKELETON_STATUS_TRACKED;
				skeleton.trackingId = humId;
				skeleton.jointCount = SENSEKIT_MAX_JOINTS;

				cout << "==================" << humId << endl;
				for (size_t j = 0; j < 8; j++){
					skeleton.joints[j].status = SENSEKIT_JOINT_STATUS_TRACKED;
					skeleton.joints[j].jointType = static_cast<sensekit_joint_type>(j + 1); //e.g, SENSEKIT_JOINT_TYPE_LEFT_SHOULDER
					//Vector3f pt = { sklt[i].x()*scaleFactor, sklt[i].y()*scaleFactor, sklt[i].z()*1.f };
					Vector3f pt(sklt[j].x() * scaleFactor, sklt[j].y() * scaleFactor, sklt[j].z() * 1.f);
					cout << "[ " << pt.x << ", " << pt.y << ", " << pt.z << " ]" << endl;

					//pt in world-scale-coord
					Vector3f ptw = mapper.convert_depth_to_world(pt);
					cout << "ws: [ " << ptw.x << ", " << ptw.y << ", " << ptw.z << " ]" << endl;

					//skeleton.joints[i].position = *(sensekit_vector3f_t*)(
					skeleton.joints[j].position = *reinterpret_cast<sensekit_vector3f_t*>(&ptw);
				}


			}

			for (int i = min(humVecSz, MAX_SKELETONS); i < MAX_SKELETONS; i++){
				sensekit_skeleton_t& skeleton = skeletonFrame->frame.skeletons[i];
				skeleton.status = SENSEKIT_SKELETON_STATUS_NOT_TRACKED;
			}
#endif // CAPG_SKEL_VERSION_0_9_1


			m_skeletonStream->end_write();
		}
	}

}}}
