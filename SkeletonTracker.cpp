#include "SkeletonTracker.h"

//zhangxaochen:
#include "SimpleSilhouette.h"
#include "./sgf_seed/sgf_segment.h"

using namespace std;
//using namespace zc;

int g_ImgIndex = 0;


namespace sensekit { namespace plugins { namespace skeleton {

    const size_t SkeletonTracker::MAX_SKELETONS = 6;

    void SkeletonTracker::on_frame_ready(StreamReader& reader, Frame& frame)
    {
        if (!m_skeletonStream->has_connections())
            return; // don't waste cycles if no one is listening

        sensekit::DepthFrame depthFrame = frame.get<DepthFrame>();

        if (!depthFrame.is_valid())
            return;

        // do something cool
		int ww = depthFrame.resolutionX(),
			hh = depthFrame.resolutionY();
		float scaleFactor = ww * 1. / IMAGE_WIDTH;

		Mat dm(hh, ww, CV_16UC1, (void*)depthFrame.data()),
			dm_draw;
		normalize(dm, dm_draw, UCHAR_MAX, 0, NORM_MINMAX, CV_8UC1);
		//imshow("dm_draw", dm_draw);

		//时序上前后帧求 abs(sub)
		Mat currPreDiffMsk = zc::simpleMask(dm, ZCDEBUG);

		int veryDepth = -1;
		Point seed = zc::simpleSeed(dm, &veryDepth, ZCDEBUG);

		const string sgf_configPath = "../../../plugins/orbbec_skeleton/sgf_seed/config.txt",
			sgf_headTemplatePath = "../../../plugins/orbbec_skeleton/sgf_seed/headtemplate.bmp";
		
		vector<Point> sgfSeeds = zc::getHeadSeeds(dm, sgf_configPath, sgf_headTemplatePath, ZCDEBUG);

		circle(dm_draw, seed, 3, 255, 2);

		if (ZCDEBUG){
			Mat vdMat;
			findNonZero(dm == veryDepth, vdMat);
			//printf("vdMat.total(): %d\n", vdMat.total());
			for (int i = 0; i < vdMat.total(); i++)
				circle(dm_draw, vdMat.at<Point>(i), 0, 122, 1);

			// 			if(vdMat.total()>0){
			// 				circle(dm_draw, vdMat.at<Point>(0), 5, 255, 2);
			// 			}
			imshow("dm_draw", dm_draw);
		}

		//Mat fgMsk = zc::simpleRegionGrow(dm, seed, 333, ZCDEBUG);
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
			silImg = fgMsk,
			maskImg = currPreDiffMsk;
		bool useDense = false,
			usePre = false;

		BPRecognizer *bpr = zc::getBprAndLoadFeature();
		//bpr->predictAndMergeJoint(&silImg, tSklt, &maskImg, false, false, true);
		bpr->predictAndMergeJoint(&depthImg, tSklt, &maskImg, false, false, true);



        sensekit_skeletonframe_wrapper_t* skeletonFrame = m_skeletonStream->begin_write(depthFrame.frameIndex());

        if (skeletonFrame != nullptr)
        {
            skeletonFrame->frame.skeletons = reinterpret_cast<sensekit_skeleton_t*>(&(skeletonFrame->frame_data));
            skeletonFrame->frame.skeletonCount = SkeletonTracker::MAX_SKELETONS;

            sensekit_skeleton_t& skeleton = skeletonFrame->frame.skeletons[0];
            skeleton.trackingId = 1;
            skeleton.status = SENSEKIT_SKELETON_STATUS_TRACKED;
            skeleton.jointCount = SENSEKIT_MAX_JOINTS;

			//zhangxaochen:
			static CoordinateMapper mapper = reader.stream<sensekit::DepthStream>().coordinateMapper();

			for (size_t i = 0; i < 8; i++){
				skeleton.joints[i].status = SENSEKIT_JOINT_STATUS_TRACKED;
				skeleton.joints[i].jointType = static_cast<sensekit_joint_type>(i + 1); //e.g, SENSEKIT_JOINT_TYPE_LEFT_SHOULDER

// 				skeleton.joints[i].position.x = 555;// tSklt[i].x() * scaleFactor;
// 				skeleton.joints[i].position.y = 555;// tSklt[i].y() * scaleFactor;
// 				skeleton.joints[i].position.z = tSklt[i].z();
				skeleton.joints[i].position = *reinterpret_cast<sensekit_vector3f_t*>(
				//skeleton.joints[i].position = *(sensekit_vector3f_t*)(
					&mapper.convert_depth_to_world({ tSklt[i].x()*scaleFactor, tSklt[i].y()*scaleFactor, tSklt[i].z()*1.f }));
// 				float wx, wy, wz;
// 				mapper.convert_depth_to_world(tSklt[i].x() * scaleFactor, tSklt[i].y() * scaleFactor, tSklt[i].z(),
// 					&wx, &wy, &wz);
// 				skeleton.joints[i].position.x = wx;
// 				skeleton.joints[i].position.y = wy;
// 				skeleton.joints[i].position.z = wz;

			}

            for(int i = 1; i < MAX_SKELETONS; i++)
            {
                sensekit_skeleton_t& skeleton = skeletonFrame->frame.skeletons[i];
                skeleton.trackingId = i + 1;
                skeleton.status = SENSEKIT_SKELETON_STATUS_NOT_TRACKED;
            }

            m_skeletonStream->end_write();
        }
    }

}}}
