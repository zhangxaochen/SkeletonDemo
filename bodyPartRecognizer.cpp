#include "bodyPartRecognizer.h"
#include "logCat.h"
//#include "MotionLog.h"

#ifndef LINC_DEBUG
//#include "util.h"
#endif
#include <time.h>
#include "Macros.h"


#ifdef LINC_DEBUG
#include <QTime>
#include "cudaRecog.h"
#endif

cv::RNG* BPRecognizer::_rng = 0;

BPRecognizer::BPRecognizer()
{
	_moduleID = LogCat::getInstancePtr()->registerModule("BPRecognizer");
	_treeNum = 0;

	_trainSamples = 0;
	_testSamples = 0;

	_value_mat = 0;
	_response_mat = 0;
	_var_type_mat = 0;
	_active_var_mask = 0;

	_sample_count = 0;
	_value_count = 0;

	_rng = new RNG(cvGetTickCount());

	DepthSample::createColorMap();
	DepthSample::createLabelMap(_para.merge_flag);

	_preLabelImg = 0;
	_curLabelImg = 0;

	_denseFeature = 0;
	_denseForest = 0;
	_denseRoot = 0;

	_erodeWin = 2;
	_preErode = 0;

	_predictStep = 2;
    //_predictStep = 1;

	clearPre();

#ifdef LINC_DEBUG
	int ret = initCuda(0);
	if(ret == 0){
		ERROR_MSG("There is no device. Turn CUDA algorithm off.\n", _moduleID);
	}else if(ret == -1){
		ERROR_MSG("There is no device supporting CUDA. Turn CUDA algorithm off.\n", _moduleID);
	}else if(ret == 1){
		IMPORT_MSG("CUDA is initialized successfully.\n\n", _moduleID);
	}
#endif

}

BPRecognizer::~BPRecognizer()
{
	clear();
	delete _rng;

	if(_preLabelImg) cvReleaseImage(&_preLabelImg);
	if(_curLabelImg) cvReleaseImage(&_curLabelImg);

	if(_denseFeature) delete [] _denseFeature;
	if(_denseForest) delete [] _denseForest;
	if(_denseRoot) delete [] _denseRoot;

	if(_preErode){ cvReleaseImage(&_preErode); _preErode = 0; }
}

void BPRecognizer::clear()
{
	for(vector<vector<DepthFeature*>*>::iterator it=_features.begin();
		it!=_features.end(); it++)
	{
		vector<DepthFeature*>& fset= **it;
		for(vector<DepthFeature*>::iterator itt=fset.begin(); itt!=fset.end(); itt++){
			delete *itt;
		}
		fset.clear();
		delete *it;
	}
	_features.clear();

	for(vector<BPRTree*>::iterator it=_forest.begin();
		it!=_forest.end(); it++){
			delete *it;
	}
	_forest.clear();

	clearTrainMat();
}

void BPRecognizer::setPara(BPRecogPara& para)
{
	_para = para;

	DepthSample::createLabelMap(_para.merge_flag);
	DepthSample::createColorMap();
}

BPRecogPara& BPRecognizer::getPara()
{
	return _para;
}

void BPRecognizer::clearTrainMat()
{
	if(_value_mat){ cvReleaseMat(&_value_mat); _value_mat = 0; }
	if(_response_mat){ cvReleaseMat(&_response_mat); _response_mat = 0; }
	if(_var_type_mat){ cvReleaseMat(&_var_type_mat); _var_type_mat = 0; }
	if(_active_var_mask){ cvReleaseMat(&_active_var_mask); _active_var_mask = 0; }
}

void BPRecognizer::addNewBestFeature(DepthFeature* f)
{
	vector<DepthFeature*>& fset = *_features[_treeNum];
	fset.push_back(f->clone());
}

void BPRecognizer::addNewNullFeature()
{
	vector<DepthFeature*>& fset = *_features[_treeNum];
	fset.push_back(0);
}

IplImage* BPRecognizer::predict(IplImage* depthImg, IplImage* maskImg, bool useDense, bool usePre)
{
	PLAIN_MSG("-- Predict Start", _moduleID);
	int valueSize = _para.theta_per_tree;
	int votes[31];
	vector<BodyLabel> voteParts;
	int halfVote = _treeNum / 2;
	if(_curLabelImg == 0){
		_curLabelImg = cvCreateImage(cvSize(depthImg->width, depthImg->height), IPL_DEPTH_8U, 1);
	}else{
		char* temp = _preLabelImg->imageData;
		_preLabelImg->imageData = _curLabelImg->imageData;
		_curLabelImg->imageData = temp;
	}

	int notBGCount = 0;
	int copyPreCount = 0;
	int copyStepCount = 0;

	int actualPredict = 0;
	int unknownCount = 0;
	int replaceErrorCount = 0;
	int twinTopCount = 0;
	int spyCount = 0;

    //---sunguofei 2015.11.16
    weight_mat.clear();
    weight_mat.resize(31);
    dis_weight_mat.clear();
    dis_weight_mat.resize(31);
    for (int i=0;i<31;++i)
    {
        weight_mat[i]=Mat::zeros(depthImg->height,depthImg->width,CV_32F);
        dis_weight_mat[i]=Mat::zeros(depthImg->height,depthImg->width,CV_32F);
    }

	for(int y=0; y<depthImg->height; y++){
		unsigned int* d_data = (unsigned int*)(depthImg->imageData + y*depthImg->widthStep);
		BodyLabel* l_data = (BodyLabel*)(_curLabelImg->imageData + y*_curLabelImg->widthStep);
		BodyLabel* l_data_2 = (BodyLabel*)(_curLabelImg->imageData + (y - y%_predictStep)*_curLabelImg->widthStep);
		uchar* mask_data = maskImg ? (uchar*)(maskImg->imageData + y*maskImg->widthStep) : 0;
		BodyLabel* pre_data = _preAvailable && usePre ? (BodyLabel*)(_preLabelImg->imageData + y*_preLabelImg->widthStep) : 0;
		uchar* e_data = _preAvailable ? (uchar*)(_preErode->imageData + y*_preErode->widthStep) : 0;
		for(int x=0; x<depthImg->width; x++){
            //---sunguofei 2015.11.18

			BodyLabel max_label = BodyLabel_Unknown;
			if(d_data[x] >= BACKGROUNG){
				max_label = BodyLabel_Background;
			}else{
				notBGCount++;
				int xx = x - x%_predictStep, yy = y - y%_predictStep;
				if(xx == x && yy == y){
					memset(votes, 0, sizeof(int)*31);
					voteParts.clear();
					int max_vote = 0;
					int nvotes, class_idx;
					int startTreeIdx = 0;
					if(pre_data && e_data[x] == NotErode){
						notBGCount++;
						// if have mask, check mask. copy previous data only when mask data = 0;
						// otherwise, copy previous data
						if(mask_data){
							if(mask_data[x] == 0){
								goto CopyPre;
							}else{
								goto PredicActually;
							}
						}else{
							goto CopyPre;
						}
CopyPre:
						int class_idx;
						int treeIdx = 0;
						if(useDense){
							class_idx = predicDense(depthImg, cvPoint(x, y), treeIdx);
						}else{
							vector<DepthFeature*>& fset = *_features[treeIdx];
							int fcount = fset.size();
							BPRTree* tree = _forest[treeIdx];
                            string s;
							CvDTreeNode* predicted_node = tree->predict(depthImg, cvPoint(x, y), fset, s);
							class_idx = predicted_node->value;
						}
						if(class_idx == pre_data[x]){
							max_label = class_idx;
							copyPreCount++;

                            //---sunguofei 2015.11.16
                            weight_mat[class_idx].at<float>(y,x) = 1.0;
                            //2015.11.18
                            if (dis_weight_mat[class_idx].at<float>(y,x)==0)
                            {
                                float weight=500;
                                if (joints.size()==31)
                                {
                                    if (x==joints[class_idx].x()&&y==joints[class_idx].y())
                                    {
                                        dis_weight_mat[class_idx].at<float>(y,x)=weight;
                                    }
                                    else
                                    {
                                        float dis=sqrt(pow(joints[class_idx].x()-x,2.0)+pow(joints[class_idx].y()-y,2.0));
                                        dis_weight_mat[class_idx].at<float>(y,x)=weight/dis;
                                    }
                                }
                            }

						}else{
							nvotes = ++votes[class_idx];
							if(nvotes==1) voteParts.push_back(class_idx);
							if(nvotes > max_vote){
								max_vote = nvotes;
								max_label = class_idx;
							}
							startTreeIdx = 1;
							spyCount++;

                            //---sunguofei 2015.11.16
                            weight_mat[class_idx].at<float>(y,x) += 1.0/_treeNum;
                            //2015.11.18
                            if (dis_weight_mat[class_idx].at<float>(y,x)==0)
                            {
                                float weight=500;
                                if (joints.size()==31)
                                {
                                    if (x==joints[class_idx].x()&&y==joints[class_idx].y())
                                    {
                                        dis_weight_mat[class_idx].at<float>(y,x)=weight;
                                    }
                                    else
                                    {
                                        float dis=sqrt(pow(joints[class_idx].x()-x,2.0)+pow(joints[class_idx].y()-y,2.0));
                                        dis_weight_mat[class_idx].at<float>(y,x)=weight*exp(-(dis*dis) / (50));//distance weight is based on the gauss kernel
                                    }
                                }
                            }

							goto PredicActually;
						}

					}else{

PredicActually:
						// predict using random forest
						actualPredict++;
                        //---sunguofei 2015.12.10
                        vector<string> leaf_id(_treeNum);
                        vector<int> leaf_value(_treeNum);
						for(int i=startTreeIdx; i<_treeNum; i++){
							if(useDense){
								class_idx = predicDense(depthImg, cvPoint(x, y), i);
							}else{
								vector<DepthFeature*>& fset = *_features[i];
								int fcount = fset.size();
								BPRTree* tree = _forest[i];
                                string s;
								CvDTreeNode* predicted_node = tree->predict(depthImg, cvPoint(x, y), fset, s);
								class_idx = predicted_node->value;

                                //---sunguofei 2015.12.10
                                leaf_id[i]=s;leaf_value[i]=class_idx;
//                                 if (s=="11111111100"&&i==4)
//                                 {
//                                     cout<<s<<endl;
//                                 }
                               /*
                                map<string,vector<vector<int>>>::iterator it;
                                it=node_ojr[i].find(s);
                                if (it!=node_ojr[i].end())
                                {
                                    double p_x,p_y,p_z;
                                    p_z=d_data[x]*262.5/329;
                                    p_x=(159.5-x)*p_z/262.5;
                                    p_y=(119.5-y)*p_z/262.5;
                                    for (int j=0;j<18;++j)
                                    {
                                        if (it->second[j][0]>50)
                                        {
                                            offset_weight[j].push_back(it->second[j][0]);
                                            vector<double> tmp(3);
                                            tmp[0]=p_x+it->second[j][1];tmp[1]=p_y+it->second[j][2];tmp[2]=p_z+it->second[j][3];
                                            if (j==13 && tmp[1]<0)
                                            {
                                                int xxxxxxx=0;
                                            }
                                            //if(i==0)
                                            offset_result[j].push_back(tmp);
                                        }
                                        if (it->second[j][4]>50)
                                        {
                                            offset_weight[j].push_back(it->second[j][4]);
                                            vector<double> tmp(3);
                                            tmp[0]=p_x+it->second[j][5];tmp[1]=p_y+it->second[j][6];tmp[2]=p_z+it->second[j][7];
                                            //if(i==0)
                                            offset_result[j].push_back(tmp);
                                        }
                                    }
                                }
                                */
                                //---sunguofei 2015.11.16
                                weight_mat[class_idx].at<float>(y,x) += 1.0/_treeNum;
                                //2015.11.18
                                if (dis_weight_mat[class_idx].at<float>(y,x)==0)
                                {
                                    float weight=500;
                                    if (joints.size()==31)
                                    {
                                        if (x==joints[class_idx].x()&&y==joints[class_idx].y())
                                        {
                                            dis_weight_mat[class_idx].at<float>(y,x)=weight;
                                        }
                                        else
                                        {
                                            float dis=sqrt(pow(joints[class_idx].x()-x,2.0)+pow(joints[class_idx].y()-y,2.0));
                                            dis_weight_mat[class_idx].at<float>(y,x)=weight*exp(-(dis*dis) / (50));
                                        }
                                    }
                                }

							}
							nvotes = ++votes[class_idx];
							if(nvotes==1) voteParts.push_back(class_idx);
							if(nvotes > max_vote){
								max_vote = nvotes;
								max_label = class_idx;
							}
						}
						if(max_vote >= halfVote && max_vote != _treeNum){
							BodyLabel choice2 = max_label; int max_vote2 = 0;
							for(int j=0; j<voteParts.size(); j++){
								BodyLabel l = voteParts[j];
								if(l != max_label && votes[l] > max_vote2){
									choice2 = l;
									max_vote2 = votes[l];
								}
							}
							if(pre_data && e_data[x] == ErodeForError){
								if(max_label == pre_data[x]){
									max_label = choice2;
									replaceErrorCount++;
								}
							}else if(max_vote == max_vote2){
								int code = (*_rng)(2);
								if(code == 0){
									max_vote = max_vote2;
								}
								twinTopCount++;
							}
                            //---sunguofei 2015.12.10
                            for (int i=0;i<leaf_id.size();++i)
                            {
                                if (leaf_value[i]==max_label)
                                {
                                    map<string,vector<vector<int>>>::iterator it;
                                    it=node_ojr[i].find(leaf_id[i]);
                                    if (it!=node_ojr[i].end())
                                    {
                                        double p_x,p_y,p_z;
                                        p_z=d_data[x]*262.5/329;
                                        p_x=(-159.5+x)*p_z/262.5;
                                        p_y=(119.5-y)*p_z/262.5;
                                        for (int j=0;j<18;++j)
                                        {
                                            if (it->second[j][0]>30)
                                            {
                                                offset_weight[j].push_back((it->second[j][0])*pow(p_z/1000.0,2.0));
                                                vector<double> tmp(3);
                                                tmp[0]=p_x+it->second[j][1];tmp[1]=p_y+it->second[j][2];tmp[2]=p_z+it->second[j][3];
                                                offset_result[j].push_back(tmp);
                                            }
                                            
                                            if (it->second[j][4]>30)
                                            {
                                                offset_weight[j].push_back((it->second[j][4])*pow(p_z/1000.0,2.0));
                                                vector<double> tmp(3);
                                                tmp[0]=p_x+it->second[j][5];tmp[1]=p_y+it->second[j][6];tmp[2]=p_z+it->second[j][7];
                                                offset_result[j].push_back(tmp);
                                            }
                                            
                                        }
                                    }
                                }
                            }
						}else if(max_vote < halfVote){
							max_label = BodyLabel_Unknown;
							unknownCount++;
						}
                        //---sunguofei 2015.12.10
					}

				}else{
					max_label = l_data_2[xx];
					copyStepCount++;
				}
			}

			l_data[x] = max_label;			
		}
	}
	if(_preLabelImg == 0){
		_preLabelImg = cvCloneImage(_curLabelImg);
	}
	PLAIN_MSG(string("Not BG: ")+LogCat::to_string(notBGCount)+
		"  Copy previous: "+LogCat::to_string(copyPreCount)+
		"  Copy step: "+LogCat::to_string(copyStepCount), _moduleID);
	PLAIN_MSG(string("Actual predict: ")+LogCat::to_string(actualPredict)+
		"  unknown: "+LogCat::to_string(unknownCount)+
		"  replace error: "+LogCat::to_string(replaceErrorCount)+
		"  twin top: "+LogCat::to_string(twinTopCount)+
		"  spy: "+LogCat::to_string(spyCount), _moduleID);

	PLAIN_MSG("-- Predict End", _moduleID);

	return _curLabelImg;
}
//---sunguofei 2015.12.9
void BPRecognizer::load_node_ojr(string path)
{
    for (int i=0;i<5;++i)
    {
        ifstream f_in;
        string file_path=path+"/meanshift_result"+string(1,'0'+i)+".txt";
        f_in.open(file_path);
        string s;
        f_in>>s;
        while (s!="end")
        {
            string node_id=s;
            vector<vector<int>> node_value;
            for (int j=0;j<18;++j)
            {
                f_in>>s;
                int joint_id;f_in>>joint_id;
                vector<int> joint_offset(8);
                for (int k=0;k<8;++k)
                {
                    f_in>>joint_offset[k];
                }
                node_value.push_back(joint_offset);
            }
            node_ojr[i].insert(map<string,vector<vector<int>>>::value_type(node_id,node_value));
            f_in>>s;
        }
        f_in.close();
    }
}
//---sunguofei 2015.11.16
void BPRecognizer::mergeJoint_meanshift(IplImage* labelImg, const Mat& depthImg, CapgSkeleton& sklt)
{
    vector<Joint> _joints;
    Mat depth_draw=Mat::zeros(240,320,CV_8U);
    //depthImg.convertTo(depth_draw,CV_8U,255/8000.0,0);
    for (int i=0;i<31;++i)
    {
        float x=0,y=0,number=0;
        for (int j=0;j<depthImg.rows;++j)
        {
            for (int k=0;k<depthImg.cols;++k)
            {
                float depth=depthImg.at<unsigned int>(j,k)*0.005;
                weight_mat[i].at<float>(j,k) *=depth*depth;
                float weight=weight_mat[i].at<float>(j,k);
                x+=weight*k;
                y+=weight*j;
                if (weight!=0)
                {
                    number+=weight;
                }
            }
        }
        //check the probability map
//         imshow("weight image",dis_weight_mat[i]);
//         waitKey(0);
        if (number!=0)
        {
            x=x/number;y=y/number;
        }
        //---sunguofei 2015.12.1
        //new method.using gauss kernal --------not good!
        /*
        Mat new_weight=weight_mat[i];//+dis_weight_mat[i];
        int valid_point_number=countNonZero(new_weight);
        vector<vector<double>> points;
        vector<double> weights;
        points.resize(valid_point_number);
        weights.resize(valid_point_number);
        int count_tmp=0;
        for (int j=0;j<new_weight.rows;++j)
        {
            for (int k=0;k<new_weight.cols;++k)
            {
                double wt=new_weight.at<float>(j,k);
                if (wt!=0)
                {
                    vector<double> pt;pt.push_back(k);pt.push_back(j);
                    weights[count_tmp]=wt;
                    points[count_tmp]=pt;
                    ++count_tmp;
                }
            }
        }
        vector<double> s_p;s_p.push_back(x);s_p.push_back(y);
        MeanShift* ms=new MeanShift();
        vector<double> shifted_point=ms->shift_point(s_p,points,weights,400);
        Joint center;
        center.x()=shifted_point[0];center.y()=shifted_point[1];
        center.z()=depthImg.at<unsigned int>(center.y(),center.x());
        //circle(depth_draw,center,2,255);
        _joints.push_back(center);
        */

        //intial window size and window location
        //method 1: using the average of label   ---not good!
        
        int window_size=50;
        int rate=10;
        Rect window = Rect(max(int(x)-window_size/2,0),max(int(y)-window_size/2,0),window_size,window_size);
        Rect window1;
        if (joints.size()==31)
        {
            //check label point number in the window. If too small, re-initialize
            window1 = Rect(max(joints[i].x()-window_size/2,0),max(joints[i].y()-window_size/2,0),window_size,window_size);
            Rect roi;
            roi.x=window1.x;roi.y=window1.y;
            roi.width=min(window_size,depthImg.cols-1-window1.x);
            roi.height=min(window_size,depthImg.rows-1-window1.y);
            int label_count=countNonZero(weight_mat[i]);
            int sub_label_count=countNonZero(Mat(weight_mat[i],roi));
            if (sub_label_count>=label_count/rate)
            {
                window=window1;
            }
        }
        TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 10, 0.01);
        

        /*
        if (joints.size()!=31||(!is_valid))
        {
            //use adaptive window size
//             int sz;
//             for (sz=20;sz<depthImg.rows/5;sz+=6)
//             {
//                 window.height=max(min(depthImg.rows-window.y-1,sz),0);
//                 window.width=max(min(depthImg.cols-window.x-1,sz),0);
//                 if (window.height==0||window.width==0)
//                 {
//                     break;
//                 }
//                 int label_count = countNonZero(weight_mat[i]);
//                 int window_label_count = countNonZero(Mat(weight_mat[i],window));
//                 if (window_label_count*4>=label_count)
//                 {
//                     break;
//                 }
//             }
//             window.height=window.width=sz;window.x=max((window.x-window.width/2),0);window.y=max((window.y-window.height/2),0);
            meanShift(weight_mat[i],window,criteria);
        }
        //method 2: using previous joints
        else
        {
            //use adaptive window size
//             int sz;
//             for (sz=20;sz<depthImg.rows/5;sz+=6)
//             {
//                 window.height=max(min(depthImg.rows-window.y-1,sz),0);
//                 window.width=max(min(depthImg.cols-window.x-1,sz),0);
//                 if (window.height==0||window.width==0)
//                 {
//                     break;
//                 }
//                 int label_count = countNonZero(weight_mat[i]);
//                 int window_label_count = countNonZero(Mat(weight_mat[i],window));
//                 if (window_label_count*4>=label_count)
//                 {
//                     break;
//                 }
//             }
//             window.height=window.width=sz;window.x=max((window.x-window.width/2),0);window.y=max((window.y-window.height/2),0);
            meanShift(weight_mat[i],window,criteria);
            //deal with the situation that the joints is not on the body
//             if (depthImg.at<unsigned int>(window.y,window.x)==INT_MAX)
//             {
//                 window = Rect(int(x)-15,int(y)-15,30,30);
//                 criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 10, 0.01);
//                 res = meanShift(weight_mat[i],window,criteria);
//             }
        }
        */
        
        meanShift(weight_mat[i]+dis_weight_mat[i],window,criteria);
        //meanShift(weight_mat[i],window,criteria);

        //check if the current window is wrong
        if (joints.size()==31)
        {
            Rect roi;
            roi.x=window.x;roi.y=window.y;
            roi.width=min(window_size,depthImg.cols-1-window.x);
            roi.height=min(window_size,depthImg.rows-1-window.y);
            int sub_label_count=countNonZero(Mat(weight_mat[i],roi));
            if (sub_label_count<=10)
            {
                window=window1;
            }
        }

        Joint center;
        center.x()=window.x+window.width/2;center.y()=window.y+window.height/2;
        center.z()=depthImg.at<unsigned int>(center.y(),center.x());
        //circle(depth_draw,center,2,255);
        _joints.push_back(center);
        
    }
    //merge joints
    joints=_joints;
    vector<Joint> ret(8);
    ret[0] = (joints[BodyLabel_L_Shoulder]);
    ret[1] = (joints[BodyLabel_L_Elbow]);
    ret[2] = (joints[BodyLabel_L_Hand]);
    ret[3] = (joints[BodyLabel_R_Shoulder]);
    ret[4] = (joints[BodyLabel_R_Elbow]);
    ret[5] = (joints[BodyLabel_R_Hand]);
    ret[6] = ((joints[BodyLabel_LU_Head]+joints[BodyLabel_LW_Head]+
        joints[BodyLabel_RU_Head]+joints[BodyLabel_RU_Head])/4.0);
    ret[7] = ((joints[BodyLabel_LU_Torso]+joints[BodyLabel_LW_Torso]+
        joints[BodyLabel_RU_Torso]+joints[BodyLabel_RU_Torso])/4.0);
//     ret.push_back(joints[BodyLabel_LU_Leg]);
//     ret.push_back(joints[BodyLabel_LW_Leg]);
//     ret.push_back(joints[BodyLabel_L_Foot]);
//     ret.push_back(joints[BodyLabel_RU_Leg]);
//     ret.push_back(joints[BodyLabel_RW_Leg]);
//     ret.push_back(joints[BodyLabel_R_Foot]);
    //more realistic joints
    ret.push_back((joints[BodyLabel_LW_Torso]+joints[BodyLabel_LU_Leg])/2.0);
    ret.push_back((joints[BodyLabel_LU_Leg]+joints[BodyLabel_LW_Leg])/2.0);
    ret.push_back(joints[BodyLabel_L_Foot]);
    ret.push_back((joints[BodyLabel_RW_Torso]+joints[BodyLabel_RU_Leg])/2.0);
    ret.push_back((joints[BodyLabel_RU_Leg]+joints[BodyLabel_RW_Leg])/2.0);
    ret.push_back(joints[BodyLabel_R_Foot]);

    //draw joints
//     for (int i=0;i<ret.size();++i)
//     {
//         Point p;
//         p.x=ret[i].x();
//         p.y=ret[i].y();
//         circle(depth_draw,p,2,255,2);
//     }
//     imshow("new joints",depth_draw);
    sklt=ret;
}
//---sunguofei 2015.12.10
void BPRecognizer::offsetJoint_meanshift(CapgSkeleton& sklt)
{
    vector<vector<double>> res;
    vector<Joint> res_2d(18);
    for (int i=0;i<18;++i)
    {
        vector<double> s_p(3);
        double weight=0;
        for (int j=0;j<offset_result[i].size();++j)
        {
            s_p[0]+=offset_result[i][j][0]*offset_weight[i][j];
            s_p[1]+=offset_result[i][j][1]*offset_weight[i][j];
            s_p[2]+=offset_result[i][j][2]*offset_weight[i][j];
            weight+=offset_weight[i][j];
        }
        if (weight!=0)
        {
            s_p[0]/=weight;s_p[1]/=weight;s_p[2]/=weight;
        }
        if (joints_3d.size()!=0)
        {
//             if (joints_3d[i][0]!=0&&joints_3d[i][1]!=0)
//             {
                s_p[0]=joints_3d[i][0];
                s_p[1]=joints_3d[i][1];
                s_p[2]=joints_3d[i][2];
            //}
        }
        MeanShift* msf=new MeanShift();
        s_p=msf->shift_point(s_p,offset_result[i],offset_weight[i],250000);
        offset_result[i].clear();
        offset_weight[i].clear();
        res.push_back(s_p);
        Joint jt;
        jt.x()=int(159.5+s_p[0]*262.5/s_p[2]);
        jt.y()=int(119.5-s_p[1]*262.5/s_p[2]);
        jt.z()=int(s_p[2]);
        res_2d[i]=jt;
    }
    joints_3d=res;
    vector<Joint> ret(8);
    ret[0] = (res_2d[8]);
    ret[1] = (res_2d[7]);
    ret[2] = (res_2d[6]);
    ret[3] = (res_2d[9]);
    ret[4] = (res_2d[10]);
    ret[5] = (res_2d[11]);
    ret[6] = (res_2d[13]);
    ret[7] = (res_2d[12]);
    ret.push_back(res_2d[2]);
    ret.push_back(res_2d[1]);
    ret.push_back(res_2d[0]);
    ret.push_back(res_2d[3]);
    ret.push_back(res_2d[4]);
    ret.push_back(res_2d[5]);
    ret.push_back(res_2d[16]);
    ret.push_back(res_2d[17]);
    sklt=ret;
}

#define TOREN 3

void BPRecognizer::mergeJoint(IplImage* labelImg, IplImage* depthImg, CapgSkeleton& ret, 
								  bool useErode, bool usePre)
{
	PLAIN_MSG("-- Merge Joint Start", _moduleID);
	//Skeleton ret;
	vector<int> pCount;
	vector<CvPoint> newSearchLeft;
	vector<CvPoint> newSearchRight;
	newSearchLeft.resize(31);
	newSearchRight.resize(31);
	pCount.resize(31);
	ret.resize(31);
	for(int i=0; i<31; i++){
		pCount[i] = 0;
		ret[i] = Vector3(0, 0, 0);
		newSearchLeft[i] = cvPoint(999, 999);
		newSearchRight[i] = cvPoint(0, 0);
	}

	if(_preErode == 0){
		_preErode = cvCreateImage(cvSize(labelImg->width, labelImg->height), IPL_DEPTH_8U, 1);
	}

	int allPixel = 0;
	int notErodeCount = 0;
	int erodeForBGCount = 0;
	int erodeForErrorCount = 0;

	int xEdge = labelImg->width - 1;
	int yEdge = labelImg->height - 1;
	int erodeFlag = _erodeWin*2+1;
	erodeFlag = erodeFlag*erodeFlag*0.2;
	erodeFlag = min(erodeFlag, 1);

	for(int y=0; y<labelImg->height; y++){
		BodyLabel* l_data = (BodyLabel*)(labelImg->imageData + y*labelImg->widthStep);
		int* d_data = (int*)(depthImg->imageData + y*depthImg->widthStep);
		BodyLabel* p_data = (BodyLabel*)(_preLabelImg->imageData + y*_preLabelImg->widthStep);
		uchar* e_data = (uchar*)(_preErode->imageData + y*_preErode->widthStep);
		for(int x=0; x<labelImg->width; x++){
			BodyLabel centerLabel = l_data[x];
			if(centerLabel == BodyLabel_Background || centerLabel == BodyLabel_Unknown){
				e_data[x] = ErodeForBackground; continue;
			}
			uchar erode = NotErode;
			allPixel++;
			if(usePre && _preAvailable){
				if(x > _skeleton[centerLabel].x()+_searchWin[centerLabel].x ||
					x < _skeleton[centerLabel].x()-_searchWin[centerLabel].x ||
					y > _skeleton[centerLabel].y()+_searchWin[centerLabel].y ||
					y < _skeleton[centerLabel].y()-_searchWin[centerLabel].y){
						erode = ErodeForError;
						erodeForErrorCount++;
						goto ErodeEnd;
				}
				//if(centerLabel == p_data[x] && e_data[x]){
				//	erode = true; goto ErodeEnd;
				//}
			}
			if(useErode){
				int xx = max(x - _erodeWin, 0); int xx_end = min(x + _erodeWin, xEdge);
				int yy = max(y - _erodeWin, 0); int yy_end = min(y + _erodeWin, yEdge);
				int flag = erodeFlag;
				for(int j=yy; j<=yy_end; j++){
					BodyLabel* check_data = (BodyLabel*)(labelImg->imageData + j*labelImg->widthStep);
					for(int i=xx; i<=xx_end; i++){
						BodyLabel checkLabel = check_data[i];
						if(checkLabel != centerLabel){
							//if(DepthSample::similarPart(centerLabel, checkLabel))
							//	continue;
							flag--;
							if(flag <= 0){
								erode = ErodeForBackground;
								erodeForBGCount++;
								goto ErodeEnd;
							}
						}
					}
				}
			}
ErodeEnd:
			e_data[x] = erode;
			if(erode == NotErode){ // into joint merge
				pCount[centerLabel]++;
				ret[centerLabel].x() += x;
				ret[centerLabel].y() += y;
				ret[centerLabel].z() += d_data[x];
				newSearchLeft[centerLabel].x = min(newSearchLeft[centerLabel].x, x);
				newSearchLeft[centerLabel].y = min(newSearchLeft[centerLabel].y, y);
				newSearchRight[centerLabel].x = max(newSearchRight[centerLabel].x, x);
				newSearchRight[centerLabel].y = max(newSearchRight[centerLabel].y, y);
			}
		}
	}

	for(int i=0; i<31; i++){
		int pcount = pCount[i];
		notErodeCount+= pcount;
		if(pcount == 0){
			if(usePre) ret[i] = _skeleton[i];
			_searchWin[i] = cvPoint(OutRangeNone, OutRangeNone);
			continue;
		}
		//if(i == BodyLabel_LU_Torso || i == BodyLabel_RU_Torso || i == BodyLabel_LW_Torso
		//	 || i == BodyLabel_RW_Torso)
		//	 _searchWin[i] = cvPoint(OutRangeBody, OutRangeBody);
		//else _searchWin[i] = cvPoint(OutRangeOther, OutRangeOther);
		ret[i].x() /= pcount;
		ret[i].y() /= pcount;
		ret[i].z() /= pcount;
		int speedx = abs(ret[i].x() - _skeleton[i].x());
		int speedy = abs(ret[i].y() - _skeleton[i].y());
		_searchWin[i] = cvPoint(newSearchRight[i].x - newSearchLeft[i].x + 2*_erodeWin + speedx+TOREN,
			newSearchRight[i].y - newSearchLeft[i].y + 2*_erodeWin + speedy+TOREN);
	}

	//int preWin = _erodeWin;
	//_erodeWin = allPixel / 1000;
	//_erodeWin = _erodeWin*_erodeWin / 40;
	//if(preWin != _erodeWin){
	//	_erodeWin = max(_erodeWin, 2);
	//	_erodeWin = min(_erodeWin, 5);
	//	if(preWin != _erodeWin){
	//		_predictStep = max(_erodeWin - 1, 2);
	//		//IMPORT_MSG(string("Erode win and predict Step change: ")+LogCat::to_string(preWin)+" to "+
	//			//LogCat::to_string(_erodeWin), _moduleID);
	//	}
	//}
	
	_skeleton = ret;
	_preAvailable = true;

	PLAIN_MSG(string("All labeled: ")+LogCat::to_string(allPixel)+
		"  Not erode: "+LogCat::to_string(notErodeCount)+
		"  Erode for BG: "+LogCat::to_string(erodeForBGCount)+
		"  Erode for error: "+LogCat::to_string(erodeForErrorCount), _moduleID);

	// convert to Shuyang Sun version
// 	ret.clear();
// 	ret.resize(8);
// 	ret[0] = (_skeleton[BodyLabel_L_Shoulder]);
// 	ret[1] = (_skeleton[BodyLabel_L_Elbow]);
// 	ret[2] = (_skeleton[BodyLabel_L_Hand]);
// 	ret[3] = (_skeleton[BodyLabel_R_Shoulder]);
// 	ret[4] = (_skeleton[BodyLabel_R_Elbow]);
// 	ret[5] = (_skeleton[BodyLabel_R_Hand]);
// 	ret[6] = ((_skeleton[BodyLabel_LU_Head]+_skeleton[BodyLabel_LW_Head]+
// 		_skeleton[BodyLabel_RU_Head]+_skeleton[BodyLabel_RU_Head])/4.0);
// 	ret[7] = ((_skeleton[BodyLabel_LU_Torso]+_skeleton[BodyLabel_LW_Torso]+
// 		_skeleton[BodyLabel_RU_Torso]+_skeleton[BodyLabel_RU_Torso])/4.0);
// 	ret.push_back(_skeleton[BodyLabel_LU_Leg]);
// 	ret.push_back(_skeleton[BodyLabel_LW_Leg]);
// 	ret.push_back(_skeleton[BodyLabel_L_Foot]);
// 	ret.push_back(_skeleton[BodyLabel_RU_Leg]);
// 	ret.push_back(_skeleton[BodyLabel_RW_Leg]);
// 	ret.push_back(_skeleton[BodyLabel_R_Foot]);

    //change the location of leg joint ---sunguofei 2015-11-5
    ret.clear();
    ret.resize(8);
    ret[0] = (_skeleton[BodyLabel_L_Shoulder]);
    ret[1] = (_skeleton[BodyLabel_L_Elbow]);
    ret[2] = (_skeleton[BodyLabel_L_Hand]);
    ret[3] = (_skeleton[BodyLabel_R_Shoulder]);
    ret[4] = (_skeleton[BodyLabel_R_Elbow]);
    ret[5] = (_skeleton[BodyLabel_R_Hand]);
    ret[6] = ((_skeleton[BodyLabel_LU_Head]+_skeleton[BodyLabel_LW_Head]+
        _skeleton[BodyLabel_RU_Head]+_skeleton[BodyLabel_RU_Head])/4.0);
    ret[7] = ((_skeleton[BodyLabel_LU_Torso]+_skeleton[BodyLabel_LW_Torso]+
        _skeleton[BodyLabel_RU_Torso]+_skeleton[BodyLabel_RU_Torso])/4.0);
    ret.push_back((_skeleton[BodyLabel_LW_Torso]+_skeleton[BodyLabel_LU_Leg])/2.0);
    ret.push_back((_skeleton[BodyLabel_LU_Leg]+_skeleton[BodyLabel_LW_Leg])/2.0);
    ret.push_back(_skeleton[BodyLabel_L_Foot]);
    ret.push_back((_skeleton[BodyLabel_RW_Torso]+_skeleton[BodyLabel_RU_Leg])/2.0);
    ret.push_back((_skeleton[BodyLabel_RU_Leg]+_skeleton[BodyLabel_RW_Leg])/2.0);
    ret.push_back(_skeleton[BodyLabel_R_Foot]);

	PLAIN_MSG("-- Merge Joint End", _moduleID);

	if(_preErodeForError > 0 && abs(erodeForErrorCount - _preErodeForError) > 0.5*_preErodeForError
		&& erodeForErrorCount >= 0.2*allPixel){
		clearPre();
	}else{
		_preErodeForError = erodeForErrorCount;
	}

	PLAIN_MSG(string("ret's size is ")+LogCat::to_string( ret.size()), _moduleID);
	//return ret;
}

//by zhangxaochen: 
// utility func, show colorful labels. 由predictAndMergeJoint绘制pLabelImg2部分改写而来
Mat label_gray2rgb(Mat &labelMat){
	CV_Assert(labelMat.type()==CV_8UC1);

	Mat out = Mat::zeros(labelMat.size(), CV_8UC3);
	for(size_t y = 0; y < labelMat.rows; y++){
		//uchar *arow = labelMat.data + y * labelMat.step; //这么写对不对？
		uchar *arow = labelMat.ptr<uchar>(y),
			*orow = out.ptr<uchar>(y);
		for(size_t x = 0; x < labelMat.cols; x++){
			//BPR init 之后就可用这个 static func
			Color c = DepthSample::getColor(arow[x]);
			orow[3*x+0]=c.get_b();
			orow[3*x+1]=c.get_g();
			orow[3*x+2]=c.get_r();
		}
	}

	return out;
}

//by zhangxaochen: 函数体移到 .cpp 来：
void BPRecognizer::predictAndMergeJoint(IplImage* depthImg, CapgSkeleton& sklt, IplImage* maskImg, bool usePre, bool useErode,bool showPic){
	IplImage* pLabelImg = predict(depthImg, maskImg, false, usePre);

	////////////////////////////////////////////////////edit by mao
	IplImage* pHandImg = cvCreateImage(cvSize(IMAGE_WIDTH, IMAGE_HEIGHT), IPL_DEPTH_8U, 1);
	////////////////////////////////////////////////////edit by mao
	
#ifdef LINCCCC_DEBUG
	if(showPic){
		IplImage* pLabelImg2 = cvCreateImage(cvSize(IMAGE_WIDTH, IMAGE_HEIGHT), IPL_DEPTH_8U, 3);
		for (size_t y = 0; y < pLabelImg2->height; y++)
		{
			for (size_t x = 0; x < pLabelImg2->width; x++)
			{
				uchar* ptr1 = (uchar*)(pLabelImg->imageData+ y * pLabelImg->widthStep + x * sizeof(uchar) * 1);
				uchar* ptr2 = (uchar*)(pLabelImg2->imageData + y * pLabelImg2->widthStep + x * sizeof(uchar) * 3);
				Color c = DepthSample::getColor(ptr1[0]);
				ptr2[0] = c.get_b();ptr2[1] = c.get_g();ptr2[2] = c.get_r();

				////////////////////////////////////////////////////edit by mao
				uchar* ptr3 = (uchar*)(pHandImg->imageData+ y * pHandImg->widthStep + x * sizeof(uchar) * 1);
				uchar* depthMap = (uchar*)(depthImg->imageData+ y * depthImg->widthStep + x * sizeof(uchar) * 1);
				if (*ptr1 == BodyLabel_L_Hand || *ptr1 == BodyLabel_R_Hand)
				{
					*ptr3 = *depthMap;
				}
				else{
					*ptr3 = 255;
				}
				////////////////////////////////////////////////////edit by mao


			}
		}

		cvShowImage("p",pLabelImg2);
		cvShowImage("HandShowImage", pHandImg);
		char tBuffer[100] = {'\0'};
		string fileName = string("e:/imagetest/") + string(itoa(g_ImgIndex++, tBuffer, 10)) + string(".jpg");
		cvSaveImage(fileName.c_str(), pHandImg);

		cvReleaseImage(&pHandImg);
		/*
		char tBuffer[100] = {'\0'};
		string fileName = string(itoa(g_ImgIndex, tBuffer, 10)) + string("p.jpg");
		cvSaveImage(fileName.c_str(), pLabelImg2);*/

		cvReleaseImage(&pLabelImg2);
	}
#endif
	mergeJoint(pLabelImg, depthImg, sklt, usePre, useErode);

	/*int nPrint = 5;
	while (nPrint--)
	{
	CapgPrintf("&sklet = %u, sklt.size() is %d\n", &sklt, sklt.size());
	xnOSSleep(50);
	}*/

#ifdef LINCCCC_DEBUG
	if(showPic){
		IplImage* simag=convertSkeletonToImage(sklt);
		cvShowImage("sk",simag);
		/*
		char tBuffer[100] = {'\0'};
		string fileName = string(itoa(g_ImgIndex, tBuffer, 10)) + string("sk.jpg");
		cvSaveImage(fileName.c_str(), simag);*/

		//g_ImgIndex++;
		cvReleaseImage(&simag);
	}
#endif
	//return skeleton;
}

//---sunguofei 2015.11.10
bool BPRecognizer::train()
{
    IMPORT_MSG("Training start", _moduleID);

    if(_trainSamples == 0){
        ERROR_MSG("Unable to train without train samples, please load", _moduleID);
        return false;
    }

    IMPORT_MSG("Clear train matrices and allocate new...", _moduleID);
    clearTrainMat();

    // memory allocation
    _sample_count = _para.img_per_tree*_para.pixel_per_img;
    _value_count = _para.theta_per_tree;
    _value_mat = cvCreateMat(_sample_count, _value_count, CV_32FC1);
    _response_mat = cvCreateMat(_sample_count, 1, CV_32SC1);
    _var_type_mat = cvCreateMat(_value_count+1, 1, CV_8UC1);
    _active_var_mask = cvCreateMat(1, _value_count, CV_8UC1);
    if(_value_mat == 0 || _response_mat == 0 ||
        _var_type_mat == 0 || _active_var_mask == 0){
            ERROR_MSG("Not enough memory to build value and response matrix", _moduleID);
            clearTrainMat();
            return false;
    }

    IMPORT_MSG("Setup value type matrix and train paramenter...", _moduleID);
    // set value type matrix
    //if(_para.feature_type == DEPTH_FEATURE_GSUB || _para.feature_type == DEPTH_FEATURE_GADD){
    //	cvSet(_var_type_mat, cvScalarAll(CV_VAR_NUMERICAL));
    //	CV_MAT_ELEM(*_var_type_mat, uchar, (_var_type_mat->rows-1), 0) = CV_VAR_CATEGORICAL;
    //}else if(_para.feature_type == DEPTH_FEATURE_GHALF){
    //	cvSet(_var_type_mat, cvScalarAll(CV_VAR_CATEGORICAL));
    //}
    cvSet(_var_type_mat, cvScalarAll(CV_VAR_NUMERICAL));
    CV_MAT_ELEM(*_var_type_mat, uchar, (_var_type_mat->rows-1), 0) = CV_VAR_CATEGORICAL;

    // setup train data
    _dPara = CvDTreeParams(_para.tree_depth, 100, 0, false, 50,
        0, false, false, 0);

    IMPORT_MSG("Growing forest...", _moduleID);
    char buf[10];
    for(int i=_treeNum; i<_para.tree_num; i++){
        BPRTree* tree = new BPRTree();
        vector<DepthFeature*>* fset = new vector<DepthFeature*>();
        _features.push_back(fset);

        IMPORT_MSG(string("Tree No.")+_itoa(i,buf,10), _moduleID);
        bool res = tree->train(this);

        if(res){
            _forest.push_back(tree);
            _treeNum++;
        }else{
            delete tree;
            clearTrainMat();
            for(vector<DepthFeature*>::iterator it=fset->begin(); it!=fset->end(); it++){
                delete *it;
            }
            fset->clear();
            _features.pop_back();
            delete fset;
            return false;
        }
    }

    clearTrainMat();

    IMPORT_MSG("Training finish", _moduleID);

    return true;
}

void BPRecognizer::test(bool save)
{
	if(_testSamples == 0){
		ERROR_MSG("Unable to test without test samples, please load", _moduleID);
		return;
	}

	if(_forest.empty() || _features.empty()){
		ERROR_MSG("Unable to test with empty forest", _moduleID);
		return;
	}

	IMPORT_MSG("Testing start", _moduleID);
	vector<DepthSample*>& testData = _testSamples->getSampleSet();

	int testCount = testData.size();
	float ana = 0;
	vector<float> partAna;
	partAna.resize(31);
	for(int i=0; i<31; i++) partAna[i] = 0;
	LogCat::getInstancePtr()->addProcessBar("Test:", 50);
	double p_add = 1.0/testCount;
	int picCount=1;
	for(vector<DepthSample*>::iterator it=testData.begin(); it!=testData.end(); it++, picCount++){
		DepthSample* img = *it;
		img->loadImage();
		IplImage* dImg = img->getDepthImg();
		IplImage* lImg_gt = img->getlabelImg();
		IplImage* colorImg = cvCreateImage(cvSize(lImg_gt->width, lImg_gt->height), IPL_DEPTH_8U, 3);
		//IplImage* lImg = predict(dImg);
		//---sunguofei 2015.11.11
        IplImage* lImg = predict(dImg,0,false,true);

        int pCount = dImg->width*dImg->height;
		int pCorrect = 0;
		int pAna = 0;
		vector<int> partCorrect;
		vector<int> partCount;
		partCorrect.resize(31); partCount.resize(31);
		for(int i=0; i<31; i++){ partCorrect[i] = 0; partCount[i] = 0; }
		BodyLabel* l_data_gt = (BodyLabel*)lImg_gt->imageData;
		BodyLabel* l_data = (BodyLabel*)lImg->imageData;
		uchar* c_data = (uchar*)colorImg->imageData;
		for(int i=0; i<pCount; i++){
			if(*l_data_gt != BodyLabel_Unknown &&
				(*l_data_gt != BodyLabel_Background || *l_data != BodyLabel_Background)){
				if(*l_data == *l_data_gt){
					pCorrect++;
					partCorrect[*l_data_gt]++;
				}
				pAna++;
				partCount[*l_data_gt]++;
			};
			if(save){
				Color c = DepthSample::getColor(*l_data);
				c_data[2] = c.get_r(); c_data[1] = c.get_g(); c_data[0] = c.get_b();
				c_data += 3;
			}
			l_data_gt++; l_data++;
		}
		ana += (float)pCorrect/(float)pAna;
		for(int i=0; i<31; i++){
			if(partCount[i] != 0)
				partAna[i] += (float)partCorrect[i]/(float)partCount[i];
		}
		img->releaseImage();
		// save label image
		if(save){
			//for(int i=0; i<31; i++){
			//	cvCircle(colorImg, _partCenter[i], 2, cvScalar(0), 2);
			//}
			string imgName("test_res/");
			imgName.append(img->_depthFileName);
			cvSaveImage(imgName.data(), colorImg);
		}
		cvReleaseImage(&colorImg);
		LogCat::getInstancePtr()->setProcessBar(picCount*p_add);
	}
	ana /= (float)testCount;
	for(int i=0; i<31; i++){
		partAna[i] /= (float)testCount;
	}
	LogCat::getInstancePtr()->removeProcessBar();

	// analysis and save and clear
	fstream anaFile;
	anaFile.open("testAna.txt", ios::out);
	if(anaFile.is_open()){
		anaFile << LogCat::to_string(ana);
		anaFile << endl << endl;
		for(int i=0; i<31; i++){
			anaFile << LogCat::to_string(i) << ": " << LogCat::to_string(partAna[i]) << endl;
		}
	}
	anaFile.close();

	IMPORT_MSG(string("Testing finish: ") + LogCat::to_string(ana) + "%", _moduleID);
}

void BPRecognizer::testRuntime(bool save)
{
	if(_testSamples == 0){
		ERROR_MSG("Unable to test without test samples, please load", _moduleID);
		return;
	}

	if(_forest.empty() || _features.empty()){
		ERROR_MSG("Unable to test with empty forest", _moduleID);
		return;
	}

	IMPORT_MSG("Testing start", _moduleID);
	vector<DepthSample*>& testData = _testSamples->getSampleSet();

	int testCount = testData.size();
	LogCat::getInstancePtr()->addProcessBar("Test:", 50);
	double p_add = 1.0/testCount;
	int picCount=1;
	string preAction("");
#ifdef LINC_DEBUG
	QTime timer;
	timer.start();
	float elapsedPredict = 0;
	float elapsedMergeJoint = 0;
#endif
	for(vector<DepthSample*>::iterator it=testData.begin(); it!=testData.end(); it++, picCount++){
		DepthSample* img = *it;
		string curAciont = img->_depthFileName.substr(0, 18);
		//if(curAciont != preAction)
		//	clearPre();
		preAction = curAciont;
		img->loadRuntimeImage();
		IplImage* dImg = img->getDepthImg();
		IplImage* colorImg = cvCreateImage(cvSize(dImg->width, dImg->height), IPL_DEPTH_8U, 3);
#ifdef LINC_DEBUG
		timer.restart();
		IplImage* lImg = predict(dImg, 0, false, true);
		elapsedPredict += timer.elapsed();
		timer.restart();
		CapgSkeleton sk = mergeJoint(lImg, dImg, true, true);
		elapsedMergeJoint += timer.elapsed();
#else
		IplImage* lImg = predict(dImg, 0, false, false);

		CapgSkeleton sk;
		mergeJoint(lImg, dImg, sk, false, false);
#endif
		int pCount = dImg->width*dImg->height;
		BodyLabel* l_data = (BodyLabel*)lImg->imageData;
		uchar* c_data = (uchar*)colorImg->imageData;
		for(int i=0; i<pCount; i++){
			if(save){
				Color c = DepthSample::getColor(*l_data);
				c_data[2] = c.get_r(); c_data[1] = c.get_g(); c_data[0] = c.get_b();
				c_data += 3;
			}
			l_data++;
		}
		img->releaseImage();
		// save label image
		if(save){
			for(int i=0; i<sk.size(); i++){
				cvCircle(colorImg, cvPoint(sk[i].x(), sk[i].y()), 2, cvScalar(0), 2);
			}
			cvLine(colorImg, cvPoint(sk[0].x(), sk[0].y()), cvPoint(sk[1].x(), sk[1].y()), cvScalar(0), 2);
			cvLine(colorImg, cvPoint(sk[1].x(), sk[1].y()), cvPoint(sk[2].x(), sk[2].y()), cvScalar(0), 2);
			cvLine(colorImg, cvPoint(sk[3].x(), sk[3].y()), cvPoint(sk[4].x(), sk[4].y()), cvScalar(0), 2);
			cvLine(colorImg, cvPoint(sk[4].x(), sk[4].y()), cvPoint(sk[5].x(), sk[5].y()), cvScalar(0), 2);
			cvLine(colorImg, cvPoint(sk[0].x(), sk[0].y()), cvPoint(sk[3].x(), sk[3].y()), cvScalar(0), 2);
			cvLine(colorImg, cvPoint(sk[6].x(), sk[6].y()), cvPoint(sk[7].x(), sk[7].y()), cvScalar(0), 2);
			string imgName("test_res/");
			cvSaveImage(string(imgName + (LogCat::to_string(picCount)+".jpg")).data(), colorImg);
			cvSaveImage(string(imgName + (LogCat::to_string(picCount)+".png")).data(), _preErode);
			//string imgName("\\\\soso-pc/RESULT/LabelData");
			//string filename = img->_depthFileName.substr(9);
			//imgName.append(filename);
			//cvSaveImage(imgName.data(), colorImg);
		}
		cvReleaseImage(&colorImg);
		LogCat::getInstancePtr()->setProcessBar(picCount*p_add);
	}
	LogCat::getInstancePtr()->removeProcessBar();

	IMPORT_MSG(string("Testing finish"), _moduleID);

#ifdef LINC_DEBUG
	elapsedPredict /= testCount;
	elapsedMergeJoint /= testCount;
	IMPORT_MSG(string("Predict: ")+LogCat::to_string(elapsedPredict)+" ms, Merge Joint: "+
		LogCat::to_string(elapsedMergeJoint)+" ms, Total: "+LogCat::to_string(elapsedMergeJoint+elapsedPredict)+
		" ms", _moduleID);
#endif
}

void BPRecognizer::clearPre()
{
	//IMPORT_MSG("!!!Clear Previous", _moduleID);

	cvZero(_preLabelImg);

	_skeleton.resize(31);
	_searchWin.resize(31);
	for(int i=0; i<31; i++){
		_skeleton[i] = Vector3(0, 0, 0);
		_searchWin[i] = cvPoint(OutRangeNone, OutRangeNone);
	}
	_uniSearchStart = cvPoint(0, 0);
	//_uniSearchEnd = cvPoint(320, 240);
	//zhangxaochen:
	_uniSearchEnd = cvPoint(IMAGE_WIDTH, IMAGE_HEIGHT);


	_preAvailable = false;

	cvZero(_preErode);

	_preErodeForError = -1;
}

bool BPRecognizer::load(const string& pathname)
{
	clear();

	// load parameter
	fstream paraFile;
	paraFile.open(string(pathname+"/forest_para").data(), ios::in);
	if(!paraFile.is_open()){
		ERROR_MSG("load failed due to unavailable parameter file", _moduleID);
		return false;
	}
	string s;
	getline(paraFile, s);
	_treeNum = LogCat::to_int(s);
	getline(paraFile, s);
	_para.tree_depth = LogCat::to_int(s);
	getline(paraFile, s);
	_para.min_sample_count = LogCat::to_int(s);
	getline(paraFile, s);
	_para.tree_num = LogCat::to_int(s);
	getline(paraFile, s);
	_para.feature_type = LogCat::to_int(s);
	getline(paraFile, s);
	_para.feature_high_bound = LogCat::to_int(s);
	getline(paraFile, s);
	_para.img_per_tree = LogCat::to_int(s);
	getline(paraFile, s);
	_para.pixel_per_img = LogCat::to_int(s);
	getline(paraFile, s);
	_para.theta_per_tree = LogCat::to_int(s);
	getline(paraFile, s);
	_para.node_active_var = LogCat::to_int(s);
	getline(paraFile, s);
	_para.merge_flag = LogCat::to_int(s);

	DepthSample::createLabelMap(_para.merge_flag);
	DepthSample::createColorMap();
	IMPORT_MSG("Load parameter successfully", _moduleID);

	// load cv forest
	for(int i=0; i<_treeNum; i++){
		string tfilename(pathname+"/cv_forest_");
		tfilename.append(LogCat::to_string(i));
		CvFileStorage *fs = cvOpenFileStorage(tfilename.data(),
			NULL, CV_STORAGE_READ);
		if(fs == 0){
			ERROR_MSG("Load failed due to unavailable cv forest file", _moduleID);
			return false;
		}
		CvFileNode *fn = cvGetFileNodeByName(fs, cvGetRootFileNode(fs), "_tree");
		BPRTree* tree = new BPRTree();
		tree->read(fs, fn, this);
		_forest.push_back(tree);
		cvReleaseFileStorage(&fs);
	}
	IMPORT_MSG("Load cv forest successfully", _moduleID);
	for(int i=0; i<_treeNum; i++){
		getline(paraFile, s);
		_forest[i]->_nodeCount = LogCat::to_int(s);
		getline(paraFile, s);
		_forest[i]->_featureCount = LogCat::to_int(s);
	}
	paraFile.close();

	// load forest feature
	for(int i=0; i<_treeNum; i++){
		string ffilename(pathname+"/forest_feature_");
		ffilename.append(LogCat::to_string(i));
		fstream fFile;
		fFile.open(ffilename.data(), ios::in);
		if(!fFile.is_open()){
			ERROR_MSG("Load failed due to unavailable forest feature file", _moduleID);
			return false;
		}
		string ss;
		vector<DepthFeature*>* fset = new vector<DepthFeature*>();
		while(getline(fFile, ss)){
			if(ss.empty()){
				fset->push_back(0);
				continue;
			}
			DepthFeature* f;
			if(_para.feature_type == DEPTH_FEATURE_GSUB){
				f =  new GSubDepthFeature();
			}else if(_para.feature_type == DEPTH_FEATURE_GADD){
				f =  new GAddDepthFeature();
			}else if(_para.feature_type == DEPTH_FEATURE_GHALF){
				f =  new GHalfDepthFeature();
			}
			f->fromString(ss);
			fset->push_back(f);
		}
		fFile.close();
		_features.push_back(fset);
	}
	IMPORT_MSG("Load forest feature successfully", _moduleID);

#ifdef LINC_DEBUG
	buideDenseForest();
	initCudaRecog(320, 240, 320, 320*4, _denseFeature, _featureSize);
#endif

	return true;
}

bool BPRecognizer::save(const string& pathname)
{
	// save parameters
	fstream paraFile;
	paraFile.open(string(pathname+"/forest_para").data(), ios::out | ios::trunc);
	if(!paraFile.is_open()){
		ERROR_MSG("Save failed due to unsuccessful parameter file creation", _moduleID);
		return false;
	}
	paraFile << LogCat::to_string(_treeNum) << endl;
	paraFile << LogCat::to_string(_para.tree_depth) << endl;
	paraFile << LogCat::to_string(_para.min_sample_count) << endl;
	paraFile << LogCat::to_string(_para.tree_num) << endl;
	paraFile << LogCat::to_string(_para.feature_type) << endl;
	paraFile << LogCat::to_string(_para.feature_high_bound) << endl;
	paraFile << LogCat::to_string(_para.img_per_tree) << endl;
	paraFile << LogCat::to_string(_para.pixel_per_img) << endl;
	paraFile << LogCat::to_string(_para.theta_per_tree) << endl;
	paraFile << LogCat::to_string(_para.node_active_var) << endl;
	paraFile << LogCat::to_string(_para.merge_flag) << endl;
	for(int i=0; i<_treeNum; i++){
		paraFile << LogCat::to_string(_forest[i]->_nodeCount) << endl;
		paraFile << LogCat::to_string(_forest[i]->_featureCount) << endl;
	}
	paraFile.flush();
	paraFile.close();
	IMPORT_MSG("Save parameter file successfully", _moduleID);

	// save cv forest
	int i=0;
	for(vector<BPRTree*>::iterator it=_forest.begin(); it!=_forest.end(); it++, i++){
		string tfilename(pathname+"/cv_forest_");
		tfilename.append(LogCat::to_string(i));
		CvFileStorage *fs = cvOpenFileStorage(tfilename.data(),
			NULL, CV_STORAGE_WRITE);
		if(fs == 0){
			ERROR_MSG("Save failed due to unsuccessful cv forest file creation", _moduleID);
			return false;
		}
		(*it)->write(fs, "_tree");
		cvReleaseFileStorage(&fs);
	}
	IMPORT_MSG("Save cv forest successfully", _moduleID);
	
	// save features
	i=0;
	for(vector<vector<DepthFeature*>*>::iterator it=_features.begin();
		it!=_features.end(); it++, i++)
	{
		vector<DepthFeature*>& fset= **it;
		fstream fFile;
		string ffilename(pathname+"/forest_feature_");
		ffilename.append(LogCat::to_string(i));
		fFile.open(ffilename.data(), ios::out | ios::trunc);
		if(!fFile.is_open()){
			ERROR_MSG("Save failed due to unsuccessful forest feature file creation", _moduleID);
			return false;
		}
		for(vector<DepthFeature*>::iterator itt=fset.begin(); itt!=fset.end(); itt++){
			if(*itt == 0){
				fFile << endl;
				continue;
			}
			fFile << (*itt)->toString() << endl;
			fFile.flush();
		}
		fFile.close();
	}
	IMPORT_MSG("Save forest feature successfully", _moduleID);

	return true;
}

DepthSampleGenerator::~DepthSampleGenerator()
{
	for(vector<DepthSample*>::iterator it=_samples.begin();
		it!=_samples.end(); it++){
			if(*it) delete *it;
	}
	_samples.clear();
}

bool DepthSampleGenerator::generateDataFromFile()
{
	if(_data_source != DATA_SOURCE_FILE) return false;
	LogCat::getInstancePtr()->addProcessBar("Generate Train Data:", 50);

	//// load from file (debug)
	//string depthFloder = _data_path + "\\depth\\";
	//string labelFloder = _data_path + "\\label\\";
	//string infoFile = _data_path + "\\sample_info";
	//fstream f;
	//f.open(infoFile.data());
	//if(!f.is_open()){
	//	ERROR_MSG("Load train data from file failed since cannot find sample_info file", _moduleID);
	//	return false;
	//}
	//string s;
	//getline(f, s); // sample count
	//int sample_count = LogCat::to_int(s);
	//getline(f, s); // file format (.jpg .bmp)
	//f.close();
	//double p_add = 1.0/sample_count;
	//double p = p_add;
	//for(int i=0; i<sample_count; i++, p+=p_add){
	//	LogCat::getInstancePtr()->setProcessBar(p);
	//	string filename = LogCat::to_string(i) + s;
	//	string depthFile = depthFloder + filename;
	//	string labelFile = labelFloder + filename;
	//	DepthSample* sample = new DepthSample(depthFile, labelFile);
	//	_samples.push_back(sample);
	//}

	// load from file (ke_ji_guan)
	string rootFile = _data_path;
	//string infoFile = _data_path + "/sample_info";
    //---sunguofei 2015.11.10  use sample_info.txt
	string infoFile = _data_path + "/sample_info.txt";

    fstream f;
	f.open(infoFile.data());
	if(!f.is_open()){
		ERROR_MSG("Load train data from file failed since cannot find sample_info file", _moduleID);
		return false;
	}
	string s;
	getline(f, s);
	double p_add = 1.0/5000;
	double p = p_add;
	while(!s.empty()){
		LogCat::getInstancePtr()->setProcessBar(p);
		string labelFile = rootFile + s + ".bmp";
        //---sunguofei 2015.11.10 our new data is .png
        //string labelFile = rootFile + s + ".png";

		getline(f, s);
		string depthFile = rootFile + s + ".bmp";
        //---sunguofei 2015.11.10
        //string depthFile = rootFile + s + ".png";

		DepthSample* sample = new DepthSample(depthFile, labelFile);
		_samples.push_back(sample);
		getline(f, s);
		p += p_add;
	}
	f.close();

	LogCat::getInstancePtr()->removeProcessBar();
	return true;
}

void DepthSampleGenerator::setDataSource( int data_source, const string& path )
{
	_data_source = data_source;
	_data_path = path;
}

void BPRecognizer::buideDenseForest()
{
	if(_denseFeature) delete [] _denseFeature;
	if(_denseForest) delete [] _denseForest;
	if(_denseRoot) delete [] _denseRoot;

	_denseRoot = new ushort [_treeNum];
	int featureCount = 0;
	int nodeCount = 0;
	for(vector<BPRTree*>::iterator it=_forest.begin();
		it!=_forest.end(); it++){
			featureCount += (*it)->_featureCount;
			nodeCount += (*it)->_nodeCount;
	}
	_denseFeature = new DenseFeature [featureCount];
	_denseForest = new DenseNode [nodeCount];
	_featureSize = 0;
	_nodeSize = 0;

	IMPORT_MSG("::buideDenseForest, memory allocation finish", _moduleID);

	memset(_denseFeature, 0, sizeof(DenseFeature)*featureCount);
	memset(_denseForest, 0, sizeof(DenseNode)*nodeCount);
	memset(_denseRoot, 0, sizeof(ushort)*_treeNum);

	IMPORT_MSG("::buideDenseForest, memory zeroing finish", _moduleID);

	_featuremap = new int [_features[0]->size()];
	IMPORT_MSG("::buideDenseForest, feature map allocation finish", _moduleID);
	for(int i=0; i<_treeNum; i++){
		IMPORT_MSG(string("::buideDenseForest, tree_")+LogCat::to_string(i), _moduleID);
		_treeCount = i;
		vector<DepthFeature*>& fset = *_features[i];
		int fcount = 0;
		for(vector<DepthFeature*>::iterator it=fset.begin(); it!=fset.end(); it++, fcount++){
			DepthFeature* f = (*it);
			if(f==0) continue;
			DenseFeature* df = &_denseFeature[_featureSize];
			df->u_x = f->_u.x / DepthModify;
			df->u_y = f->_u.y / DepthModify;
			df->v_x = f->_v.x / DepthModify;
			df->v_y = f->_v.y / DepthModify;
			_featuremap[fcount] = _featureSize;
			_featureSize++;
		}
		IMPORT_MSG("::buideDenseForest, copy feature finish", _moduleID);
		_denseRoot[i] = buildDenseNode(_forest[i]->root);
		IMPORT_MSG("::buideDenseForest, copy nodes finish", _moduleID);
	}
	delete [] _featuremap;
}

ushort BPRecognizer::buildDenseNode(CvDTreeNode* node)
{
	// set dense node
	int nodeIndex = _nodeSize;
	DenseNode* dn = &_denseForest[_nodeSize];
	_nodeSize++;

	// set dense feature
	if(node->split){
		dn->tValue = node->split->ord.c;

		int feature = node->split->var_idx;
		dn->fIndex = _featuremap[feature];

		// set left and right
		if(node->left){
			buildDenseNode(node->left);
			dn->rightIndex = buildDenseNode(node->right);
		}
	}
	else{
		dn->rightIndex = 0xffff;
		dn->tValue = node->value;
	}

	return nodeIndex;
}

BodyLabel BPRecognizer::predicDense(IplImage* img, CvPoint p, int treeNum)
{
	BodyLabel label = BodyLabel_Unknown;

	DenseNode* node = &_denseForest[_denseRoot[treeNum]];
	while(node->rightIndex != 0xffff)
	{
		DenseFeature* f = &_denseFeature[node->fIndex];
		GAddDepthFeature fadd(cvPoint(f->u_x, f->u_y), cvPoint(f->v_x, f->v_y));
		float v = fadd.getValue(img, p);
		v <= node->tValue ? node = node+1 :
			node = &_denseForest[node->rightIndex];
	}
	
	label = node->tValue;
	return label;
}

void generateMaskImage( const IplImage* pDepthImg,IplImage* pMaskImg )
{
	static bool bFirst = true;
	static IplImage* prev_depth_img = NULL;
	if (bFirst)
	{
		memset(pMaskImg->imageData, 0x01, pMaskImg->imageSize);
		prev_depth_img = cvCreateImage(cvSize(IMAGE_WIDTH, IMAGE_HEIGHT), IPL_DEPTH_32S, 1);
		bFirst = false;
		//CapgPrintf("%s", "First BodyPartRecognized");
	}
	else
	{
		memset(pMaskImg->imageData, 0x00, pMaskImg->imageSize);

		int nDiffCount = 0;
		for (size_t y = 0; y < pDepthImg->height; y++)
		{
			int *pPrevDepth = (int*)(prev_depth_img->imageData + y * prev_depth_img->widthStep);
			int *pCurDepth = (int*)(pDepthImg->imageData + y * pDepthImg->widthStep);
			uchar *pMask = (uchar*)(pMaskImg->imageData + y * pMaskImg->widthStep);

			for (size_t x = 0; x < pDepthImg->width; x++)
			{
				if (abs(pPrevDepth[x] - pCurDepth[x]) > 50)
				{
					pMask[x] = 0xFF;
					nDiffCount++;
				}
			}
		}

		//CapgPrintf("different points is %d", nDiffCount);
	}
	memcpy(prev_depth_img->imageData, pDepthImg->imageData, pDepthImg->imageSize);
}

IplImage* convertSkeletonToImage( const CapgSkeleton&sk )
{
	IplImage* colorImg=cvCreateImage(cvSize(IMAGE_WIDTH,IMAGE_HEIGHT),IPL_DEPTH_8U, 3);
	cvThreshold(colorImg,colorImg,255,255,CV_THRESH_BINARY_INV);
	//CvScalar color1=cvScalar(255,0,0);
	//CvScalar color2=cvScalar(0,255,0);
	//CvScalar color3=cvScalar(0,0,255);
	//cvLine(imag,cvPoint(s[7].x(),s[7].y()),cvPoint(s[6].x(),s[6].y()),color1,4);
	///*cvLine(imag,cvPoint(s[7].x(),s[7].y()),cvPoint(s[0].x(),s[0].y()),color,4);*/
	//cvLine(imag,cvPoint(s[0].x(),s[0].y()),cvPoint(s[1].x(),s[1].y()),color2,4);
	//cvLine(imag,cvPoint(s[1].x(),s[1].y()),cvPoint(s[2].x(),s[2].y()),color2,4);
	///*cvLine(imag,cvPoint(s[7].x(),s[7].y()),cvPoint(s[3].x(),s[3].y()),color,4);*/
	//cvLine(imag,cvPoint(s[3].x(),s[3].y()),cvPoint(s[4].x(),s[4].y()),color3,4);
	//cvLine(imag,cvPoint(s[4].x(),s[4].y()),cvPoint(s[5].x(),s[5].y()),color3,4);
	for(int i=0; i<8; i++){
		cvCircle(colorImg, cvPoint(sk[i].x(), sk[i].y()), 2, cvScalar(0), 2);
	}
	cvLine(colorImg, cvPoint(sk[0].x(), sk[0].y()), cvPoint(sk[1].x(), sk[1].y()), cvScalar(0), 2);
	cvLine(colorImg, cvPoint(sk[1].x(), sk[1].y()), cvPoint(sk[2].x(), sk[2].y()), cvScalar(0), 2);
	cvLine(colorImg, cvPoint(sk[3].x(), sk[3].y()), cvPoint(sk[4].x(), sk[4].y()), cvScalar(0), 2);
	cvLine(colorImg, cvPoint(sk[4].x(), sk[4].y()), cvPoint(sk[5].x(), sk[5].y()), cvScalar(0), 2);
	cvLine(colorImg, cvPoint(sk[0].x(), sk[0].y()), cvPoint(sk[3].x(), sk[3].y()), cvScalar(0), 2);
	cvLine(colorImg, cvPoint(sk[6].x(), sk[6].y()), cvPoint(sk[7].x(), sk[7].y()), cvScalar(0), 2);

	//if(sk[9].x()!=0&&sk[9].y()!=0&&sk[8].x()!=0&&sk[8].y()!=0)
	//	cvLine(colorImg, cvPoint(sk[8].x(), sk[8].y()), cvPoint(sk[9].x(), sk[9].y()), cvScalar(0), 2);
	//if(sk[9].x()!=0&&sk[9].y()!=0&&sk[10].x()!=0&&sk[10].y()!=0)
	//	cvLine(colorImg, cvPoint(sk[9].x(), sk[9].y()), cvPoint(sk[10].x(), sk[10].y()), cvScalar(0), 2);
	//if(sk[11].x()!=0&&sk[11].y()!=0&&sk[12].x()!=0&&sk[12].y()!=0)
	//	cvLine(colorImg, cvPoint(sk[11].x(), sk[11].y()), cvPoint(sk[12].x(), sk[12].y()), cvScalar(0), 2);
	//if(sk[12].x()!=0&&sk[12].y()!=0&&sk[13].x()!=0&&sk[13].y()!=0)
	//	cvLine(colorImg, cvPoint(sk[12].x(), sk[12].y()), cvPoint(sk[13].x(), sk[13].y()), cvScalar(0), 2);

	cvCircle(colorImg, cvPoint(sk[8].x(), sk[8].y()), 2, cvScalar(0), 2);
	cvCircle(colorImg, cvPoint(sk[11].x(), sk[11].y()), 2, cvScalar(0), 2);
	if(sk[8].x()!=0&&sk[8].y()!=0&&sk[11].x()!=0&&sk[11].y()!=0)
	{
		cvLine(colorImg, cvPoint(sk[8].x(), sk[8].y()), cvPoint(sk[11].x(), sk[11].y()), cvScalar(0), 2);
		cvLine(colorImg, cvPoint(sk[7].x(), sk[7].y()),
			cvPoint((sk[8].x()+sk[11].x())/2, (sk[8].y()+sk[11].y())/2), cvScalar(0), 2);
	}

	return colorImg;
}
