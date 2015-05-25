#include "depthSample.h"
#include "bodyPartRecognizer.h"

int DepthSample::BodyLabel_Count = 31;
map<BodyLabel, Color> DepthSample::_labelColorMap = map<BodyLabel, Color>();
map<Color, BodyLabel> DepthSample::_colorLabelMap = map<Color, BodyLabel>();
bool DepthSample::_similarMatrix [31][31] = {};

string DepthSample::_rootPath = string("\\\\soso-pc\\RESULT\\");

void DepthSample::createColorMap(){
	_labelColorMap.clear();

	_labelColorMap.insert(make_pair(BodyLabel_LU_Head, Color(60, 60, 120)));
	_labelColorMap.insert(make_pair(BodyLabel_RU_Head, Color(60, 60, 240)));
	_labelColorMap.insert(make_pair(BodyLabel_LW_Head, Color(60, 60, 180)));
	_labelColorMap.insert(make_pair(BodyLabel_RW_Head, Color(60, 120, 60)));
	_labelColorMap.insert(make_pair(BodyLabel_Neck, Color(60, 180, 60)));
	_labelColorMap.insert(make_pair(BodyLabel_L_Shoulder, Color(60, 240, 60)));
	_labelColorMap.insert(make_pair(BodyLabel_R_Shoulder, Color(120, 60, 60)));
	_labelColorMap.insert(make_pair(BodyLabel_LU_Arm, Color(180, 60, 60)));
	_labelColorMap.insert(make_pair(BodyLabel_RU_Arm, Color(120, 120, 180)));
	_labelColorMap.insert(make_pair(BodyLabel_LW_Arm, Color(240, 60, 60)));
	_labelColorMap.insert(make_pair(BodyLabel_RW_Arm, Color(120, 120, 240)));
	_labelColorMap.insert(make_pair(BodyLabel_L_Elbow, Color(120, 60, 120)));
	_labelColorMap.insert(make_pair(BodyLabel_R_Elbow, Color(120, 180, 120)));
	_labelColorMap.insert(make_pair(BodyLabel_L_Wrist, Color(240, 120, 120)));
	_labelColorMap.insert(make_pair(BodyLabel_R_Wrist, Color(120, 240, 120)));
	_labelColorMap.insert(make_pair(BodyLabel_L_Hand, Color(180, 120, 120)));
	_labelColorMap.insert(make_pair(BodyLabel_R_Hand, Color(60, 120, 120)));
	_labelColorMap.insert(make_pair(BodyLabel_LU_Torso, Color(240, 240, 60)));
	_labelColorMap.insert(make_pair(BodyLabel_RU_Torso, Color(240, 240, 180)));
	_labelColorMap.insert(make_pair(BodyLabel_LW_Torso, Color(240, 180, 240)));
	_labelColorMap.insert(make_pair(BodyLabel_RW_Torso, Color(240, 60, 240)));
	_labelColorMap.insert(make_pair(BodyLabel_LU_Leg, Color(180, 240, 240)));
	_labelColorMap.insert(make_pair(BodyLabel_RU_Leg, Color(240, 120, 60)));
	_labelColorMap.insert(make_pair(BodyLabel_LW_Leg, Color(60, 240, 240)));
	_labelColorMap.insert(make_pair(BodyLabel_RW_Leg, Color(240, 60, 120)));
	_labelColorMap.insert(make_pair(BodyLabel_L_Knee, Color(60, 180, 240)));
	_labelColorMap.insert(make_pair(BodyLabel_R_Knee, Color(180, 60, 120)));
	_labelColorMap.insert(make_pair(BodyLabel_L_Ankle, Color(180, 240, 60)));
	_labelColorMap.insert(make_pair(BodyLabel_R_Ankle, Color(240, 180, 60)));
	_labelColorMap.insert(make_pair(BodyLabel_L_Foot, Color(60, 240, 120)));
	_labelColorMap.insert(make_pair(BodyLabel_R_Foot, Color(240, 60, 180)));

	_labelColorMap.insert(make_pair(BodyLabel_Unknown, Color(0, 0, 0)));
	_labelColorMap.insert(make_pair(BodyLabel_Background, Color(255, 255, 255)));
}

void DepthSample::createLabelMap(int flag)
{
	_colorLabelMap.clear();

	for(int i=0; i<31; i++){
		for(int j=0; j<31; j++){
			if(i == j) _similarMatrix[i][j] = true;
			else _similarMatrix[i][j] = false;
		}
	}

	if(flag & Head_Merge){
		_colorLabelMap.insert(make_pair(Color(60, 60, 120), BodyLabel_LU_Head));
		_colorLabelMap.insert(make_pair(Color(60, 60, 240), BodyLabel_LU_Head));
		_colorLabelMap.insert(make_pair(Color(60, 60, 180), BodyLabel_LU_Head));
		_colorLabelMap.insert(make_pair(Color(60, 120, 60), BodyLabel_LU_Head));
		_colorLabelMap.insert(make_pair(Color(60, 180, 60), BodyLabel_LU_Head));
		DepthSample::BodyLabel_Count -= 4;
		_similarMatrix[BodyLabel_LU_Head][BodyLabel_Neck] = true;
		_similarMatrix[BodyLabel_LU_Head][BodyLabel_Neck] = true;
	}else{
		_colorLabelMap.insert(make_pair(Color(60, 60, 120), BodyLabel_LU_Head));
		_colorLabelMap.insert(make_pair(Color(60, 60, 240), BodyLabel_RU_Head));
		_colorLabelMap.insert(make_pair(Color(60, 60, 180), BodyLabel_LW_Head));
		_colorLabelMap.insert(make_pair(Color(60, 120, 60), BodyLabel_RW_Head));
		_colorLabelMap.insert(make_pair(Color(60, 180, 60), BodyLabel_Neck));
		_similarMatrix[BodyLabel_LU_Head][BodyLabel_RU_Head] = true;
		_similarMatrix[BodyLabel_LU_Head][BodyLabel_LW_Head] = true;
		_similarMatrix[BodyLabel_LU_Head][BodyLabel_RW_Head] = true;
		_similarMatrix[BodyLabel_LW_Head][BodyLabel_Neck] = true;
		_similarMatrix[BodyLabel_RW_Head][BodyLabel_Neck] = true;
	}
	if(flag & Arm_Merge){
		_colorLabelMap.insert(make_pair(Color(60, 240, 60), BodyLabel_L_Shoulder));
		_colorLabelMap.insert(make_pair(Color(120, 60, 60), BodyLabel_R_Shoulder));
		_colorLabelMap.insert(make_pair(Color(180, 60, 60), BodyLabel_L_Elbow));
		_colorLabelMap.insert(make_pair(Color(120, 120, 180), BodyLabel_R_Elbow));
		_colorLabelMap.insert(make_pair(Color(240, 60, 60), BodyLabel_L_Elbow));
		_colorLabelMap.insert(make_pair(Color(120, 120, 240), BodyLabel_R_Elbow));
		_colorLabelMap.insert(make_pair(Color(120, 60, 120), BodyLabel_L_Elbow));
		_colorLabelMap.insert(make_pair(Color(120, 180, 120), BodyLabel_R_Elbow));
		_colorLabelMap.insert(make_pair(Color(240, 120, 120), BodyLabel_L_Hand));
		_colorLabelMap.insert(make_pair(Color(120, 240, 120), BodyLabel_R_Hand));
		_colorLabelMap.insert(make_pair(Color(180, 120, 120), BodyLabel_L_Hand));
		_colorLabelMap.insert(make_pair(Color(60, 120, 120), BodyLabel_R_Hand));
		DepthSample::BodyLabel_Count -= 4;
	}else{
		_colorLabelMap.insert(make_pair(Color(60, 240, 60), BodyLabel_L_Shoulder));
		_colorLabelMap.insert(make_pair(Color(120, 60, 60), BodyLabel_R_Shoulder));
		_colorLabelMap.insert(make_pair(Color(180, 60, 60), BodyLabel_LU_Arm));
		_colorLabelMap.insert(make_pair(Color(120, 120, 180), BodyLabel_RU_Arm));
		_colorLabelMap.insert(make_pair(Color(240, 60, 60), BodyLabel_LW_Arm));
		_colorLabelMap.insert(make_pair(Color(120, 120, 240), BodyLabel_RW_Arm));
		_colorLabelMap.insert(make_pair(Color(120, 60, 120), BodyLabel_L_Elbow));
		_colorLabelMap.insert(make_pair(Color(120, 180, 120), BodyLabel_R_Elbow));
		_colorLabelMap.insert(make_pair(Color(240, 120, 120), BodyLabel_L_Wrist));
		_colorLabelMap.insert(make_pair(Color(120, 240, 120), BodyLabel_R_Wrist));
		_colorLabelMap.insert(make_pair(Color(180, 120, 120), BodyLabel_L_Hand));
		_colorLabelMap.insert(make_pair(Color(60, 120, 120), BodyLabel_R_Hand));
	}
	if(flag & Body_Merge){
		_colorLabelMap.insert(make_pair(Color(240, 240, 60), BodyLabel_LU_Torso));
		_colorLabelMap.insert(make_pair(Color(240, 240, 180), BodyLabel_LU_Torso));
		_colorLabelMap.insert(make_pair(Color(240, 180, 240), BodyLabel_LU_Torso));
		_colorLabelMap.insert(make_pair(Color(240, 60, 240), BodyLabel_LU_Torso));
		DepthSample::BodyLabel_Count -= 3;
	}else{
		_colorLabelMap.insert(make_pair(Color(240, 240, 60), BodyLabel_LU_Torso));
		_colorLabelMap.insert(make_pair(Color(240, 240, 180), BodyLabel_RU_Torso));
		_colorLabelMap.insert(make_pair(Color(240, 180, 240), BodyLabel_LW_Torso));
		_colorLabelMap.insert(make_pair(Color(240, 60, 240), BodyLabel_RW_Torso));
	}
	if(flag & Leg_Merge){
		//_colorLabelMap.insert(make_pair(Color(180, 240, 240), BodyLabel_LU_Leg));
		//_colorLabelMap.insert(make_pair(Color(240, 120, 60), BodyLabel_LU_Leg));
		//_colorLabelMap.insert(make_pair(Color(60, 240, 240), BodyLabel_LU_Leg));
		//_colorLabelMap.insert(make_pair(Color(240, 60, 120), BodyLabel_LU_Leg));
		//_colorLabelMap.insert(make_pair(Color(60, 180, 240), BodyLabel_LU_Leg));
		//_colorLabelMap.insert(make_pair(Color(180, 60, 120), BodyLabel_LU_Leg));
		//_colorLabelMap.insert(make_pair(Color(180, 240, 60), BodyLabel_LU_Leg));
		//_colorLabelMap.insert(make_pair(Color(240, 180, 60), BodyLabel_LU_Leg));
		//_colorLabelMap.insert(make_pair(Color(60, 240, 120), BodyLabel_LU_Leg));
		//_colorLabelMap.insert(make_pair(Color(240, 60, 180), BodyLabel_LU_Leg));
		//DepthSample::BodyLabel_Count -= 9;
		_colorLabelMap.insert(make_pair(Color(180, 240, 240), BodyLabel_LU_Leg));
		_colorLabelMap.insert(make_pair(Color(240, 120, 60), BodyLabel_RU_Leg));
		_colorLabelMap.insert(make_pair(Color(60, 240, 240), BodyLabel_LW_Leg));
		_colorLabelMap.insert(make_pair(Color(240, 60, 120), BodyLabel_RW_Leg));
		_colorLabelMap.insert(make_pair(Color(60, 180, 240), BodyLabel_LW_Leg));
		_colorLabelMap.insert(make_pair(Color(180, 60, 120), BodyLabel_RW_Leg));
		_colorLabelMap.insert(make_pair(Color(180, 240, 60), BodyLabel_LW_Leg));
		_colorLabelMap.insert(make_pair(Color(240, 180, 60), BodyLabel_RW_Leg));
		_colorLabelMap.insert(make_pair(Color(60, 240, 120), BodyLabel_L_Foot));
		_colorLabelMap.insert(make_pair(Color(240, 60, 180), BodyLabel_R_Foot));
		DepthSample::BodyLabel_Count -= 4;
	}else{
		_colorLabelMap.insert(make_pair(Color(180, 240, 240), BodyLabel_LU_Leg));
		_colorLabelMap.insert(make_pair(Color(240, 120, 60), BodyLabel_RU_Leg));
		_colorLabelMap.insert(make_pair(Color(60, 240, 240), BodyLabel_LW_Leg));
		_colorLabelMap.insert(make_pair(Color(240, 60, 120), BodyLabel_RW_Leg));
		_colorLabelMap.insert(make_pair(Color(60, 180, 240), BodyLabel_L_Knee));
		_colorLabelMap.insert(make_pair(Color(180, 60, 120), BodyLabel_R_Knee));
		_colorLabelMap.insert(make_pair(Color(180, 240, 60), BodyLabel_L_Ankle));
		_colorLabelMap.insert(make_pair(Color(240, 180, 60), BodyLabel_R_Ankle));
		_colorLabelMap.insert(make_pair(Color(60, 240, 120), BodyLabel_L_Foot));
		_colorLabelMap.insert(make_pair(Color(240, 60, 180), BodyLabel_R_Foot));
	}

	_colorLabelMap.insert(make_pair(Color(0, 0, 0), BodyLabel_Unknown));
	_colorLabelMap.insert(make_pair(Color(255, 255, 255), BodyLabel_Background));

}

bool DepthSample::similarPart(BodyLabel l1, BodyLabel l2)
{
	if(l1 > l2){
		BodyLabel temp = l2;
		l2 = l1;
		l1 = temp;
	}
	return _similarMatrix[l1][l2];
}

DepthSample::DepthSample(IplImage* depthImg, IplImage* lableImg)
{
	_depthImg = cvCloneImage(depthImg);
	_labelImg = cvCloneImage(lableImg);
}

DepthSample::~DepthSample()
{
	releaseImage();
}

void DepthSample::loadImage()
{
	if(_depthFileName.empty() || _labelFileName.empty())
		return;

	if(_depthImg || _labelImg) releaseImage();

	string filename = _rootPath + _depthFileName;
	IplImage* colorDepth = cvLoadImage(filename.data());
	if(colorDepth == 0){
		ERROR_MSG(string("Cannot open image: ")+_depthFileName, 0);
		return;
	}
	filename = _rootPath + _labelFileName;
	IplImage* colorLabel = cvLoadImage(filename.data());
	if(colorLabel == 0){
		ERROR_MSG(string("Cannot open image: ")+_labelFileName, 0);
		releaseImage();
		return;
	}

	cv::RNG* rng = BPRecognizer::getRng();
	int code = (*rng).uniform(0, 10);
	if(code == 0)
		_cut_y = (*rng).uniform((int)(colorDepth->height*0.5), (int)(colorDepth->height*0.6));
	else if(code >= 1 && code <= 2)
		_cut_y = (*rng).uniform((int)(colorDepth->height*0.6), (int)(colorDepth->height*0.7));
	else if(code >= 3 && code <= 5)
		_cut_y = (*rng).uniform((int)(colorDepth->height*0.7), (int)(colorDepth->height*0.9));
	else _cut_y = (*rng).uniform((int)(colorDepth->height*0.9), colorDepth->height);

	_depthImg = cvCreateImage(cvSize(colorDepth->width, colorDepth->height), IPL_DEPTH_32S, 1);
	_labelImg = cvCreateImage(cvSize(colorLabel->width, colorLabel->height), IPL_DEPTH_8U, 1);
	translateColorLabelImg(colorLabel);
	normalColorDepthImg(colorDepth);

	for(int y=0; y<colorDepth->height; y++){
		uchar* d_data = (uchar*)(colorDepth->imageData + y*colorDepth->widthStep);
		unsigned int* nd_data = (unsigned int*)(_depthImg->imageData + y*_depthImg->widthStep);
		for(int x=0; x<colorDepth->width; x++){
			// b+256*g
			if(nd_data[x]==255+256*255+65536*255)
			{
				d_data[3*x] = d_data[3*x+1] = d_data[3*x+2] = 255;
			}else{
				d_data[3*x+2] = 0;
				d_data[3*x+1] = (nd_data[x] << 16) >> 24;
				d_data[3*x] = (nd_data[x] << 24) >> 24;
			}
		}
	}
	cvSaveImage("temp_depth.png", colorDepth);

	for(int y=0; y<colorLabel->height; y++){
		uchar* d_data = (uchar*)(colorLabel->imageData + y*colorLabel->widthStep);
		uchar* nd_data = (uchar*)(_labelImg->imageData + y*_labelImg->widthStep);
		for(int x=0; x<colorLabel->width; x++){
			Color c = getColor(nd_data[x]);
			d_data[3*x] = c.get_b();
			d_data[3*x+1] = c.get_g();
			d_data[3*x+2] = c.get_r();
		}
	}
	//cvSaveImage("temp_color.png", colorLabel);

	cvReleaseImage(&colorLabel);
	cvReleaseImage(&colorDepth);
}

void DepthSample::loadRuntimeImage()
{
	if(_depthFileName.empty())
		return;

	if(_depthImg || _labelImg) releaseImage();

	string filename = _rootPath + _depthFileName;
	IplImage* colorDepth = cvLoadImage(filename.data());
	if(colorDepth == 0){
		ERROR_MSG(string("Cannot open image: ")+_depthFileName, 0);
		return;
	}

	_depthImg = cvCreateImage(cvSize(colorDepth->width, colorDepth->height), IPL_DEPTH_32S, 1);
	normalColorDepthImg_r(colorDepth);

	cvReleaseImage(&colorDepth);
}

void DepthSample::releaseImage()
{
	if(_depthImg){
		cvReleaseImage(&_depthImg);
		_depthImg = 0;
	}
	if(_labelImg){
		cvReleaseImage(&_labelImg);
		_labelImg = 0;
	}
}

void DepthSample::randomTrainPixel(int pCount)
{
	clearTrainPixel();
	cv::RNG* rng = BPRecognizer::getRng();
	map<BodyLabel, int> randomCount;
	int maxCount = pCount / BodyLabel_Count;
	int avgCount = maxCount*BodyLabel_Count;
	while(_trainPixel.size() < avgCount){
		int p_x = (*rng)(_depthImg->width);
		int p_y = (*rng)(_depthImg->height);
		BodyLabel l = CV_IMAGE_ELEM(_labelImg, BodyLabel, p_y, p_x);
		if(l != BodyLabel_Background && l!=BodyLabel_Unknown){
			map<BodyLabel, int>::iterator it = randomCount.find(l);
			if(it==randomCount.end()){
				randomCount.insert(make_pair(l, 1));
				_trainPixel.push_back(cvPoint(p_x, p_y));
				if(_trainPixel.size() == maxCount*randomCount.size())
					break;
			}else{
				it->second++;
				if(it->second <= maxCount){
					_trainPixel.push_back(cvPoint(p_x, p_y));
					if(_trainPixel.size() == maxCount*randomCount.size())
						break;
				}
				else
					it->second = maxCount;
			}
		}
	}
	maxCount = pCount / randomCount.size();
	avgCount = maxCount*randomCount.size();
	while(_trainPixel.size() < avgCount){
		int p_x = (*rng)(_depthImg->width);
		int p_y = (*rng)(_depthImg->height);
		BodyLabel l = CV_IMAGE_ELEM(_labelImg, BodyLabel, p_y, p_x);
		if(l != BodyLabel_Background && l!=BodyLabel_Unknown){
			map<BodyLabel, int>::iterator it = randomCount.find(l);
			if(it==randomCount.end()){
				randomCount.insert(make_pair(l, 1));
				_trainPixel.push_back(cvPoint(p_x, p_y));
			}else{
				it->second++;
				if(it->second <= maxCount)
					_trainPixel.push_back(cvPoint(p_x, p_y));
				else
					it->second = maxCount;
			}
		}
	}
	while(_trainPixel.size() < pCount){
		int p_x = (*rng)(_depthImg->width);
		int p_y = (*rng)(_depthImg->height);
		BodyLabel l = CV_IMAGE_ELEM(_labelImg, BodyLabel, p_y, p_x);
		if(l != BodyLabel_Background && l!=BodyLabel_Unknown)
			_trainPixel.push_back(cvPoint(p_x, p_y));
	}
}

void DepthSample::clearTrainPixel()
{
	_trainPixel.clear();
	_trainPixel.resize(0);
}

list<CvPoint>& DepthSample::getTrainPixel()
{
	return _trainPixel;
}

#define AlignDepth 1800

void DepthSample::normalColorDepthImg(IplImage* dImg)
{
	cv::RNG* rng = BPRecognizer::getRng();
	for(int y=0; y<_cut_y; y++){
		uchar* d_data = (uchar*)(dImg->imageData + y*dImg->widthStep);
		unsigned int* nd_data = (unsigned int*)(_depthImg->imageData + y*_depthImg->widthStep);
		for(int x=0; x<dImg->width; x++){
			// b+256*g
			nd_data[x] = (d_data[3*x] + 256*d_data[3*x+1] + 65536*d_data[3*x+2]);
			if(nd_data[x]!=BACKGROUNG){
				// shift depth
				//nd_data[x] = nd_data[x]*2+400;
				nd_data[x] = (nd_data[x] - _minDepth) * 2 + AlignDepth;
				// reduce sampling rate and add noise
				nd_data[x] = ((int)(nd_data[x]/13))*13 + (*rng)(3);
			}
		}
	}
	// set cut to background
	for(int y=_cut_y; y<dImg->height; y++){
		unsigned int* nd_data = (unsigned int*)(_depthImg->imageData + y*_depthImg->widthStep);
		for(int x=0; x<dImg->width; x++){
			nd_data[x] = BACKGROUNG;
		}
	}
	// expansion
	IplImage* tempImg = cvCloneImage(_depthImg);
	for(int y=0; y<tempImg->height; y++){
		unsigned int* nd_data = (unsigned int*)(tempImg->imageData + y*tempImg->widthStep);
		unsigned int* rd_data = (unsigned int*)(_depthImg->imageData + y*_depthImg->widthStep);
		for(int x=0; x<tempImg->width; x++){
			if(nd_data[x]==BACKGROUNG){
				unsigned int d = 0; int count = 0;
				for(int yy=-3; yy<=3; yy++){
					int yyy = y+yy;
					if(yyy < 0 || yyy >= tempImg->height) continue;
					unsigned int* d_data = (unsigned int*)(tempImg->imageData + yyy*tempImg->widthStep);
					for(int xx=-3; xx<=3; xx++){
						int xxx = x+xx;
						if(xxx < 0 || xxx >= tempImg->width) continue;
						if(d_data[xxx]!=BACKGROUNG){
							count++;
							d+=d_data[xxx];
						}
					}
				}
				if(count!=0){
					rd_data[x] = d / count;
				}
			}
		}
	}
	cvReleaseImage(&tempImg);
}

void DepthSample::normalColorDepthImg_r(IplImage* dImg)
{
	for(int y=0; y<dImg->height; y++){
		uchar* d_data = (uchar*)(dImg->imageData + y*dImg->widthStep);
		unsigned int* nd_data = (unsigned int*)(_depthImg->imageData + y*_depthImg->widthStep);
		for(int x=0; x<dImg->width; x++){
			// b+256*g
			nd_data[x] = (d_data[3*x] + 256*d_data[3*x+1] + 65536*d_data[3*x+2]);
		}
	}
}

void DepthSample::translateColorLabelImg(IplImage* lImg)
{
	for(int y=0; y<_cut_y; y++){
		uchar* l_data = (uchar*)(_labelImg->imageData + y*_labelImg->widthStep);
		uchar* c_data = (uchar*)(lImg->imageData + y*lImg->widthStep);
		for(int x=0; x<_labelImg->width; x++){
			Color c(c_data[3*x+2], c_data[3*x+1], c_data[3*x]);
			BodyLabel label = getLabel(c);
			l_data[x] = label;
		}
	}
	// set cut to background
	for(int y=_cut_y; y<lImg->height; y++){
		uchar* l_data = (uchar*)(_labelImg->imageData + y*_labelImg->widthStep);
		for(int x=0; x<_labelImg->width; x++){
			l_data[x] = BodyLabel_Background;
		}
	}
	//// expansion
	//IplImage* tempImg = cvCloneImage(_labelImg);
	//for(int y=0; y<tempImg->height; y++){
	//	uchar* l_data = (uchar*)(tempImg->imageData + y*tempImg->widthStep);
	//	uchar* rl_data = (uchar*)(_labelImg->imageData + y*_labelImg->widthStep);
	//	for(int x=0; x<tempImg->width; x++){
	//		if(l_data[x]==BodyLabel_Background || l_data[x]==BodyLabel_Unknown){
	//			for(int yy=-3; yy<=3; yy++){
	//				int yyy = y+yy;
	//				if(yyy < 0 || yyy >= tempImg->height) continue;
	//				uchar* ll_data = (uchar*)(tempImg->imageData + yyy*tempImg->widthStep);
	//				vector<int> count;
	//				count.resize(31);
	//				for(int i=0; i<31; i++) count[i]=0;
	//				for(int xx=-3; xx<=3; xx++){
	//					int xxx = x+xx;
	//					if(xxx < 0 || xxx >= tempImg->height) continue;
	//					if(ll_data[xxx]!=BodyLabel_Background && ll_data[xxx]!=BodyLabel_Unknown){
	//						count[ll_data[xxx]]++;
	//					}
	//				}
	//				BodyLabel maxLabel = BodyLabel_Background;
	//				int maxCount = 0;
	//				for(int i=0; i<31; i++){
	//					if(count[i] > maxCount){
	//						maxCount = count[i];
	//						maxLabel = i;
	//					}
	//				}
	//				rl_data[x] = maxLabel;
	//			}
	//		}
	//	}
	//}
	//cvReleaseImage(&tempImg);
}

BodyLabel DepthSample::getLabel(Color& color)
{
	map<Color, BodyLabel>::iterator it = _colorLabelMap.find(color);
	if(it != _colorLabelMap.end()) return it->second;
	else return BodyLabel_Unknown;
}

Color DepthSample::getColor(BodyLabel bodyLabel)
{
	return _labelColorMap[bodyLabel];
}