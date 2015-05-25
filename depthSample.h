#ifndef __depthSample_h_
#define __depthSample_h_

#include "opencv2/opencv.hpp"
#include <list>
#include <math.h>

using namespace std;

#ifndef BodyLabel_LU_Head
typedef uchar BodyLabel;
#define BodyLabel_LU_Head 0
#define BodyLabel_RU_Head 1
#define BodyLabel_LW_Head 2
#define BodyLabel_RW_Head 3
#define BodyLabel_Neck 4
#define BodyLabel_L_Shoulder 5
#define BodyLabel_R_Shoulder 6
#define BodyLabel_LU_Arm 7
#define BodyLabel_RU_Arm 8
#define BodyLabel_LW_Arm 9
#define BodyLabel_RW_Arm 10
#define BodyLabel_L_Elbow 11
#define BodyLabel_R_Elbow 12
#define BodyLabel_L_Wrist 13
#define BodyLabel_R_Wrist 14
#define BodyLabel_L_Hand 15
#define BodyLabel_R_Hand 16
#define BodyLabel_LU_Torso 17
#define BodyLabel_RU_Torso 18
#define BodyLabel_LW_Torso 19
#define BodyLabel_RW_Torso 20
#define BodyLabel_LU_Leg 21
#define BodyLabel_RU_Leg 22
#define BodyLabel_LW_Leg 23
#define BodyLabel_RW_Leg 24
#define BodyLabel_L_Knee 25
#define BodyLabel_R_Knee 26
#define BodyLabel_L_Ankle 27
#define BodyLabel_R_Ankle 28
#define BodyLabel_L_Foot 29
#define BodyLabel_R_Foot 30
#define BodyLabel_Unknown 99
#define BodyLabel_Background 100
#endif

#define Head_Merge 0xf000
#define Body_Merge 0x0f00
#define Leg_Merge  0x00f0
#define Arm_Merge  0x000f

struct Color
{
	unsigned int rgb;

	Color() : rgb(0){}
	Color(uchar rr, uchar gg, uchar bb) : rgb((rr<<16) + (gg<<8) + bb){}
	Color(const Color& c) : rgb(c.rgb){}

	uchar get_r(){ return rgb>>16; }
	uchar get_g(){ return (rgb<<16)>>24; }
	uchar get_b(){ return (rgb<<24)>>24; }

	bool equal(Color& c){
		if(abs(c.get_r() - get_r()) <= 10 &&
			abs(c.get_g() - get_g()) <= 10 &&
			abs(c.get_b() - get_b()) <= 10)
			return true;
		else return false;
	}

	bool operator<(const Color&  c) const{
		return (rgb< c.rgb);
	}
};

//***************************
//** DepthImage **
//** Base class of training samples / test samples / runtime input
//***************************
class DepthSample
{
public:
	string _depthFileName;
	string _labelFileName;

	int _minDepth;

	DepthSample(IplImage* depth, IplImage* label);
	DepthSample(string& depthfile, string& labelfile, int minDepth = 0)
		: _depthFileName(depthfile), _labelFileName(labelfile), _minDepth(minDepth),
		  _depthImg(0), _labelImg(0){}
	~DepthSample();

	void loadImage();
	void loadRuntimeImage();
	void releaseImage();

	void randomTrainPixel(int pCount);
	void clearTrainPixel();
	list<CvPoint>& getTrainPixel();

	IplImage* getDepthImg(){ return _depthImg; }
	IplImage* getlabelImg(){ return _labelImg; }

	static Color getColor(BodyLabel bodyLabel);
	static BodyLabel getLabel(Color& color);

	static void createLabelMap(int flag = 0);
	static void createColorMap();

	static bool similarPart(BodyLabel l1, BodyLabel l2);

	static int BodyLabel_Count;

private:
	void normalColorDepthImg(IplImage* dImg);
	void normalColorDepthImg_r(IplImage* dImg);
	void translateColorLabelImg(IplImage* lImg);

private:
	IplImage* _depthImg;
	IplImage* _labelImg;

	static string _rootPath;

	list<CvPoint> _trainPixel;

	static map<BodyLabel, Color> _labelColorMap;
	static map<Color, BodyLabel> _colorLabelMap;

	static bool _similarMatrix [31][31];

	int _cut_y;
};

#endif
