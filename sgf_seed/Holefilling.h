
#if !defined(_HOLEFILLING_H_INCLUDED_)
#define _HOLEFILLING_H_INCLUDED_

#include<opencv2/opencv.hpp>
#include<vector>
#include<queue>
#include"Node.h"
using namespace std;


class Holefilling{
public:
	Holefilling();
	virtual ~Holefilling();
	//get Gaussian function with parameter sigma
   double get_Guassian_Function(double x,double y,double x1,double y1,double sigma);

   void  Normalized_Convolution(const int& r,const int& c,vector<int>& vec);
   
   void BFS();

   void setX(int x);
   void setY(int y);
   void setHeight(int h);
   void setWidth(int w);
   void setMat(cv::Mat& mat);
   

private:
   int x;
   int y;
   int height;
   int width;
   cv::Mat img;
   queue<Node>q;
};



#endif

