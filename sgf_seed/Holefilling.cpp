
#include<iostream>
#include<cmath>
#include<queue>
#include<vector>
#include<opencv2/opencv.hpp>
#include"Holefilling.h"

using namespace std;
using namespace cv;

Holefilling::Holefilling():x(0),y(0),height(0),width(0){
	
}

Holefilling::~Holefilling(){

}
//get Gaussian function with parameter sigma
double Holefilling::get_Guassian_Function(double x,double y,double x1,double y1,double sigma){
	double distance = pow(x - x1, 2) + pow(y - y1, 2);
	return exp(-(distance / pow(sigma,2)));

}

void Holefilling::Normalized_Convolution(const int& r,const int& c,vector<int>& vec){
	
	
	if(r>0&&r<height&&c>0&&c<width){
		int mark=c+r*width;
	if(img.at<sgf_uchar>(r,c)==0&&!vec[mark]){
			sgf_uchar data_left =0;
			sgf_uchar data_right =0;
			sgf_uchar data_top =0;
			sgf_uchar data_bottom = 0;
			sgf_uchar data_left_top=0;
			sgf_uchar data_left_bottom=0;
			sgf_uchar data_right_top=0;
			sgf_uchar data_right_bottom=0;
			double left_Gaussian =0.0;
			double right_Gaussian =0.0;
			double top_Gaussian =0.0;
			double bottom_Gaussian =0.0;
			double left_top_Gaussian =0.0;
			double right_top_Gaussian =0.0;
			double left_bottom_Gaussian =0.0;
			double right_bottom_Gaussian =0.0;

			double sum_Gaussian=0.0;
			double 	left_Gaus_multi =0.0;
			double  right_Gaus_multi =0.0;
			double  top_Gaus_multi = 0.0;
			double  bottom_Gaus_multi =0.0;
			double 	left_top_Gaus_multi =0.0;
			double  right_top_Gaus_multi =0.0;
			double  left_bottom_Gaus_multi = 0.0;
			double  right_bottom_Gaus_multi =0.0;
			double  depth=0.0;
			double  sum_Gaus_multi=0.0;
			double sigma=10; 
			if(c-1>=0)data_left = img.at<sgf_uchar>(r,c-1);
			if(c+1<width)data_right = img.at<sgf_uchar>(r,c+1);
			if(r-1>=0)data_top = img.at<sgf_uchar>(r-1,c);
			if(r+1<height)data_bottom = img.at<sgf_uchar>(r+1,c);
			if(c-1>=0&&r-1>=0)data_left_top = img.at<sgf_uchar>(r-1,c-1);
			if(c+1<width&&r-1>=0)data_right_top = img.at<sgf_uchar>(r-1,c+1);
			if(c-1>=0&&r+1<height)data_left_bottom = img.at<sgf_uchar>(r+1,c-1);
			if(r+1<height&&c+1<width)data_right_bottom = img.at<sgf_uchar>(r+1,c+1);


			if(data_left||data_right||data_top||data_bottom
			||data_left_top||data_right_top||data_left_bottom||data_right_bottom){
			  
			   if(data_left)left_Gaussian = get_Guassian_Function(r, c, r, c-1, sigma);
			   if(data_right)right_Gaussian = get_Guassian_Function(r, c, r, c+1, sigma);
			   if(data_top)top_Gaussian = get_Guassian_Function(r, c, r-1, c, sigma);
			   if(data_bottom)bottom_Gaussian = get_Guassian_Function(r, c, r+1,c,sigma);
			   if(data_left_top)left_top_Gaussian = get_Guassian_Function(r, c, r-1, c-1, sigma);
			   if(data_right_top)right_top_Gaussian = get_Guassian_Function(r, c, r-1, c+1, sigma);
			   if(data_left_bottom)left_bottom_Gaussian = get_Guassian_Function(r, c, r+1, c-1, sigma);
			   if(data_right_bottom)right_bottom_Gaussian = get_Guassian_Function(r, c, r+1,c+1,sigma);
			   sum_Gaussian = (left_Gaussian + right_Gaussian + top_Gaussian + bottom_Gaussian
				   +left_top_Gaussian + right_top_Gaussian + left_bottom_Gaussian + right_bottom_Gaussian);
			   left_Gaus_multi = data_left*left_Gaussian;
			   right_Gaus_multi = data_right*right_Gaussian;
			   top_Gaus_multi = data_top*top_Gaussian;
			   bottom_Gaus_multi = data_bottom*bottom_Gaussian;
			   left_top_Gaus_multi = data_left_top*left_top_Gaussian;
			   right_top_Gaus_multi = data_right_top*right_top_Gaussian;
			   left_bottom_Gaus_multi = data_left_bottom*left_bottom_Gaussian;
			   right_bottom_Gaus_multi = data_right_bottom*right_bottom_Gaussian;
			   sum_Gaus_multi = (left_Gaus_multi + right_Gaus_multi + top_Gaus_multi + bottom_Gaus_multi
				   +left_top_Gaus_multi + right_top_Gaus_multi + left_bottom_Gaus_multi + right_bottom_Gaus_multi);
			   depth = sum_Gaus_multi / sum_Gaussian;
			  // img.at<cv::Vec3b>(r,c)[0]=saturate_cast<uchar>(depth);
			   //img.at<cv::Vec3b>(r,c)[1]=saturate_cast<uchar>(depth);
			   //img.at<cv::Vec3b>(r,c)[2]=saturate_cast<uchar>(depth);

			  img.at<sgf_uchar>(r,c)=saturate_cast<sgf_uchar>(depth);
			   vec[mark]=1;
			   Node n;
			   n.setX(r);
			   n.setY(c);
			  // n.x=r;
			   //n.y=c;
			   q.push(n);
			}
				         
	}
		
			}}



void Holefilling::BFS(){
	
	height=img.size().height;
	width=img.size().width;
	int sz=height*width;
	vector<int>vec(sz,0);
	for(int i=0;i<img.rows;i++){
		for(int j=0;j<img.cols;j++){
			  Normalized_Convolution(i,j,vec);  
			   
			   while(!q.empty()){
				   Node node=q.front();
			       q.pop();
				   Normalized_Convolution(i,j-1,vec);
				   Normalized_Convolution(i,j+1,vec);
				   Normalized_Convolution(i-1,j,vec);
				   Normalized_Convolution(i+1,j,vec);

				   Normalized_Convolution(i-1,j-1,vec);
				   Normalized_Convolution(i-1,j+1,vec);
				   Normalized_Convolution(i+1,j-1,vec);
				   Normalized_Convolution(i+1,j+1,vec);
		
			   }
			}

			}
		}

void Holefilling::setX(int x){
	this->x=x;

}
void Holefilling::setY(int y){

     this->y=y;

}
void Holefilling::setHeight(int h){
	this->height=h;
  
}
void Holefilling::setWidth(int w){
	this->width=w;
}
void Holefilling::setMat(cv::Mat& mat){
	this->img=mat;
}

