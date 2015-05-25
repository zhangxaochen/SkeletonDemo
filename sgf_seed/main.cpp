#include <stdlib.h>  
#include <iostream>  
#include <string> 
#include <fstream>
#include <time.h>
#include <vector>
//¡¾1¡¿  
#include <XnCppWrapper.h> 
#include "opencv2/opencv.hpp"  
#include "opencv/cv.h"  
#include "opencv/highgui.h" 


#include "sgf_segment.h"

using namespace std;  
using namespace cv;  
using namespace sgf;

void CheckOpenNIError( XnStatus result, string status )  
{   
	if( result != XN_STATUS_OK )   
		cerr << status << " Error: " << xnGetStatusString( result ) << endl;  
}  

int main( int argc, char** argv )  
{  
	XnStatus result = XN_STATUS_OK;    
	xn::DepthMetaData depthMD;  
	xn::ImageMetaData imageMD;  

	//OpenCV  
	IplImage*  imgDepth16u=cvCreateImage(cvSize(320,240),IPL_DEPTH_16U,1);  
	//IplImage* imgRGB8u=cvCreateImage(cvSize(320,240),IPL_DEPTH_8U,3);  
	IplImage*  depthShow=cvCreateImage(cvSize(320,240),IPL_DEPTH_8U,1);  
	//IplImage* imageShow=cvCreateImage(cvSize(320,240),IPL_DEPTH_8U,3);  
	cvNamedWindow("depth",1);  
	//cvNamedWindow("image",1);  
	char key=0;  

	//¡¾2¡¿  
	// context   
	xn::Context context;   
	result = context.Init();   
	CheckOpenNIError( result, "initialize context" );    

	// creategenerator    
	xn::DepthGenerator depthGenerator;    
	result = depthGenerator.Create( context );   
	CheckOpenNIError( result, "Create depth generator" );    
	//xn::ImageGenerator imageGenerator;  
	//result = imageGenerator.Create( context );   
	//CheckOpenNIError( result, "Create image generator" );  

	//¡¾3¡¿  
	//map mode    
	XnMapOutputMode mapMode;   
	mapMode.nXRes = 320;    
	mapMode.nYRes = 240;   
	mapMode.nFPS = 30;   
	result = depthGenerator.SetMapOutputMode( mapMode );    
	//result = imageGenerator.SetMapOutputMode( mapMode );    

	//¡¾4¡¿  
	// correct view port    
	//depthGenerator.GetAlternativeViewPointCap().SetViewPoint( imageGenerator );   

	//¡¾5¡¿  
	//read data  
	result = context.StartGeneratingAll();    
	//¡¾6¡¿  
	result = context.WaitNoneUpdateAll();    

	//sgf
	segment my_seg(true);
	my_seg.read_config();
	my_seg.set_headTemplate2D();

	while( (key!=27) && !(result = context.WaitNoneUpdateAll( ))  )   
	{    
		int time=clock();
		//get meta data  
		depthGenerator.GetMetaData(depthMD);   
		//imageGenerator.GetMetaData(imageMD);  

		//¡¾7¡¿  
		//OpenCV output  
		memcpy(imgDepth16u->imageData,depthMD.Data(),320*240*2);  
		cvConvertScale(imgDepth16u,depthShow,255/4096.0,0);  
		//memcpy(imgRGB8u->imageData,imageMD.Data(),320*240*3);  
		//cvCvtColor(imgRGB8u,imageShow,CV_RGB2BGR);  
		cvShowImage("depth", depthShow);  
		//cvShowImage("image",imageShow);  

		//sgf
		Mat sgf_depth=Mat(imgDepth16u);
//		imshow("sgf_depth", sgf_depth);
		sgf_depth.convertTo(sgf_depth,CV_32F);
// 		Mat roimat = sgf_depth(Rect(100,100, 30, 30));
// 		cout<<"roimat:\n"<<roimat<<endl;
// 		imshow("roimat", roimat);
// 		for (int i=100;i<110;++i)
// 		{
// 			for (int j=100;j<110;++j)
// 				cout<<sgf_depth.at<ushort>(i,j)<<' ';
// 			cout<<endl;
// 		}
// 		double dmin,dmax;
// 		minMaxLoc(sgf_depth,&dmin,&dmax);
// 		cout<<"---------------dmin,dmax: "<<dmin<<", "<<dmax<<endl;

// 		imshow("depth -------",sgf_depth);
// 		waitKey(1);
		my_seg.set_depthMap(sgf_depth);
		my_seg.compute();
		Mat result=my_seg.get_result();
// 		minMaxLoc(sgf_depth,&dmin,&dmax);
// 		cout<<"---------------dmin,dmax: "<<dmin<<", "<<dmax<<endl;
// 		imshow("my result",result);
// 		waitKey(1);
//		cout<<"time cost:"<<clock()-time<<" ms\n";
		key=cvWaitKey(1);  
	}  

	//destroy  
	cvDestroyWindow("depth");  
	//cvDestroyWindow("image");  
	cvReleaseImage(&imgDepth16u);  
	//cvReleaseImage(&imgRGB8u);  
	cvReleaseImage(&depthShow);  
	//cvReleaseImage(&imageShow);  
	context.StopGeneratingAll();  
	context.Shutdown();  
	return 0;  
} 