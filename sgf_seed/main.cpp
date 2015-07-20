#include <XnCppWrapper.h> 
#include "opencv/cv.h"  
#include "opencv2/opencv.hpp"
#include "opencv/highgui.h"  

#include "sgf_segment.h"

#include<string>
#include<iostream>
#include<vector>
#include<fstream>
#include<time.h>
using namespace std;
using namespace cv;
using namespace sgf;
vector<IplImage*> depths,images;
void OpenNItoOpenCV();
int main(int argc, char* argv[])
{
	//string video = "segment_test_1.oni";
	//string video="sun_han_short.oni";
	//string video="sgf-sss.oni";
	//string video="sgf-lyh-stand.oni";
	//string video="sun_walk.oni";
	//string video="zc-stand-wo-feet.oni";
	OpenNItoOpenCV();     //����ȡ����oni��Ϣת��Ϊopencv�ܹ������ͼ��
	return 0;
}
void OpenNItoOpenCV()
{
	segment my_seg;
	my_seg.read_config("config.txt");
	string sFilename=my_seg.videoname;

	XnStatus nRetVal = XN_STATUS_OK;
	xn::Context g_context;     //���������Ķ���
	nRetVal=g_context.Init();     //�����Ķ����ʼ��          
	xn::Player plyr;
	g_context.OpenFileRecording(sFilename.c_str(), plyr);     //�����е�oni�ļ�
plyr.SeekToFrame("MyDepth", 999, XN_PLAYER_SEEK_SET);
	xn::ImageGenerator g_image;     //����image generator
	nRetVal=g_context.FindExistingNode(XN_NODE_TYPE_IMAGE,g_image);     //��ȡoni�ļ��е�image�ڵ�
	xn::DepthGenerator g_depth;     //����depth generator
	nRetVal=g_context.FindExistingNode(XN_NODE_TYPE_DEPTH,g_depth);     //��ȡoni�ļ��е�depth�ڵ�
	xn::ImageMetaData g_imd;     //����image�ڵ�Ԫ���ݶ���    
	xn::DepthMetaData g_dmd;     //����depth�ڵ�Ԫ���ݶ���

	int nWidth;     //oni�����лҶ�/��ɫͼ��Ŀ��
	int nHeight;    //oni�����лҶ�/��ɫͼ��ĸ߶�
	IplImage *g_pImgColor=0;     //��������opencv��ʾ�Ĳ�ɫͼ��
	IplImage *g_pImgDepth=0;     //��������opencv��ʾ�����ͼ��
	IplImage *imgDepth16u=0;     //�洢�Ҷ���Ϣ
	IplImage *imgRGB8u=0;        //�洢��ɫ��Ϣ
	//����oni���ͼ��IplImage*��ʽvector
	XnUInt32 frameDepth = 0;
	int number=0;

	//������segment���������ݴ������ʾ
	//segment my_seg(true,true,true,true,true,false);

// 	int fourcc    = CV_FOURCC('D','I','V','X'); //���������
// 	VideoWriter writer1,writer2,writer3,writer4;
// 	writer1=VideoWriter("only_resopnse_threshold.avi",fourcc,30,Size(320,240),false);
// 	writer2=VideoWriter("response_and_ckb.avi",fourcc,30,Size(320,240),false);
// 	writer3=VideoWriter("response_ckb_and_headsize.avi",fourcc,30,Size(320,240),false);
// 	writer4=VideoWriter("all_methods.avi",fourcc,30,Size(320,240),false);

	int begin=clock();
	while(1)
	{    
		nRetVal = g_context.WaitOneUpdateAll(g_depth);     //�����������
		if(nRetVal!=XN_STATUS_OK)     //�ж��Ƿ���³ɹ�
		{
			printf("failed update depth image/n");
			continue;
		}
		if(g_depth.GetFrameID()<frameDepth)break;     //�ж��Ƿ�ѭ������
		else
		{
			g_depth.GetMetaData(g_dmd);     //��ȡg_depth��Ԫ����g_dmd
			if(g_pImgDepth==0)     //���ݵ�ǰ�����ͼ��Ԫ��������g_pImgDepth�Ĵ�С��ͨ����
			{
				nWidth=g_dmd.XRes();
				nHeight=g_dmd.YRes();
				g_pImgDepth=cvCreateImage(cvSize(nWidth,nHeight),8,1);
				imgDepth16u=cvCreateImage(cvSize(nWidth,nHeight),IPL_DEPTH_16U,1);
				cvZero(g_pImgDepth);
				cvZero(imgDepth16u);
			}
			memcpy(imgDepth16u->imageData,g_dmd.Data(),nWidth*nHeight*2);
			cvConvertScale(imgDepth16u,g_pImgDepth,255/9000.0,0);
			//cvEqualizeHist(g_pImgDepth,g_pImgDepth);     //ֱ��ͼ���⻯��������Ȳ��
			depths.push_back(g_pImgDepth);     //���浽vector��
			frameDepth = g_depth.GetFrameID();     //��¼��ǰframeID
			cvShowImage("depthTest",depths.front());     //�������
			char c=cvWaitKey(10);

			//sgf_segmentation
			Mat depth=Mat(imgDepth16u);
			/*--------*/
			double m1,m2;
			minMaxLoc(depth,&m1,&m2);
			
			//depth.convertTo(depth,CV_32F);
// 			Mat head=imread("headtemplate.bmp");
// 			if(head.channels()!=1)
// 			 cvtColor(head,head,CV_BGR2GRAY);
			//threshold(head,head,128,255,cv::THRESH_BINARY_INV);

			char name[5];
			itoa(number,name,10);
			string s=name;

			my_seg.set_name(s);
			//my_seg.set_depthMap(depth);
			//my_seg.set_background(depth);
			my_seg.set_headTemplate2D("headtemplate.bmp");

			vector<Point> seed;
			Mat mask;

			seed=my_seg.seed_method1(depth,true,true);
			//seed=my_seg.seedSGF(depth,true);
			//seed=my_seg.seedSGF(depth,true,mask);


			//imshow("mask",mask);waitKey(1);
			cout<<"seed number: "<<seed.size()<<endl;
			//my_seg.find_circles();
			//my_seg.find_realHead();

			
			my_seg.output(my_seg.videoname+".txt");

			//break;
			if(27==c)break;
		}  
 		++number;
// 		if (number==800||number==200||number==500)
// 		{
// 			cout<<number<<endl;
// 			cout<<my_seg.accurate<<endl;
// 		}
// 		if (number==50)
// 		{
// 			/*waitKey(0);*/
// 		}
		waitKey(10);
// 		Mat res1=my_seg.get_result1();
// 		Mat res2=my_seg.get_result2();
// 		Mat res3=my_seg.get_result3();
// 		Mat res=my_seg.get_result();
// 
// 		imshow("res1",res1);waitKey(10);
// 		imshow("res2",res2);waitKey(10);
// 		imshow("res3",res3);waitKey(10);
// 		imshow("res",res);waitKey(10);
// 
// 		writer1.write(res1);
// 		writer2.write(res2);	
// 		writer3.write(res3);
// 		writer4.write(res);
	}
	cout<<"time cost:"<<clock()-begin<<endl;
	system("pause");
	//����oni��ɫͼ��IplImage*��ʽvector
// 	XnUInt32 frameImage = 0;
// 	while(1)
// 	{    
// 		nRetVal = g_context.WaitOneUpdateAll(g_depth);     //�����������
// 		if(nRetVal!=XN_STATUS_OK)     //�ж��Ƿ���³ɹ�
// 		{
// 			printf("failed update color image/n");
// 			continue;
// 		}
// 		if(g_image.GetFrameID()<frameImage)break;     //�ж��Ƿ�ѭ������
// 		else
// 		{
// 			g_image.GetMetaData(g_imd);     //��ȡg_image��Ԫ����g_imd
// 			if(g_pImgColor==0)     //���ݵ�ǰ�Ĳ�ɫͼ��Ԫ��������g_pImgColor�Ĵ�С��ͨ����
// 			{
// 				nWidth=g_imd.XRes();
// 				nHeight=g_imd.YRes();
// 				g_pImgColor=cvCreateImage(cvSize(nWidth,nHeight),8,3);
// 				imgRGB8u=cvCreateImage(cvSize(nWidth,nHeight),8,3);
// 				cvZero(g_pImgColor);
// 				cvZero(imgRGB8u);
// 			}
// 			memcpy(imgRGB8u->imageData,g_imd.Data(),nWidth*nHeight*3);
// 			cvCvtColor(imgRGB8u,g_pImgColor,CV_RGB2BGR);
// 			images.push_back(g_pImgColor);     //���浽vector��
// 			frameImage = g_image.GetFrameID();     //��¼��ǰframeID
// 			cvShowImage("imgTest",images.front());     //�������
// 			char c=cvWaitKey(30);
// 			if(27==c)break;
// 		}  
// 	}
	//һ������
	if(g_pImgDepth)cvReleaseImage(&g_pImgDepth);
	if(g_pImgColor)cvReleaseImage(&g_pImgColor);
	if(imgDepth16u)cvReleaseImage(&imgDepth16u);
	if(imgRGB8u)cvReleaseImage(&imgRGB8u);
	g_context.Shutdown();
}