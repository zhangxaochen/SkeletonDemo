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
#include<map>
using namespace std;
using namespace cv;
using namespace sgf;
vector<IplImage*> depths,images;
void OpenNItoOpenCV();
vector<Mat> makeBackgroundModel(const std::vector<cv::Mat>& dmats)
{
	int row=dmats[0].rows,col=dmats[0].cols;
	Mat res_avg=Mat::zeros(row,col,CV_16U);
	Mat res_maxdepth=Mat::zeros(row,col,CV_16U);
	Mat res_maxnum=Mat::zeros(row,col,CV_16U);
	for (int i=0;i<row;++i)
	{
		for (int j=0;j<col;++j)
		{
			//对每一个像素，计算该像素背景值，可以使用的方法：均值，最大值，众数
			int num=0;
			int max_depth=0;
			int avg_depth=0;
			//map<int,int> most_depth;
			for (int k=0;k<dmats.size();++k)
			{
				int depth=dmats[k].at<ushort>(i,j);
				if (depth!=0)
				{
					++num;
					avg_depth+=depth;
					//most_depth.insert(depth);
				}
				max_depth=max(max_depth,depth);
				//找最大的众数
// 				map<int,int>::iterator it;
// 				int 
// 				for (it=most_depth.begin();it!=most_depth.end();++it)
// 				{
// 
// 				}
			}
			if (num!=0)
			{
				avg_depth/=num;
			}
			res_avg.at<ushort>(i,j)=avg_depth;
			res_maxdepth.at<ushort>(i,j)=max_depth;
		}
	}
	vector<Mat> res;
	res.push_back(res_avg);
	res.push_back(res_maxdepth);

	return res;
}
int main(int argc, char* argv[])
{
	//string video = "segment_test_1.oni";
	//string video="sun_han_short.oni";
	//string video="sgf-sss.oni";
	//string video="sgf-lyh-stand.oni";
	//string video="sun_walk.oni";
	//string video="zc-stand-wo-feet.oni";
	OpenNItoOpenCV();     //将获取到的oni信息转换为opencv能够处理的图像
	return 0;
}
void OpenNItoOpenCV()
{
	segment my_seg;
	my_seg.read_config("config.txt");
	string sFilename=my_seg.videoname;

	XnStatus nRetVal = XN_STATUS_OK;
	xn::Context g_context;     //创建上下文对象
	nRetVal=g_context.Init();     //上下文对象初始化          
	xn::Player plyr;
	g_context.OpenFileRecording(sFilename.c_str(), plyr);     //打开已有的oni文件
	plyr.SeekToFrame("MyDepth", 999, XN_PLAYER_SEEK_SET);
	xn::ImageGenerator g_image;     //创建image generator
	nRetVal=g_context.FindExistingNode(XN_NODE_TYPE_IMAGE,g_image);     //获取oni文件中的image节点
	xn::DepthGenerator g_depth;     //创建depth generator
	nRetVal=g_context.FindExistingNode(XN_NODE_TYPE_DEPTH,g_depth);     //获取oni文件中的depth节点
	xn::ImageMetaData g_imd;     //创建image节点元数据对象    
	xn::DepthMetaData g_dmd;     //创建depth节点元数据对象

	int nWidth;     //oni数据中灰度/彩色图像的宽度
	int nHeight;    //oni数据中灰度/彩色图像的高度
	IplImage *g_pImgColor=0;     //定义用于opencv显示的彩色图像
	IplImage *g_pImgDepth=0;     //定义用于opencv显示的深度图像
	IplImage *imgDepth16u=0;     //存储灰度信息
	IplImage *imgRGB8u=0;        //存储彩色信息
	//保存oni深度图像到IplImage*格式vector
	XnUInt32 frameDepth = 0;
	int number=0;
	int spend=0;

	//构造类segment，进行数据处理和显示
	//segment my_seg(true,true,true,true,true,false);

// 	int fourcc    = CV_FOURCC('D','I','V','X'); //编解码类型
// 	VideoWriter writer1,writer2,writer3,writer4;
// 	writer1=VideoWriter("only_resopnse_threshold.avi",fourcc,30,Size(320,240),false);
// 	writer2=VideoWriter("response_and_ckb.avi",fourcc,30,Size(320,240),false);
// 	writer3=VideoWriter("response_ckb_and_headsize.avi",fourcc,30,Size(320,240),false);
// 	writer4=VideoWriter("all_methods.avi",fourcc,30,Size(320,240),false);

	int begin=clock();
	vector<Mat> depth_array;
	vector<Mat> res;
	Mat depth_old;
	Mat max_depth;
	Mat fgMask_old;
	Mat max_dmat_mask;
	while(1)
	{    
		nRetVal = g_context.WaitOneUpdateAll(g_depth);     //更新深度数据
		if(nRetVal!=XN_STATUS_OK)     //判断是否更新成功
		{
			printf("failed update depth image/n");
			continue;
		}
		if(g_depth.GetFrameID()<frameDepth)break;     //判断是否循环结束
		else
		{
			g_depth.GetMetaData(g_dmd);     //获取g_depth的元数据g_dmd
			if(g_pImgDepth==0)     //根据当前的深度图像元数据设置g_pImgDepth的大小和通道数
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
			//cvEqualizeHist(g_pImgDepth,g_pImgDepth);     //直方图均衡化，拉开深度层次
			depths.push_back(g_pImgDepth);     //保存到vector中
			frameDepth = g_depth.GetFrameID();     //记录当前frameID
			cvShowImage("depthTest",depths.front());     //输出测试
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
			if (depth_old.data==NULL)
			{
				depth_old=depth.clone();
			}

			//seed=my_seg.seedHeadTempMatch(depth,true);
			//mask=my_seg.buildMaxDepth(depth,max_depth,depth_old,fgMask_old);
			int t=clock();
			sgf::buildMaxDepth(depth,depth_old,max_depth,max_depth,fgMask_old,fgMask_old,max_dmat_mask,max_dmat_mask);
			cout<<"time: "<<clock()-t<<endl;
			spend+=clock()-t;
			waitKey(1);
			depth_old=depth.clone();
			//seed=my_seg.seedSGF(depth,true);
			//seed=my_seg.seedSGF(depth,true,mask);


			//imshow("mask",mask);
			cout<<"seed number: "<<seed.size()<<endl;
			//my_seg.find_circles();
			//my_seg.find_realHead();

			
			//my_seg.output(my_seg.videoname+".txt");

// 			if (number==0)
// 			{
// 				depth_old=depth;
// 			}
// 			else
// 			{
// 				Mat depth_dif=depth-depth_old;
// 			}
// 
// 			if (number<20)
// 			{
// 				depth_array.push_back(depth);
// 			}
// 			else if (number==20)
// 			{
// 				res=makeBackgroundModel(depth_array);
// 				Mat gray_avg,gray_max;
// 				res[0].convertTo(gray_avg,CV_8U,255/9000.0,0);
// 				res[1].convertTo(gray_max,CV_8U,255/9000.0,0);
// 				imshow("background avg",gray_avg);
// 				imshow("background max depth",gray_max);
// 			}
// 			else
// 			{
// 				Mat difference_avg=Mat::zeros(depth.rows,depth.cols,CV_8U);
// 				Mat difference_max=Mat::zeros(depth.rows,depth.cols,CV_8U);
// 				for (int i=0;i<depth.rows;++i)
// 				{
// 					for (int j=0;j<depth.cols;++j)
// 					{
// 						if (abs(depth.at<ushort>(i,j)-res[0].at<ushort>(i,j))>50&&depth.at<ushort>(i,j)!=0)
// 						{
// 							difference_avg.at<uchar>(i,j)=255;
// 						}
// 						if (abs(depth.at<ushort>(i,j)-res[1].at<ushort>(i,j))>50&&depth.at<ushort>(i,j)!=0)
// 						{
// 							difference_max.at<uchar>(i,j)=255;
// 						}
// 					}
// 				}
// 				imshow("difference avg",difference_avg);
// 				imshow("difference max",difference_max);
// 			}
// 			waitKey(1);

			//break;
			if(27==c) break;
			++number;
		}  
// 		if (number==800||number==200||number==500)
// 		{
// 			cout<<number<<endl;
// 			cout<<my_seg.accurate<<endl;
// 		}
// 		if (number==50)
// 		{
// 			/*waitKey(0);*/
// 		}
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
	cout<<"avg time:"<<spend/number<<endl;
	cout<<"time cost:"<<clock()-begin<<endl;
	system("pause");
	//保存oni彩色图像到IplImage*格式vector
// 	XnUInt32 frameImage = 0;
// 	while(1)
// 	{    
// 		nRetVal = g_context.WaitOneUpdateAll(g_depth);     //更新深度数据
// 		if(nRetVal!=XN_STATUS_OK)     //判断是否更新成功
// 		{
// 			printf("failed update color image/n");
// 			continue;
// 		}
// 		if(g_image.GetFrameID()<frameImage)break;     //判断是否循环结束
// 		else
// 		{
// 			g_image.GetMetaData(g_imd);     //获取g_image的元数据g_imd
// 			if(g_pImgColor==0)     //根据当前的彩色图像元数据设置g_pImgColor的大小和通道数
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
// 			images.push_back(g_pImgColor);     //保存到vector中
// 			frameImage = g_image.GetFrameID();     //记录当前frameID
// 			cvShowImage("imgTest",images.front());     //输出测试
// 			char c=cvWaitKey(30);
// 			if(27==c)break;
// 		}  
// 	}
	//一起清理
	if(g_pImgDepth)cvReleaseImage(&g_pImgDepth);
	if(g_pImgColor)cvReleaseImage(&g_pImgColor);
	if(imgDepth16u)cvReleaseImage(&imgDepth16u);
	if(imgRGB8u)cvReleaseImage(&imgRGB8u);
	g_context.Shutdown();
}