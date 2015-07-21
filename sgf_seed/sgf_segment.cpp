#include "sgf_segment.h"
//#include "SimpleSilhouette.h"
#include <time.h>
#include <cmath>

using namespace sgf;
using namespace std;
using namespace cv;

// segment::segment(bool show_result1,bool show_distance_map1,bool show_edge1,
// 	bool do_region_grow1,bool show_responses1,bool show_histogram1)
// {
// 	show_result=show_result1;
// 	show_distance_map=show_distance_map1;
// 	show_edge=show_edge1;
// 	do_region_grow=do_region_grow1;
// 	show_responses=show_responses1;
// 	show_histogram=show_histogram1;
// 	accurate=0;
// 	bg_count=0;
// }

// segment::segment(bool show_result1,bool show_depth_without_bg1,
// 	bool show_topdown_view1,bool show_topdown_binary1)
// {
// 	show_result=show_result1;
// 	show_depth_without_bg=show_depth_without_bg1;
// 	show_topdown_view=show_topdown_view1;
// 	show_topdown_binary=show_topdown_binary1;
// }

segment::segment()
{
	_SGF_MODE=1;
	_SGF_DEBUG=1;
	_SGF_SHOW=1;
	_MOG=1;

	has_set_depth=false;

#ifdef CV_VERSION_EPOCH
	my_MOG = BackgroundSubtractorMOG2(200, 20, false);
#elif CV_VERSION_MAJOR >= 3
	//TODO: cv3
#endif //CV_VERSION_EPOCH

}

void segment::set_mog_par(int history,int varthreshold,bool detect_shadow,int learningRate)
{
	mog_history=history;
	mog_varthreshold=varthreshold;
	mog_lr=learningRate;
	mog_detect_shadow=detect_shadow;
}

void segment::set_depthMap(const cv::Mat& depth)
{
	depth_map=depth.clone();
	if (!has_set_depth)
	{
		depth_map_old=depth.clone();
		depth_map_old.convertTo(depth_map_old,CV_32FC1);
		has_set_depth=true;
	}
	depth_map.convertTo(depth_map,CV_32FC1);
	double dmax,dmin;
	minMaxLoc(depth_map,&dmin,&dmax);
	max_depth=dmax;min_depth=dmin;
	//depth_map.convertTo(gray_clone,CV_8U,255.0/(dmax-dmin),-dmin*255.0/(dmax-dmin));
	depth_map.convertTo(gray_clone,CV_8UC1,255.0/8000,0);
	gray_map=gray_clone.clone();
}
bool segment::set_headTemplate2D(const std::string &headTemplatePath)
{
	Mat head=imread(headTemplatePath,0);
	bool readSucceed = !head.empty();
	if(readSucceed){
		head_template=head.clone();
		threshold(head_template,head_template,128,255,THRESH_BINARY);
	}
	return readSucceed;
}
void segment::fill_holes()
{
	Holefilling hole;
	hole.setMat(depth_map);
	hole.BFS();
}
void segment::smooth_image()
{
	for ( int i = 1; i < 4; i = i + 2 )
	{
		medianBlur ( depth_map, depth_map, i );
	}

	double dmax,dmin;
	minMaxLoc(depth_map,&dmin,&dmax);
	max_depth=dmax;min_depth=dmin;
	depth_map.convertTo(gray_map,CV_8U,255.0/(dmax-dmin),-dmin*255.0/(dmax-dmin));

//	imwrite("depth_"+name+".jpeg",gray_map);
// 	cv::imshow("depth map after pre-process",gray_map);
// 	cv::waitKey(1);
}
void segment::compute_edge(double threshold1,double threshold2)
{
	Canny(gray_map, edge_map_thresh, threshold1, threshold2);
	threshold(edge_map_thresh,edge_map_thresh,128,255,cv::THRESH_BINARY);
}
void segment::compute_distanceMap_2D()
{
// 	edge_map_thresh=Mat::ones(200,200,CV_8U);
// 	edge_map_thresh.at<uchar>(100,100)=0;
//	int time_begin=clock();
	distanceTransform(edge_map_thresh,distance_map,distance_type,mask_type);
//	cout<<"time cost in distance transform:"<<clock()-time_begin<<endl;
	double dmax,dmin;
	minMaxLoc(distance_map,&dmin,&dmax);
	distance_map.convertTo(distance_map_uchar,CV_8U,255.0/(dmax-dmin),-dmin*255.0/(dmax-dmin));
}
void segment::compute_filter_map_edge(double threshold1,double threshold2)
{
	Canny(filter_map, filter_edge, threshold1, threshold2);
	threshold(filter_edge,filter_edge,128,255,cv::THRESH_BINARY);
// 	imshow("filter map edge",filter_edge);
// 	waitKey(1);
}
void segment::compute_subsamples()
{
	sub_distance_maps.clear();
	Mat src=distance_map_uchar.clone();
	//sub_distance_maps.push_back(src.clone());

	//自己实现采样率为3/4的金字塔
	while (src.rows>150&&src.cols>150)
	{
		int c=src.cols*3/4,r=src.rows*3/4;
		//cout<<" width: "<<c<<" height: "<<r<<endl;
		resize(src,src,Size(c,r));
// 		Mat sub=Mat::zeros(r,c,CV_8U);
// 		for (int i=0;i<r;++i)
// 		{
// 			for (int j=0;j<c;++j)
// 				sub.at<uchar>(i,j)=src.at<uchar>(i+i/3,j+j/3);
// 		}
		sub_distance_maps.push_back(src.clone());
		//src=sub;
	}
}
void segment::compute_response()
{
	sub_responses.clear();
	sub_responses_binary.clear();
	for (int i=0;i<sub_distance_maps.size();++i)
	{
		cv::Mat matching_space;
		matching_space.create(sub_distance_maps[i].cols-head_template.cols+1,sub_distance_maps[i].rows-head_template.rows+1, CV_32FC1 );
		int t=clock();
		matchTemplate(sub_distance_maps[i],head_template,matching_space, CV_TM_CCORR_NORMED );
//		cout<<"template time:"<<clock()-t<<endl;

		double dmax,dmin;
		minMaxLoc(matching_space,&dmin,&dmax);
		Mat matching_space1;
		matching_space.convertTo(matching_space1,CV_8U,255.0/(dmax-dmin),-dmin*255.0/(dmax-dmin));

		//二值化
		Mat matching_space_binary,matching_space1_binary;

		threshold(matching_space,matching_space_binary,threshold_binary_response,255,THRESH_BINARY);//绝对二值化
		matching_space_binary.convertTo(matching_space_binary,CV_8U);

 		sub_responses.push_back(matching_space.clone());
 		sub_responses_binary.push_back(matching_space_binary.clone());
	}
}
double segment::get_headheight(double d)
{
	double res;
	res=1.3838*pow(10.0,-9)*pow(d,3)-1.8435*pow(10.0,-5)*pow(d,2)+0.091403*d-189.38;
	return -res;
}

Mat segment::get_edgeMapWithThresh()
{
	return edge_map_thresh.clone();
}
Mat segment::get_distanceMap()
{
	return distance_map.clone();
}
Mat segment::get_result()
{
	return interest_point.clone();
}Mat segment::get_result1()
{
	return interest_point_raw.clone();
}Mat segment::get_result2()
{
	return interest_point1.clone();
}Mat segment::get_result3()
{
	return interest_point2.clone();
}
vector<Point> segment::seedHeadTempMatch(cv::Mat dmat,bool showResult/* =false */,bool showTime/*=false*/,bool showDemo/*=false*/) //利用模板匹配
{
	set_depthMap(dmat);
	int begin=clock();
	seperate_foot_and_ground();

	compute_edge(threshold_depth_min,threshold_depth_max);
	threshold(edge_map_thresh,edge_map_thresh,128,255,THRESH_BINARY_INV);

	compute_distanceMap_2D();
	equalizeHist(distance_map_uchar,distance_map_uchar);
	compute_subsamples();
	compute_response();
	region_of_interest.clear();
	region_of_interest_raw.clear();
	interest_point=edge_map_thresh.clone();
	interest_point_raw=edge_map_thresh.clone();
	interest_point2=edge_map_thresh.clone();
	find_and_draw_countours();
	choose_and_draw_interest_region();

	if (!showDemo)
	{
		return headpoints_location;
	}

	//下边是自己的区域增长和分割
	region_grow_map=Mat::zeros(distance_map.rows,distance_map.cols,CV_8U);
	mask_of_distance=Mat::zeros(distance_map.rows,distance_map.cols,CV_8U);
	for (int i=0;i<headpoints_location.size();++i)
	{
		seed_queue.empty();
		region_grow(headpoints_location[i].y,headpoints_location[i].x,50);
	}
	int dilate_size=1;
	Mat element1 = getStructuringElement( MORPH_RECT,
		Size( 2*dilate_size + 1, 2*dilate_size+1 ),
		Point( dilate_size, dilate_size ) );
	dilate( region_grow_map,region_grow_map,element1);
	if (showResult)
	{
		imshow("region grow",region_grow_map);waitKey(1);
// 		imwrite("region_grow_"+name+".jpg",region_grow_map);
// 		cout<<name<<endl;
	}

	//根据前后两帧深度数据，进行差分（一阶导）,从而将运动区域中的背景剔除


	//更新深度数据
	depth_map_old=depth_map.clone();


	Mat fg_depth;
#ifdef CV_VERSION_EPOCH
	my_MOG.operator()(gray_map, fg_depth, 0.005);
#elif CV_VERSION_MAJOR >= 3
	//TODO: cv3
#endif //CV_VERSION_EPOCH
	if (showResult)
	{
		imshow("foeground mask",fg_depth);waitKey(1);
	}

	//根据区域内动点个数进行分割长在一起的人
	vector<Mat> res_tmp=get_seperate_masks(region_grow_map,fg_depth,headpoints_location,headpoints_radius,true,true);

	/*--------*/
	//寻找轮廓的极值点，用来分割长在一起的人
	int t=clock();
	vector<Mat> Masks=get_seperate_masks(region_grow_map,true);
	/*--------*/
	
	if (showTime)
	{
		cout<<"time spend in seperating masks: "<<clock()-t<<endl;
		cout<<"time spend:"<<clock()-begin<<endl;
	}
	if (showResult)
	{
		display();
	}
	return headpoints_location;
}
vector<Point> segment::seed_method2(cv::Mat dmat,bool showResult/* =false */,bool showTime/* =false */)
{
	set_depthMap(dmat);
	int begin=clock();
	compute_hist();
	compute_height();
	seperate_foot_and_ground();
	sjbh();
	compute_cost();
	if (showTime)
	{
		cout<<"time spend:"<<clock()-begin<<endl;
	}
	if (showResult)
	{
		imshow("result",gray_clone);
		waitKey(1);
		imshow("depth without bg mask",depth_mask);
		waitKey(1);
		imshow("top down view",depth_sjbh);
		waitKey(1);
		imshow("top down view binary",sjbh_binary);
		waitKey(1);
	}
		return get_seed();
}
vector<Point> segment::seedSGF(Mat dmat,bool showResult,bool seed_raw,Mat& depth_without_bg)
{
	set_depthMap(dmat);
	
	//fill_holes();
	//smooth_image();
	if (_MOG)
	{
		useMOG();
		vector<Point> res;
		return res;
	}

	int begin=clock();
	int t1=clock();

	if (_SGF_MODE)
	{
		compute_hist();

		compute_height();

		//smooth_image();

		seperate_foot_and_ground();

		sjbh();
// 		imshow("depth mask",depth_mask);
// 		waitKey(1);
		compute_cost();
		if (_SGF_DEBUG)
		{
			cout<<"time spend:"<<clock()-begin<<endl;
		}
		if (_SGF_SHOW)
		{
			imshow("result",gray_clone);
			waitKey(1);
			imshow("depth without bg mask",depth_mask);
			waitKey(1);
			imshow("top down view",depth_sjbh);
			waitKey(1);
			imshow("top down view binary",sjbh_binary);
			waitKey(1);
		}
		return get_seed();
	}


// 
// 	depth_without_bg=depth_mask.clone();
// 	if (!seed_raw)
// 	{
// 		return get_seed();
// 	}
// 	else
// 	{
// 		return get_seed_raw();
// 	}
// 	show_difference();
// 
// 	double my_max=0;
// 	double c=0;
// 	for(int i=0;i<depth_map.rows;++i)
// 	{
// 		for (int j=0;j<depth_map.cols;++j)
// 		{
// 			if (filter_map.at<uchar>(i,j)!=0)
// 			{
// 				my_max+=depth_map.at<float>(i,j);
// 				c++;
// 			}
// 		}
// 	}
// 	my_max/=c;
// 	cout<<"average depth: "<<my_max<<endl;
// // 	for(int i=0;i<depth_map.rows;++i)
// // 		for (int j=0;j<depth_map.cols;++j)
// // 			depth_map.at<float>(i,j)=min(int(my_max),int(depth_map.at<float>(i,j)));
// 	Mat depth_show;
// 	depth_map.convertTo(depth_show,CV_8U,255.0/9000,0);
// 	imshow("depth",depth_show);
// 
// 
// 	find_bg();
// 
// 	//imwrite("depth_"+name+".jpeg",gray_map);
// 
// 	waitKey(1);


//	smooth_image();
	

//	seperate_foot_and_ground();
// 	compute_filter_map_edge(threshold_filter_min,threshold_filter_max);
// // 	imshow("seperate foot and ground",filter_map);
// // 	waitKey(1);
// 
	else
	{
 	compute_edge(threshold_depth_min,threshold_depth_max);
// // 	imshow("depth edge",edge_map_thresh);
// // 	waitKey(1);
// 	edge_map_thresh=edge_map_thresh+filter_edge;
// 	imshow("edge",edge_map_thresh);
// 	waitKey(1);
	threshold(edge_map_thresh,edge_map_thresh,128,255,THRESH_BINARY_INV);

//	edge_map_thresh=zc::getHumanEdge(depth_map);

//	cout<<"canny and seperate foot time cost:"<<clock()-t1<<endl;

	t1=clock();
	compute_distanceMap_2D();
//	cout<<"distance transform time:"<<clock()-t1<<endl;
// 	imshow("distance 1",distance_map_uchar);
// 
// 
// 
// 	waitKey(1);
// 
// 	//直方图变换
// 	Mat lut=Mat::zeros(1,256,CV_8U);
// 	for (int i=0;i<256;++i)
// 	{
// 		lut.at<uchar>(0,i)=255*pow(i*1.0/255,1.0/3);
// 	}
// 	Mat distance_map_lut;
// 	LUT(distance_map_uchar,lut,distance_map_lut);
// 	imshow("distance 2",distance_map_lut);
// 	waitKey(1);
 
	t1=clock();
 	equalizeHist(distance_map_uchar,distance_map_uchar);
//	cout<<"hist equalize time:"<<clock()-t1<<endl;
// 	imshow("distance 3",distance_map_uchar);
// 	waitKey(1);
// 
// 
	t1=clock();
	compute_subsamples();
//	cout<<"pyr time:"<<clock()-t1<<endl;

	t1=clock();
	compute_response();
//	cout<<"response time:"<<clock()-t1<<endl;

	t1=clock();
	region_of_interest.clear();
	region_of_interest_raw.clear();
	interest_point=edge_map_thresh.clone();
	interest_point_raw=edge_map_thresh.clone();
// 	interest_point1=edge_map_thresh.clone();
 	interest_point2=edge_map_thresh.clone();
// 	interest_point3=edge_map_thresh.clone();
// 	interest_point4=edge_map_thresh.clone();
	find_and_draw_countours();
//	cout<<"raw contours time:"<<clock()-t1<<endl;
	t1=clock();
	choose_and_draw_interest_region();

	region_grow_map=Mat::zeros(distance_map.rows,distance_map.cols,CV_8U);
	mask_of_distance=Mat::zeros(distance_map.rows,distance_map.cols,CV_8U);
	for (int i=0;i<region_of_interest.size();++i)
	{
		seed_queue.empty();
		region_grow(region_of_interest[i].y,region_of_interest[i].x,1);
	}
	imshow("region grow",region_grow_map);waitKey(1);
//	cout<<"choose seed time:"<<clock()-t1<<endl;
	if (_SGF_DEBUG)
	{
		cout<<"time spend:"<<clock()-begin<<endl;
	}
	if (_SGF_SHOW)
	{
		display();
	}
	return headpoints_location;
	}
}
void segment::output(string name)
{
	ofstream output(name,fstream::app);
	for (int i=0;i<headpoints_location.size();++i)
	{
		output<<headpoints_location[i].x<<' '<<headpoints_location[i].y<<' ';
	}
	output<<endl;
	output.close();
}
void segment::find_and_draw_countours()
{
	sub_contours.clear();
	int k=sub_responses_binary.size();
	
	int i=0;
	
	for (int i=0;i<k;++i)
	{
		Mat tmp;
		threshold(sub_responses_binary[i],tmp,128,255,THRESH_BINARY_INV);
		double tmp_min,tmp_max;
		minMaxLoc(tmp,&tmp_min,&tmp_max);

		Mat contour_image=Mat::zeros(tmp.rows,tmp.cols,CV_8U);
		threshold(contour_image,contour_image,128,255,THRESH_BINARY_INV);
		
		if (tmp_max>0)
		{
			vector<vector<Point> > contours;
			findContours(tmp,contours, CV_RETR_LIST , CV_CHAIN_APPROX_NONE );

			//对找到的轮廓进行处理，比如去掉太小的轮廓或者大致形状不满足要求的轮廓，或者均值方差等不满足要求
			deal_with_contours(contours,i+1);

			/*drawContours(contour_image,contours,-1,0,CV_FILLED);*/
		}
		sub_contours.push_back(contour_image);
	}
}
void segment::set_name(std::string s)
{
	name=s;
}
bool segment::read_config(const std::string &configPath)
{
	ifstream config_file;
	config_file.open(configPath);
	bool openSucceed = !config_file.bad();
	if(openSucceed){
		config_file>>videoname>>threshold_depth_min>>threshold_depth_max
			>>threshold_binary_filter>>threshold_filter_min>>threshold_filter_max
			>>threshold_binary_response>>threshold_contour_size
			>>threshold_contour_ckb>>threshold_headsize_min>>threshold_headsize_max
			>>a>>const_depth
			>>distance_type>>mask_type
			>>height>>bar_width>>depth_rate1>>depth_rate2
			>>height_center>>cost_threshold;

		config_file.close();
	}
	return openSucceed;
}
void segment::deal_with_contours(vector<vector<Point> >& contours,int k)
{ //兴趣区域轮廓形状大小规则，参数自己调整
	double thres=threshold_contour_size*pow(3.0/4,k);
	vector<vector<Point>>::iterator it;
// 	for (it=contours.begin();it!=contours.end();)
// 	{
// 		int min_x=1000,min_y=1000,max_x=0,max_y=0;
// 		int x_avg=0,y_avg=0;
// 		for (int i=0;i<(*it).size();++i)
// 		{
// 			min_x=min((*it)[i].x,min_x);
// 			min_y=min((*it)[i].y,min_y);
// 			max_x=max((*it)[i].x,max_x);
// 			max_y=max((*it)[i].y,max_y);
// 			x_avg += int((*it)[i].x);
// 			y_avg += int((*it)[i].y);
// 		}
// 		x_avg/=(*it).size();y_avg/=(*it).size();
// 		Point2i p;
// 		p.x=(x_avg+(head_template.cols)/2)*pow(4.0/3,k);
// 		p.y=(y_avg+(head_template.rows)/2)*pow(4.0/3,k);
// 		region_of_interest_raw.push_back(p);
// 
// 		if ((*it).size()<thres)
// 			it=contours.erase(it);
// 		else
// 		{
// 			double ckb=(max_x-min_x)*1.0/(max_y-min_y);
// 			if (ckb>0.3&&ckb<3)
// 			{
// 				region_of_interest.push_back(p);
// 				++it;
// 			}
// 			else
// 				it=contours.erase(it);
// 		}
// 	}
	//新方法，找最小包围矩形
	for (it=contours.begin();it!=contours.end();)
	{
		int min_x=1000,min_y=1000,max_x=0,max_y=0;
		int x_avg=0,y_avg=0;
		for (int i=0;i<(*it).size();++i)
		{
			min_x=min((*it)[i].x,min_x);
			min_y=min((*it)[i].y,min_y);
			max_x=max((*it)[i].x,max_x);
			max_y=max((*it)[i].y,max_y);
			x_avg += int((*it)[i].x);
			y_avg += int((*it)[i].y);
		}
		x_avg/=(*it).size();y_avg/=(*it).size();
		Point2i p;
		p.x=(x_avg+(head_template.cols)/2)*pow(4.0/3,k);
		p.y=(y_avg+(head_template.rows)/2)*pow(4.0/3,k);
		region_of_interest_raw.push_back(p);

		if ((*it).size()<thres)
			it=contours.erase(it);
		else
		{
			RotatedRect rr;
			rr=minAreaRect(*it);
			double width=rr.size.width,height=rr.size.height;
			double ckb=width*1.0/height;
			if (ckb>1.0/threshold_contour_ckb&&ckb<threshold_contour_ckb)
			{
				region_of_interest.push_back(p);
				++it;
			}
			else
				it=contours.erase(it);
		}
	}

}
void segment::region_grow(int x,int y,int thres)
{
	Point2i p;
	p.x=x;p.y=y;
	seed_queue.push(p);
	while (seed_queue.size()!=0)
	{
		Point2i p_tmp;

		p_tmp=seed_queue.front();
		seed_queue.pop();
		region_grow_map.at<uchar>(p_tmp.x,p_tmp.y)=255;
		for (int i=-1;i<2;i+=2)
		{
			double tmp_d1=depth_map.at<float>(p_tmp.x,p_tmp.y);
			double x=p_tmp.x+i,y=p_tmp.y;
			if (x>=0 && x<distance_map.rows && y>=0 && y<distance_map.cols)
			{
				if (mask_of_distance.at<uchar>(x,y)==0)
				{
					mask_of_distance.at<uchar>(x,y)=1;
					//double tmp=distance_map.at<float>(x,y);
					double tmp1=filter_map.at<uchar>(x,y);
					double tmp_d2=depth_map.at<float>(x,y);
					//if (tmp>thres&&tmp1<100)
					if (abs(tmp_d1-tmp_d2)<thres&&tmp1<100)
					{
						Point2i p1;
						p1.x=x;p1.y=y;
						seed_queue.push(p1);
					}
				}
			}
			x=p_tmp.x,y=p_tmp.y+i;
			if (x>=0 && x<distance_map.rows && y>=0 && y<distance_map.cols)
			{
				if (mask_of_distance.at<uchar>(x,y)==0)
				{
					mask_of_distance.at<uchar>(x,y)=1;
					//double tmp=distance_map.at<float>(x,y);
					double tmp1=filter_map.at<uchar>(x,y);
					double tmp_d2=depth_map.at<float>(x,y);
					//if (tmp>thres&&tmp1<100)
					if (abs(tmp_d1-tmp_d2)<thres&&tmp1<100)
					{
						Point2i p1;
						p1.x=x;p1.y=y;
						seed_queue.push(p1);
					}
				}
			}
		}
	}
	//再进行膨胀操作，去掉图像中的小部分孔洞

}
void segment::seperate_foot_and_ground()
{
	Mat kernal=Mat::zeros(6,1,CV_32F);
	for (int i=0;i<kernal.rows;++i)
	{
		if (i<3)
			kernal.at<float>(i,0)=-1;
		else
			kernal.at<float>(i,0)=1;
	}
	Mat kernal1=Mat::zeros(1,6,CV_32F);
	for (int i=0;i<kernal1.cols;++i)
	{
		if (i<3)
			kernal1.at<float>(0,i)=1;
		else
			kernal1.at<float>(0,i)=-1;
	}
	filter_map1=Mat::zeros(depth_map.rows,depth_map.cols,CV_32F);
	filter2D(depth_map,filter_map1,-1,kernal1);
	filter_map=Mat::zeros(depth_map.rows,depth_map.cols,CV_32F);
	filter2D(depth_map,filter_map,-1,kernal);
	for (int i=0;i<filter_map.rows;++i)
	{
		for (int j=0;j<filter_map.cols;++j)
			filter_map.at<float>(i,j)=abs(filter_map.at<float>(i,j));
	}
	//abs(filter_map);
	double dmax,dmin;
	minMaxLoc(filter_map,&dmin,&dmax);
	//imshow("filter map",filter_map);
	filter_map.convertTo(filter_map,CV_8U);
	for (int i=0;i<filter_map.rows;++i)
	{
		for (int j=0;j<filter_map.cols;++j)
		{
			if (i<150)
			{
				filter_map.at<uchar>(i,j)=0;
			}
		}
	}
	/*imshow("filter map1",filter_map);*/

	waitKey(1);

// 	for ( int i = 1; i < 4; i = i + 2 )
// 	{
// 		medianBlur ( filter_map, filter_map, i );
// 	}

	Mat tmp;
	Canny(filter_map,tmp,threshold_filter_min,threshold_filter_max);
// 	imshow("raw canny from filter map",tmp);
// 	waitKey(1);

	threshold(filter_map,filter_map,threshold_binary_filter,255,THRESH_BINARY);
	
	int erode_size=4;
	Mat element = getStructuringElement( MORPH_RECT,
		Size( 2*erode_size + 1, 2*erode_size+1 ),
		Point( erode_size, erode_size ) );
	int dilate_size=3;
	Mat element1 = getStructuringElement( MORPH_RECT,
		Size( 2*dilate_size + 1, 2*dilate_size+1 ),
		Point( dilate_size, dilate_size ) );
	erode( filter_map, filter_map,element);
	dilate( filter_map, filter_map,element1);
 	/*imshow("filter map binary",filter_map);*/
 	waitKey(1);
}
void segment::compute_hist()
{
	double step_length=bar_width;
	minMaxLoc(depth_map,&min_depth,&max_depth);
	float range[]={min_depth,max_depth};
	int histSize=(max_depth-min_depth)/step_length;
	const float* histRange={range};
	bool uniform=true,accumulate=false;
// 	threshold(filter_map,filter_map,128,255,THRESH_BINARY_INV);
// 	imshow("filter",filter_map);
// 	waitKey(1);
	calcHist( &depth_map, 1, 0, Mat(), histogram_image, 1, &histSize, &histRange, uniform, accumulate );

/*	cout<<histogram_image<<endl;*/

	double tmp_max,tmp_min;
	minMaxLoc(histogram_image,&tmp_min,&tmp_max);

	double trs=depth_map.rows*depth_map.cols*1.0/depth_rate1;
	double sep=max_depth;
	double total_trs=depth_map.rows*depth_map.cols*1.0/depth_rate2;
	double total_number=0;
	for (int i=histogram_image.rows-1;i>=0;--i)
	{
		bool find_max=false;
		double cur_number=histogram_image.at<float>(i,0);
		if (cur_number>=trs)
		{
			find_max=true;
			for (int j=-5;j<=5;++j)
			{
				int k=i-j;
				if (k>=0&&k<histogram_image.rows)
				{
					if (cur_number<histogram_image.at<float>(k,0))
					{
						find_max=false;break;
					}
				}
			}
		}
		if (find_max)
		{
			sep=(i-1)*step_length;break;
		}
		total_number+=cur_number;
		if (total_number>=total_trs)
		{
			sep=(i-1)*step_length;break;
		}
	}
//	cout<<"depth threshold: "<<sep<<endl;
	depth_mask=Mat::zeros(depth_map.rows,depth_map.cols,CV_8U);
// 	imshow("11111",depth_mask);
// 	waitKey(1);
	for (int i=0;i<depth_map.rows;++i)
	{
		for (int j=0;j<depth_map.cols;++j)
		{
			double depth=depth_map.at<float>(i,j);
			if (depth>sep||depth==0)
			{
				depth_mask.at<uchar>(i,j)=255;
			}
		}
	}

	//int histSize=10;
// 	int hist_w = 900; int hist_h = 500;
// 	int bin_w = cvRound( (double) hist_w/histSize );
// 	Mat histImage=Mat::zeros( hist_h, hist_w, CV_8U);
// 	normalize(histogram_image, histogram_image, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
// 	for( int i = 1; i < histSize; i++ )
// 	{
// 		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(histogram_image.at<float>(i-1,0)) ) ,
// 			Point( bin_w*(i), hist_h - cvRound(histogram_image.at<float>(i,0)) ),
// 			255, 2, 8, 0  );
// 	}
// 	imshow("histogram", histImage );
// 	waitKey(0);
}
bool segment::compute_trueHead(const Point2i& p)
{
	int x = depth_map.at<float>(p.y,p.x)/10;
	double number=histogram_image.at<float>(x,0);
	if (histogram_image.at<float>(x,0)<700 || histogram_image.at<float>(x,0)>3000)
		return false;
	int i_min=max(0,x-10),i_max=min(x+10,histogram_image.rows-1);
	double sum=0;
	for (int i=i_min;i<=i_max;++i)
	{
		sum+=histogram_image.at<float>(i,0);
	}
	sum/=(i_max-i_min);
	if (sum<400) return false;

	return true;
}
void segment::choose_and_draw_interest_region()
{
	headpoints_location.clear();
	headpoints_radius.clear();
	vector<Point>::iterator it;
// 	for (int i=0;i<region_of_interest_raw.size();++i)
// 	{
// 		Point2i p=region_of_interest_raw[i];
// 		int x=p.x;
// 		int y=p.y;
// 		double _r=distance_map.at<float>(y,x);
// 		circle(interest_point_raw,p,_r,0,2);
//	}
	for (it=region_of_interest.begin();it!=region_of_interest.end();)
	{
		Point2i p=*it;
		int x=p.x;
		int y=p.y;
		double depth=depth_map.at<float>(y,x);
		double _h=get_headheight(depth_map.at<float>(y,x));
		double _R=1.33*_h/4;//根据图像分辨率进行调整
		//double _R=1.33*_h/2;
		double _r=distance_map.at<float>(y,x);

		//circle(interest_point1,p,_r,0,2);
		int x_min=max(0,int(x-_r*a)),x_max=min(int(x+_r*a),depth_map.cols-1);
		int y_min=max(0,int(y-_r*a));

// 		int x_down=x,x_down_min=max(0,int(x-_r*a*0.7)),x_down_max=min(depth_map.cols-1,int(x+_r*a*0.7));
// 		int y_down=min(int(y+_r*a),depth_map.rows-1);

		double bz=_r/_R;
		if (bz>threshold_headsize_min&&bz<threshold_headsize_max&&_r>5)
		{
			circle(interest_point3,p,_r,0,2);
			if (depth+const_depth<depth_map.at<float>(y,x_min)&&depth+const_depth<depth_map.at<float>(y,x_max)
			&&depth+const_depth<depth_map.at<float>(y_min,x))//头部大小与深度值关系规则，需要调参)
			{
				//判断一个种子点是不是头部，补充方法

// 				if (2*const_depth>abs(depth_map.at<float>(y_down,x_down)-depth)
// 					&&2*const_depth>abs(depth_map.at<float>(y_down,x_down_min)-depth)
// 					&&2*const_depth>abs(depth_map.at<float>(y_down,x_down_max)-depth))
// 				{

				
				circle(interest_point4,p,_r,0,2);
			//去掉表示同一个头部的多个圆，只保留半径最大的一个
			Point2i p_2i;
			p_2i.x=(*it).x;p_2i.y=(*it).y;
			bool push_back=true;
			for (int j=0;j<headpoints_location.size();++j)
			{
				double max_r=_r+headpoints_radius[j];
				if (pow(int(p_2i.x-headpoints_location[j].x),2.0)+pow(int(p_2i.y-headpoints_location[j].y),2.0)<max_r*max_r)
				{
					if (_r>headpoints_radius[j])
					{
						headpoints_location[j]=p_2i;
						headpoints_radius[j]=_r;
					}
					push_back=false;
					break;
				}
			}
			if (push_back)
			{
				headpoints_location.push_back(p_2i);
				headpoints_radius.push_back(_r);
			}
//			}
			}
			++it;
// 			if (compute_trueHead(p))//直方图统计规则，需要调整参数
// 			{
// 				/*cout<<x<<"---"<<y<<endl;*/
// 				circle(interest_point,p,_r,0,2);
// 				++it;
// 			}
// 			else
// 				it=region_of_interest.erase(it);
   		}
		else
			it=region_of_interest.erase(it);
 	}
	for (int i=0;i<headpoints_location.size();++i)
	{
		circle(interest_point2,headpoints_location[i],headpoints_radius[i],0,2);
	}

	//分析结果是否正确
// 	if (headpoints_location.size()==2)
// 	{
// /*		accurate++;*/
// 		bool right=true;
// 		for (int i=0;i<headpoints_location.size();++i)
// 		{
// 			Point2i p=headpoints_location[i];
// 			if ((p.x>90&&p.x<150)||(p.x>170&&p.x<230))
// 			{
// 				if (p.y>0&&p.y<60)
// 				{
// 					right=true;
// 				}
// 				else
// 				{
// 					right=false;
// 					break;
// 				}
// 			}
// 			else
// 			{
// 				right=false;
// 				break;
// 			}
// 		}
// 		if (right)
// 		{
// 			accurate++;
// 		}
// 	}
}
void segment::display()
{
// 	if (show_result)
// 	{	
// 		imshow("result",interest_point);
// 		waitKey(1);
// 		imshow("result of raw headpoints",interest_point_raw);
// 		waitKey(1);
// 		imshow("result with thresh ckb",interest_point1);
// 		waitKey(1);
		imshow("result",interest_point2);
		waitKey(1);
		imwrite("headPoints_"+name+".jpg",interest_point2);
// 		imshow("result with all methods",interest_point4);
// 		waitKey(1);
// 		imshow("result with thresh ckb and headsize",interest_point3);
// 		waitKey(1);
//	}
// 	if (do_region_grow)
// 	{
// 		imshow("region grow",region_grow_map);
// 		waitKey(1);
// 	}
// 	if (show_edge)
// 	{
// 		imshow("edges from Canny",edge_map_thresh);
// 		waitKey(1);
// 	}
// 	if (show_distance_map)
// 	{
// 		imshow("distance map",sub_distance_maps[0]);
// 		waitKey(1);
// 	}
// 	if (show_responses)
// 	{
// 		for (int i=0;i<sub_responses.size();++i)
// 		{
// 			imshow("response"+string('0'+i,1),sub_responses[i]);
// 			waitKey(1);
// 		}
// 	}
// 	if (show_histogram)
// 	{
// 		int histSize=10;
// 		int hist_w = 900; int hist_h = 500;
// 		int bin_w = cvRound( (double) hist_w/histSize );
// 		Mat histImage=Mat::zeros( hist_h, hist_w, CV_8U);
// 		normalize(histogram_image, histogram_image, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
// 		for( int i = 1; i < histSize; i++ )
// 		{
// 			line( histImage, Point( bin_w*(i-1), hist_h - cvRound(histogram_image.at<float>(i-1,0)) ) ,
// 				Point( bin_w*(i), hist_h - cvRound(histogram_image.at<float>(i,0)) ),
// 				255, 2, 8, 0  );
// 		}
// 		imshow("histogram", histImage );
// 		waitKey(1);
// 	}
}
void segment::sjbh()
{
	double dmin,dmax;
	minMaxLoc(depth_map,&dmin,&dmax);
//	cout<<dmax<<endl;
	depth_sjbh=Mat::zeros(int(255)+1,depth_map.cols,CV_32F);
	for (int i=0;i<depth_map.cols;++i)
	{
		for (int j=depth_map.rows-1;j>=0;--j)
		{
			int flag=filter_map.at<uchar>(j,i);
			int flag1=depth_mask.at<uchar>(j,i);
			//cout<<flag<<endl;
 			if (flag<100&&flag1!=0)
 			{
				int val=gray_map.at<uchar>(j,i);
				depth_sjbh.at<float>(255-val,i)=255-j;
			}
// 			if (j<100)
// 			{
// 				depth_sjbh.at<float>(val,i)=240-j;
// 			}
		}
	}
	depth_sjbh.convertTo(depth_sjbh,CV_8U);
// 	imshow("shi jiao bian huan no operation",depth_sjbh);
// 	waitKey(1);
	//threshold(depth_sjbh,depth_sjbh,50,255,CV_8U);

	Mat tmp;
	int erode_size=2;
	Mat element = getStructuringElement( MORPH_RECT,
		Size( 2*erode_size + 1, 2*erode_size+1 ),
		Point( erode_size, erode_size ) );
	int dilate_size=1;
	Mat element1 = getStructuringElement( MORPH_RECT,
		Size( 2*dilate_size + 1, 2*dilate_size+1 ),
		Point( dilate_size, dilate_size ) );
	dilate( depth_sjbh, tmp,element1);
	//erode( depth_sjbh, depth_sjbh,element);
	//dilate( depth_sjbh, depth_sjbh,element);
	//resize(depth_sjbh,depth_sjbh,Size(depth_sjbh.cols,depth_sjbh.rows/4));
	//depth_sjbh=depth_sjbh.t();
// 	imshow("shi jiao bian huan",tmp);
// 	imwrite("topdown-view_"+name+".jpeg",tmp);
// 	waitKey(1);
	
	threshold(tmp,sjbh_binary,180,255,THRESH_BINARY);
// 	imshow("sjbh binary",sjbh_binary);
// 	waitKey(1);
// 	imwrite("topdown-binary_"+name+".jpeg",sjbh_binary);

	vector<vector<Point> > contours;
	Mat sjbh_binary_clone=sjbh_binary.clone();
	findContours(sjbh_binary_clone,contours, CV_RETR_LIST , CV_CHAIN_APPROX_NONE );
	//deal_with_contours(contours,0);
	//Mat gray_clone=gray_map.clone();

	headpoints_location.clear();
	for (int i=0;i<contours.size();++i)
	{
		if (contours[i].size()>30)
		{
			double x=0,y=0;
			for (int j=0;j<contours[i].size();++j)
			{
				x+=contours[i][j].x;
				y+=contours[i][j].y;
			}
			x=x/contours[i].size();y=y/contours[i].size();
			if (y<depth_map.rows-20)
			{
				for (int y=0;y<depth_map.rows;++y)
				{
					if (depth_mask.at<uchar>(y,x)!=0)
					{
						Point2i p;
						p.x=x;p.y=y;
						headpoints_location.push_back(p);
						break;
					}
				}
			}
		}
	}
// 	imshow("depth with seed",gray_clone);
// 	waitKey(1);
// 	imwrite("seed_"+name+".jpeg",gray_clone);
}
void segment::set_background(const Mat& bg)
{
	Mat bg_tmp=bg.clone();
	bg_tmp.convertTo(bg_tmp,CV_32F);
	if (background_depth.data==NULL)
	{
		background_depth=bg_tmp.clone();
	}
	if (bg_count<50)
	{
		for (int i=0;i<bg.rows;++i)
		{
			for (int j=0;j<bg.cols;++j)
			{
				background_depth.at<float>(i,j)=
					max(background_depth.at<float>(i,j),bg_tmp.at<float>(i,j));
			}
		}
		bg_count++;
	}
}
void segment::show_difference()
{
	difference_map=Mat::zeros(depth_map.rows,depth_map.cols,CV_8U);
	for (int i=0;i<depth_map.rows;++i)
	{
		for (int j=0;j<depth_map.cols;++j)
		{
			if (depth_map.at<float>(i,j)!=0)//&&background_depth.at<float>(i,j)!=0)
			{
				double dis=depth_map.at<float>(i,j)-background_depth.at<float>(i,j);
				if (dis<-500)
				{
					difference_map.at<uchar>(i,j)=255;
				}
			}
		}
	}
	imshow("background substract",difference_map);
	waitKey(1);
}
void segment::find_bg()
{
	int bg_dis=0,bg_number=0;
	Mat res=Mat::zeros(gray_map.rows,gray_map.cols,CV_8U);
	for (int dis=100;dis<255;++dis)
	{
		int c=0;
		Mat tmp=Mat::zeros(gray_map.rows,gray_map.cols,CV_8U);
		for (int i=0;i<gray_map.rows;++i)
		{
			for (int j=0;j<gray_map.cols;++j)
			{
				int val=abs(gray_map.at<uchar>(i,j)-dis);
				if (val<10)
				{
					++c;
					tmp.at<uchar>(i,j)=255;
				}
			}
		}
		if (c>bg_number&&c>gray_map.rows*gray_map.cols/4)
		{
			res=tmp;
			bg_number=c;
			bg_dis=dis;
		}
	}
	imshow("bg plane",res);
	waitKey(1);
}
void segment::compute_height()
{
	double center_x=depth_map.cols/2,center_y=depth_map.rows/2;
	double focal_y=300;
	height_map=Mat::zeros(depth_map.rows,depth_map.cols,CV_32F);
	for (int i=0;i<depth_map.cols;++i)
	{
		for (int j=0;j<depth_map.rows;++j)
		{
			double depth=depth_map.at<float>(j,i);
			if (depth!=0)
			{
				height_map.at<float>(j,i)=(j-center_y)*depth/focal_y;
			}
		}
	}
	//cout<<"--------"<<endl;
	for (int i=height_map.rows-1;i>=0;--i)
	{
		//cout<<"height:"<<height_map.at<float>(i,160)<<endl;
	}
// 	cout<<"height 1:"<<height_map.at<float>(180,150)<<endl
// 		<<"height 2:"<<height_map.at<float>(220,150)<<endl
// 		<<"height 3:"<<height_map.at<float>(180,110)<<endl
// 		<<"height 4:"<<height_map.at<float>(220,110)<<endl;
	double height_min,height_max;
	minMaxLoc(height_map,&height_min,&height_max);
/*	cout<<"height min: "<<height_min<<endl<<"height max: "<<height_max<<endl;*/
	height_absolute=height_max-height_min;

	double abs_height=2000;
	for (int i=0;i<depth_map.cols;++i)
	{
		for (int j=0;j<depth_map.rows;++j)
		{
			double height=height_map.at<float>(j,i);
			if (height<height_max-abs_height)
			{
				depth_mask.at<uchar>(j,i)=255;
			}
			height_map.at<float>(j,i)=height_max-height;
		}
	}
	
	threshold(depth_mask,depth_mask,128,255,THRESH_BINARY_INV);
	int erode_size=3;
	Mat element = getStructuringElement( MORPH_RECT,
		Size( 2*erode_size + 1, 2*erode_size+1 ),
		Point( erode_size, erode_size ) );
	int dilate_size=3;
	Mat element1 = getStructuringElement( MORPH_RECT,
		Size( 2*dilate_size + 1, 2*dilate_size+1 ),
		Point( dilate_size, dilate_size ) );
	erode( depth_mask, depth_mask,element);
	dilate( depth_mask, depth_mask,element1);
// 	imshow("depth after cut",depth_map);
// 	waitKey(1);
}

void segment::compute_cost()
{
	/*
	三个方面考虑每一个候选的种子点的正确性
	1、高度信息，高度不能太高也不能太低
	2、深度信息，尽量在屏幕中比较靠前的位置
	3、左右信息，比较靠近屏幕的中心
	*/
	points_cost.clear();

	//计算处理后的深度图的深度均值
	double depth_avg=0;int depth_num=0;
	double d_min=8000,d_max=0;
	for (int i=0;i<depth_map.rows;++i)
	{
		for (int j=0;j<depth_map.cols;++j)
		{
			double d=depth_map.at<float>(i,j);
			if (d>0)
			{
				d_min=min(d,d_min);
			}
			d_max=max(d,d_max);
			int flag1=filter_map.at<uchar>(i,j);
			int flag2=depth_mask.at<uchar>(i,j);
			if (flag1==0&&flag2!=0)
			{
				depth_avg+=d;
				++depth_num;
			}
		}
	}
	depth_avg/=depth_num;
	//depth_avg-=500;

	headpoints_location_1.clear();
	//计算每一个点对应的cost

	int num=0;
	for (int i=0;i<headpoints_location.size();++i)
	{
		double cost=0;
		Point p=headpoints_location[i];
		
		//深度惩罚项
		double depth=depth_map.at<float>(p.y,p.x);
		cost+=abs(depth-depth_avg)/(d_max-d_min);

		//高度惩罚项
		double height=height_map.at<float>(p.y,p.x);
		cost+=abs(height-height_center)/height_absolute;

		//左右惩罚项
		cost+=abs(p.x-depth_map.cols/2)*1.0/depth_map.cols;

		points_cost.push_back(cost);
/*		cout<<"points cost: "<<cost<<endl;*/

		circle(gray_clone,p,10,0,10);

		if (cost<cost_threshold)
		{
			headpoints_location_1.push_back(p);
			circle(gray_clone,p,5,255,5);
			++num;
		}
	}


}

vector<Point2i> segment::get_seed_raw()
{
	return headpoints_location;
}
vector<double> segment::get_headSize()
{
	return headpoints_radius;
}
vector<Point2i> segment::get_seed()
{
	return headpoints_location_1;
}
void segment::useMOG()
{
	imshow("raw image",depth_map);waitKey(1);
	//BackgroundSubtractorMOG2 bgMOG(10, 2, false );
	Mat fg_depth;
#ifdef CV_VERSION_EPOCH
	my_MOG.operator()(depth_map, fg_depth, 0.005);
#elif CV_VERSION_MAJOR >= 3
	//TODO: cv3
#endif //CV_VERSION_EPOCH

	//bgMOG.operator()(gray_map,fg_depth);
	Mat bg_depth;
	//pMOG->getBackgroundImage(bg_depth);
	//bgMOG.getBackgroundImage(bg_depth);
	//bg_depth.convertTo(bg_depth,CV_8U);
	//fg_depth.convertTo(fg_depth,CV_8U);
	imshow("foreground of MOG",fg_depth);waitKey(1);
}

vector<Point> segment::get_seperate_points(const Mat& M,bool showResult,bool Delay)
{
	Mat tmp=M.clone();
	vector<vector<Point> > contours;
	findContours(tmp,contours, CV_RETR_LIST , CV_CHAIN_APPROX_NONE );
	if (contours.size()==0)
	{
		cout<<"no contours find!"<<endl;
		vector<Point> res;
		return res;
	}
	//找最大的轮廓
	int idx=0;
	int size_max=0;
	for (int i=0;i<contours.size();++i)
	{
		if (contours[i].size()>size_max)
		{
			size_max=contours[i].size();
			idx=i;
		}
	}
	vector<Point> P=contours[idx];
	//循环遍历P，找到最左边和最右边的点
	Point p_left,p_right;
	Point p_top,p_down;
	int index_left,index_right;
	p_left.x=region_grow_map.cols;
	p_right.x=0;
	p_top.y=region_grow_map.cols;
	p_down.y=0;
	int size_p=P.size();
	int num=size_p/20;
	for (int i=0;i<size_p;++i)
	{
		if (P[i].x<p_left.x)
		{
			p_left=P[i];index_left=i;
		}
		if (P[i].x>p_right.x)
		{
			p_right=P[i];index_right=i;
		}
		if (P[i].y<p_top.y)
		{
			p_top=P[i];
		}
		if (P[i].y>p_down.y)
		{
			p_down=P[i];
		}
	}
	int region_height=p_down.y-p_top.y;
	//因为不知道轮廓点是顺时针还是逆时针，这里要判断一下，保证以顺时针的顺序遍历从左到右的点
	int flag=-1;
	int test=(index_left-1)%size_p;
	if (P[(index_left+1)%size_p].y<P[(index_left-1+size_p)%size_p].y)
	{
		flag=1;
	}
	//从左到右开始寻找局部最高和最低点，局部最低点作为输出结果
	vector<Point> local_minimum;vector<int> local_min_index;
	vector<Point> local_maximum;
	for (int i=index_left;i!=index_right;i=(i+flag+size_p)%size_p)
	{
		Point p=P[i];
		bool is_min=true;
		bool is_max=true;
		int min_y=region_grow_map.rows;
		for (int j=1;j<=num;++j)
		{
			if (p.y>P[(i+j)%size_p].y||p.y>P[(i-j+size_p)%size_p].y)
			{
				is_max=false;
			}
			if (p.y<P[(i+j)%size_p].y||p.y<P[(i-j+size_p)%size_p].y)
			{
				is_min=false;
			}
			if ((!is_max)&&(!is_min))
			{
				break;
			}
			min_y=min(min_y,min(P[(i+j)%size_p].y,P[(i-j+size_p)%size_p].y));
		}
		if (is_min)
		{
			if (local_minimum.size()==0||abs(local_minimum[local_minimum.size()-1].x-p.x)>5)
			{
				if (p.y-p_top.y>region_height/3)
				{

					if (p.y-min_y/(2*num)>10)
					{
						local_minimum.push_back(p);local_min_index.push_back(i);
					}
				}
			}
		}
		if (is_max)
		{
			if (local_maximum.size()==0||abs(local_maximum[local_maximum.size()-1].x-p.x)>5)
			{
				local_maximum.push_back(p);
			}
		}
	}
	if (showResult)
	{
		cout<<"minimum numbers: "<<local_minimum.size()<<endl;
		for (int i=0;i<local_minimum.size();++i)
		{
			circle(tmp,local_minimum[i],5,128,5);
		}
		imshow("minimum points",tmp);
		if (Delay)
			waitKey(0);
		else
			waitKey(1);
	}
	return local_minimum;
}

vector<Mat> segment::get_seperate_masks(const Mat& M,bool showResult,bool Delay)
{
	//为避免出错（奇怪的错误），对tmp进行膨胀,避免出现一个像素宽度的轮廓
	Mat tmp=M.clone();
	int dilate_size=1;
	Mat element1 = getStructuringElement( MORPH_RECT,
		Size( 2*dilate_size + 1, 2*dilate_size+1 ),
		Point( dilate_size, dilate_size ) );
	dilate( tmp,tmp,element1);
	vector<vector<Point> > contours;
	findContours(tmp,contours, CV_RETR_LIST , CV_CHAIN_APPROX_NONE );
	if (contours.size()==0)
	{
		cout<<"no contours find!"<<endl;
		vector<Mat> res;
		return res;
	}
	//找最大的轮廓
	int idx=0;
	int size_max=0;
	for (int i=0;i<contours.size();++i)
	{
		if (contours[i].size()>size_max)
		{
			size_max=contours[i].size();
			idx=i;
		}
	}
	vector<Point> P=contours[idx];
	double total_area=contourArea(P);
	double area_rate=0.2;
	//循环遍历P，找到最左边和最右边的点
	Point p_left,p_right;
	Point p_top,p_down;
	int index_left,index_right;
	p_left.x=tmp.cols;
	p_right.x=0;
	p_top.y=tmp.cols;
	p_down.y=0;
	int size_p=P.size();
	int num=size_p/20;
	for (int i=0;i<size_p;++i)
	{
		if (P[i].x<p_left.x)
		{
			p_left=P[i];index_left=i;
		}
		if (P[i].x>p_right.x)
		{
			p_right=P[i];index_right=i;
		}
		if (P[i].y<p_top.y)
		{
			p_top=P[i];
		}
		if (P[i].y>p_down.y)
		{
			p_down=P[i];
		}
	}
	int region_height=p_down.y-p_top.y;
	//因为不知道轮廓点是顺时针还是逆时针，这里要判断一下，保证以顺时针的顺序遍历从左到右的点
	int flag=-1;
	int test=(index_left-1)%size_p;
	if (P[(index_left+1)%size_p].y<P[(index_left-1+size_p)%size_p].y)
	{
		flag=1;
	}
	//从左到右开始寻找局部最高和最低点，局部最低点作为输出结果
	vector<Point> local_minimum;vector<int> local_min_index;
	vector<Point> local_maximum;
	for (int i=index_left;i!=index_right;i=(i+flag+size_p)%size_p)
	{
		Point p=P[i];
		bool is_min=true;
		bool is_max=true;
		int min_y=tmp.rows;
		for (int j=1;j<=num;++j)
		{
			if (p.y>P[(i+j)%size_p].y||p.y>P[(i-j+size_p)%size_p].y)
			{
				is_max=false;
			}
			if (p.y<P[(i+j)%size_p].y||p.y<P[(i-j+size_p)%size_p].y)
			{
				is_min=false;
			}
			if ((!is_max)&&(!is_min))
			{
				break;
			}
			min_y=min(min_y,min(P[(i+j)%size_p].y,P[(i-j+size_p)%size_p].y));
		}
		if (is_min)
		{
			if (local_minimum.size()==0||abs(local_minimum[local_minimum.size()-1].x-p.x)>5)
			{
				if (p.y-p_top.y>region_height/3)
				{
					if (p.y-min_y>10)
					{
						local_minimum.push_back(p);local_min_index.push_back(i);
					}
				}
			}
		}
		if (is_max)
		{
			if (local_maximum.size()==0||abs(local_maximum[local_maximum.size()-1].x-p.x)>5)
			{
				local_maximum.push_back(p);
			}
		}
	}
	if (showResult)
	{
		cout<<"minimum numbers: "<<local_minimum.size()<<endl;
		for (int i=0;i<local_minimum.size();++i)
		{
			circle(tmp,local_minimum[i],5,128,5);
		}
		imshow("minimum points",tmp);waitKey(1);
	}
	vector<Mat> Masks;
	if (local_minimum.size()==0)
	{
		if (showResult)
		{
			imshow("demo show seperate region",M);
			imwrite("seperate_mask_"+name+".jpg",M);
			if (Delay)
				waitKey(0);
			else
				waitKey(1);
		}
		Masks.push_back(M.clone());
		return Masks;
	}
	int color_1=255/(local_minimum.size()+1);
	int color_2=1;
	Mat demo_show=Mat::zeros(tmp.rows,tmp.cols,CV_8U);
	//根据每一个最小值点，将原始mask分成很多小块
	if (local_minimum.size()>=1)
	{
		//先把最左边分开
		vector<vector<Point>> c;
		vector<Point> contour;
		Mat mask=Mat::zeros(tmp.rows,tmp.cols,CV_8U);
		Point p_mid1,p_mid2;
		int index_mid1,index_mid2;
		p_mid1=P[local_min_index[0]];index_mid1=local_min_index[0];
		int _c=0;

		for (int j=local_min_index[0];;j=(j-flag+size_p)%size_p)
		{
			Point p=P[j];
			contour.push_back(p);
			if (p.x==P[local_min_index[0]].x&&p.y>P[local_min_index[0]].y)
			{
				p_mid2=p;index_mid2=j;
				break;
			}
			++_c;
			if (_c>size_p)
			{
				p_mid2=p;index_mid2=j;
				break;
			}
			
		}
		double area=contourArea(contour);
		if (area>total_area*area_rate)
		{
		c.push_back(contour);
		drawContours(mask,c,-1,255,CV_FILLED);
		mask=mask&M;
		demo_show.setTo(color_1*color_2,mask);++color_2;
		/*drawContours(demo_show,c,-1,color_1*color_2,CV_FILLED);++color_2;*/
// 		if (showResult)
// 		{
// 			imshow("seperated mask left",mask);waitKey(1);
// 		}
		Masks.push_back(mask.clone());
		}

		//中间部分分开
		for (int i=0;i<local_minimum.size()-1;++i)
		{
			mask=Mat::zeros(tmp.rows,tmp.cols,CV_8U);
			c.clear();
			contour.clear();

			for (int j=local_min_index[i+1];j!=index_mid1;j=(j-flag+size_p)%size_p)
			{
				Point p=P[j];
				contour.push_back(p);
			}
			contour.push_back(p_mid1);
			contour.push_back(p_mid2);

			p_mid1=P[local_min_index[i+1]];index_mid1=local_min_index[i+1];

			for (int j=index_mid2;;j=(j-flag+size_p)%size_p)
			{
				Point p=P[j];
				contour.push_back(p);
				if (p.x==P[local_min_index[i+1]].x&&p.y>P[local_min_index[i+1]].y)
				{
					p_mid2=p;index_mid2=j;
					break;
				}
			}
			area=contourArea(contour);
			if (area>total_area*area_rate)
			{
			c.push_back(contour);
			drawContours(mask,c,-1,255,CV_FILLED);
			mask=mask&M;
			demo_show.setTo(color_1*color_2,mask);++color_2;
			/*drawContours(demo_show,c,-1,color_1*color_2,CV_FILLED);++color_2;*/
// 			if (showResult)
// 			{
// 				imshow("seperated mask "+string('0'+i,1),mask);waitKey(1);
// 			}
			Masks.push_back(mask.clone());
			}
		}

		//最右边分开
		mask=Mat::zeros(tmp.rows,tmp.cols,CV_8U);
		c.clear();
		contour.clear();
		for (int j=index_mid1;j!=index_mid2;j=(j+flag+size_p)%size_p)
		{
			Point p=P[j];
			contour.push_back(p);
			if (p.x==P[local_min_index[0]].x&&p.y>P[local_min_index[0]].y)
			{
				contour.push_back(P[index_mid2]);
				break;
			}
		}
		area=contourArea(contour);
		if (area>total_area*area_rate)
		{
		c.push_back(contour);
		drawContours(mask,c,-1,255,CV_FILLED);
		mask=mask&M;
		demo_show.setTo(color_1*color_2,mask);++color_2;
		/*drawContours(demo_show,c,-1,color_1*color_2,CV_FILLED);++color_2;*/
// 		if (showResult)
// 		{
// 			imshow("seperated mask right",mask);waitKey(1);
// 		}
		Masks.push_back(mask.clone());
		}
	}
	if (showResult)
	{
		imshow("demo show seperate region",demo_show);
		imwrite("seperate_mask_"+name+".jpg",demo_show);
		if (Delay)
			waitKey(0);
		else
			waitKey(1);
	}
	return Masks;
}
vector<Mat> segment::get_seperate_masks(const cv::Mat& fgMask,const cv::Mat& mogMask,vector<Point> headPoints,std::vector<double> headSize,bool showResult/* =false */,bool drawHist/* =false */,bool Delay/* =false */)
{
	//首先统计fgMask中每条竖线上的前景点个数
	Mat mog_in_fg=fgMask&mogMask;
	vector<int> points;
	for (int i=0;i<mog_in_fg.cols;++i)
	{
		int num=0;
		for (int j=0;j<mog_in_fg.rows;++j)
		{
			if (mog_in_fg.at<uchar>(j,i)!=0)
			{
				++num;
			}
		}
		points.push_back(num);
		//cout<<num<<endl;
	}
	//画出上述统计点的直方图
	if (drawHist)
	{
		imshow("motion points",mog_in_fg);waitKey(1);
		imwrite("motionPoints_"+name+".jpg",mog_in_fg);
		Mat hist=Mat::zeros(mog_in_fg.rows,mog_in_fg.cols,CV_8U);
		for (int i=0;i<points.size();++i)
		{
			Point p1;p1.x=i;p1.y=mog_in_fg.rows;
			Point p2;p2.x=i;p2.y=mog_in_fg.rows-points[i];
			line(hist,p1,p2,255);
		}
		imshow("histogram of motion points in fgMask",hist);waitKey(1);
		imwrite("mog_histogram_"+name+".jpg",hist);
	}
	/*
	如何分割：
	两种方法：
	1、在直方图中找极大值点，从这些极大值点处直接进行分割；
	2、利用头部点的信息，看头部点位置处直方图的点数信息，
	   若大于一个阈值，则从头部点处往左右分割；
	*/
	vector<Mat> res;
	//寻找直方图中的极大值点

	//判断头部点竖线上的的动点数量并根据阈值筛掉不满足要求的头部点
	if (headPoints.size()==0)
	{
		res.push_back(fgMask.clone());
		return res;
	}
	vector<int> seperate_line;
	vector<int> seperate_line_width;
	double head_body_rate=4;
	int pointNumberThreshold=10;
	int pointNumber=5;
	for (int i=0;i<headPoints.size();++i)
	{
		int hist_left=max(0,headPoints[i].x-5);
		int hist_right=min(headPoints[i].x+5,int(mog_in_fg.cols)-1);
		int num=0;
		for (int j=hist_left;j<=hist_right;++j)
		{
			num+=points[j];
		}
		cout<<"head motion points number: "<<num<<endl;
		num/=hist_right-hist_left+1;
		if (num>=pointNumberThreshold)
		{
			seperate_line.push_back(headPoints[i].x);
			seperate_line_width.push_back(int(headSize[i]*head_body_rate));
		}
	}
	//根据seperate_line对原始的前景mask进行划分

	Mat tmp=fgMask.clone();
	vector<vector<Point> > contours;
	findContours(tmp,contours, CV_RETR_LIST , CV_CHAIN_APPROX_NONE );
	if (contours.size()==0)
	{
		cout<<"no contours find!"<<endl;
		vector<Mat> res;
		return res;
	}
	//找最大的轮廓
	int idx=0;
	int size_max=0;
	for (int i=0;i<contours.size();++i)
	{
		if (contours[i].size()>size_max)
		{
			size_max=contours[i].size();
			idx=i;
		}
	}
	vector<Point> P=contours[idx];
	//循环遍历P，找到最左边和最右边的点
	Point p_left,p_right;
	int index_left,index_right;
	p_left.x=region_grow_map.cols;
	p_right.x=0;
	int size_p=P.size();
	int num=size_p/20;
	for (int i=0;i<size_p;++i)
	{
		if (P[i].x<p_left.x)
		{
			p_left=P[i];index_left=i;
		}
		if (P[i].x>p_right.x)
		{
			p_right=P[i];index_right=i;
		}
	}
	//因为不知道轮廓点是顺时针还是逆时针，这里要判断一下，保证以顺时针的顺序遍历从左到右的点
	int flag=-1;
	int test=(index_left-1)%size_p;
	if (P[(index_left+1+size_p)%size_p].y<P[(index_left-1+size_p)%size_p].y)
	{
		flag=1;
	}
	//确定要分割的区域的左右边界
	int seperate_left=p_left.x,seperate_right=p_right.x;

	Mat mask=Mat::zeros(fgMask.rows,fgMask.cols,CV_8U);
	for (int i=0;i<seperate_line.size();++i)
	{
		int left_line=max(seperate_line[i]-seperate_line_width[i],seperate_left);
		int right_line=min(seperate_line[i]+seperate_line_width[i],seperate_right);
		//seperate_left=right_line;
		Point left_top;left_top.y=0;left_top.x=left_line;
		int index_left_top=0;
		Point right_top;right_top.y=0;right_top.x=right_line;
		int index_right_top=0;
		//对分割线，开始寻找轮廓上边对应的最低的点和下边对应的最高的点
		for (int j=index_left;j!=index_right;j=(j+flag+size_p)%size_p)
		{
			if (P[j].x==left_line&&P[j].y>=left_top.y)
			{
				left_top=P[j];
				index_left_top=j;
			}
			if (P[j].x==right_line&&P[j].y>=right_top.y)
			{
				right_top=P[j];
				index_right_top=j;
			}
		}
		Point left_down;left_down.y=fgMask.rows;left_down.x=left_line;
		int index_left_down=0;
		Point right_down;right_down.y=fgMask.rows;right_down.x=right_line;
		int index_right_down=0;
		for (int j=index_right;j!=index_left;j=(j+flag+size_p)%size_p)
		{
			if (P[j].x==left_line&&P[j].y<=left_down.y)
			{
				left_down=P[j];
				index_left_down=j;
			}
			if (P[j].x==right_line&&P[j].y<=right_down.y)
			{
				right_down=P[j];
				index_right_down=j;
			}
		}
		//画出该部分的轮廓
		vector<Point> contour;
		for (int j=index_left_top;j!=index_right_top;j=(j+flag+size_p)%size_p)
		{
			contour.push_back(P[j]);
		}
		//contour.push_back(right_top);
		for (int j=index_right_down;j!=index_left_down;j=(j+flag+size_p)%size_p)
		{
			contour.push_back(P[j]);
		}
		//contour.push_back(left_down);
		vector<vector<Point>> c;c.push_back(contour);
		drawContours(mask,c,-1,255,CV_FILLED);
		mask=mask&fgMask;
		res.push_back(mask);
	}
	imshow("seperate mask",mask);waitKey(1);
	imwrite("seperate_mask_"+name+".jpg",mask);
	return res;
}