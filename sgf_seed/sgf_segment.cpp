#include "sgf_segment.h"
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

segment::segment(){}

void segment::set_depthMap(const cv::Mat& depth)
{
	depth_map=depth.clone();
	depth_map.convertTo(depth_map,CV_32F);
	double dmax,dmin;
	minMaxLoc(depth_map,&dmin,&dmax);
	max_depth=dmax;min_depth=dmin;
	depth_map.convertTo(gray_clone,CV_8U,255.0/(dmax-dmin),-dmin*255.0/(dmax-dmin));
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
	int time_begin=clock();
	distanceTransform(edge_map_thresh,distance_map,distance_type,mask_type);
	cout<<"time cost in distance transform:"<<clock()-time_begin<<endl;
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
	sub_distance_maps.push_back(src.clone());

	//自己实现采样率为3/4的金字塔
	while (src.rows>150&&src.cols>150)
	{
		int c=src.cols*3/4,r=src.rows*3/4;
		Mat sub=Mat::zeros(r,c,CV_8U);
		for (int i=0;i<r;++i)
		{
			for (int j=0;j<c;++j)
				sub.at<uchar>(i,j)=src.at<uchar>(i+i/3,j+j/3);
		}
		sub_distance_maps.push_back(sub.clone());
		src=sub;
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
		matchTemplate(sub_distance_maps[i],head_template,matching_space, CV_TM_CCORR_NORMED );

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
vector<Point> segment::seedSGF(Mat dmat,bool showResult,bool seed_raw,Mat& depth_without_bg)
{
	set_depthMap(dmat);
	
	//fill_holes();

// 	compute_hist();
// 
// 	compute_height();
// 
// 	smooth_image();
// 
// 	seperate_foot_and_ground();
// 
// 	sjbh();
// // 	imshow("depth mask",depth_mask);
// // 	waitKey(1);
// 	compute_cost();
// 
// 
// 	if (showResult)
// 	{
// 		imshow("result",gray_clone);
// 		waitKey(1);
// 		imshow("depth without bg mask",depth_mask);
// 		waitKey(1);
// 		imshow("top down view",depth_sjbh);
// 		waitKey(1);
// 		imshow("top down view binary",sjbh_binary);
// 		waitKey(1);
// 	}
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


	int begin=clock();
	smooth_image();

	seperate_foot_and_ground();
 	compute_filter_map_edge(threshold_filter_min,threshold_filter_max);
// // 	imshow("seperate foot and ground",filter_map);
// // 	waitKey(1);
// 
 	compute_edge(threshold_depth_min,threshold_depth_max);
// // 	imshow("depth edge",edge_map_thresh);
// // 	waitKey(1);
 	edge_map_thresh=edge_map_thresh+filter_edge;
// 	imshow("edge",edge_map_thresh);
// 	waitKey(1);
	threshold(edge_map_thresh,edge_map_thresh,128,255,THRESH_BINARY_INV);

	cout<<"canny and seperate foot time cost:"<<clock()-begin<<endl;
	compute_distanceMap_2D();
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
// 
 	equalizeHist(distance_map_uchar,distance_map_uchar);
// 	imshow("distance 3",distance_map_uchar);
// 	waitKey(1);
// 
// 
	compute_subsamples();

	compute_response();

	region_of_interest.clear();
	region_of_interest_raw.clear();
	interest_point=edge_map_thresh.clone();
	interest_point_raw=edge_map_thresh.clone();
	interest_point1=edge_map_thresh.clone();
	interest_point2=edge_map_thresh.clone();
	interest_point3=edge_map_thresh.clone();
	interest_point4=edge_map_thresh.clone();
	find_and_draw_countours();
	choose_and_draw_interest_region();

	region_grow_map=Mat::zeros(distance_map.rows,distance_map.cols,CV_8U);
	mask_of_distance=Mat::zeros(distance_map.rows,distance_map.cols,CV_8U);
	for (int i=0;i<region_of_interest.size();++i)
	{
		stack_list.clear();
		/*region_grow(region_of_interest[i].y,region_of_interest[i].x,1);*/
	}
	cout<<"time spend:"<<clock()-begin<<endl;

	display();
	vector<Point> res;
	return res;
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
			deal_with_contours(contours,i);

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
	stack_list.push_front(p);
	while (stack_list.size()!=0)
	{
		Point2i p_tmp;
		p_tmp=stack_list.front();
		stack_list.pop_front();
		region_grow_map.at<uchar>(p_tmp.x,p_tmp.y)=255;
		for (int i=-1;i<2;i+=2)
		{
			double x=p_tmp.x+i,y=p_tmp.y;
			if (x>=0 && x<distance_map.rows && y>=0 && y<distance_map.cols)
			{
				if (mask_of_distance.at<uchar>(x,y)==0)
				{
					mask_of_distance.at<uchar>(x,y)=1;
					double tmp=distance_map.at<float>(x,y);
					//double tmp1=filter_map.at<uchar>(x,y);
					if (tmp>thres)//&&tmp1<100)
					{
						Point2i p1;
						p1.x=x;p1.y=y;
						stack_list.push_front(p1);
					}
				}
			}
			x=p_tmp.x,y=p_tmp.y+i;
			if (x>=0 && x<distance_map.rows && y>=0 && y<distance_map.cols)
			{
				if (mask_of_distance.at<uchar>(x,y)==0)
				{
					mask_of_distance.at<uchar>(x,y)=1;
					double tmp=distance_map.at<float>(x,y);
					double tmp1=filter_map.at<uchar>(x,y);
					if (tmp>thres&&tmp1<100)
					{
						Point2i p1;
						p1.x=x;p1.y=y;
						stack_list.push_front(p1);
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
	imshow("filter map1",filter_map);

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
 	imshow("filter map binary",filter_map);
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
	for (int i=0;i<region_of_interest_raw.size();++i)
	{
		Point2i p=region_of_interest_raw[i];
		int x=p.x;
		int y=p.y;
		double _r=distance_map.at<float>(y,x);
		circle(interest_point_raw,p,_r,0,2);
	}
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

		circle(interest_point1,p,_r,0,2);
		int x_min=max(0,int(x-_r*a)),x_max=min(int(x+_r*a),depth_map.cols-1);
		int y_min=max(0,int(y-_r*a));
		double bz=_r/_R;
		if (bz>threshold_headsize_min&&bz<threshold_headsize_max&&_r>5)
		{
			circle(interest_point3,p,_r,0,2);
			if (depth+const_depth<depth_map.at<float>(y,x_min)&&depth+const_depth<depth_map.at<float>(y,x_max)
			&&depth+const_depth<depth_map.at<float>(y_min,x))//头部大小与深度值关系规则，需要调参)
			{
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
	if (headpoints_location.size()==2)
	{
/*		accurate++;*/
		bool right=true;
		for (int i=0;i<headpoints_location.size();++i)
		{
			Point2i p=headpoints_location[i];
			if ((p.x>90&&p.x<150)||(p.x>170&&p.x<230))
			{
				if (p.y>0&&p.y<60)
				{
					right=true;
				}
				else
				{
					right=false;
					break;
				}
			}
			else
			{
				right=false;
				break;
			}
		}
		if (right)
		{
			accurate++;
		}
	}
}
void segment::display()
{
// 	if (show_result)
// 	{	
// 		imshow("result",interest_point);
// 		waitKey(1);
		imshow("result of raw headpoints",interest_point_raw);
		waitKey(1);
		imshow("result with thresh ckb",interest_point1);
		waitKey(1);
		imshow("result",interest_point2);
		waitKey(1);
		//imwrite(name+".jpg",interest_point2);
		imshow("result with all methods",interest_point4);
		waitKey(1);
		imshow("result with thresh ckb and headsize",interest_point3);
		waitKey(1);
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
	cout<<"--------"<<endl;
	for (int i=height_map.rows-1;i>=0;--i)
	{
		cout<<"height:"<<height_map.at<float>(i,160)<<endl;
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
vector<Point2i> segment::get_seed()
{
	return headpoints_location_1;
}