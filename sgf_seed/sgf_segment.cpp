#include "sgf_segment.h"
#include <time.h>
#include <cmath>

using namespace sgf;
using namespace std;
using namespace cv;

segment::segment(bool show_result1,bool show_distance_map1,bool show_edge1,
	bool do_region_grow1,bool show_responses1,bool show_histogram1)
{
	show_result=show_result1;
	show_distance_map=show_distance_map1;
	show_edge=show_edge1;
	do_region_grow=do_region_grow1;
	show_responses=show_responses1;
	show_histogram=show_histogram1;
	accurate=0;
}
void segment::set_depthMap(const cv::Mat& depth)
{
	depth_map=depth.clone();
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
	distanceTransform(edge_map_thresh,distance_map,CV_DIST_L2,CV_DIST_MASK_PRECISE);
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

	//�Լ�ʵ�ֲ�����Ϊ3/4�Ľ�����
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

		//��ֵ��
		Mat matching_space_binary,matching_space1_binary;

		threshold(matching_space,matching_space_binary,threshold_binary_response,255,THRESH_BINARY);//���Զ�ֵ��
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
	return interest_point2.clone();
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
void segment::compute()
{
	fill_holes();
	smooth_image();

// 	imshow("depth image",gray_map);
// 	waitKey(1);

	//compute_hist();
// 	double dmin,dmax;
// 	minMaxLoc(depth_map,&dmin,&dmax);
// 	cout<<"+++++dmin,dmax: "<<dmin<<", "<<dmax<<endl;
	seperate_foot_and_ground();
	compute_filter_map_edge(threshold_filter_min,threshold_filter_max);
// 	imshow("seperate foot and ground",filter_map);
// 	waitKey(1);

	compute_edge(threshold_depth_min,threshold_depth_max);
// 	imshow("depth edge",edge_map_thresh);
// 	waitKey(1);
	//��Ҫ�� filter_edge
	//edge_map_thresh=edge_map_thresh+filter_edge;
// 	imshow("edge",edge_map_thresh);
// 	waitKey(1);
	threshold(edge_map_thresh,edge_map_thresh,128,255,THRESH_BINARY_INV);

	compute_distanceMap_2D();
// 	imshow("distance map without equalization",distance_map_uchar);
// 	waitKey(1);
	equalizeHist(distance_map_uchar,distance_map_uchar);

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

// 	region_grow_map=Mat::zeros(distance_map.rows,distance_map.cols,CV_8U);
// 	mask_of_distance=Mat::zeros(distance_map.rows,distance_map.cols,CV_8U);
// 	for (int i=0;i<region_of_interest.size();++i)
// 	{
// 		stack_list.clear();
// 		region_grow(region_of_interest[i].y,region_of_interest[i].x,1);
// 	}

	display();
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

			//���ҵ����������д�������ȥ��̫С���������ߴ�����״������Ҫ������������߾�ֵ����Ȳ�����Ҫ��
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
		>>a>>const_depth;

	config_file.close();
	}
	return openSucceed;
}
void segment::deal_with_contours(vector<vector<Point> >& contours,int k)
{ //��Ȥ����������״��С���򣬲����Լ�����
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
	//�·���������С��Χ����
	for (it=contours.begin();it!=contours.end();)
	{
		int min_x=1000,min_y=1000,max_x=0,max_y=0;
		int x_avg=0,y_avg=0;
		double s=(*it).size();
		for (int i=0;i<(*it).size();++i)
		{
			min_x=min((*it)[i].x,min_x);
			min_y=min((*it)[i].y,min_y);
			max_x=max((*it)[i].x,max_x);
			max_y=max((*it)[i].y,max_y);
			x_avg += int((*it)[i].x);
			y_avg += int((*it)[i].y);
		}
		if ((*it).size()!=0)
		{
			x_avg/=(*it).size();y_avg/=(*it).size();
		}
		else
		{
			x_avg=0;y_avg=0;
		}
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
	//�ٽ������Ͳ�����ȥ��ͼ���е�С���ֿ׶�

}
void segment::seperate_foot_and_ground()
{
	Mat kernal=Mat::zeros(6,1,CV_32F);
	for (int i=0;i<kernal.rows;++i)
	{
		if (i<3)
			kernal.at<float>(i,0)=1;
		else
			kernal.at<float>(i,0)=-1;
	}
	filter_map=Mat::zeros(depth_map.rows,depth_map.cols,CV_32F);
	filter2D(depth_map,filter_map,-1,kernal);
	//abs(filter_map);//ȡ����ֵ
	filter_map.convertTo(filter_map,CV_8U);
// 	double dmax,dmin;
// 	minMaxLoc(filter_map,&dmin,&dmax);
// 	cout<<"*****dmax,dmin:"<<dmax<<"--"<<dmin<<endl;


	for ( int i = 1; i < 4; i = i + 2 )
	{
		medianBlur ( filter_map, filter_map, i );
	}
// 	imshow("filter map",filter_map);
// 	waitKey(1);

	Mat tmp;
	Canny(filter_map,tmp,threshold_filter_min,threshold_filter_max);
// 	imshow("raw canny from filter map",tmp);
// 	waitKey(1);

	threshold(filter_map,filter_map,threshold_binary_filter,255,THRESH_BINARY);
	
	int erode_size=3;
	Mat element = getStructuringElement( MORPH_RECT,
		Size( 2*erode_size + 1, 2*erode_size+1 ),
		Point( erode_size, erode_size ) );
	erode( filter_map, filter_map,element);
	dilate( filter_map, filter_map,element);
// 	imshow("filter map binary",filter_map);
// 	waitKey(1);
}
void segment::compute_hist()
{
	double step_length=10;
	float range[]={min_depth,max_depth};
	int histSize=(max_depth-min_depth)/step_length;
	const float* histRange={range};
	bool uniform=true,accumulate=false;
	calcHist( &depth_map, 1, 0, Mat(), histogram_image, 1, &histSize, &histRange, uniform, accumulate );
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
		double _R=1.33*_h/4;//����ͼ��ֱ��ʽ��е���
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
			&&depth+const_depth<depth_map.at<float>(y_min,x))//ͷ����С�����ֵ��ϵ������Ҫ����)
			{
				circle(interest_point4,p,_r,0,2);
			//ȥ����ʾͬһ��ͷ���Ķ��Բ��ֻ�����뾶����һ��
			Point p_2i;
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
// 			if (compute_trueHead(p))//ֱ��ͼͳ�ƹ�����Ҫ��������
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

	//��������Ƿ���ȷ
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
	if (show_result)
	{	
// 		imshow("result",interest_point);
// 		waitKey(1);
// 		imshow("result of raw headpoints",interest_point_raw);
// 		waitKey(1);
// 		imshow("result with thresh ckb",interest_point1);
// 		waitKey(1);
		imshow("result",interest_point2);
		waitKey(1);
		//imwrite(name+".jpg",interest_point2);
// 		imshow("result with all methods",interest_point4);
// 		waitKey(1);
// 		imshow("result with thresh ckb and headsize",interest_point3);
// 		waitKey(1);
	}
	if (do_region_grow)
	{
		imshow("region grow",region_grow_map);
		waitKey(1);
	}
	if (show_edge)
	{
		imshow("edges from Canny",edge_map_thresh);
		waitKey(1);
	}
	if (show_distance_map)
	{
		imshow("distance map",sub_distance_maps[0]);
		waitKey(1);
	}
	if (show_responses)
	{
		for (int i=0;i<sub_responses.size();++i)
		{
			imshow("response"+string('0'+i,1),sub_responses[i]);
			waitKey(1);
		}
	}
	if (show_histogram)
	{
		int histSize=10;
		int hist_w = 900; int hist_h = 500;
		int bin_w = cvRound( (double) hist_w/histSize );
		Mat histImage=Mat::zeros( hist_h, hist_w, CV_8U);
		normalize(histogram_image, histogram_image, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
		for( int i = 1; i < histSize; i++ )
		{
			line( histImage, Point( bin_w*(i-1), hist_h - cvRound(histogram_image.at<float>(i-1,0)) ) ,
				Point( bin_w*(i), hist_h - cvRound(histogram_image.at<float>(i,0)) ),
				255, 2, 8, 0  );
		}
		imshow("histogram", histImage );
		waitKey(1);
	}
}
vector<Point2i> segment::get_seed()
{
	return headpoints_location;
}