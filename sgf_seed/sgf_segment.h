#ifndef _SGF_SEGMENT
#define _SGF_SEGMENT
#include <opencv2/opencv.hpp>
#include <vector>
#include "Holefilling.h"
#include <fstream>
#include <string>
#include <list>



/*
implement of human detection using depth information by Kinect.
*/

namespace sgf
{
	class segment
	{
	public:
// 		segment(bool show_result1=true,bool show_distance_map1=false,bool show_edge1=false,
// 			bool do_region_grow1=false,bool show_responses1=false,bool show_histogram1=false);
// 		segment(bool show_result1=false,bool show_depth_without_bg1=false,
// 			bool show_topdown_view1=false,bool show_topdown_binary1=false);
		segment();
		void set_depthMap(const cv::Mat&);
		void set_mog_par(int history,int varthreshold,bool detect_shadow,int learningRate);

		void set_background(const cv::Mat&);
		bool set_headTemplate2D(const std::string &headTemplatePath);
		cv::Mat get_edgeMapWithThresh();
		cv::Mat get_distanceMap();
		cv::Mat get_result();
		cv::Mat get_result1();
		cv::Mat get_result2();
		cv::Mat get_result3();
		void set_name(std::string);
		void output(string);
		std::vector<cv::Point> seedSGF(cv::Mat dmat,bool showResult=false,bool raw_seed=false,cv::Mat& depth_without_bg=cv::Mat());
		std::vector<cv::Point> seed_method1(cv::Mat dmat,bool showResult=false,bool showTime=false);
		std::vector<cv::Point> seed_method2(cv::Mat dmat,bool showResult=false,bool showTime=false);
		bool read_config(const std::string &configPath);
		int accurate;
		std::string videoname;
	private:
		std::vector<cv::Point2i> get_seed();
		std::vector<cv::Point2i> get_seed_raw();
		void fill_holes();
		void show_difference();
		void smooth_image();
		void compute_edge();
		void compute_edge(double,double);
		void compute_filter_map_edge(double,double);
		void compute_distanceMap_2D();
		void compute_subsamples();
		void compute_response();
		double get_headheight(double);
		void find_and_draw_countours();
		void deal_with_contours(std::vector<std::vector<cv::Point> >&,int);
		void region_grow(int,int,int);
		void seperate_foot_and_ground();
		void compute_hist();
		bool compute_trueHead(const cv::Point2i& p);
		void choose_and_draw_interest_region();
		void sjbh();
		void display();
		void find_bg();
		void compute_height();
		void compute_cost();
		void useMOG();


	private:
		//一些配置信息
		/*-----*/
		double threshold_depth_min,threshold_depth_max; //绝对阈值
		double threshold_binary_filter;
		double threshold_filter_min,threshold_filter_max;
		double threshold_binary_response;
		double threshold_contour_size;
		double threshold_contour_ckb;
		double threshold_headsize_min,threshold_headsize_max;
		double a,const_depth;
		int distance_type,mask_type;
		double height,depth_rate1,depth_rate2,bar_width;
		double height_center,cost_threshold;
		/*-----*/

		bool _SGF_DEBUG,_SGF_SHOW,_SGF_MODE,_MOG;
		int mog_history,mog_varthreshold,mog_lr;
		bool mog_detect_shadow;

		double max_depth,min_depth;
		double height_absolute;

		bool has_set_depth;

		bool do_region_grow,show_histogram,show_distance_map,
			show_edge,show_responses,show_result,show_depth_without_bg,
			show_topdown_view,show_topdown_binary;
		
		std::string name;
#ifdef CV_VERSION_EPOCH
		cv::BackgroundSubtractorMOG2 my_MOG;
#elif CV_VERSION_MAJOR >= 3
		//TODO: cv3
#endif //CV_VERSION_EPOCH

		cv::Mat background_depth;
		cv::Mat difference_map;
		int bg_count;

		cv::Mat depth_map;
		cv::Mat depth_mask;

		cv::Mat height_map;
		cv::Mat filter_map1;
		cv::Mat filter_map;
		cv::Mat filter_map_binary;
		cv::Mat histogram_image;
		cv::Mat gray_map;
		cv::Mat gray_clone;
		cv::Mat depth_sjbh;
		cv::Mat sjbh_binary;

		cv::Mat head_template;
		cv::Mat edge_map_thresh;
		cv::Mat filter_edge;
		cv::Mat distance_map;
		cv::Mat mask_of_distance;
		cv::Mat region_grow_map;
		cv::Mat distance_map_uchar;
		cv::Mat interest_point;
		cv::Mat interest_point_raw;
		cv::Mat interest_point1;
		cv::Mat interest_point2;
		cv::Mat interest_point3;
		cv::Mat interest_point4;
		std::vector<cv::Mat> sub_distance_maps;
		std::vector<cv::Mat> sub_responses;
		std::vector<cv::Mat> sub_responses_binary;
		std::vector<cv::Point2i> region_of_interest;
		std::vector<cv::Point2i> headpoints_location;
		std::vector<cv::Point2i> headpoints_location_1;
		std::vector<double> points_cost;
		std::vector<double> headpoints_radius;
		std::vector<cv::Point2i> region_of_interest_raw;
		std::vector<cv::Vec3f> real_head;
		std::vector<cv::Mat> sub_contours;

		std::list<cv::Point2i> stack_list;
	};
}

#endif //_SGE_SEGMENT