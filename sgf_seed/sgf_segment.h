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
		segment(bool show_result1=false,bool show_distance_map1=false,bool show_edge1=false,
			bool do_region_grow1=false,bool show_responses1=false,bool show_histogram1=false);
		void set_depthMap(const cv::Mat&);
		void set_headTemplate2D();
		cv::Mat get_edgeMapWithThresh();
		cv::Mat get_distanceMap();
		cv::Mat get_result();
		cv::Mat get_result1();
		cv::Mat get_result2();
		cv::Mat get_result3();
		void set_name(std::string);
		void output(string);
		void compute();
		void read_config();
		std::vector<cv::Point> get_seed();
		int accurate;
		std::string videoname;
	private:
		void fill_holes();
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
		void display();

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
		/*-----*/

		double max_depth,min_depth;

		bool do_region_grow,show_histogram,show_distance_map,show_edge,show_responses,show_result;

		std::string name;
		cv::Mat depth_map;
		cv::Mat filter_map;
		cv::Mat histogram_image;
		cv::Mat gray_map;
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
		std::vector<cv::Point> headpoints_location;
		std::vector<double> headpoints_radius;
		std::vector<cv::Point2i> region_of_interest_raw;
		std::vector<cv::Vec3f> real_head;
		std::vector<cv::Mat> sub_contours;

		std::list<cv::Point2i> stack_list;
	};
}