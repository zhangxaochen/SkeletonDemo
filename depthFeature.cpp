#include "depthFeature.h"

GSubDepthFeature::GSubDepthFeature(CvPoint u,
										 CvPoint v)
{
	_u.x = u.x * DepthModify;
	_u.y = u.y * DepthModify;
	_v.x = v.x * DepthModify;
	_v.y = v.y * DepthModify;
}

GSubDepthFeature::~GSubDepthFeature()
{
	;
}

GSubDepthFeature::GSubDepthFeature(const GSubDepthFeature& f)
{
	_u = f._u;
	_v = f._v;
}

float GSubDepthFeature::getValue(IplImage* img, CvPoint pixel)
{
	double depth = CV_IMAGE_ELEM(img, unsigned int, pixel.y, pixel.x);
	if(depth == BACKGROUNG) return 0;
	double _depth = 1.0/depth;
	int x, y;
	double depth_u, depth_v;
	x = (int)(pixel.x+_u.x*_depth);
	y = (int)(pixel.y+_u.y*_depth);
	(x >= 0 && x < img->width && y >= 0 && y < img->height) ? 
		depth_u = CV_IMAGE_ELEM(img, unsigned int, y, x) : depth_u = BACKGROUNG;
	x = (int)(pixel.x+_v.x*_depth);
	y = (int)(pixel.y+_v.y*_depth);
	(x >= 0 && x < img->width && y >= 0 && y < img->height) ? 
		depth_v = CV_IMAGE_ELEM(img, unsigned int, y, x) : depth_v = BACKGROUNG;
	if(depth_u == BACKGROUNG || depth_v == BACKGROUNG)
		return VERY_LARGE_POSITIVE;
	else return (depth_u - depth_v);
}

string GSubDepthFeature::toString()
{
	string ret;
	ret.append(LogCat::to_string(_u.x/DepthModify));
	ret.append(" ");
	ret.append(LogCat::to_string(_u.y/DepthModify));
	ret.append(" ");
	ret.append(LogCat::to_string(_v.x/DepthModify));
	ret.append(" ");
	ret.append(LogCat::to_string(_v.y/DepthModify));
	ret.append(" ");
	return ret;
}

void GSubDepthFeature::fromString(string s)
{
	stringstream ss(s);
	string sub_s;
	getline(ss, sub_s, ' ');
	_u.x = LogCat::to_int(sub_s)*DepthModify;
	getline(ss, sub_s, ' ');
	_u.y = LogCat::to_int(sub_s)*DepthModify;
	getline(ss, sub_s, ' ');
	_v.x = LogCat::to_int(sub_s)*DepthModify;
	getline(ss, sub_s, ' ');
	_v.y = LogCat::to_int(sub_s)*DepthModify;
}

DepthFeature* GSubDepthFeature::clone()
{
	return new GSubDepthFeature(*this);
}

GAddDepthFeature::GAddDepthFeature(CvPoint u, CvPoint v)
{
	_u.x = u.x * DepthModify;
	_u.y = u.y * DepthModify;
	_v.x = v.x * DepthModify;
	_v.y = v.y * DepthModify;
}

GAddDepthFeature::~GAddDepthFeature()
{
	;
}

GAddDepthFeature::GAddDepthFeature(const GAddDepthFeature& f)
{
	_u = f._u;
	_v = f._v;
}

float GAddDepthFeature::getValue(IplImage* img, CvPoint pixel)
{
	double depth = CV_IMAGE_ELEM(img, unsigned int, pixel.y, pixel.x);
	if(depth == BACKGROUNG) return 0;
	double _depth = 1.0/depth;
	int x, y;
	double depth_u, depth_v;
	x = (int)(pixel.x+_u.x*_depth);
	y = (int)(pixel.y+_u.y*_depth);
	(x >= 0 && x < img->width && y >= 0 && y < img->height) ? 
		depth_u = CV_IMAGE_ELEM(img, unsigned int, y, x) : depth_u = BACKGROUNG;
	int x2 = (int)(pixel.x+_v.x*_depth);
	int y2 = (int)(pixel.y+_v.y*_depth);
	(x2 >= 0 && x2 < img->width && y2 >= 0 && y2 < img->height) ? 
		depth_v = CV_IMAGE_ELEM(img, unsigned int, y2, x2) : depth_v = BACKGROUNG;
	if(depth_u == BACKGROUNG || depth_v == BACKGROUNG)
		return VERY_LARGE_POSITIVE;
	else return (depth_u + depth_v - 2.0*depth);
}

string GAddDepthFeature::toString()
{
	string ret;
	ret.append(LogCat::to_string(_u.x/DepthModify));
	ret.append(" ");
	ret.append(LogCat::to_string(_u.y/DepthModify));
	ret.append(" ");
	ret.append(LogCat::to_string(_v.x/DepthModify));
	ret.append(" ");
	ret.append(LogCat::to_string(_v.y/DepthModify));
	ret.append(" ");
	return ret;
}

void GAddDepthFeature::fromString(string s)
{
	stringstream ss(s);
	string sub_s;
	getline(ss, sub_s, ' ');
	_u.x = LogCat::to_int(sub_s)*DepthModify;
	getline(ss, sub_s, ' ');
	_u.y = LogCat::to_int(sub_s)*DepthModify;
	getline(ss, sub_s, ' ');
	_v.x = LogCat::to_int(sub_s)*DepthModify;
	getline(ss, sub_s, ' ');
	_v.y = LogCat::to_int(sub_s)*DepthModify;
}

DepthFeature* GAddDepthFeature::clone()
{
	return new GAddDepthFeature(*this);
}

GHalfDepthFeature::GHalfDepthFeature(CvPoint u,
								   CvPoint v)
{
	_u.x = u.x * DepthModify;
	_u.y = u.y * DepthModify;
	_v.x = v.x * DepthModify;
	_v.y = v.y * DepthModify;
}

GHalfDepthFeature::~GHalfDepthFeature()
{
	;
}

GHalfDepthFeature::GHalfDepthFeature(const GHalfDepthFeature& f)
{
	_u = f._u;
	_v = f._v;
}

float GHalfDepthFeature::getValue(IplImage* img, CvPoint pixel)
{
	double depth = CV_IMAGE_ELEM(img, unsigned int, pixel.y, pixel.x);
	if(depth == BACKGROUNG) return 0;
	double _depth = 1.0/depth;
	int x, y;
	double depth_u, depth_v, depth_half;
	x = (int)(pixel.x+_u.x*_depth);
	y = (int)(pixel.y+_u.y*_depth);
	(x >= 0 && x < img->width && y >= 0 && y < img->height) ? 
		depth_u = CV_IMAGE_ELEM(img, unsigned int, y, x) : depth_u = BACKGROUNG;
	int x2 = (int)(pixel.x+_v.x*_depth);
	int y2 = (int)(pixel.y+_v.y*_depth);
	(x2 >= 0 && x2 < img->width && y2 >= 0 && y2 < img->height) ? 
		depth_v = CV_IMAGE_ELEM(img, unsigned int, y2, x2) : depth_v = BACKGROUNG;
	x = (x+x2)/2;
	y = (y+y2)/2;
	(x >= 0 && x < img->width && y >= 0 && y < img->height) ? 
		depth_half = CV_IMAGE_ELEM(img, unsigned int, y, x) : depth_u = BACKGROUNG;
	if(depth_u == BACKGROUNG || depth_v == BACKGROUNG || depth_half == BACKGROUNG)
		return VERY_LARGE_POSITIVE;
	else return (depth_u + depth_v - 2.0*depth_half);
}

string GHalfDepthFeature::toString()
{
	string ret;
	ret.append(LogCat::to_string(_u.x/DepthModify));
	ret.append(" ");
	ret.append(LogCat::to_string(_u.y/DepthModify));
	ret.append(" ");
	ret.append(LogCat::to_string(_v.x/DepthModify));
	ret.append(" ");
	ret.append(LogCat::to_string(_v.y/DepthModify));
	ret.append(" ");
	return ret;
}

void GHalfDepthFeature::fromString(string s)
{
	stringstream ss(s);
	string sub_s;
	getline(ss, sub_s, ' ');
	_u.x = LogCat::to_int(sub_s)*DepthModify;
	getline(ss, sub_s, ' ');
	_u.y = LogCat::to_int(sub_s)*DepthModify;
	getline(ss, sub_s, ' ');
	_v.x = LogCat::to_int(sub_s)*DepthModify;
	getline(ss, sub_s, ' ');
	_v.y = LogCat::to_int(sub_s)*DepthModify;
}

DepthFeature* GHalfDepthFeature::clone()
{
	return new GHalfDepthFeature(*this);
}