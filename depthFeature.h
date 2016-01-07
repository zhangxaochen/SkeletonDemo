#ifndef __depthFeature_h_ 
#define __depthFeature_h_

#include "opencv2/opencv.hpp"
#include "logCat.h"

#define VERY_LARGE_POSITIVE 0xffffff
#define BACKGROUNG 0xffffffu

#define DepthModify 3000

//***************************
//** DepthFeature **
//** Interface of various kind of depth features
//***************************
class DepthFeature
{
public:
	DepthFeature(){}
	virtual ~DepthFeature(){}

	// get feature value of an pixel
	// must be implemented by every depth feature
	virtual float getValue(IplImage* img, CvPoint pixel) = 0;

	virtual string toString() = 0;

	virtual void fromString(string s)= 0;

	virtual DepthFeature* clone() = 0;

	CvPoint _u;
	CvPoint _v;
};

//***************************
//** GSubDepthFeature **
//** Simple depth comparison feature
//** At pixel x, feature sub depth difference of two pixels offset from x
//** f(I, x) = d(x + u/d(x)) - d(x + v/d(x))
//** 1/d(x) to normalize offsets, ensures feature is depth invariant
//***************************
class GSubDepthFeature : public DepthFeature
{
public:
	GSubDepthFeature(){}
	GSubDepthFeature(CvPoint u, CvPoint v);
	GSubDepthFeature(const GSubDepthFeature& f);
	~GSubDepthFeature();

	float getValue(IplImage* img, CvPoint pixel);

	string toString();

	void fromString(string s);

	DepthFeature* clone();
};

//***************************
//** GAddDepthFeature **
//** Simple depth gradient comparison feature
//** At pixel x, feature compute the depth difference of the mean of two pixels' depth and
//** their middle point
//***************************
class GAddDepthFeature : public DepthFeature
{
public:
	GAddDepthFeature(){}
	GAddDepthFeature(CvPoint u, CvPoint v);
	GAddDepthFeature(const GAddDepthFeature& f);
	~GAddDepthFeature();

	float getValue(IplImage* img, CvPoint pixel);

	string toString();

	void fromString(string s);

	DepthFeature* clone();
};

//***************************
//** GCompDepthFeature **
//** Simple depth gradient comparison feature
//** At pixel x, feature compare the depth gradient of two pixels offset from x
//***************************
class GHalfDepthFeature : public DepthFeature
{
public:
	GHalfDepthFeature(){}
	GHalfDepthFeature(CvPoint u, CvPoint v);
	GHalfDepthFeature(const GHalfDepthFeature& f);
	~GHalfDepthFeature();

	float getValue(IplImage* img, CvPoint pixel);

	string toString();

	void fromString(string s);

	DepthFeature* clone();
};

#endif