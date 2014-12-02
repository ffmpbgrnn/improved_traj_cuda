#ifndef DENSETRACKSTAB_H_
#define DENSETRACKSTAB_H_

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <ctype.h>
#include <unistd.h>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <list>
#include <string>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/nonfree/gpu.hpp"
#include "DataTypes.h"

using namespace cv;
extern int start_frame;
extern int end_frame;
extern int scale_num;
extern const float scale_stride;
extern char* bb_file;

// parameters for descriptors
extern int patch_size;
extern int nxy_cell;
extern int nt_cell;
extern float epsilon;
extern const float min_flow;

// parameters for tracking
extern double quality;
extern int min_distance;
extern int init_gap;
extern int track_length;

// parameters for rejecting trajectory
extern const float min_var;
extern const float max_var;
extern const float max_dis;


typedef struct {
	int x;       // top left corner
	int y;
	int width;
	int height;
}RectInfo;

typedef struct {
    int width;   // resolution of the video
    int height;
    int length;  // number of frames
}SeqInfo;

typedef struct {
    int length;  // length of the trajectory
    int gap;     // initialization gap for feature re-sampling
}TrackInfo;

typedef struct {
    int nBins;   // number of bins for vector quantization
    bool isHof;
    int nxCells; // number of cells in x direction
    int nyCells;
    int ntCells;
    int dim;     // dimension of the descriptor
    int height;  // size of the block for computing the descriptor
    int width;
}DescInfo;

// integral histogram for the descriptors
typedef struct {
    int height;
    int width;
    int nBins;
    float* desc;
}DescMat;

class Track
{
public:
    std::vector<Point2f> point;
    std::vector<Point2f> disp;
    std::vector<float> hog;
    std::vector<float> hof;
    std::vector<float> mbhX;
    std::vector<float> mbhY;
    int index;

    Track(const Point2f& point_, const TrackInfo& trackInfo, const DescInfo& hogInfo,
          const DescInfo& hofInfo, const DescInfo& mbhInfo)
        : point(trackInfo.length+1), disp(trackInfo.length), hog(hogInfo.dim*trackInfo.length),
          hof(hofInfo.dim*trackInfo.length), mbhX(mbhInfo.dim*trackInfo.length), mbhY(mbhInfo.dim*trackInfo.length)
    {
        index = 0;
        point[0] = point_;
    }

    void addPoint(const Point2f& point_)
    {
        index++;
        point[index] = point_;
    }
};

class BoundBox
{
public:
	Point2f TopLeft;
	Point2f BottomRight;
	float confidence;

	BoundBox(float a1, float a2, float a3, float a4, float a5)
	{
		TopLeft.x = a1;
		TopLeft.y = a2;
		BottomRight.x = a3;
		BottomRight.y = a4;
		confidence = a5;
	}
};

class Frame
{
public:
	int frameID;
	std::vector<BoundBox> BBs;

	Frame(const int& frame_)
	{
		frameID = frame_;
		BBs.clear();
	}
};

#endif /*DENSETRACKSTAB_H_*/
