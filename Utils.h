#include "DenseTrackStab.h"
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

void download(const gpu::GpuMat& d_mat, vector<Point2f>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
    d_mat.download(mat);
}
