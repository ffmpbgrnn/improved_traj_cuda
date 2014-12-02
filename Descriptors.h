#ifndef DESCRIPTORS_H_
#define DESCRIPTORS_H_

#include "DenseTrackStab.h"
#include "Utils.h"

using namespace cv;
using namespace gpu;
// get the rectangle for computing the descriptor
void GetRect(const Point2f& point, RectInfo& rect, const int width, const int height, const DescInfo& descInfo);
void BuildDescMat(const Mat& xComp, const Mat& yComp, float* desc, const DescInfo& descInfo);
// get a descriptor from the integral histogram
void GetDesc(const DescMat* descMat, RectInfo& rect, DescInfo descInfo, std::vector<float>& desc, const int index);
// for HOG descriptor
void HogComp(const Mat& img, float* desc, DescInfo& descInfo);
// for HOF descriptor
void HofComp(const Mat& flow, float* desc, DescInfo& descInfo);
// for MBH descriptor
void MbhComp(const Mat& flow, float* descX, float* descY, DescInfo& descInfo);
// check whether a trajectory is valid or not
bool IsValid(std::vector<Point2f>& track, float& mean_x, float& mean_y, float& var_x, float& var_y, float& length);
bool IsCameraMotion(std::vector<Point2f>& disp);
// detect new feature points in an image without overlapping to previous points
void DenseSample(const GpuMat& d_grey, std::vector<Point2f>& points, const double quality, const int min_distance);

void InitPry(const Mat& frame, std::vector<float>& scales, std::vector<Size>& sizes);
void BuildPry(const std::vector<Size>& sizes, const int type, std::vector<GpuMat>& pyr);
void DrawTrack(const std::vector<Point2f>& point, const int index, const float scale, Mat& image);
void PrintDesc(std::vector<float>& desc, DescInfo& descInfo, TrackInfo& trackInfo);
void LoadBoundBox(char* file, std::vector<Frame>& bb_list);
void InitMaskWithBox(Mat& mask, std::vector<BoundBox>& bbs);
static void MyWarpPerspective(Mat& prev_src, Mat& src, Mat& dst, Mat& M0, int flags = INTER_LINEAR,
	            			 int borderType = BORDER_CONSTANT, const Scalar& borderValue = Scalar());
void ComputeMatch(const std::vector<KeyPoint>& prev_kpts, const std::vector<KeyPoint>& kpts,
				  const GpuMat& prev_desc, const GpuMat& desc, std::vector<Point2f>& prev_pts, std::vector<Point2f>& pts, Stream& stream);
void MergeMatch(const std::vector<Point2f>& prev_pts1, const std::vector<Point2f>& pts1,
				const std::vector<Point2f>& prev_pts2, const std::vector<Point2f>& pts2,
				std::vector<Point2f>& prev_pts_all, std::vector<Point2f>& pts_all, Stream& stream);
void MatchFromFlow(const GpuMat& d_prev_grey, const GpuMat& d_flow_x,
	               const GpuMat& d_flow_y, std::vector<Point2f>& v_prev_pts,
	               std::vector<Point2f>& pts, const GpuMat& d_mask, Stream& stream);
#endif /*DESCRIPTORS_H_*/
