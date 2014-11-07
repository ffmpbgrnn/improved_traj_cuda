#ifndef UNITILS_H_
#define UNITILS_H_
#include "DenseTrackStab.h"
void download(const gpu::GpuMat& d_mat, vector<Point2f>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
    d_mat.download(mat);
}

#endif