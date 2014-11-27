#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/nonfree/gpu.hpp"
using namespace cv;
using namespace gpu;

__host__ __device__ __forceinline__ int divUp(int total, int grain)
{
    return (total + grain - 1) / grain;
}

__global__ void DCUDA_windowedMatchingMask(const PtrStepSzf& src, const PtrStepSzf& dst, GpuMat& mask)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const float maxDeltaX = 25.0, maxDeltaY = 25.0;
    float x_diff = dst(x, 0) - src(y, 0);
    float y_diff = dst(x, 1) - src(y, 1);
    *(mask.datastart + y * mask.step + x * sizeof(uchar)) = fabsf(x_diff) < maxDeltaX && fabsf(y_diff) < maxDeltaY;
}

__global__ void CUDA_MatchFromFlow(const PtrStepSzf& d_prev_pts, const PtrStepSzf& )
{
    int i = threadIdx.x;
    int x = std::min<int>(std::max<int>(cvRound(d_prev_pts(0, i).x), 0), width-1);
    int y = std::min<int>(std::max<int>(cvRound(d_prev_pts(1, [i].y), 0), height-1);
    *(d_pts.datastart + i * sizeof(CV_32FC1)) = x + d_flow_x
    *(d_pts.datastart + d_pts.step + i * sizeof(CV_32FC1)) = 
    // const float* f = flow.ptr<float>(y);
    pts.push_back(Point2f(x+flow_x.ptr<float>(y)[x], y+flow_y.ptr<float>(y)[x]));
}

GpuMat CUDA_windowedMatchingMask( const GpuMat& keypoints1, const GpuMat& keypoints2)
{
    if( keypoints1.cols == 0 || keypoints2.cols == 0 )
        return GpuMat();

    int rows = (int)keypoints1.cols, cols = (int)keypoints2.cols;
    GpuMat mask( rows, cols, CV_8UC1 );

    const dim3 block(32, 8);
    const dim3 grid(divUp(cols, block.x), divUp(rows, block.y));
 
    // PtrStepSzb src(keypoints1.cols, keypoints1.rows, keypoints1.datastart, keypoints1.step);
    // PtrStepSzb dst(keypoints2.cols, keypoints2.rows, keypoints2.datastart, keypoints2.step);
    DCUDA_windowedMatchingMask<<<grid, block>>>(keypoints1, keypoints2, mask);

    return mask;
}


// int main()
// {
//     CUDA_windowedMatchingMask();
// }