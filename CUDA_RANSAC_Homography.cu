#include "CUDA_RANSAC_Homography.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <numeric>
#include <omp.h>
#include <assert.h>
#include "CUDA_SVD.h"

static const int NTHREADS = 256; // threads per block

#define SQ(x) (x)*(x)

static void CheckCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

__device__ int CalcHomography(const Point2Df src[4], const Point2Df dst[4], float ret_H[9])
{
    // This version does not normalised the input data, which is contrary to what Multiple View Geometry says.
    // I included it to see what happens when you don't do this step.

    GPU_Matrix X;
    GPU_Matrix V;
    GPU_Vector S;

    X.rows = 9;
    X.cols = 9;

    V.rows = 9;
    V.cols = 9;

    S.size = 3;

    for(int i=0; i < 4; i++) {
        float srcx = src[i].x;
        float srcy = src[i].y;
        float dstx = dst[i].x;
        float dsty = dst[i].y;

        int y1 = (i*2 + 0)*9;
        int y2 = (i*2 + 1)*9;

        // First row
        X.data[y1+0] = 0.f;
        X.data[y1+1] = 0.f;
        X.data[y1+2] = 0.f;

        X.data[y1+3] = -srcx;
        X.data[y1+4] = -srcy;
        X.data[y1+5] = -1.f;

        X.data[y1+6] = dsty*srcx;
        X.data[y1+7] = dsty*srcy;
        X.data[y1+8] = dsty;

        // Second row
        X.data[y2+0] = srcx;
        X.data[y2+1] = srcy;
        X.data[y2+2] = 1.f;

        X.data[y2+3] = 0.f;
        X.data[y2+4] = 0.f;
        X.data[y2+5] = 0.f;

        X.data[y2+6] = -dstx*srcx;
        X.data[y2+7] = -dstx*srcy;
        X.data[y2+8] = -dstx;
    }

    // Fill the last row - redundant but makes the matrix a nice 9x9 system
    float srcx = src[3].x;
    float srcy = src[3].y;
    float dstx = dst[3].x;
    float dsty = dst[3].y;

    int y = 8*9;
    X.data[y+0] = -dsty*srcx;
    X.data[y+1] = -dsty*srcy;
    X.data[y+2] = -dsty;

    X.data[y+3] = dstx*srcx;
    X.data[y+4] = dstx*srcy;
    X.data[y+5] = dstx;

    X.data[y+6] = 0.0f;
    X.data[y+7] = 0.0f;
    X.data[y+8] = 0.0f;

    bool ret = linalg_SV_decomp_jacobi(&X, &V, &S);

    ret_H[0] = V.data[0*9 + 8];
    ret_H[1] = V.data[1*9 + 8];
    ret_H[2] = V.data[2*9 + 8];
    ret_H[3] = V.data[3*9 + 8];
    ret_H[4] = V.data[4*9 + 8];
    ret_H[5] = V.data[5*9 + 8];
    ret_H[6] = V.data[6*9 + 8];
    ret_H[7] = V.data[7*9 + 8];
    ret_H[8] = V.data[8*9 + 8];

    return ret;
}

__device__ int CalcHomography2(const Point2Df src[4], const Point2Df dst[4], float ret_H[9])
{
    // This version normalises the data before processing as recommended in the book Multiple View Geometry
    GPU_Matrix X;
    GPU_Matrix V;
    GPU_Vector S;

    X.rows = 9;
    X.cols = 9;

    V.rows = 9;
    V.cols = 9;

    S.size = 3;

    // Normalise the data
    Point2Df src_mean, dst_mean;
    float src_var = 0.0f;
    float dst_var = 0.0f;

    src_mean.x = (src[0].x + src[1].x + src[2].x + src[3].x)*0.25f;
    src_mean.y = (src[0].y + src[1].y + src[2].y + src[3].y)*0.25f;

    dst_mean.x = (dst[0].x + dst[1].x + dst[2].x + dst[3].x)*0.25f;
    dst_mean.y = (dst[0].y + dst[1].y + dst[2].y + dst[3].y)*0.25f;

    for(int i=0; i < 4; i++) {
        src_var += SQ(src[i].x - src_mean.x) + SQ(src[i].y - src_mean.y);
        dst_var += SQ(dst[i].x - dst_mean.x) + SQ(dst[i].y - dst_mean.y);
    }

    src_var *= 0.25f;
    dst_var *= 0.25f;

    float src_scale = sqrt(2.0f) / sqrt(src_var);
    float dst_scale = sqrt(2.0f) / sqrt(dst_var);


    /*
    float src_avg_dist = 0.0f;
    float dst_avg_dist = 0.0f;

    for(int i=0; i < 4; i++) {
        src_avg_dist += sqrt(SQ(src[i].x - src_mean.x) + SQ(src[i].y - src_mean.y));
        dst_avg_dist += sqrt(SQ(dst[i].x - dst_mean.x) + SQ(dst[i].y - dst_mean.y));
    }

    src_avg_dist *= 0.25f;
    dst_avg_dist *= 0.25f;

    float src_scale = sqrt(2.0f) / src_avg_dist;
    float dst_scale = sqrt(2.0f) / dst_avg_dist;
    */

    for(int i=0; i < 4; i++) {
        float srcx = (src[i].x - src_mean.x)*src_scale;
        float srcy = (src[i].y - src_mean.y)*src_scale;

        float dstx = (dst[i].x - dst_mean.x)*dst_scale;
        float dsty = (dst[i].y - dst_mean.y)*dst_scale;

        int y1 = (i*2 + 0)*9;
        int y2 = (i*2 + 1)*9;

         // First row
        X.data[y1 + 0] = 0.0f;
        X.data[y1 + 1] = 0.0f;
        X.data[y1 + 2] = 0.0f;

        X.data[y1 + 3] = -srcx;
        X.data[y1 + 4] = -srcy;
        X.data[y1 + 5] = -1.0f;

        X.data[y1 + 6] = dsty*srcx;
        X.data[y1 + 7] = dsty*srcy;
        X.data[y1 + 8] = dsty;

        // Second row
        X.data[y2 + 0] = srcx;
        X.data[y2 + 1] = srcy;
        X.data[y2 + 2] = 1.0f;

        X.data[y2 + 3] = 0.0f;
        X.data[y2 + 4] = 0.0f;
        X.data[y2 + 5] = 0.0f;

        X.data[y2 + 6] = -dstx*srcx;
        X.data[y2 + 7] = -dstx*srcy;
        X.data[y2 + 8] = -dstx;
    }

    // Fill the last row
    float srcx = (src[3].x - src_mean.x)*src_scale;
    float srcy = (src[3].y - src_mean.y)*src_scale;
    float dstx = (dst[3].x - dst_mean.x)*dst_scale;
    float dsty = (dst[3].y - dst_mean.y)*dst_scale;

    X.data[8*9 + 0] = -dsty*srcx;
    X.data[8*9 + 1] = -dsty*srcy;
    X.data[8*9 + 2] = -dsty;

    X.data[8*9 + 3] = dstx*srcx;
    X.data[8*9 + 4] = dstx*srcy;
    X.data[8*9 + 5] = dstx;

    X.data[8*9 + 6] = 0.0f;
    X.data[8*9 + 7] = 0.0f;
    X.data[8*9 + 8] = 0.0f;

    bool ret = linalg_SV_decomp_jacobi(&X, &V, &S);

    float H[9];

    H[0] = V.data[0*9 + 8];
    H[1] = V.data[1*9 + 8];
    H[2] = V.data[2*9 + 8];
    H[3] = V.data[3*9 + 8];
    H[4] = V.data[4*9 + 8];
    H[5] = V.data[5*9 + 8];
    H[6] = V.data[6*9 + 8];
    H[7] = V.data[7*9 + 8];
    H[8] = V.data[8*9 + 8];

    // Undo the transformation using inv(dst_transform) * H * src_transform
    // Matrix operation expanded out using wxMaxima
    float s1 = src_scale;
    float s2 = dst_scale;

    float tx1 = src_mean.x;
    float ty1 = src_mean.y;

    float tx2 = dst_mean.x;
    float ty2 = dst_mean.y;

    ret_H[0] = s1*tx2*H[6] + s1*H[0]/s2;
    ret_H[1] = s1*tx2*H[7] + s1*H[1]/s2;
    ret_H[2] = tx2*(H[8] - s1*ty1*H[7] - s1*tx1*H[6]) + (H[2] - s1*ty1*H[1] - s1*tx1*H[0])/s2;

    ret_H[3] = s1*ty2*H[6] + s1*H[3]/s2;
    ret_H[4] = s1*ty2*H[7] + s1*H[4]/s2;
    ret_H[5] = ty2*(H[8] - s1*ty1*H[7] - s1*tx1*H[6]) + (H[5] - s1*ty1*H[4] - s1*tx1*H[3])/s2;

    ret_H[6] = s1*H[6];
    ret_H[7] = s1*H[7];
    ret_H[8] = H[8] - s1*ty1*H[7] - s1*tx1*H[6];

    return ret;
}

__device__ int EvalHomography(const Point2Df *src, const Point2Df *dst, int pts_num, const float H[9], float inlier_threshold)
{
    int inliers = 0;

    for(int i=0; i < pts_num; i++) {
        float x = H[0]*src[i].x + H[1]*src[i].y + H[2];
        float y = H[3]*src[i].x + H[4]*src[i].y + H[5];
        float z = H[6]*src[i].x + H[7]*src[i].y + H[8];

        x /= z;
        y /= z;

        float dist_sq = (dst[i].x - x)*(dst[i].x- x) + (dst[i].y - y)*(dst[i].y - y);

        if(dist_sq < inlier_threshold) {
            inliers++;
        }
    }

    return inliers;
}

__global__ void RANSAC_Homography(const Point2Df *src, const Point2Df *dst,int pts_num, const int *rand_list, float inlier_threshold, int iterations, int *ret_inliers, float *ret_homography)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx >= iterations) {
        return;
    }

    ret_inliers[idx] = 0;

    int rand_idx[4];
    Point2Df _src[4];
    Point2Df _dst[4];
    float *H = &ret_homography[idx*9];

    rand_idx[0] = rand_list[idx*4];
    rand_idx[1] = rand_list[idx*4+1];
    rand_idx[2] = rand_list[idx*4+2];
    rand_idx[3] = rand_list[idx*4+3];

    // Check for duplicates
    if(rand_idx[0] == rand_idx[1]) return;
    if(rand_idx[0] == rand_idx[2]) return;
    if(rand_idx[0] == rand_idx[3]) return;
    if(rand_idx[1] == rand_idx[2]) return;
    if(rand_idx[1] == rand_idx[3]) return;
    if(rand_idx[2] == rand_idx[3]) return;

    for(int i=0; i < 4; i++) {
        _src[i].x = src[rand_idx[i]].x;
        _src[i].y = src[rand_idx[i]].y;
        _dst[i].x = dst[rand_idx[i]].x;
        _dst[i].y = dst[rand_idx[i]].y;
    }

#ifdef NORMALISE_INPUT_POINTS
    int ret = CalcHomography2(_src, _dst, H);
#else
    int ret = CalcHomography(_src, _dst, H);
#endif

    ret_inliers[idx] = EvalHomography(src, dst, pts_num, H, inlier_threshold);
}

void CUDA_RANSAC_Homography(const std::vector<Point2Df> &src, const std::vector<Point2Df> &dst, /*const std::vector <float> &match_score,*/
                            float inlier_threshold, int iterations,
                            /*int *best_inliers, */float *best_H, std::vector <char> *inlier_mask)
{
    int best_inliers_i;
    int *best_inliers = &best_inliers_i;
    assert(src.size() == dst.size());

#ifdef BIAS_RANDOM_SELECTION
    assert(match_score.size() == dst.size());
#endif

    int RANSAC_threshold = inlier_threshold*inlier_threshold;
    int threads = NTHREADS;
    int blocks = iterations/threads + ((iterations % threads)?1:0);

    Point2Df *gpu_src;
    Point2Df *gpu_dst;
    int *gpu_rand_list;
    int *gpu_ret_inliers;
    float *gpu_ret_H;
    std::vector <int> rand_list(iterations*4);
    std::vector <int> ret_inliers(iterations);
    std::vector <float> ret_H(iterations*9);

    cudaMalloc((void**)&gpu_src, sizeof(Point2Df)*src.size());
    cudaMalloc((void**)&gpu_dst, sizeof(Point2Df)*dst.size());
    cudaMalloc((void**)&gpu_rand_list, sizeof(int)*iterations*4);
    cudaMalloc((void**)&gpu_ret_inliers, sizeof(int)*iterations);
    cudaMalloc((void**)&gpu_ret_H, sizeof(float)*iterations*9);
    CheckCUDAError("cudaMalloc");

    cudaMemcpy(gpu_src, &src[0], sizeof(Point2Df)*src.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_dst, &dst[0], sizeof(Point2Df)*dst.size(), cudaMemcpyHostToDevice);

    // Generate random numbers on host
    // Using a bias version when randomly selecting points
    // Point with better matching score have a highr chance of getting picked
    {
#ifdef BIAS_RANDOM_SELECTION
        std::vector <float> cummulative = match_score;
        double sum = accumulate(match_score.begin(), match_score.end(), 0.0);

        // Normalise the scores
        for(unsigned int i=0; i < cummulative.size(); i++) {
            cummulative[i] /= sum;
        }

        // Calc the cummulative distribution
        for(unsigned int i=1; i < cummulative.size(); i++) {
            cummulative[i] += cummulative[i-1];
        }

        for(unsigned int i=0; i < rand_list.size(); i++) {
            float x = rand()/(1.0 + RAND_MAX); // random between [0,1)

            // Binary search to find which index x lands on
            int min = 0;
            int max = src.size();
            int index = 0;

            while(true) {
                int mid = (min + max) / 2;

                if(min == max - 1) {
                    if(x < cummulative[min]) {
                        index = min;
                    }
                    else {
                        index = max;
                    }
                    break;
                }

                if(x > cummulative[mid]) {
                    min = mid;
                }
                else {
                    max = mid;
                }
            }

            rand_list[i] = index;
        }
#else
        for(unsigned int i=0; i < rand_list.size(); i++) {
            rand_list[i] = src.size() * (rand()/(1.0 + RAND_MAX));
        }
#endif
        cudaMemcpy(gpu_rand_list, &rand_list[0], sizeof(int)*rand_list.size(), cudaMemcpyHostToDevice);
        CheckCUDAError("cudaMemcpy");
    }

    RANSAC_Homography<<<blocks, threads>>>(gpu_src, gpu_dst, src.size(), gpu_rand_list, RANSAC_threshold, iterations, gpu_ret_inliers, gpu_ret_H);
    cudaThreadSynchronize();
    CheckCUDAError("RANSAC_Homography");

    cudaMemcpy(&ret_inliers[0], gpu_ret_inliers, sizeof(int)*ret_inliers.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ret_H[0], gpu_ret_H, sizeof(float)*ret_H.size(), cudaMemcpyDeviceToHost);

    *best_inliers = 0;
    int best_idx = 0;

    for(int i=0; i < ret_inliers.size(); i++) {
        /*
        printf("ret %d: %d\n", i, ret_inliers[i]);

        for(int j=0; j< 9; j++) {
            printf("%.3f ",  ret_H[i*9+j]);
        }
        printf("\n");
        */
        if(ret_inliers[i] > *best_inliers) {
            *best_inliers = ret_inliers[i];
            best_idx = i;
        }
    }

    memcpy(best_H, &ret_H[best_idx*9], sizeof(float)*9);

    // Fill the mask
    std::vector <char> &_inlier_mask = *inlier_mask;
    _inlier_mask.resize(src.size(), 0);

    for(int i=0; i < src.size(); i++) {
        float x = best_H[0]*src[i].x + best_H[1]*src[i].y + best_H[2];
        float y = best_H[3]*src[i].x + best_H[4]*src[i].y + best_H[5];
        float z = best_H[6]*src[i].x + best_H[7]*src[i].y + best_H[8];

        x /= z;
        y /= z;

        float dist_sq = (dst[i].x - x)*(dst[i].x- x) + (dst[i].y - y)*(dst[i].y - y);

        if(dist_sq < RANSAC_threshold) {
            _inlier_mask[i] = 1;
        }
    }

    *best_inliers = accumulate(_inlier_mask.begin(), _inlier_mask.end(), 0);

    //printf("CUDA blocks/threads: %d %d\n", blocks, threads);

    cudaFree(gpu_src);
    cudaFree(gpu_dst);
    cudaFree(gpu_rand_list);
    cudaFree(gpu_ret_inliers);
    cudaFree(gpu_ret_H);
}
