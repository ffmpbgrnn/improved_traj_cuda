#include "opencv2/core/gpumat.hpp"
using namespace cv;
using namespace gpu;


namespace cv { namespace gpu {
class SURF_GPU
{
public:
    enum KeypointLayout
    {
        X_ROW = 0,
        Y_ROW,
        LAPLACIAN_ROW,
        OCTAVE_ROW,
        SIZE_ROW,
        ANGLE_ROW,
        HESSIAN_ROW,
        ROWS_COUNT
    };

    //! the default constructor
    SURF_GPU();
    //! the full constructor taking all the necessary parameters
    explicit SURF_GPU(double _hessianThreshold, int _nOctaves=4,
         int _nOctaveLayers=2, bool _extended=false, float _keypointsRatio=0.01f, bool _upright = false);

    //! returns the descriptor size in float's (64 or 128)
    int descriptorSize() const;

    //! upload host keypoints to device memory
    void uploadKeypoints(const std::vector<KeyPoint>& keypoints, GpuMat& keypointsGPU);
    //! download keypoints from device to host memory
    void downloadKeypoints(const GpuMat& keypointsGPU, std::vector<KeyPoint>& keypoints);

    //! download descriptors from device to host memory
    void downloadDescriptors(const GpuMat& descriptorsGPU, std::vector<float>& descriptors);

    //! finds the keypoints using fast hessian detector used in SURF
    //! supports CV_8UC1 images
    //! keypoints will have nFeature cols and 6 rows
    //! keypoints.ptr<float>(X_ROW)[i] will contain x coordinate of i'th feature
    //! keypoints.ptr<float>(Y_ROW)[i] will contain y coordinate of i'th feature
    //! keypoints.ptr<float>(LAPLACIAN_ROW)[i] will contain laplacian sign of i'th feature
    //! keypoints.ptr<float>(OCTAVE_ROW)[i] will contain octave of i'th feature
    //! keypoints.ptr<float>(SIZE_ROW)[i] will contain size of i'th feature
    //! keypoints.ptr<float>(ANGLE_ROW)[i] will contain orientation of i'th feature
    //! keypoints.ptr<float>(HESSIAN_ROW)[i] will contain response of i'th feature
    void operator()(const GpuMat& img, const GpuMat& mask, GpuMat& keypoints);
    //! finds the keypoints and computes their descriptors.
    //! Optionally it can compute descriptors for the user-provided keypoints and recompute keypoints direction
    void operator()(const GpuMat& img, const GpuMat& mask, GpuMat& keypoints, GpuMat& descriptors,
        bool useProvidedKeypoints = false);

    void operator()(const GpuMat& img, const GpuMat& mask, std::vector<KeyPoint>& keypoints);
    void operator()(const GpuMat& img, const GpuMat& mask, std::vector<KeyPoint>& keypoints, GpuMat& descriptors,
        bool useProvidedKeypoints = false);

    void operator()(const GpuMat& img, const GpuMat& mask, std::vector<KeyPoint>& keypoints, std::vector<float>& descriptors,
        bool useProvidedKeypoints = false);

    void releaseMemory();

    // SURF parameters
    double hessianThreshold;
    int nOctaves;
    int nOctaveLayers;
    bool extended;
    bool upright;

    //! max keypoints = min(keypointsRatio * img.size().area(), 65535)
    float keypointsRatio;

    GpuMat sum, mask1, maskSum, intBuffer;

    GpuMat det, trace;

    GpuMat maxPosBuffer;
};
} } // // namespace gpu } namespace cv }