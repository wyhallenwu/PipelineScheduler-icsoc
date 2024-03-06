#include <microservice.h>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>

typedef uint16_t BatchSizeType;

cv::cuda::GpuMat resizePadRightBottom(
    const cv::cuda::GpuMat &input,
    size_t height,
    size_t width,
    const cv::Scalar &bgcolor = cv::Scalar(128, 128, 128),
    bool toNormalize = true
);

void normalize(
    cv::cuda::GpuMat &input,
    const std::array<float, 3>& subVals,
    const std::array<float, 3>& divVals
);


class BasePreprocessor : public Microservice {
public:
    BasePreprocessor(const BaseMicroserviceConfigs &configs);
    ~BasePreprocessor();
protected:
    BatchSizeType msvc_onBufferBatchSize;
    std::vector<cv::cuda::GpuMat> msvc_batchBuffer;
    bool isTimeToBatch() override;
    bool checkReqEligibility(ClockType currReq_genTime) override;
};


typedef uint16_t BatchSizeType;

class BaseProcessor : public Microservice {
public:
    BaseProcessor(const BaseMicroserviceConfigs &configs);
    ~BaseProcessor();
protected:
    BatchSizeType msvc_onBufferBatchSize;
    bool checkReqEligibility(ClockType currReq_genTime) override;
};

/**
 * @brief crop from input image all bounding boxes whose coordinates are provided by `bbox_coorList`
 * 
 * @param image 
 * @param infer_h the height of the image during inference, used to scale to bounding boxes to their original coordinates for cropping
 * @param infer_w the height of the image during inference, used to scale to bounding boxes to their original coordinates for cropping
 * @param bbox_coorList pointer to a 2d `float` array of bounding box coordinates of size (TopK, 4). The box format is 
 *                      [x1, y1, x2, y2] (e.g., [0, 266, 260, 447])
 * @return cv::cuda::GpuMat
 */
void crop(
    const cv::cuda::GpuMat &image,
    uint16_t infer_h,
    uint16_t infer_w,
    RequestShapeType numDetections,
    const float *bbox_coorList,
    std::vector<cv::cuda::GpuMat> &croppedBBoxes
);

class BasePostprocessor : public Microservice {
public:
    BasePostprocessor(const BaseMicroserviceConfigs &configs);
    ~BasePostprocessor();
};