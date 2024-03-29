#include <microservice.h>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <thread>

typedef uint16_t BatchSizeType;

cv::cuda::GpuMat resizePadRightBottom(
    cv::cuda::GpuMat &input,
    size_t height,
    size_t width,
    const cv::Scalar &bgcolor = cv::Scalar(128, 128, 128),
    cv::cuda::Stream &stream = cv::cuda::Stream::Null()
);

cv::cuda::GpuMat normalize(
    cv::cuda::GpuMat &input,
    cv::cuda::Stream &stream = cv::cuda::Stream::Null(),
    const std::array<float, 3>& subVals = {0.f, 0.f, 0.f},
    const std::array<float, 3>& divVals = {1.f, 1.f, 1.f}
);

cv::cuda::GpuMat cvtHWCToCHW(
    cv::cuda::GpuMat &input,
    cv::cuda::Stream &stream = cv::cuda::Stream::Null()
);


class BasePreprocessor : public Microservice {
public:
    BasePreprocessor(const BaseMicroserviceConfigs &configs);
    ~BasePreprocessor() {

    };
protected:
    BatchSizeType msvc_onBufferBatchSize = 0;
    std::vector<cv::cuda::GpuMat> msvc_batchBuffer;
    bool isTimeToBatch() override;
    bool checkReqEligibility(ClockType currReq_genTime) override;
};


typedef uint16_t BatchSizeType;

class BaseProcessor : public Microservice {
public:
    BaseProcessor(const BaseMicroserviceConfigs &configs);
    ~BaseProcessor() {

    };
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
    int orig_h,
    int orig_w,
    int infer_h,
    int infer_w,
    int numDetections,
    const float *bbox_coorList,
    std::vector<cv::cuda::GpuMat> &croppedBBoxes
);

void cropOneBox(
    const cv::cuda::GpuMat &image,
    int infer_h,
    int infer_w,
    int numDetections,
    const float *bbox_coorList,
    cv::cuda::GpuMat &croppedBBoxes
);

class BasePostprocessor : public Microservice {
public:
    BasePostprocessor(const BaseMicroserviceConfigs &configs);
    ~BasePostprocessor() {

    };
};