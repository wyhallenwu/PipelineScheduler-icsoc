#include <microservice.h>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <thread>
#include <misc.h>
#include <trtengine.h>
#include <random>

typedef uint16_t BatchSizeType;
using namespace msvcconfigs;
using json = nlohmann::json;

cv::cuda::GpuMat resizePadRightBottom(
    cv::cuda::GpuMat &input,
    size_t height,
    size_t width,
    const cv::Scalar &bgcolor = cv::Scalar(128, 128, 128),
    cv::cuda::Stream &stream = cv::cuda::Stream::Null(),
    uint8_t IMG_TYPE = 16, //CV_8UC3
    uint8_t COLOR_CVT_TYPE = 4, //CV_BGR2RGB
    uint8_t RESIZE_INTERPOL_TYPE = 3 //INTER_AREA
);

cv::cuda::GpuMat normalize(
    cv::cuda::GpuMat &input,
    cv::cuda::Stream &stream = cv::cuda::Stream::Null(),
    const std::array<float, 3>& subVals = {0.f, 0.f, 0.f},
    const std::array<float, 3>& divVals = {1.f, 1.f, 1.f},
    const float normalized_scale = 1.f / 255.f
);

cv::cuda::GpuMat cvtHWCToCHW(
    cv::cuda::GpuMat &input,
    cv::cuda::Stream &stream = cv::cuda::Stream::Null(),
    uint8_t IMG_TYPE = 16 //CV_8UC3
);


class BaseReqBatcher : public Microservice {
public:
    BaseReqBatcher(const BaseMicroserviceConfigs &configs);
    ~BaseReqBatcher() = default;

    virtual void batchRequests();
protected:
    /**
     * @brief 
     * 
     */
    struct BaseReqBatcherConfigs {
        uint8_t msvc_imgType = 16; //CV_8UC3
        uint8_t msvc_colorCvtType = 4; //CV_BGR2RGB
        uint8_t msvc_resizeInterpolType = 3; //INTER_AREA
        float msvc_imgNormScale = 1.f / 255.f;
        std::array<float, 3> msvc_subVals = {0.f, 0.f, 0.f};
        std::array<float, 3> msvc_divVals = {1.f, 1.f, 1.f};
    };

    void readConfigsFromJson(std::string cfgPath);

    BatchSizeType msvc_onBufferBatchSize = 0;
    std::vector<cv::cuda::GpuMat> msvc_batchBuffer;
    bool isTimeToBatch() override;
    bool checkReqEligibility(ClockType currReq_genTime) override;

    uint8_t msvc_imgType, msvc_colorCvtType, msvc_resizeInterpolType;
    float msvc_imgNormScale;
    std::array<float, 3> msvc_subVals, msvc_divVals;

};


typedef uint16_t BatchSizeType;

class BaseBatchInferencer : public Microservice {
public:
    BaseBatchInferencer(const BaseMicroserviceConfigs &configs);
    ~BaseBatchInferencer() = default;
    virtual void inference();

    RequestShapeType getInputShapeVector();
    RequestShapeType getOutputShapeVector();
protected:
    BatchSizeType msvc_onBufferBatchSize;
    std::vector<void *> msvc_engineInputBuffers, msvc_engineOutputBuffers;
    TRTConfigs msvc_engineConfigs;
    Engine* msvc_inferenceEngine;

    TRTConfigs readConfigsFromJson(const std::string cfgPath);

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

class BaseBBoxCropper : public Microservice {
public:
    BaseBBoxCropper(const BaseMicroserviceConfigs &configs);
    ~BaseBBoxCropper() = default;

    void cropping();
    void setInferenceShape(RequestShapeType shape) {
        msvc_inferenceShape = shape;
    }

    void generateRandomBBox(
        float *bboxList,
        const uint16_t height,
        const uint16_t width,
        const uint16_t numBboxes,
        const uint16_t seed = 2024
    );

    void cropProfiling();

protected:
    RequestShapeType msvc_inferenceShape;
};