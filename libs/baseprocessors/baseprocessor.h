#ifndef BASEPROCESSOR_H
#define BASEPROCESSOR_H

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


inline uint64_t getNumberAtIndex(const std::string& str, int index);
inline std::string getTimeDifString(const ClockType &start, const ClockType &end) {
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    return std::to_string(duration.count());
}

inline cv::Scalar vectorToScalar(const std::vector<float>& vec);

inline cv::cuda::GpuMat resizePadRightBottom(
    cv::cuda::GpuMat &input,
    size_t height,
    size_t width,
    const std::vector<float>& bgcolor = {128, 128, 128},
    cv::cuda::Stream &stream = cv::cuda::Stream::Null(),
    uint8_t IMG_TYPE = 16, //CV_8UC3
    uint8_t COLOR_CVT_TYPE = 4, //CV_BGR2RGB
    uint8_t RESIZE_INTERPOL_TYPE = 3 //INTER_AREA
);

inline cv::cuda::GpuMat normalize(
    cv::cuda::GpuMat &input,
    cv::cuda::Stream &stream = cv::cuda::Stream::Null(),
    const std::vector<float>& subVals = {0.f, 0.f, 0.f},
    const std::vector<float>& divVals = {1.f, 1.f, 1.f},
    const float normalized_scale = 1.f / 255.f
);

inline cv::cuda::GpuMat cvtHWCToCHW(
    cv::cuda::GpuMat &input,
    cv::cuda::Stream &stream = cv::cuda::Stream::Null(),
    uint8_t IMG_TYPE = 16 //CV_8UC3
);

/**
 * @brief 
 * 
 */
struct BaseReqBatcherConfigs : BaseMicroserviceConfigs{
    uint8_t msvc_imgType = 16; //CV_8UC3
    uint8_t msvc_colorCvtType = 4; //CV_BGR2RGB
    uint8_t msvc_resizeInterpolType = 3; //INTER_AREA
    float msvc_imgNormScale = 1.f / 255.f;
    std::vector<float> msvc_subVals = {0.f, 0.f, 0.f};
    std::vector<float> msvc_divVals = {1.f, 1.f, 1.f};
};

/**
 * @brief 
 * 
 */
struct BaseBatchInferencerConfigs : BaseMicroserviceConfigs {
    TRTConfigs msvc_engineConfigs;
    Engine* msvc_inferenceEngine;
};

struct BaseBBoxCropperConfigs : BaseMicroserviceConfigs {
    RequestShapeType msvc_inferenceShape;
};

struct BaseBBoxCropperVerifierConfigs : BaseBBoxCropperConfigs {
};


class BaseReqBatcher : public Microservice {
public:
    BaseReqBatcher(const json &jsonConfigs);
    ~BaseReqBatcher() = default;

    virtual void batchRequests();
    virtual void batchRequestsProfiling();

    void dispatchThread() override {
        if (msvc_RUNMODE == RUNMODE::PROFILING) {
            spdlog::trace("{0:s} dispatching profiling thread.", __func__);
            std::thread batcher(&BaseReqBatcher::batchRequestsProfiling, this);
            batcher.detach();
            return;
        }
        spdlog::trace("{0:s} dispatching batching thread.", __func__);
        std::thread batcher(&BaseReqBatcher::batchRequests, this);
        batcher.detach();
    }

    BaseReqBatcherConfigs loadConfigsFromJson(const json &jsonConfigs);

    virtual void loadConfigs(const json &jsonConfigs, bool isConstructing = false) override;
protected:

    BatchSizeType msvc_onBufferBatchSize = 0;
    std::vector<cv::cuda::GpuMat> msvc_batchBuffer;
    bool isTimeToBatch() override;
    bool checkReqEligibility(ClockType currReq_genTime) override;

    uint8_t msvc_imgType, msvc_colorCvtType, msvc_resizeInterpolType;
    float msvc_imgNormScale;
    std::vector<float> msvc_subVals, msvc_divVals;

};


typedef uint16_t BatchSizeType;

class BaseBatchInferencer : public Microservice {
public:
    BaseBatchInferencer(const json &jsonConfigs);
    ~BaseBatchInferencer() = default;
    virtual void inference();
    virtual void inferenceProfiling();

    RequestShapeType getInputShapeVector();
    RequestShapeType getOutputShapeVector();

    void dispatchThread() override {
        if (msvc_RUNMODE == RUNMODE::PROFILING) {
            spdlog::trace("{0:s} dispatching profiling thread.", __func__);
            std::thread inferencer(&BaseBatchInferencer::inferenceProfiling, this);
            inferencer.detach();
            return;
        }
        spdlog::trace("{0:s} dispatching inference thread.", __func__);
        std::thread inferencer(&BaseBatchInferencer::inference, this);
        inferencer.detach();
    }

    virtual void loadConfigs(const json &jsonConfigs, bool isConstructing = false) override;
protected:
    BatchSizeType msvc_onBufferBatchSize;
    std::vector<void *> msvc_engineInputBuffers, msvc_engineOutputBuffers;
    TRTConfigs msvc_engineConfigs;
    Engine* msvc_inferenceEngine;

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
inline void crop(
    const cv::cuda::GpuMat &image,
    int orig_h,
    int orig_w,
    int infer_h,
    int infer_w,
    int numDetections,
    const float *bbox_coorList,
    std::vector<cv::cuda::GpuMat> &croppedBBoxes
);

inline void cropOneBox(
    const cv::cuda::GpuMat &image,
    int infer_h,
    int infer_w,
    int numDetections,
    const float *bbox_coorList,
    cv::cuda::GpuMat &croppedBBoxes
);

class BaseBBoxCropper : public Microservice {
public:
    BaseBBoxCropper(const json &jsonConfigs);
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

    void dispatchThread() override {
        if (msvc_RUNMODE == RUNMODE::PROFILING) {
            spdlog::trace("{0:s} dispatching profiling thread.", __func__);
            std::thread postprocessor(&BaseBBoxCropper::cropProfiling, this);
            postprocessor.detach();
            return;
        }
        spdlog::trace("{0:s} dispatching cropping thread.", __func__);
        std::thread postprocessor(&BaseBBoxCropper::cropping, this);
        postprocessor.detach();
    }

    virtual void loadConfigs(const json &jsonConfigs, bool isConstructing = false) override;

protected:
    RequestShapeType msvc_inferenceShape;
};

class BaseBBoxCropperVerifier : public Microservice {
public:
    BaseBBoxCropperVerifier(const json& jsonConfigs);
    ~BaseBBoxCropperVerifier() = default;

    void cropping();
    void setInferenceShape(RequestShapeType shape) {
        msvc_inferenceShape = shape;
    }

    void cropProfiling();

    virtual void dispatchThread() override {
        std::thread postprocessor(&BaseBBoxCropperVerifier::cropping, this);
        postprocessor.detach();
    }

    BaseBBoxCropperVerifierConfigs loadConfigsFromJson(const json &jsonConfigs);

    virtual void loadConfigs(const json &jsonConfigs, bool isConstructing = false) override;

protected:
    RequestShapeType msvc_inferenceShape;
};

class BaseClassifier : public Microservice {
public:
    BaseClassifier(const BaseMicroserviceConfigs &configs);
    ~BaseClassifier() = default;

    virtual void classify() ;

    virtual void dispatchThread() override {
        std::thread classifier(&BaseClassifier::classify, this);
        classifier.detach();
    }

protected:
    RequestShapeType msvc_inferenceShape;
    uint16_t msvc_numClasses;
};

class BaseSoftmaxClassifier : public BaseClassifier {
public:
    BaseSoftmaxClassifier(const BaseMicroserviceConfigs &configs);
    ~BaseSoftmaxClassifier() = default;

    virtual void classify() override;
};

class BaseKPointExtractor : public Microservice{
public:
    BaseKPointExtractor(const BaseMicroserviceConfigs &configs);
    ~BaseKPointExtractor() = default;

    virtual void extractor();

    virtual void dispatchThread() override {
        std::thread extractor(&BaseKPointExtractor::extractor, this);
        extractor.detach();
    }
};

#endif //BASEPROCESSOR_H