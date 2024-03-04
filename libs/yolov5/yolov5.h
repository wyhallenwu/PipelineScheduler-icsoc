#include <basepreprocessor.h>
#include <baseprocessor.h>
#include <basepostprocessor.h>
#include <trtengine.h>
#include <misc.h>
#include <chrono>
#include <thread>

struct yoloV5Configs {
    const TRTConfigs engineConfigs;
    const std::vector<std::string> classNames = cocoClassNames;
};


template<typename InType>
class YoloV5Preprocessor : public BasePreprocessor<InType> {
public:
    YoloV5Preprocessor(const BaseMicroserviceConfigs &config);
    ~YoloV5Preprocessor();
protected:
    void batchRequests();
    // 
    // bool isTimeToBatch() override;
    //
    // bool checkReqEligibility(uint64_t currReq_genTime) override;
    //
    // void updateReqRate(ClockTypeTemp lastInterReqDuration) override;
};

template<typename InType>
class YoloV5Inference : public BaseProcessor<InType> {
public:
    YoloV5Inference(const BaseMicroserviceConfigs &config, const TRTConfigs &engineConfigs);
    ~YoloV5Inference();
protected:
    void inference();
    std::vector<void *> msvc_engineInputBuffers, msvc_engineOutputBuffers;
    std::vector<LocalGPUReqDataType> batchedOutBuffer;
    TRTConfigs msvc_engineConfigs;
    Engine msvc_inferenceEngine;
};

template<typename InType>
class YoloV5Postprocessor : public BasePostprocessor<InType> {
public:
    YoloV5Postprocessor(const BaseMicroserviceConfigs &config);
    ~YoloV5Postprocessor();
protected:
    void postProcessing();
};