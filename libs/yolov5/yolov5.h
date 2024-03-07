#include <baseprocessor.h>
#include <trtengine.h>
#include <misc.h>
#include <chrono>

struct yoloV5Configs {
    const TRTConfigs engineConfigs;
    const std::vector<std::string> classNames = cocoClassNames;
};


class YoloV5Preprocessor : public BasePreprocessor {
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

class YoloV5Inference : public BaseProcessor {
public:
    YoloV5Inference(const BaseMicroserviceConfigs &config, const TRTConfigs &engineConfigs);
    ~YoloV5Inference();
protected:
    void inference();
    std::vector<void *> msvc_engineInputBuffers, msvc_engineOutputBuffers;
    TRTConfigs msvc_engineConfigs;
    Engine* msvc_inferenceEngine;
};

class YoloV5Postprocessor : public BasePostprocessor {
public:
    YoloV5Postprocessor(const BaseMicroserviceConfigs &config);
    ~YoloV5Postprocessor();
protected:
    void postProcessing();
};