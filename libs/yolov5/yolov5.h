#include "../baseprocessors/basepreprocessor.h"
#include "../baseprocessors/baseprocessor.h"
#include "../trtengine/trtengine.h"

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