#include <baseprocessor.h>
#include <trtengine.h>
#include <misc.h>

class ArcFacePreprocessor : public BasePreprocessor {
public:
    ArcFacePreprocessor(const BaseMicroserviceConfigs &config);
    ~ArcFacePreprocessor();
protected:
    void batchRequests();
    // 
    // bool isTimeToBatch() override;
    //
    // bool checkReqEligibility(uint64_t currReq_genTime) override;
    //
    // void updateReqRate(ClockTypeTemp lastInterReqDuration) override;
};

class ArcFaceInference : public BaseProcessor {
public:
    ArcFaceInference(const BaseMicroserviceConfigs &config, const TRTConfigs &engineConfigs);
    ~ArcFaceInference();
protected:
    void inference();
    std::vector<void *> msvc_engineInputBuffers, msvc_engineOutputBuffers;
    TRTConfigs msvc_engineConfigs;
    Engine* msvc_inferenceEngine;
};

class ArcFacePostprocessor : public BasePostprocessor {
public:
    ArcFacePostprocessor(const BaseMicroserviceConfigs &config);
    ~ArcFacePostprocessor();
protected:
    void postProcessing();
};