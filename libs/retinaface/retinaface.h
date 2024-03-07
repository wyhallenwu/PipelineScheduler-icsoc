#include <baseprocessor.h>
#include <trtengine.h>
#include <misc.h>

class RetinaFacePreprocessor : public BasePreprocessor {
public:
    RetinaFacePreprocessor(const BaseMicroserviceConfigs &config);
    ~RetinaFacePreprocessor();
protected:
    void batchRequests();
    // 
    // bool isTimeToBatch() override;
    //
    // bool checkReqEligibility(uint64_t currReq_genTime) override;
    //
    // void updateReqRate(ClockTypeTemp lastInterReqDuration) override;
};

class RetinaFaceInference : public BaseProcessor {
public:
    RetinaFaceInference(const BaseMicroserviceConfigs &config, const TRTConfigs &engineConfigs);
    ~RetinaFaceInference();
protected:
    void inference();
    std::vector<void *> msvc_engineInputBuffers, msvc_engineOutputBuffers;
    TRTConfigs msvc_engineConfigs;
    Engine* msvc_inferenceEngine;
};

class RetinaFacePostprocessor : public BasePostprocessor {
public:
    RetinaFacePostprocessor(const BaseMicroserviceConfigs &config);
    ~RetinaFacePostprocessor();
protected:
    void postProcessing();
};