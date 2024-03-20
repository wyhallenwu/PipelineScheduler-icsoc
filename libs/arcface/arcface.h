#include <baseprocessor.h>
#include <trtengine.h>
#include <misc.h>
#include "container_agent.h"
#include "receiver.h"
#include "sender.h"

class ArcFacePreprocessor : public BasePreprocessor {
public:
    ArcFacePreprocessor(const BaseMicroserviceConfigs &config);
    ~ArcFacePreprocessor() = default;
protected:
    friend class ArcFaceAgent;
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
    ~ArcFaceInference() = default;
protected:
    friend class ArcFaceAgent;
    void inference();
    std::vector<void *> msvc_engineInputBuffers, msvc_engineOutputBuffers;
    TRTConfigs msvc_engineConfigs;
    Engine* msvc_inferenceEngine;
};

class ArcFacePostprocessor : public BasePostprocessor {
public:
    ArcFacePostprocessor(const BaseMicroserviceConfigs &config);
    ~ArcFacePostprocessor() = default;
protected:
    friend class ArcFaceAgent;
    void postProcessing();
};

class ArcFaceAgent : public ContainerAgent {
public:
    ArcFaceAgent(const std::string &name, uint16_t own_port, std::vector<Microservice*> services);
};