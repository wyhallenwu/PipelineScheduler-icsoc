#include <baseprocessor.h>
#include <trtengine.h>
#include <misc.h>
#include <chrono>
#include "container_agent.h"
#include "receiver.h"
#include "sender.h"


class YoloV5Preprocessor : public BasePreprocessor {
public:
    YoloV5Preprocessor(const BaseMicroserviceConfigs &config);
    ~YoloV5Preprocessor() {

    };
protected:
    friend class YoloV5Agent;
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
    ~YoloV5Inference() {

    };
protected:
    friend class YoloV5Agent;
    void inference();
    std::vector<void *> msvc_engineInputBuffers, msvc_engineOutputBuffers;
    TRTConfigs msvc_engineConfigs;
    Engine* msvc_inferenceEngine;
};

class YoloV5Postprocessor : public BasePostprocessor {
public:
    friend class YoloV5Agent;
    YoloV5Postprocessor(const BaseMicroserviceConfigs &config);
    ~YoloV5Postprocessor() {
        
    };
protected:
    void postProcessing();
};

class YoloV5Agent : public ContainerAgent {
public:
    YoloV5Agent(const std::string &name, uint16_t own_port,
                std::vector<BaseMicroserviceConfigs> &msvc_configs, const TRTConfigs &yoloConfigs, Microservice *receiver);
};