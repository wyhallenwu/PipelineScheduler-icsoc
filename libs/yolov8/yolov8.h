#include <baseprocessor.h>
#include <trtengine.h>
#include <misc.h>
#include <chrono>
#include "container_agent.h"
#include "receiver.h"
#include "sender.h"
#include "spdlog/spdlog.h"

using namespace spdlog;


// class YoloV8Preprocessor : public BaseReqBatcher {
// public:
//     YoloV8Preprocessor(const BaseMicroserviceConfigs &config);
//     ~YoloV8Preprocessor() = default;
// protected:
//     friend class YoloV8Agent;
//     void batchRequests();
//     // 
//     // bool isTimeToBatch() override;
//     //
//     // bool checkReqEligibility(uint64_t currReq_genTime) override;
//     //
//     // void updateReqRate(ClockTypeTemp lastInterReqDuration) override;
// };

// class YoloV8Inference : public BaseBatchInferencer {
// public:
//     YoloV8Inference(const BaseMicroserviceConfigs &config, const TRTConfigs &engineConfigs);
//     ~YoloV8Inference() = default;
// protected:
//     friend class YoloV8Agent;
//     void inference();
//     std::vector<void *> msvc_engineInputBuffers, msvc_engineOutputBuffers;
//     TRTConfigs msvc_engineConfigs;
//     Engine* msvc_inferenceEngine;
// };

// class YoloV8Postprocessor : public BaseBBoxCropper {
// public:
//     friend class YoloV8Agent;
//     YoloV8Postprocessor(const BaseMicroserviceConfigs &config);
//     ~YoloV8Postprocessor() = default;
// protected:
//     void postProcessing();
// };

class YoloV8Agent : public ContainerAgent {
public:
    YoloV8Agent(const std::string &name, uint16_t own_port, int8_t devIndex, std::vector<Microservice*> services);
};