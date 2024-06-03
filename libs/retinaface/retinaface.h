#include <baseprocessor.h>
#include <trtengine.h>
#include <misc.h>
#include "container_agent.h"
#include "data_reader.h"
#include "receiver.h"

// class RetinaFacePreprocessor : public BasePreprocessor {
// public:
//     RetinaFacePreprocessor(const BaseMicroserviceConfigs &config);
//     ~RetinaFacePreprocessor() = default;
// protected:
//     friend class RetinaFaceAgent;
//     void batchRequests();
//     // 
//     // bool isTimeToBatch() override;
//     //
//     // bool checkReqEligibility(uint64_t currReq_genTime) override;
//     //
//     // void updateReqRate(ClockTypeTemp lastInterReqDuration) override;
// };

// class RetinaFaceInference : public BaseProcessor {
// public:
//     RetinaFaceInference(const BaseMicroserviceConfigs &config, const TRTConfigs &engineConfigs);
//     ~RetinaFaceInference() = default;
// protected:
//     friend class RetinaFaceAgent;
//     void inference();
//     std::vector<void *> msvc_engineInputBuffers, msvc_engineOutputBuffers;
//     TRTConfigs msvc_engineConfigs;
//     Engine* msvc_inferenceEngine;
// };

// class RetinaFacePostprocessor : public BasePostprocessor {
// public:
//     RetinaFacePostprocessor(const BaseMicroserviceConfigs &config);
//     ~RetinaFacePostprocessor() = default;
// protected:
//     friend class RetinaFaceAgent;
//     void postProcessing();
// };

class RetinaFaceAgent : public ContainerAgent {
public:
    RetinaFaceAgent(
        const json &configs
    );
};

class RetinaFaceDataSource : public ContainerAgent {
public:
    RetinaFaceDataSource(
        const json &configs
    );
};