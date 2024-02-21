#include "yolov5.h"

template<typename InType>
YoloV5Inference<InType>::YoloV5Inference(
    const BaseMicroserviceConfigs &config, 
    const TRTConfigs &engineConfigs) : BaseProcessor<InType>(config), msvc_engineConfigs(engineConfigs) {
    
    msvc_inferenceEngine = Engine(engineConfigs);
    msvc_idealBatchSize = engineConfigs.maxBatchSize;

    msvc_engineInputBuffers = msvc_inferenceEngine.getInputBuffers();
    msvc_engineOutputBuffers = msvc_inferenceEngine.getOutputBuffers();
}

template<typename InType>
void YoloV5Inference<InType>::inference() {
    // The time where the last request was generated.
    ClockTypeTemp lastReq_genTime;
    // The time where the current incoming request was generated.
    ClockTypeTemp currReq_genTime;
    // The time where the current incoming request arrives
    ClockTypeTemp currReq_recvTime;
    std::vector<LocalGPUReqDataType> outBuffer;

    BatchSizeType currReq_batchSize;
    while (true) {
        if (!this->RUN_THREADS) {
            break;
        }
        // Processing the next incoming request
        InType currReq = this->InQueue.pop();
        this->msvc_inReqCount++;
        currReq_genTime = currReq.req_origGenTime;
        // We need to check if the next request is worth processing.
        // If it's too late, then we can drop and stop processing this request.
        if (!this->checkReqEligibility(currReq_genTime)) {
            continue;
        }
        // The generated time of this incoming request will be used to determine the rate with which the microservice should
        // check its incoming queue.
        currReq_recvTime = std::chrono::high_resolution_clock::now();
        if (this->msvc_inReqCount > 1) {
            this->updateReqRate(currReq_genTime);
        }
        // Do batched inference
        std::vector<LocalGPUReqDataType> batch;
        currReq_batchSize = currReq.req_data.size();
        for (std::size_t i; i < currReq_batchSize; ++i) {
            batch.emplace_back(currReq.req_data[i].content);
        }
        msvc_inferenceEngine.runInference(batch, outBuffer, msvc_idealBatchSize);

        
    }
}
