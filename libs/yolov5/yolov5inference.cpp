#include "yolov5.h"

using namespace spdlog;

YoloV5Inference::YoloV5Inference(
    const BaseMicroserviceConfigs &config, 
    const TRTConfigs &engineConfigs) : BaseProcessor(config), msvc_engineConfigs(engineConfigs) {
    
    msvc_inferenceEngine = new Engine(engineConfigs);

    msvc_engineInputBuffers = msvc_inferenceEngine->getInputBuffers();
    msvc_engineOutputBuffers = msvc_inferenceEngine->getOutputBuffers();

    info("{0:s} is created.", msvc_name); 
}

void YoloV5Inference::inference() {
    // The time where the last request was generated.
    ClockType lastReq_genTime;
    // The time where the current incoming request was generated.
    ClockType currReq_genTime;
    // The time where the current incoming request arrives
    ClockType currReq_recvTime;

    // Vector of GPU Mat to copy data into and out of the TRT Engine
    std::vector<LocalGPUReqDataType> trtInBuffer, trtOutBuffer;

    // Data package to be sent to and processed at the next microservice
    std::vector<RequestData<LocalGPUReqDataType>> outReqData;   

    // Instance of data to be packed into `outReqData`
    RequestData<LocalGPUReqDataType> data;

    Request<LocalGPUReqDataType> outReq;

    // Batch size of current request
    BatchSizeType currReq_batchSize;
    spdlog::info("{0:s} STARTS.", msvc_name); 

    cudaStream_t inferenceStream;
    checkCudaErrorCode(cudaStreamCreate(&inferenceStream), __func__);
    while (true) {
        // Allowing this thread to naturally come to an end
        if (this->STOP_THREADS) {
            spdlog::info("{0:s} STOPS.", msvc_name);
            break;
        }
        else if (this->PAUSE_THREADS) {
            spdlog::info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }

        // Processing the next incoming request
        // Current incoming equest and request to be sent out to the next
        Request<LocalGPUReqDataType> currReq = msvc_InQueue.at(0)->pop2();
        msvc_inReqCount++;

        // The generated time of this incoming request will be used to determine the rate with which the microservice should
        // check its incoming queue.
        currReq_recvTime = std::chrono::system_clock::now();
        if (this->msvc_inReqCount > 1) {
            this->updateReqRate(currReq_genTime);
        }

        // Do batched inference with TRT
        currReq_batchSize = currReq.req_batchSize;
        trace("{0:s} popped a request of batch size {1:d}", msvc_name, currReq_batchSize);

        for (std::size_t i = 0; i < currReq_batchSize; ++i) {
            trtInBuffer.emplace_back(currReq.req_data[i].data);
        }
        info("{0:s} extracts inference data from message. Run inference!", msvc_name);
        msvc_inferenceEngine->runInference(trtInBuffer, trtOutBuffer, currReq_batchSize, inferenceStream);
        trace("{0:s} finished INFERENCE.", msvc_name);


        // After inference, 4 buffers are filled with memory, which we need to carry to post processor.
        // We put 4 buffers into a vector along with their respective shapes for the post processor to interpret.
        for (std::size_t i = 0; i < this->msvc_outReqShape.at(0).size(); ++i) {
            data = {
                this->msvc_outReqShape.at(0).at(i),
                trtOutBuffer[i]
            };
            outReqData.emplace_back(data);
        }
        


        // Packing everything inside the `outReq` to be sent to and processed at the next microservice
        outReq = {
            std::chrono::_V2::system_clock::now(),
            currReq.req_e2eSLOLatency,
            "",
            currReq_batchSize,
            outReqData, //req_data
            currReq.upstreamReq_data // upstreamReq_data
        };
        // // After inference, the gpumat inside `inbuffer` is no longer used and can be freed.
        // for (std::size_t i = 0; i < trtInBuffer.size(); i++) {
        //     checkCudaErrorCode(cudaFree(trtInBuffer.at(i).cudaPtr()));
        // }
        info("{0:s} emplaced a request for a batch size of {1:d}", msvc_name, currReq_batchSize);

        msvc_OutQueue[0]->emplace(outReq);
        outReqData.clear();
        trtInBuffer.clear();
        trtOutBuffer.clear();

        trace("{0:s} sleeps for {1:d} millisecond", msvc_name, msvc_interReqTime);
        std::this_thread::sleep_for(std::chrono::milliseconds(this->msvc_interReqTime));
    }
    checkCudaErrorCode(cudaStreamDestroy(inferenceStream), __func__);
}
