#include <yolov5.h>
#include <chrono>
#include <thread>

template<typename InType>
YoloV5Preprocessor<InType>::YoloV5Preprocessor(const BaseMicroserviceConfigs &configs) : BasePreprocessor<InType>(configs) {
    
}

template<typename InType>
void YoloV5Preprocessor<InType>::batchRequests() {
    // The time where the last request was generated.
    ClockTypeTemp lastReq_genTime;
    // The time where the current incoming request was generated.
    ClockTypeTemp currReq_genTime;
    // The time where the current incoming request arrives
    ClockTypeTemp currReq_recvTime;
    // Buffer memory for each batch
    std::vector<LocalGPUReqDataType> batchBuffer;
    
    while (true) {
        // Allowing this thread to naturally come to an end
        if (not this->RUN_THREADS) {
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
        // Resize the incoming request image the padd with the grey color
        // The resize image will be copied into a reserved buffer
        batchBuffer.emplace_back(
            {
                resizePadRightBottom(
                    currReq.req_data.content,
                    this->msvc_outReqShape[0][1],
                    this->msvc_outReqShape[0][2],
                    cv::Scalar(128, 128, 128)
                )
            }
        );
        cudaFree(currReq.req_data.cudaPtr());
        // First we need to decide if this is an appropriate time to batch the buffered data or if we can wait a little more.
        // Waiting more means there is a higher chance the earliest request in the buffer will be late eventually.
        if (this->isTimeToBatch()) {
            // If true, copy the buffer data into the out queue
            std::vector<Data<LocalGPUReqDataType>> batchedData;
            for (std::size_t i = 0; i < batchBuffer.size(); ++i) {
                batchedData.emplace_back(
                    {
                        this->msvc_outReqShape[0],
                        batchBuffer[i]
                    }
                );
            }
            
            DataRequest<Data<LocalGPUReqDataType>> outReq(
                std::chrono::high_resolution_clock::now(),
                currReq.req_e2eSLOLatency,
                "",
                this->msvc_onBufferBatchSize,
                batchedData
            );
            this->OutQueue.emplace(outReq);
            this->mscv_onBufferBatchSize = 0;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(this->msvc_interReqTime));
    }
}