#include <yolov5.h>

YoloV5Preprocessor::YoloV5Preprocessor(const BaseMicroserviceConfigs &configs) : BasePreprocessor(configs) {
    
}

void YoloV5Preprocessor::batchRequests() {
    // The time where the last request was generated.
    ClockType lastReq_genTime;
    // The time where the current incoming request was generated.
    ClockType currReq_genTime;
    // The time where the current incoming request arrives
    ClockType currReq_recvTime;
    // Buffer memory for each batch
    std::vector<RequestData<LocalGPUReqDataType>> bufferData;

    // // Data carried from upstream microservice to be processed at a downstream
    std::vector<RequestData<LocalGPUReqDataType>> prevData;
    RequestData<LocalGPUReqDataType> data;

    // Incoming request
    Request<LocalGPUReqDataType> currReq;

    // Request sent to a downstream microservice
    Request<LocalGPUReqDataType> outReq;   

    // Batch size of current request
    BatchSizeType currReq_batchSize;
    while (true) {
        // Allowing this thread to naturally come to an end
        if (this->STOP_THREADS) {
            break;
        }
        else if (this->PAUSE_THREADS) {
            continue;
        }
        // Processing the next incoming request
        currReq = msvc_InQueue.at(0)->pop2();
        msvc_inReqCount++;
        currReq_genTime = currReq.req_origGenTime;
        // We need to check if the next request is worth processing.
        // If it's too late, then we can drop and stop processing this request.
        if (!this->checkReqEligibility(currReq_genTime)) {
            continue;
        }
        // The generated time of this incoming request will be used to determine the rate with which the microservice should
        // check its incoming queue.
        currReq_recvTime = std::chrono::system_clock::now();
        if (this->msvc_inReqCount > 1) {
            this->updateReqRate(currReq_genTime);
        }

        msvc_onBufferBatchSize++;
        // Resize the incoming request image the padd with the grey color
        // The resize image will be copied into a reserved buffer
        data.data = resizePadRightBottom(
            currReq.req_data[0].data,
            this->msvc_outReqShape[0][1],
            this->msvc_outReqShape[0][2],
            cv::Scalar(128, 128, 128)
        );
        data.shape = RequestShapeType({3, msvc_outReqShape[0][1], msvc_outReqShape[0][2]});
        bufferData.emplace_back(data);
        // prevData.emplace_back(currReq.req_data[0]);

        cudaFree(currReq.req_data[0].data.cudaPtr());
        // First we need to decide if this is an appropriate time to batch the buffered data or if we can wait a little more.
        // Waiting more means there is a higher chance the earliest request in the buffer will be late eventually.
        if (this->isTimeToBatch()) {
            // If true, copy the buffer data into the out queue
            outReq = {
                std::chrono::_V2::system_clock::now(),
                9999,
                "",
                msvc_onBufferBatchSize,
                bufferData,
                prevData
            };
            msvc_OutQueue[0]->emplace(outReq);
            msvc_onBufferBatchSize = 0;
            bufferData.clear();
            prevData.clear();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(this->msvc_interReqTime));
    }
}