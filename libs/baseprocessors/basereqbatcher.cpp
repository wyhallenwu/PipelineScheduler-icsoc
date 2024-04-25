#include "baseprocessor.h"

using namespace spdlog;

/**
 * @brief Get number at index from a string of comma separated numbers
 * 
 * @param str 
 * @param index 
 * @return uint64_t 
 */
inline uint64_t getNumberAtIndex(const std::string& str, int index) {
    int currentIndex = 0;
    int startPos = 0;
    uint64_t number = 0;

    if (index == -1) {
        // Handle the case where the index is -1 (last number)
        for (size_t i = 0; i < str.length(); i++) {
            if (str[i] == ',') {
                startPos = i + 1;
            }
        }
        std::string numberStr = str.substr(startPos);
        number = std::stoull(numberStr);
        return number;
    }

    for (size_t i = 0; i < str.length(); i++) {
        if (str[i] == ',') {
            if (currentIndex == index) {
                std::string numberStr = str.substr(startPos, i - startPos);
                number = std::stoull(numberStr);
                return number;
            }
            startPos = i + 1;
            currentIndex++;
        }
    }

    // Handle the last number in the string
    if (currentIndex == index) {
        std::string numberStr = str.substr(startPos);
        number = std::stoull(numberStr);
        return number;
    }

    return 0; // Return 0 if the index is out of range
}

BaseReqBatcherConfigs BaseReqBatcher::loadConfigsFromJson(const json &jsonConfigs) {
    BaseReqBatcherConfigs configs;

    jsonConfigs.at("msvc_imgType").get_to(configs.msvc_imgType);
    jsonConfigs.at("msvc_colorCvtType").get_to(configs.msvc_colorCvtType);
    jsonConfigs.at("msvc_resizeInterpolType").get_to(configs.msvc_resizeInterpolType);
    std::string normVal;
    jsonConfigs.at("msvc_imgNormScale").get_to(normVal);
    msvc_imgNormScale = fractionToFloat(normVal);
    jsonConfigs.at("msvc_subVals").get_to(configs.msvc_subVals);
    jsonConfigs.at("msvc_divVals").get_to(configs.msvc_divVals);
    return configs;
}

/**
 * @brief Load the configurations from the json file
 * 
 * @param jsonConfigs 
 * @param isConstructing
 * 
 * @note The function is called from the constructor or when the microservice is to be reloaded
 */
void BaseReqBatcher::loadConfigs(const json &jsonConfigs, bool isConstructing) {
    // Load the configs from the json file for Microservice class
    if (!isConstructing) { // If the function is not called from the constructor
        Microservice::loadConfigs(jsonConfigs, true);
    }

    BaseReqBatcherConfigs configs = loadConfigsFromJson(jsonConfigs);

    msvc_imgType = configs.msvc_imgType;
    msvc_colorCvtType = configs.msvc_colorCvtType;
    msvc_resizeInterpolType = configs.msvc_resizeInterpolType;
    msvc_imgNormScale = configs.msvc_imgNormScale;
    msvc_subVals = configs.msvc_subVals;
    msvc_divVals = configs.msvc_divVals;
}

/**
 * @brief Construct a new Base Preprocessor that inherites the LocalGPUDataMicroservice given the `InType`
 * 
 * @param configs 
 */
BaseReqBatcher::BaseReqBatcher(const json &jsonConfigs) : Microservice(jsonConfigs){
    loadConfigs(jsonConfigs);
    info("{0:s} is created.", msvc_name); 
}

void BaseReqBatcher::batchRequests() {
    msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);
    setDevice();
    // The time where the last request was generated.
    ClockType lastReq_genTime;
    // The time where the current incoming request was generated.
    ClockType currReq_genTime;
    // The time where the current incoming request arrives
    ClockType currReq_recvTime;

    // Batch reqs' gen time
    RequestTimeType outReq_genTime;

    // Batch reqs' slos
    RequestSLOType outReq_slo;

    // Batch reqs' paths
    RequestPathType outReq_path;

    // Buffer memory for each batch
    std::vector<RequestData<LocalGPUReqDataType>> bufferData;

    // // Data carried from upstream microservice to be processed at a downstream
    std::vector<RequestData<LocalGPUReqDataType>> prevData;
    RequestData<LocalGPUReqDataType> data;

    // Incoming request
    Request<LocalGPUReqDataType> currReq;

    Request<LocalCPUReqDataType> currCPUReq;

    // Request sent to a downstream microservice
    Request<LocalGPUReqDataType> outReq;   

    // Batch size of current request
    BatchSizeType currReq_batchSize;
    info("{0:s} STARTS.", msvc_name); 
    cv::cuda::Stream preProcStream;
    READY = true;
    while (true) {
        // Allowing this thread to naturally come to an end
        if (this->STOP_THREADS) {
            info("{0:s} STOPS.", msvc_name);
            break;
        }
        else if (this->PAUSE_THREADS) {
            //info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }
        // Processing the next incoming request
        if (msvc_InQueue.at(0)->getActiveQueueIndex() != msvc_activeInQueueIndex.at(0)) {
            if (msvc_InQueue.at(0)->size(msvc_activeInQueueIndex.at(0)) == 0) {
                msvc_activeInQueueIndex.at(0) = msvc_InQueue.at(0)->getActiveQueueIndex();
                trace("{0:s} Set current active queue index to {1:d}.", msvc_name, msvc_activeInQueueIndex.at(0));
            }
        }
        trace("{0:s} Current active queue index {1:d}.", msvc_name, msvc_activeInQueueIndex.at(0));
        if (msvc_activeInQueueIndex.at(0) == 1) {
            currCPUReq = msvc_InQueue.at(0)->pop1();
            currReq = uploadReq(currCPUReq);
        } else if (msvc_activeInQueueIndex.at(0) == 2) {
            currReq = msvc_InQueue.at(0)->pop2();
        }
        msvc_inReqCount++;
        currReq_genTime = currReq.req_origGenTime[0];

        // We need to check if the next request is worth processing.
        // If it's too late, then we can drop and stop processing this request.
        if (!this->checkReqEligibility(currReq_genTime)) {
            continue;
        }
        // The generated time of this incoming request will be used to determine the rate with which the microservice should
        // check its incoming queue.
        currReq_recvTime = {std::chrono::high_resolution_clock::now()};
        if (this->msvc_inReqCount > 1) {
            this->updateReqRate(currReq_genTime);
        }
        currReq_batchSize = currReq.req_batchSize;

        outReq_genTime.emplace_back(currReq_genTime);
        outReq_slo.emplace_back(currReq.req_e2eSLOLatency[0]);
        outReq_path.emplace_back(currReq.req_travelPath[0] + "[" + msvc_containerName + "_" + std::to_string(msvc_inReqCount) + "]");
        trace("{0:s} popped a request of batch size {1:d}. In queue size is {2:d}.", msvc_name, currReq_batchSize, msvc_InQueue.at(0)->size());

        msvc_onBufferBatchSize++;
        // Resize the incoming request image the padd with the grey color
        // The resize image will be copied into a reserved buffer


        prevData.emplace_back(currReq.req_data[0]);

        trace("{0:s} resizing a frame of [{1:d}, {2:d}] -> [{3:d}, {4:d}]",
            msvc_name,
            currReq.req_data[0].data.rows,
            currReq.req_data[0].data.cols,
            (this->msvc_outReqShape.at(0))[0][1],
            (this->msvc_outReqShape.at(0))[0][2]
        );
        data.data = resizePadRightBottom(
            currReq.req_data[0].data,
            (this->msvc_outReqShape.at(0))[0][1],
            (this->msvc_outReqShape.at(0))[0][2],
            cv::Scalar(128, 128, 128),
            preProcStream,
            msvc_imgType,
            msvc_colorCvtType,
            msvc_resizeInterpolType
        );

        data.data = cvtHWCToCHW(data.data, preProcStream, msvc_imgType);

        data.data = normalize(data.data, preProcStream, msvc_subVals, msvc_divVals, msvc_imgNormScale);

        trace("{0:s} finished resizing a frame", msvc_name);
        data.shape = RequestDataShapeType({3, (this->msvc_outReqShape.at(0))[0][1], (this->msvc_outReqShape.at(0))[0][2]});
        bufferData.emplace_back(data);
        trace("{0:s} put an image into buffer. Current batch size is {1:d} ", msvc_name, msvc_onBufferBatchSize);

        // std::cout << "Time taken to preprocess a req is " << stopwatch.elapsed_seconds() << std::endl;
        // cudaFree(currReq.req_data[0].data.cudaPtr());
        // First we need to decide if this is an appropriate time to batch the buffered data or if we can wait a little more.
        // Waiting more means there is a higher chance the earliest request in the buffer will be late eventually.
        if (this->isTimeToBatch()) {
            // If true, copy the buffer data into the out queue
            outReq = {
                outReq_genTime,
                outReq_slo,
                outReq_path,
                msvc_onBufferBatchSize,
                bufferData,
                prevData
            };
            trace("{0:s} emplaced a request of batch size {1:d} ", msvc_name, msvc_onBufferBatchSize);
            msvc_OutQueue[0]->emplace(outReq);
            msvc_onBufferBatchSize = 0;
            outReq_genTime.clear();
            outReq_path.clear();
            outReq_slo.clear();
            bufferData.clear();
            prevData.clear();
        }
        trace("{0:s} sleeps for {1:d} millisecond", msvc_name, msvc_interReqTime);
        std::this_thread::sleep_for(std::chrono::milliseconds(this->msvc_interReqTime));
    }
    msvc_logFile.close();
}


void BaseReqBatcher::batchRequestsProfiling() {
    msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);
    setDevice();
    // The time where the last request was generated.
    ClockType lastReq_genTime;
    // The time where the current incoming request was generated.
    ClockType currReq_genTime;
    // The time where the current incoming request arrives
    ClockType currReq_recvTime;

    // Batch reqs' gen time
    RequestTimeType outReq_genTime;

    // Batch reqs' slos
    RequestSLOType outReq_slo;

    // Batch reqs' paths
    RequestPathType outReq_path;

    // Buffer memory for each batch
    std::vector<RequestData<LocalGPUReqDataType>> bufferData;

    // // Data carried from upstream microservice to be processed at a downstream
    std::vector<RequestData<LocalGPUReqDataType>> prevData;
    RequestData<LocalGPUReqDataType> data;

    // Incoming request
    Request<LocalGPUReqDataType> currReq;

    Request<LocalCPUReqDataType> currCPUReq;

    // Request sent to a downstream microservice
    Request<LocalGPUReqDataType> outReq;   

    // Batch size of current request
    BatchSizeType currReq_batchSize;
    info("{0:s} STARTS.", msvc_name); 
    cv::cuda::Stream preProcStream;

    auto timeNow = std::chrono::high_resolution_clock::now();

    READY = true;
    while (true) {
        // Allowing this thread to naturally come to an end
        if (this->STOP_THREADS) {
            info("{0:s} STOPS.", msvc_name);
            break;
        }
        else if (this->PAUSE_THREADS) {
            //info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }
        // Processing the next incoming request
        if (msvc_InQueue.at(0)->getActiveQueueIndex() != msvc_activeInQueueIndex.at(0)) {
            if (msvc_InQueue.at(0)->size(msvc_activeInQueueIndex.at(0)) == 0) {
                msvc_activeInQueueIndex.at(0) = msvc_InQueue.at(0)->getActiveQueueIndex();
                trace("{0:s} Set current active queue index to {1:d}.", msvc_name, msvc_activeInQueueIndex.at(0));
            }
        }
        trace("{0:s} Current active queue index {1:d}.", msvc_name, msvc_activeInQueueIndex.at(0));
        if (msvc_activeInQueueIndex.at(0) == 1) {
            currCPUReq = msvc_InQueue.at(0)->pop1();
            currReq = uploadReq(currCPUReq);
        } else if (msvc_activeInQueueIndex.at(0) == 2) {
            currReq = msvc_InQueue.at(0)->pop2();
        }
        msvc_inReqCount++;
        currReq_genTime = currReq.req_origGenTime[0];

        // We need to check if the next request is worth processing.
        // If it's too late, then we can drop and stop processing this request.
        if (!this->checkReqEligibility(currReq_genTime)) {
            continue;
        }
        // The generated time of this incoming request will be used to determine the rate with which the microservice should
        // check its incoming queue.
        currReq_recvTime = {std::chrono::high_resolution_clock::now()};
        if (this->msvc_inReqCount > 1) {
            this->updateReqRate(currReq_genTime);
        }
        currReq_batchSize = currReq.req_batchSize;

        trace("{0:s} popped a request of batch size {1:d}. In queue size is {2:d}.", msvc_name, currReq_batchSize, msvc_InQueue.at(0)->size());

        msvc_onBufferBatchSize++;
        // Resize the incoming request image the padd with the grey color
        // The resize image will be copied into a reserved buffer


        prevData.emplace_back(currReq.req_data[0]);

        trace("{0:s} resizing a frame of [{1:d}, {2:d}] -> [{3:d}, {4:d}]",
            msvc_name,
            currReq.req_data[0].data.rows,
            currReq.req_data[0].data.cols,
            (this->msvc_outReqShape.at(0))[0][1],
            (this->msvc_outReqShape.at(0))[0][2]
        );
        data.data = resizePadRightBottom(
            currReq.req_data[0].data,
            (this->msvc_outReqShape.at(0))[0][1],
            (this->msvc_outReqShape.at(0))[0][2],
            cv::Scalar(128, 128, 128),
            preProcStream,
            msvc_imgType,
            msvc_colorCvtType,
            msvc_resizeInterpolType
        );

        data.data = cvtHWCToCHW(data.data, preProcStream, msvc_imgType);

        data.data = normalize(data.data, preProcStream, msvc_subVals, msvc_divVals, msvc_imgNormScale);

        trace("{0:s} finished resizing a frame", msvc_name);
        data.shape = RequestDataShapeType({3, (this->msvc_outReqShape.at(0))[0][1], (this->msvc_outReqShape.at(0))[0][2]});
        bufferData.emplace_back(data);
        trace("{0:s} put an image into buffer. Current batch size is {1:d} ", msvc_name, msvc_onBufferBatchSize);


        // Set the ideal batch size for this microservice using the signal from the receiver.
        // Only used during profiling time.
        msvc_idealBatchSize = getNumberAtIndex(currReq.req_travelPath[0], 0);

        outReq_slo.emplace_back(currReq.req_e2eSLOLatency[0]);
        outReq_path.emplace_back(currReq.req_travelPath[0]);


        timeNow = std::chrono::high_resolution_clock::now();

        // Add the whole time vector of currReq to outReq
        outReq_genTime.insert(outReq_genTime.end(), currReq.req_origGenTime.begin(), currReq.req_origGenTime.end());
        outReq_genTime.emplace_back(timeNow);

        // First we need to decide if this is an appropriate time to batch the buffered data or if we can wait a little more.
        // Waiting more means there is a higher chance the earliest request in the buffer will be late eventually.
        if (this->isTimeToBatch()) { // If true, copy the buffer data into the out queue

            // Moment of batching
            timeNow =  std::chrono::high_resolution_clock::now();

            /**
             * @brief At the moment of batching we stick this time stamp into each request in the batch.
             * This lets us know how much each individual request has to wait and how much is the batched inference
             * time exactly.
             * 
             */
            uint8_t numTimeStampPerReq = (uint8_t)(outReq_genTime.size() / msvc_onBufferBatchSize);
            uint16_t insertPos = numTimeStampPerReq;
            while (insertPos < outReq_genTime.size()) {
                outReq_genTime.insert(outReq_genTime.begin() + insertPos, timeNow);
                insertPos += numTimeStampPerReq + 1;
            }
            if (insertPos == outReq_genTime.size()) {
                outReq_genTime.push_back(timeNow);
            }
            outReq = {
                outReq_genTime,
                outReq_slo,
                outReq_path,
                msvc_onBufferBatchSize,
                bufferData,
                prevData
            };
            trace("{0:s} emplaced a request of batch size {1:d} ", msvc_name, msvc_onBufferBatchSize);
            msvc_OutQueue[0]->emplace(outReq);
            msvc_onBufferBatchSize = 0;
            outReq_genTime.clear();
            outReq_path.clear();
            outReq_slo.clear();
            bufferData.clear();
            prevData.clear();
        }
        trace("{0:s} sleeps for {1:d} millisecond", msvc_name, msvc_interReqTime);
        std::this_thread::sleep_for(std::chrono::milliseconds(this->msvc_interReqTime));
    }
    msvc_logFile.close();
}

/**
 * @brief Simplest function to decide if the requests should be batched and sent to the main processor.
 * 
 * @return true True if its time to batch
 * @return false if otherwise
 */
bool BaseReqBatcher::isTimeToBatch() {
    if (msvc_onBufferBatchSize == this->msvc_idealBatchSize) {
        return true;
    }
    return false;
}

/**
 * @brief Check if the request is still worth being processed.
 * For instance, if the request is already late at the moment of checking, there is no value in processing it anymore.
 * 
 * @return true 
 * @return false 
 */
bool BaseReqBatcher::checkReqEligibility(ClockType currReq_gentime) {
    return true;
}