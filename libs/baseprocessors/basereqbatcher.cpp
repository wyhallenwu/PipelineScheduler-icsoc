#include "baseprocessor.h"

using namespace spdlog;

/**
 * @brief Construct a new Base Preprocessor that inherites the LocalGPUDataMicroservice given the `InType`
 * 
 * @param configs 
 */
BaseReqBatcher::BaseReqBatcher(const BaseMicroserviceConfigs &configs) : Microservice(configs){
    readConfigsFromJson(configs.msvc_appLvlConfigs);
    info("{0:s} is created.", msvc_name); 
}

void BaseReqBatcher::batchRequests() {
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

        Stopwatch stopwatch;
        stopwatch.start();


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

        stopwatch.stop();
        std::cout << "Time taken to preprocess a req is " << stopwatch.elapsed_seconds() << std::endl;
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

void BaseReqBatcher::readConfigsFromJson(std::string cfgPath) {
    spdlog::trace("{0:s} attempts to parse Config from json file.", __func__);
    std::ifstream file(cfgPath);
    json j = json::parse(file);

    j.at("msvc_imgType").get_to(msvc_imgType);
    j.at("msvc_colorCvtType").get_to(msvc_colorCvtType);
    j.at("msvc_resizeInterpolType").get_to(msvc_resizeInterpolType);
    std::string normVal;
    j.at("msvc_imgNormScale").get_to(normVal);
    msvc_imgNormScale = fractionToFloat(normVal);

    // Assuming msvc_subVals and msvc_divVals are std::vector<double>
    j.at("msvc_subVals").get_to(msvc_subVals);
    j.at("msvc_divVals").get_to(msvc_divVals);

    spdlog::trace("{0:s} finished parsing Config from file.", __func__);
}