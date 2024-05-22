#include "baseprocessor.h"

using namespace spdlog;

inline cv::Scalar vectorToScalar(const std::vector<float>& vec) {
    // Ensure the vector has exactly 4 elements
    if (vec.size() == 1) {
        return cv::Scalar(vec[0]);
    } else if (vec.size() == 3) {
        return cv::Scalar(vec[0], vec[1], vec[2]);
    } else {
        throw std::runtime_error("Unsupported number of channels");
    }
}

/**
 * @brief normalize the input data by subtracting and dividing values from the original pixesl
 * 
 * @param input the input data
 * @param subVals values to be subtracted
 * @param divVals values to be dividing by
 * @param stream an opencv stream for asynchronous operation on cuda
 */
inline cv::cuda::GpuMat normalize(
    cv::cuda::GpuMat &input,
    cv::cuda::Stream &stream,
    const std::vector<float>& subVals,
    const std::vector<float>& divVals,
    const float normalized_scale
) {
    trace("Going into {0:s}", __func__);
    cv::cuda::GpuMat normalized;
    cv::Scalar subValsScalar = vectorToScalar(subVals);
    cv::Scalar divValsScalar = vectorToScalar(divVals);
    if (input.channels() == 1) {
        input.convertTo(normalized, CV_32FC1, normalized_scale, stream);
        cv::cuda::subtract(normalized, subValsScalar, normalized, cv::noArray(), -1, stream);
        cv::cuda::divide(normalized, divValsScalar, normalized, 1, -1, stream);
    } else if (input.channels() == 3) {
        input.convertTo(normalized, CV_32FC3, normalized_scale, stream);    
        cv::cuda::subtract(normalized, subValsScalar, normalized, cv::noArray(), -1, stream);
        cv::cuda::divide(normalized, divValsScalar, normalized, 1, -1, stream);
    } else {
        throw std::runtime_error("Unsupported number of channels");
    }

    stream.waitForCompletion();
    trace("Finished {0:s}", __func__);

    return normalized;
}

inline cv::cuda::GpuMat cvtHWCToCHW(
    cv::cuda::GpuMat &input,
    cv::cuda::Stream &stream,
    uint8_t IMG_TYPE
) {

    trace("Going into {0:s}", __func__);
    uint16_t height = input.rows;
    uint16_t width = input.cols;
    /**
     * @brief TODOs
     * This is the correct way but since we haven't figured out how to carry to image to be cropped
     * it screws things up a bit.
     * cv::cuda::GpuMat transposed(1, height * width, CV_8UC3);
     */
    // cv::cuda::GpuMat transposed(height, width, CV_8UC3);
    cv::cuda::GpuMat transposed(1, height * width, IMG_TYPE);
    std::vector<cv::cuda::GpuMat> channels;
    if (input.channels() == 1) {
        uint8_t IMG_SINGLE_CHANNEL_TYPE = IMG_TYPE;
        channels = {
            cv::cuda::GpuMat(height, width, IMG_SINGLE_CHANNEL_TYPE, &(transposed.ptr()[0]))
        };
    } else if (input.channels() == 3) {
        uint8_t IMG_SINGLE_CHANNEL_TYPE = IMG_TYPE ^ 16;
        size_t channel_mem_width = height * width;
        
        channels = {
            cv::cuda::GpuMat(height, width, IMG_SINGLE_CHANNEL_TYPE, &(transposed.ptr()[0])),
            cv::cuda::GpuMat(height, width, IMG_SINGLE_CHANNEL_TYPE, &(transposed.ptr()[channel_mem_width])),
            cv::cuda::GpuMat(height, width, IMG_SINGLE_CHANNEL_TYPE, &(transposed.ptr()[channel_mem_width * 2]))
        };
    } else {
        throw std::runtime_error("Unsupported number of channels");
    }
    cv::cuda::split(input, channels, stream);

    stream.waitForCompletion();    

    trace("Finished {0:s}", __func__);

    return transposed;
}

/**
 * @brief resize the input data without changing the aspect ratio and pad the empty area with a designated color
 * 
 * @param input the input data
 * @param height the expected height after processing
 * @param width  the expect width after processing
 * @param bgcolor color to pad the empty area with
 * @return cv::cuda::GpuMat 
 */
inline cv::cuda::GpuMat resizePadRightBottom(
    cv::cuda::GpuMat &input,
    const size_t height,
    const size_t width,
    const std::vector<float> &bgcolor,
    cv::cuda::Stream &stream,
    uint8_t IMG_TYPE,
    uint8_t COLOR_CVT_TYPE,
    uint8_t RESIZE_INTERPOL_TYPE

) {
    trace("Going into {0:s}", __func__);

    uint16_t TARGET_IMG_TYPE;

    // If the image is grayscale, then the target image type should be 0
    if (GRAYSCALE_CONVERSION_CODES.count(COLOR_CVT_TYPE)) {
        TARGET_IMG_TYPE = 0;
    } else {
        TARGET_IMG_TYPE = IMG_TYPE;
    }
    cv::cuda::GpuMat color_cvt_image(input.rows, input.cols, TARGET_IMG_TYPE);
    cv::cuda::cvtColor(input, color_cvt_image, COLOR_CVT_TYPE, 0, stream);

    float r = std::min(width / (input.cols * 1.0), height / (input.rows * 1.0));
    int unpad_w = r * input.cols;
    int unpad_h = r * input.rows;
    //Create a new GPU Mat 
    cv::cuda::GpuMat resized(unpad_h, unpad_w, TARGET_IMG_TYPE);
    cv::cuda::resize(color_cvt_image, resized, resized.size(), 0, 0, RESIZE_INTERPOL_TYPE, stream);
    cv::cuda::GpuMat out(height, width, TARGET_IMG_TYPE, vectorToScalar(bgcolor));
    // Creating an opencv stream for asynchronous operation on cuda
    resized.copyTo(out(cv::Rect(0, 0, resized.cols, resized.rows)), stream);

    stream.waitForCompletion();
    trace("Finished {0:s}", __func__);

    return out;
}

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
    configs.msvc_imgNormScale = fractionToFloat(normVal);
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
        Microservice::loadConfigs(jsonConfigs, isConstructing);
    }

    BaseReqBatcherConfigs configs = loadConfigsFromJson(jsonConfigs);

    msvc_imgType = configs.msvc_imgType;
    msvc_colorCvtType = configs.msvc_colorCvtType;
    msvc_resizeInterpolType = configs.msvc_resizeInterpolType;
    msvc_imgNormScale = configs.msvc_imgNormScale;
    msvc_subVals = configs.msvc_subVals;
    msvc_divVals = configs.msvc_divVals;
    msvc_arrivalRecords.setKeepLength(jsonConfigs.at("cont_metricsScrapeIntervalMillisec"));
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
    // The time where the last request was generated.
    ClockType lastReq_genTime;
    // The time where the current incoming request was generated.
    ClockType currReq_genTime;
    // The time where the current incoming request arrives
    ClockType currReq_recvTime;

    // Batch reqs' gen time
    BatchTimeType outBatch_genTime;

    // Batch reqs' slos
    RequestSLOType outReq_slo;

    // Batch reqs' paths
    RequestPathType outBatch_path;

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

    auto timeNow = std::chrono::high_resolution_clock::now();

    // Batch size of current request
    BatchSizeType currReq_batchSize;
    info("{0:s} STARTS.", msvc_name); 
    cv::cuda::Stream *preProcStream;
    while (true) {
        // Allowing this thread to naturally come to an end
        if (this->STOP_THREADS) {
            info("{0:s} STOPS.", msvc_name);
            break;
        }
        else if (this->PAUSE_THREADS) {
            if (RELOADING) {
                /**
                 * @brief Opening a new log file
                 * During runtime: log file should come with a new timestamp everytime the microservice is reloaded
                 * 
                 */

                if (msvc_logFile.is_open()) {
                    msvc_logFile.close();
                }
                msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);

                setDevice();
                preProcStream = new cv::cuda::Stream();

                outBatch_genTime.clear();
                outBatch_path.clear();
                outReq_slo.clear();
                bufferData.clear();
                prevData.clear();

                info("{0:s} is (RE)LOADED.", msvc_name);
                RELOADING = false;
                READY = true;
            }
            //info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }
        // Processing the next incoming request
        if (msvc_InQueue.at(0)->getActiveQueueIndex() != msvc_activeInQueueIndex.at(0)) {
            if (msvc_InQueue.at(0)->size(msvc_activeInQueueIndex.at(0)) == 0) {
                msvc_activeInQueueIndex.at(0) = msvc_InQueue.at(0)->getActiveQueueIndex();
                trace("{0:s} Set current active queue index to {1:d}.", msvc_name, msvc_activeInQueueIndex.at(0));
            }
            trace("{0:s} Current active queue index {1:d}.", msvc_name, msvc_activeInQueueIndex.at(0));
        }
        if (msvc_activeInQueueIndex.at(0) == 1) {
            currCPUReq = msvc_InQueue.at(0)->pop1();
            if (!validateRequest<LocalCPUReqDataType>(currCPUReq)) {
                continue;
            }
            currReq = uploadReq(currCPUReq);
        } else if (msvc_activeInQueueIndex.at(0) == 2) {
            currReq = msvc_InQueue.at(0)->pop2();
            if (!validateRequest<LocalGPUReqDataType>(currReq)) {
                continue;
            }
        }
        
        msvc_inReqCount++;

        // Keeping record of the arrival requests
        msvc_arrivalRecords.addRecord(currReq.req_origGenTime[0], msvc_inReqCount);

        // The generated time of this incoming request will be used to determine the rate with which the microservice should
        // check its incoming queue.
        currReq_genTime = currReq.req_origGenTime[0][0];
        if (this->msvc_inReqCount > 1) {
            this->updateReqRate(currReq_genTime);
        }

        currReq_batchSize = currReq.req_batchSize;

        outReq_slo.emplace_back(currReq.req_e2eSLOLatency[0]);
        outBatch_path.emplace_back(currReq.req_travelPath[0] + "[" + msvc_containerName + "_" + std::to_string(msvc_inReqCount) + "]");
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
            {128, 128, 128},
            *preProcStream,
            msvc_imgType,
            msvc_colorCvtType,
            msvc_resizeInterpolType
        );

        data.data = cvtHWCToCHW(data.data, *preProcStream, msvc_imgType);

        data.data = normalize(data.data, *preProcStream, msvc_subVals, msvc_divVals, msvc_imgNormScale);

        trace("{0:s} finished resizing a frame", msvc_name);
        data.shape = RequestDataShapeType({(this->msvc_outReqShape.at(0))[0][1], (this->msvc_outReqShape.at(0))[0][1], (this->msvc_outReqShape.at(0))[0][2]});
        bufferData.emplace_back(data);
        trace("{0:s} put an image into buffer. Current batch size is {1:d} ", msvc_name, msvc_onBufferBatchSize);

        // Consider this the moment the request preprocessed and is waiting to be batched
        timeNow = std::chrono::high_resolution_clock::now();

        // Add the whole time vector of currReq to outReq
        currReq.req_origGenTime[0].emplace_back(timeNow);
        outBatch_genTime.emplace_back(currReq.req_origGenTime[0]);

        // std::cout << "Time taken to preprocess a req is " << stopwatch.elapsed_seconds() << std::endl;
        // cudaFree(currReq.req_data[0].data.cudaPtr());
        // First we need to decide if this is an appropriate time to batch the buffered data or if we can wait a little more.
        // Waiting more means there is a higher chance the earliest request in the buffer will be late eventually.
        if (this->isTimeToBatch()) {
            // If true, copy the buffer data into the out queue
            timeNow = std::chrono::high_resolution_clock::now();

            for (auto& req_genTime : outBatch_genTime) {
                req_genTime.emplace_back(timeNow);
            }

            outReq = {
                outBatch_genTime,
                outReq_slo,
                outBatch_path,
                msvc_onBufferBatchSize,
                bufferData,
                prevData
            };
            trace("{0:s} emplaced a request of batch size {1:d} ", msvc_name, msvc_onBufferBatchSize);
            msvc_OutQueue[0]->emplace(outReq);
            msvc_onBufferBatchSize = 0;
            outBatch_genTime.clear();
            outBatch_path.clear();
            outReq_slo.clear();
            bufferData.clear();
            prevData.clear();
        }
        trace("{0:s} sleeps for {1:d} millisecond", msvc_name, msvc_interReqTime);
        std::this_thread::sleep_for(std::chrono::milliseconds(this->msvc_interReqTime));
    }
    msvc_logFile.close();
}

template <typename T>
bool BaseReqBatcher::validateRequest(Request<T> &req) {
    // Meaning the the timeout in pop() has been reached and no request was actually popped
    if (strcmp(req.req_travelPath[0].c_str(), "empty") == 0) {
        return false;
    }

    // We need to check if the next request is worth processing.
    // If it's too late, then we can drop and stop processing this request.
    return this->checkReqEligibility(req.req_origGenTime[0]);
}

void BaseReqBatcher::batchRequestsProfiling() {
    msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);
    // The time where the last request was generated.
    ClockType lastReq_genTime;
    // The time where the current incoming request was generated.
    ClockType currReq_genTime;
    // The time where the current incoming request arrives

    // Batch reqs' gen time
    BatchTimeType outBatch_genTime;

    // Batch reqs' slos
    RequestSLOType outBatch_slo;

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
    cv::cuda::Stream *preProcStream;

    auto timeNow = std::chrono::high_resolution_clock::now();
    while (true) {
        // Allowing this thread to naturally come to an end
        if (this->STOP_THREADS) {
            info("{0:s} STOPS.", msvc_name);
            break;
        }
        else if (this->PAUSE_THREADS) {
            if (this->RELOADING) {
                /**
                 * @brief Opening a new log file
                 * During runtime: log file should come with a new timestamp everytime the microservice is reloaded
                 * 
                 */

                if (msvc_logFile.is_open()) {
                    msvc_logFile.close();
                }
                msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);

                // delete preProcStream;
                setDevice();
                preProcStream = new cv::cuda::Stream();

                outBatch_genTime.clear();
                outReq_path.clear();
                outBatch_slo.clear();
                bufferData.clear();
                prevData.clear();

                this->RELOADING = false;
                this->READY = true;
                info("{0:s} is (RE)LOADED.", msvc_name);
            }
            //info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }
        // Processing the next incoming request
        if (msvc_InQueue.at(0)->getActiveQueueIndex() != msvc_activeInQueueIndex.at(0)) {
            if (msvc_InQueue.at(0)->size(msvc_activeInQueueIndex.at(0)) == 0) {
                msvc_activeInQueueIndex.at(0) = msvc_InQueue.at(0)->getActiveQueueIndex();
                // trace("{0:s} Set current active queue index to {1:d}.", msvc_name, msvc_activeInQueueIndex.at(0));
            }
        }
        trace("{0:s} Current active queue index {1:d}.", msvc_name, msvc_activeInQueueIndex.at(0));
        if (msvc_activeInQueueIndex.at(0) == 1) {
            currCPUReq = msvc_InQueue.at(0)->pop1();
            if (!validateRequest<LocalCPUReqDataType>(currCPUReq)) {
                continue;
            }
            currReq = uploadReq(currCPUReq);
        } else if (msvc_activeInQueueIndex.at(0) == 2) {
            currReq = msvc_InQueue.at(0)->pop2();
            if (!validateRequest<LocalGPUReqDataType>(currReq)) {
                continue;
            }
        }

        msvc_inReqCount++;

        // The generated time of this incoming request will be used to determine the rate with which the microservice should
        // check its incoming queue.
        currReq_genTime = currReq.req_origGenTime[0][0];
        if (this->msvc_inReqCount > 1) {
            this->updateReqRate(currReq_genTime);
        }

        currReq_batchSize = currReq.req_batchSize;

        trace("{0:s} popped a request of batch size {1:d}. In queue size is {2:d}.", msvc_name, currReq_batchSize, msvc_InQueue.at(0)->size());

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
            {128, 128, 128},
            *preProcStream,
            msvc_imgType,
            msvc_colorCvtType,
            msvc_resizeInterpolType
        );

        data.data = cvtHWCToCHW(data.data, *preProcStream, msvc_imgType);

        data.data = normalize(data.data, *preProcStream, msvc_subVals, msvc_divVals, msvc_imgNormScale);

        trace("{0:s} finished resizing a frame", msvc_name);
        data.shape = RequestDataShapeType({3, (this->msvc_outReqShape.at(0))[0][1], (this->msvc_outReqShape.at(0))[0][2]});
        bufferData.emplace_back(data);
        trace("{0:s} put an image into buffer. Current batch size is {1:d} ", msvc_name, msvc_onBufferBatchSize);


        /**
         * @brief At the moment of batching we stick this time stamp into each request in the batch.
         * This lets us know how much each individual request has to wait and how much is the batched inference
         * time exactly.
         * 
         */
        outBatch_slo.emplace_back(currReq.req_e2eSLOLatency[0]);
        outReq_path.emplace_back(currReq.req_travelPath[0]);

        // Consider this the moment the request preprocessed and is waiting to be batched
        timeNow = std::chrono::high_resolution_clock::now();

        // Add the whole time vector of currReq to outReq
        currReq.req_origGenTime[0].emplace_back(timeNow); // THIRD_TIMESTAMP
        outBatch_genTime.emplace_back(currReq.req_origGenTime[0]);

        // Set the ideal batch size for this microservice using the signal from the receiver.
        // Only used during profiling time.
        msvc_idealBatchSize = getNumberAtIndex(currReq.req_travelPath[0], 1);

        msvc_onBufferBatchSize++;

        // First we need to decide if this is an appropriate time to batch the buffered data or if we can wait a little more.
        // Waiting more means there is a higher chance the earliest request in the buffer will be late eventually.
        if (this->isTimeToBatch()) { // If true, copy the buffer data into the out queue

            // Moment of batching
            timeNow =  std::chrono::high_resolution_clock::now();

            for (auto& req_genTime : outBatch_genTime) {
                req_genTime.emplace_back(timeNow); //FOURTH_TIMESTAMP
            }

            outReq = {
                outBatch_genTime,
                outBatch_slo,
                outReq_path,
                msvc_onBufferBatchSize,
                bufferData,
                prevData
            };
            trace("{0:s} emplaced a request of batch size {1:d} ", msvc_name, msvc_onBufferBatchSize);
            msvc_OutQueue[0]->emplace(outReq);
            msvc_onBufferBatchSize = 0;
            outBatch_genTime.clear();
            outReq_path.clear();
            outBatch_slo.clear();
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
bool BaseReqBatcher::checkReqEligibility(std::vector<ClockType> &currReq_time) {
    auto now = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(now - currReq_time[0]).count();
    if (diff > this->msvc_svcLevelObjLatency) {
        this->droppedReqCount++;
        return false;
    }
    // `currReq_recvTime` will also be used to measured how much for the req to sit in queue and
    // how long it took for the request to be preprocessed
    currReq_time.emplace_back(now); // SECOND_TIMESTAMP
    return true;
}