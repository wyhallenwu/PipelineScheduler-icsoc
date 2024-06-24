#include "baseprocessor.h"

using namespace spdlog;


BaseKPointExtractorConfigs BaseKPointExtractor::loadConfigsFromJson(const json &config) {
    BaseKPointExtractorConfigs configs;
    return configs;
}

void BaseKPointExtractor::loadConfigs(const json &jsonConfigs, bool isConstructing) {
    if (!isConstructing) { // If the microservice is being reloaded
        BasePostprocessor::loadConfigs(jsonConfigs, isConstructing);
    }
    BaseKPointExtractorConfigs configs = loadConfigsFromJson(jsonConfigs);
}


BaseKPointExtractor::BaseKPointExtractor(const json &jsonConfigs) : BasePostprocessor(jsonConfigs) {
    loadConfigs(jsonConfigs, true);
    spdlog::get("container_agent")->info("{0:s} is created.", msvc_name); 
}

inline uint16_t maxIndex(float* arr, size_t size) {
    float* max_ptr = std::max_element(arr, arr + size);
    return max_ptr - arr;
}
void BaseKPointExtractor::extractor() {

    // The time where the last request was generated.
    ClockType lastReq_genTime;
    // The time where the current incoming request was generated.
    ClockType currReq_genTime;
    // The time where the current incoming request arrives
    ClockType currReq_recvTime;
    // Path
    std::string currReq_path;

    // Data package to be sent to and processed at the next microservice
    std::vector<RequestData<LocalGPUReqDataType>> outReqData;
    // List of images carried from the previous microservice here to be cropped from.
    std::vector<RequestData<LocalGPUReqDataType>> imageList;
    // Instance of data to be packed into `outReqData`
    RequestData<LocalGPUReqDataType> reqData;

    std::vector<RequestData<LocalGPUReqDataType>> currReq_data;

    // List of bounding boxes cropped from one single image
    std::vector<cv::cuda::GpuMat> singleImageBBoxList;

    // Current incoming equest and request to be sent out to the next
    Request<LocalGPUReqDataType> currReq, outReq;

    // Batch size of current request
    BatchSizeType currReq_batchSize;

    // Shape of cropped bounding boxes
    RequestDataShapeType bboxShape;
    spdlog::get("container_agent")->info("{0:s} STARTS.", msvc_name); 


    cudaStream_t postProcStream;

    NumQueuesType queueIndex;

    size_t bufferSize;
    RequestDataShapeType shape;


    float *keyPoints;
    // uint16_t predictedClass[msvc_idealBatchSize];

    auto timeNow = std::chrono::high_resolution_clock::now();

    while (true) {
        // Allowing this thread to naturally come to an end
        if (STOP_THREADS) {
            spdlog::get("container_agent")->info("{0:s} STOPS.", msvc_name);
            break;
        }
        else if (PAUSE_THREADS) {
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
                checkCudaErrorCode(cudaStreamCreate(&postProcStream), __func__);

                BatchSizeType batchSize;
                if (msvc_allocationMode == AllocationMode::Conservative) {
                    batchSize = msvc_idealBatchSize;
                } else if (msvc_allocationMode == AllocationMode::Aggressive) {
                    batchSize = msvc_maxBatchSize;
                }
                keyPoints = new float[batchSize * msvc_dataShape[0][0] * msvc_dataShape[0][1] * msvc_dataShape[0][2]];

                spdlog::get("container_agent")->info("{0:s} is (RE)LOADED.", msvc_name);
                RELOADING = false;
                READY = true;
            }
            //info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }

        // Processing the next incoming request
        currReq = msvc_InQueue.at(0)->pop2();
        // Meaning the the timeout in pop() has been reached and no request was actually popped
        if (strcmp(currReq.req_travelPath[0].c_str(), "empty") == 0) {
            continue;
        /**
         * @brief ONLY IN PROFILING MODE
         * Check if the profiling is to be stopped, if true, then send a signal to the downstream microservice to stop profiling
         */
        } else if (strcmp(currReq.req_travelPath[0].c_str(), "STOP_PROFILING") == 0) {
            STOP_THREADS = true;
            msvc_OutQueue[0]->emplace(currReq);
            continue;
        }
        msvc_inReqCount++;

        // The generated time of this incoming request will be used to determine the rate with which the microservice should
        // check its incoming queue.
        currReq_recvTime = std::chrono::high_resolution_clock::now();
        if (msvc_inReqCount > 1) {
            updateReqRate(currReq_genTime);
        }

        currReq_batchSize = currReq.req_batchSize;
        spdlog::get("container_agent")->trace("{0:s} popped a request of batch size {1:d}", msvc_name, currReq_batchSize);

        currReq_data = currReq.req_data;

        bufferSize = msvc_modelDataType * (size_t)currReq_batchSize;
        shape = currReq_data[0].shape;
        for (uint8_t j = 0; j < shape.size(); ++j) {
            bufferSize *= shape[j];
        }
        checkCudaErrorCode(cudaMemcpyAsync(
            (void *) keyPoints,
            currReq_data[0].data.cudaPtr(),
            bufferSize,
            cudaMemcpyDeviceToHost,
            postProcStream
        ), __func__);

        cudaStreamSynchronize(postProcStream);

        for (BatchSizeType i = 0; i < currReq.req_batchSize; i++) {
            // We consider this when the request was received by the postprocessor
            currReq.req_origGenTime[i].emplace_back(std::chrono::high_resolution_clock::now());

            uint32_t totalInMem = currReq.upstreamReq_data[i].data.rows * currReq.upstreamReq_data[i].data.cols * currReq.upstreamReq_data[i].data.channels() * CV_ELEM_SIZE1(currReq.upstreamReq_data[i].data.type());
            currReq.req_travelPath[i] += "|1|1|" + std::to_string(totalInMem) + "]";

            if (msvc_activeOutQueueIndex.at(queueIndex) == 1) { //Local CPU
                cv::Mat out(currReq.upstreamReq_data[i].data.size(), currReq.upstreamReq_data[i].data.type());
                checkCudaErrorCode(cudaMemcpyAsync(
                    out.ptr(),
                    currReq.upstreamReq_data[i].data.cudaPtr(),
                    currReq.upstreamReq_data[i].data.rows * currReq.upstreamReq_data[i].data.cols * currReq.upstreamReq_data[i].data.channels() * CV_ELEM_SIZE1(currReq.upstreamReq_data[i].data.type()),
                    cudaMemcpyDeviceToHost,
                    postProcStream
                ), __func__);
                checkCudaErrorCode(cudaStreamSynchronize(postProcStream), __func__);
                msvc_OutQueue.at(0)->emplace(
                    Request<LocalCPUReqDataType>{
                        {{currReq.req_origGenTime[i].front(), std::chrono::high_resolution_clock::now()}},
                        {currReq.req_e2eSLOLatency[i]},
                        {currReq.req_travelPath[i]},
                        1,
                        {
                            {currReq.upstreamReq_data[i].shape, out}
                        } //req_data
                    }
                );
                spdlog::get("container_agent")->trace("{0:s} emplaced an image to CPU queue.", msvc_name);
            } else {
                msvc_OutQueue.at(0)->emplace(
                    Request<LocalGPUReqDataType>{
                        {{currReq.req_origGenTime[i].front(), std::chrono::high_resolution_clock::now()}},
                        {currReq.req_e2eSLOLatency[i]},
                        {currReq.req_travelPath[i]},
                        1,
                        {
                            currReq.upstreamReq_data[i]
                        }
                    }
                );
                spdlog::get("container_agent")->trace("{0:s} emplaced an image to GPU queue.", msvc_name);
            }

            uint32_t totalOutMem = totalInMem;

            /**
             * @brief There are 7 important timestamps to be recorded:
             * 1. When the request was generated
             * 2. When the request was received by the batcher
             * 3. When the request was done preprocessing by the batcher
             * 4. When the request, along with all others in the batch, was batched together and sent to the inferencer
             * 5. When the batch inferencer was completed by the inferencer 
             * 6. When the request was received by the postprocessor
             * 7. When each request was completed by the postprocessor
             */
            // If the number of warmup batches has been passed, we start to record the latency
            if (msvc_batchCount > msvc_numWarmupBatches) {
                currReq.req_origGenTime[i].emplace_back(std::chrono::high_resolution_clock::now());
                // TODO: Add the request number
                msvc_processRecords.addRecord(currReq.req_origGenTime[i], currReq_batchSize, totalInMem, totalOutMem, 0, getOriginStream(currReq.req_travelPath[i]));
            }
        }

        
        msvc_batchCount++;

        spdlog::get("container_agent")->trace("{0:s} sleeps for {1:d} millisecond", msvc_name, msvc_interReqTime);
        std::this_thread::sleep_for(std::chrono::milliseconds(msvc_interReqTime));

    }
    checkCudaErrorCode(cudaStreamDestroy(postProcStream), __func__);
    msvc_logFile.close();
}

void BaseKPointExtractor::extractorProfiling() {

    // The time where the last request was generated.
    ClockType lastReq_genTime;
    // The time where the current incoming request was generated.
    ClockType currReq_genTime;
    // The time where the current incoming request arrives
    ClockType currReq_recvTime;
    // Path
    std::string currReq_path;

    // Data package to be sent to and processed at the next microservice
    std::vector<RequestData<LocalGPUReqDataType>> outReqData;
    // List of images carried from the previous microservice here to be cropped from.
    std::vector<RequestData<LocalGPUReqDataType>> imageList;
    // Instance of data to be packed into `outReqData`
    RequestData<LocalGPUReqDataType> reqData;

    std::vector<RequestData<LocalGPUReqDataType>> currReq_data;

    // List of bounding boxes cropped from one single image
    std::vector<cv::cuda::GpuMat> singleImageBBoxList;

    // Current incoming equest and request to be sent out to the next
    Request<LocalGPUReqDataType> currReq, outReq;

    // Batch size of current request
    BatchSizeType currReq_batchSize;

    // Shape of cropped bounding boxes
    RequestDataShapeType bboxShape;
    spdlog::get("container_agent")->info("{0:s} STARTS.", msvc_name); 


    cudaStream_t postProcStream;

    NumQueuesType queueIndex;

    size_t bufferSize;
    RequestDataShapeType shape;


    float *keyPoints;
    // uint16_t predictedClass[msvc_idealBatchSize];

    auto timeNow = std::chrono::high_resolution_clock::now();

    while (true) {
        // Allowing this thread to naturally come to an end
        if (STOP_THREADS) {
            spdlog::get("container_agent")->info("{0:s} STOPS.", msvc_name);
            break;
        }
        else if (PAUSE_THREADS) {
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
                checkCudaErrorCode(cudaStreamCreate(&postProcStream), __func__);

                keyPoints = new float[msvc_idealBatchSize * msvc_dataShape[0][0] * msvc_dataShape[0][1] * msvc_dataShape[0][2]];

                spdlog::get("container_agent")->info("{0:s} is (RE)LOADED.", msvc_name);
                RELOADING = false;
                READY = true;
            }
            //info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }

        // Processing the next incoming request
        currReq = msvc_InQueue.at(0)->pop2();
        // Meaning the the timeout in pop() has been reached and no request was actually popped
        if (strcmp(currReq.req_travelPath[0].c_str(), "empty") == 0) {
            continue;
        }
        msvc_inReqCount++;

        // The generated time of this incoming request will be used to determine the rate with which the microservice should
        // check its incoming queue.
        currReq_recvTime = std::chrono::high_resolution_clock::now();
        if (msvc_inReqCount > 1) {
            updateReqRate(currReq_genTime);
        }

        currReq_batchSize = currReq.req_batchSize;
        spdlog::get("container_agent")->trace("{0:s} popped a request of batch size {1:d}", msvc_name, currReq_batchSize);

        currReq_data = currReq.req_data;

        bufferSize = msvc_modelDataType * (size_t)currReq_batchSize;
        shape = currReq_data[0].shape;
        for (uint8_t j = 0; j < shape.size(); ++j) {
            bufferSize *= shape[j];
        }
        checkCudaErrorCode(cudaMemcpyAsync(
            (void *) keyPoints,
            currReq_data[0].data.cudaPtr(),
            bufferSize,
            cudaMemcpyDeviceToHost,
            postProcStream
        ), __func__);

        cudaStreamSynchronize(postProcStream);

        for (BatchSizeType i = 0; i < currReq.req_batchSize; i++) {
            timeNow = std::chrono::high_resolution_clock::now();
            currReq.req_origGenTime[i].emplace_back(timeNow);

            // TODO: DRAW KEYPOINTS
        }

        msvc_OutQueue[0]->emplace(
            Request<LocalCPUReqDataType>{
                currReq.req_origGenTime,
                {},
                currReq.req_travelPath,
                currReq.req_batchSize,
                {}
            }
        );

        spdlog::get("container_agent")->trace("{0:s} sleeps for {1:d} millisecond", msvc_name, msvc_interReqTime);
        std::this_thread::sleep_for(std::chrono::milliseconds(msvc_interReqTime));

    }
    checkCudaErrorCode(cudaStreamDestroy(postProcStream), __func__);
    msvc_logFile.close();
}