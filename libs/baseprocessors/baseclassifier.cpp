#include "baseprocessor.h"

using namespace spdlog;

BaseClassifierConfigs BaseClassifier::loadConfigsFromJson(const json &jsonConfigs) {
    BaseClassifierConfigs configs;
    return configs;
}

void BaseClassifier::loadConfigs(const json &jsonConfigs, bool isConstructing) {
    if (!isConstructing) { // If the microservice is being reloaded
        Microservice::loadConfigs(jsonConfigs, isConstructing);
    }
    BaseClassifierConfigs configs = loadConfigsFromJson(jsonConfigs);
    msvc_numClasses = msvc_dataShape[0][0];
}

BaseClassifier::BaseClassifier(const json &jsonConfigs) : Microservice(jsonConfigs) {
    loadConfigs(jsonConfigs, true);
    info("{0:s} is created.", msvc_name); 
}

inline uint16_t maxIndex(float* arr, size_t size) {
    float* max_ptr = std::max_element(arr, arr + size);
    return max_ptr - arr;
}

void BaseClassifier::classify() {
    msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);

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
    info("{0:s} STARTS.", msvc_name); 


    cudaStream_t postProcStream;

    // TODO: remove potentially unused variable
    NumQueuesType queueIndex;

    size_t bufferSize;
    RequestDataShapeType shape;

    float *predictedProbs;
    // TODO: remove potentially unused variables
    uint16_t *predictedClass;

    while (true) {
        // Allowing this thread to naturally come to an end
        if (this->STOP_THREADS) {
            info("{0:s} STOPS.", msvc_name);
            break;
        }
        else if (this->PAUSE_THREADS) {
            if (RELOADING) {
                setDevice();
                checkCudaErrorCode(cudaStreamCreate(&postProcStream), __func__);
                
                predictedProbs = new float[msvc_idealBatchSize * msvc_numClasses];
                predictedClass = new uint16_t[msvc_idealBatchSize];
                info("{0:s} is (RE)LOADED.", msvc_name);
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
        if (this->msvc_inReqCount > 1) {
            this->updateReqRate(currReq_genTime);
        }

        currReq_batchSize = currReq.req_batchSize;
        trace("{0:s} popped a request of batch size {1:d}", msvc_name, currReq_batchSize);

        currReq_data = currReq.req_data;

        bufferSize = this->msvc_modelDataType * (size_t)currReq_batchSize;
        shape = currReq_data[0].shape;
        for (uint8_t j = 0; j < shape.size(); ++j) {
            bufferSize *= shape[j];
        }
        checkCudaErrorCode(cudaMemcpyAsync(
            (void *) predictedProbs,
            currReq_data[0].data.cudaPtr(),
            bufferSize,
            cudaMemcpyDeviceToHost,
            postProcStream
        ), __func__);

        cudaStreamSynchronize(postProcStream);

        for (uint8_t i = 0; i < currReq_batchSize; ++i) {
            predictedClass[i] = maxIndex(predictedProbs + i * msvc_numClasses, msvc_numClasses);
        }

        trace("{0:s} sleeps for {1:d} millisecond", msvc_name, msvc_interReqTime);
        std::this_thread::sleep_for(std::chrono::milliseconds(this->msvc_interReqTime));

    }
    checkCudaErrorCode(cudaStreamDestroy(postProcStream), __func__);
    msvc_logFile.close();
}