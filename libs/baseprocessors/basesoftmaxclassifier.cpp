#include "baseprocessor.h"

using namespace spdlog;

BaseSoftmaxClassifier::BaseSoftmaxClassifier(const BaseMicroserviceConfigs &config) : BaseClassifier(config) {
    msvc_numClasses = config.msvc_dataShape[0][0];
    info("{0:s} is created.", msvc_name); 
}

inline uint16_t maxIndex(float* arr, size_t size) {
    float* max_ptr = std::max_element(arr, arr + size);
    return max_ptr - arr;
}

inline void softmax(const float* logits, float* probabilities, int size) {
    float max_logit = logits[0]; // Initialize max_logit with the first element
    for (int i = 1; i < size; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i]; // Update max_logit if a larger element is found
        }
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum_exp += exp(logits[i] - max_logit);
    }

    for (int i = 0; i < size; ++i) {
        probabilities[i] = exp(logits[i] - max_logit) / sum_exp;
    }
}

void BaseSoftmaxClassifier::classify() {
    msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);

    setDevice();

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
    checkCudaErrorCode(cudaStreamCreate(&postProcStream), __func__);

    NumQueuesType queueIndex;

    size_t bufferSize;
    RequestDataShapeType shape;

    float predictedLogits[msvc_idealBatchSize][msvc_numClasses], predictedProbs[msvc_idealBatchSize][msvc_numClasses];
    uint16_t predictedClass[msvc_idealBatchSize];

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
        currReq = msvc_InQueue.at(0)->pop2();
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
            (void *) predictedLogits[0],
            currReq_data[0].data.cudaPtr(),
            bufferSize,
            cudaMemcpyDeviceToHost,
            postProcStream
        ), __func__);

        cudaStreamSynchronize(postProcStream);

        for (uint8_t i = 0; i < currReq_batchSize; ++i) {
            softmax(predictedLogits[i], predictedProbs[i], msvc_numClasses);
            predictedClass[i] = maxIndex(predictedProbs[i], msvc_numClasses);
        }

        trace("{0:s} sleeps for {1:d} millisecond", msvc_name, msvc_interReqTime);
        std::this_thread::sleep_for(std::chrono::milliseconds(this->msvc_interReqTime));

    }
    checkCudaErrorCode(cudaStreamDestroy(postProcStream), __func__);
    msvc_logFile.close();
}