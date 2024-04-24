#include <arcface.h>

ArcFacePostprocessor::ArcFacePostprocessor(const BaseMicroserviceConfigs &config) : BasePostprocessor(config) {
}

void ArcFacePostprocessor::postProcessing() {
    // The time where the last request was generated.
    ClockType lastReq_genTime;
    // The time where the current incoming request was generated.
    ClockType currReq_genTime;
    // The time where the current incoming request arrives
    ClockType currReq_recvTime;

    // Data package to be sent to and processed at the next microservice
    std::vector<RequestData<LocalGPUReqDataType>> outReqData;
    // List of images carried from the previous microservice here to be cropped from.
    std::vector<RequestData<LocalGPUReqDataType>> imageList;
    // Instance of data to be packed into `outReqData`
    RequestData<LocalGPUReqDataType> reqData;

    // Current incoming equest and request to be sent out to the next
    Request<LocalGPUReqDataType> currReq, outReq;

    // Batch size of current request
    BatchSizeType currReq_batchSize;

    // Shape of cropped bounding boxes
    RequestShapeType bboxShape;

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
        currReq_recvTime = std::chrono::high_resolution_clock::now();
        if (this->msvc_inReqCount > 1) {
            this->updateReqRate(currReq_genTime);
        }
        currReq_batchSize = currReq.req_batchSize;

        /**
         * @brief Each request to the postprocessing microservice of ArcFace contains the buffers which are results of TRT inference 
         * The buffer is of size (batch, 512) carrying embedding features of each input face.
         * The embedding feature will be used to recognize the face.
         */

        
        // Let the poor boy sleep for some time before he has to wake up and do postprocessing again.
        std::this_thread::sleep_for(std::chrono::milliseconds(this->msvc_interReqTime));
    }


}