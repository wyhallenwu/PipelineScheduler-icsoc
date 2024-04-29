#include "baseprocessor.h"

using namespace spdlog;

BaseBBoxCropperVerifierConfigs BaseBBoxCropperVerifier::loadConfigsFromJson(const json &jsonConfigs) {
    BaseBBoxCropperVerifierConfigs configs;
    return configs;
}

void BaseBBoxCropperVerifier::loadConfigs(const json &jsonConfigs, bool isConstructing) {
    if (!isConstructing) { // If the microservice is being reloaded
        Microservice::loadConfigs(jsonConfigs, isConstructing);
    }
    BaseBBoxCropperVerifierConfigs configs = loadConfigsFromJson(jsonConfigs);

}

BaseBBoxCropperVerifier::BaseBBoxCropperVerifier(const json &jsonConfigs) : Microservice(jsonConfigs) {
    loadConfigs(jsonConfigs, true);
    info("{0:s} is created.", msvc_name); 
}

void BaseBBoxCropperVerifier::cropping() {
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

    // Height and width of the image used for inference
    int orig_h, orig_w, infer_h, infer_w;

    /**
     * @brief Each request to the cropping microservice of YOLOv5 contains the buffers which are results of TRT inference 
     * as well as the images from which bounding boxes will be cropped.
     * `The buffers` are a vector of 4 small raw memory (float) buffers (since using `BatchedNMSPlugin` provided by TRT), which are
     * 1/ `num_detections` (Batch, 1): number of objects detected in this frame
     * 2/ `nmsed_boxes` (Batch, TopK, 4): the boxes remaining after nms is performed.
     * 3/ `nmsed_scores` (Batch, TopK): the scores of these boxes.
     * 4/ `nmsed_classes` (Batch, TopK):
     * We need to bring these buffers to CPU in order to process them.
     */

    uint16_t maxNumDets;
    
    int32_t *num_detections;
    float *nmsed_boxes;
    float *nmsed_scores;
    float *nmsed_classes;

    std::vector<float *> ptrList;

    size_t bufferSize;

    // class of the bounding box cropped from one the images in the image list
    int16_t bboxClass;
    // The index of the queue we are going to put data on based on the value of `bboxClass`
    NumQueuesType queueIndex;

    // To whole the shape of data sent from the inferencer
    RequestDataShapeType shape;

    while (true) {
        // Allowing this thread to naturally come to an end
        if (this->STOP_THREADS) {
            info("{0:s} STOPS.", msvc_name);
            break;
        }
        else if (this->PAUSE_THREADS) {
            if (RELOADING){

                setDevice();
                checkCudaErrorCode(cudaStreamCreate(&postProcStream), __func__);
                
                maxNumDets = msvc_dataShape[2][0];

                delete num_detections;
                delete nmsed_boxes;
                delete nmsed_scores;
                delete nmsed_classes;

                num_detections = new int32_t[msvc_idealBatchSize];
                nmsed_boxes = new float[msvc_idealBatchSize * maxNumDets * 4];
                nmsed_scores = new float[msvc_idealBatchSize * maxNumDets];
                nmsed_classes = new float[msvc_idealBatchSize * maxNumDets];

                ptrList = {nmsed_boxes, nmsed_scores, nmsed_classes};

                outReqData.clear();
                singleImageBBoxList.clear();

                RELOADING = false;
                READY = true;
                info("{0:s} is (RE)LOADED.", msvc_name);
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

        for (std::size_t i = 0; i < (currReq_data.size() - 1); ++i) {
            bufferSize = this->msvc_modelDataType * (size_t)currReq_batchSize;
            RequestDataShapeType shape = currReq_data[i].shape;
            for (uint8_t j = 0; j < shape.size(); ++j) {
                bufferSize *= shape[j];
            }
            if (i == 0) {
                checkCudaErrorCode(cudaMemcpyAsync(
                    (void *) num_detections,
                    currReq_data[i].data.cudaPtr(),
                    bufferSize,
                    cudaMemcpyDeviceToHost,
                    postProcStream
                ), __func__);
            } else {
                checkCudaErrorCode(cudaMemcpyAsync(
                    (void *) ptrList[i - 1],
                    currReq_data[i].data.cudaPtr(),
                    bufferSize,
                    cudaMemcpyDeviceToHost,
                    postProcStream
                ), __func__);
            }
        }

        checkCudaErrorCode(cudaStreamSynchronize(postProcStream), __func__);
        trace("{0:s} unloaded 4 buffers to CPU {1:d}", msvc_name, currReq_batchSize);

        // List of images to be cropped from
        imageList = currReq.upstreamReq_data; 

        // Doing post processing for the whole batch
        for (BatchSizeType i = 0; i < currReq_batchSize; ++i) {


            currReq_genTime = currReq.req_origGenTime[i];
            currReq_path = currReq.req_travelPath[i];

            // If there is no object in frame, we don't have to do nothing.
            int numDetsInFrame = (int)num_detections[i];

            // Otherwise, we need to do some cropping.
            orig_h = imageList[i].shape[1];
            orig_w = imageList[i].shape[2];

            // crop(imageList[i].data, orig_h, orig_w, infer_h, infer_w, numDetsInFrame, nmsed_boxes[i][0], singleImageBBoxList);
            trace("{0:s} cropped {1:d} bboxes in image {2:d}", msvc_name, numDetsInFrame, i);

            outReq = {
                {currReq_genTime},
                currReq.req_e2eSLOLatency,
                {currReq_path},
                1,
                currReq.upstreamReq_data, //req_data
            };
            msvc_OutQueue.at(0)->emplace(outReq);
            trace("{0:s} emplaced an image to queue {2:d}.", msvc_name, bboxClass, queueIndex);
            // // After cropping is done for this image in the batch, the image's cuda memory can be freed.
            // checkCudaErrorCode(cudaFree(imageList[i].data.cudaPtr()));
            // Clearing out data of the vector

            outReqData.clear();
            singleImageBBoxList.clear();
        }
        // // Free all the output buffers of trtengine after cropping is done.
        // for (size_t i = 0; i < currReq_data.size(); i++) {
        //     checkCudaErrorCode(cudaFree(currReq_data.at(i).data.cudaPtr()));
        // }


        
        trace("{0:s} sleeps for {1:d} millisecond", msvc_name, msvc_interReqTime);
        std::this_thread::sleep_for(std::chrono::milliseconds(this->msvc_interReqTime));
        // Synchronize the cuda stream
    }


    checkCudaErrorCode(cudaStreamDestroy(postProcStream), __func__);
    msvc_logFile.close();
}