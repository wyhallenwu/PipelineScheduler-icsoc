#include "baseprocessor.h"

using namespace spdlog;

void BaseBBoxCropper::loadConfigs(const json &jsonConfigs, bool isConstructing) {
    if (!isConstructing) { // If this is not called from the constructor
        Microservice::loadConfigs(jsonConfigs, isConstructing);
    }
}

BaseBBoxCropper::BaseBBoxCropper(const json &jsonConfigs) : Microservice(jsonConfigs) {
    loadConfigs(jsonConfigs, true);
    info("{0:s} is created.", msvc_name); 
}

void BaseBBoxCropper::cropping() {
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

    READY = true;

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

                infer_h = msvc_inferenceShape[0][1];
                infer_w = msvc_inferenceShape[0][2];
                
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
                info("{0:s} is (RE)LOADED.", msvc_name);
            }
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
            if (numDetsInFrame <= 0) {
                outReqData.clear();
                continue;
            }

            // Otherwise, we need to do some cropping.
            orig_h = imageList[i].shape[1];
            orig_w = imageList[i].shape[2];

            crop(imageList[i].data, orig_h, orig_w, infer_h, infer_w, numDetsInFrame, nmsed_boxes + i * maxNumDets * 4, singleImageBBoxList);
            trace("{0:s} cropped {1:d} bboxes in image {2:d}", msvc_name, numDetsInFrame, i);

            // After cropping, we need to find the right queues to put the bounding boxes in
            for (int j = 0; j < numDetsInFrame; ++j) {
                bboxClass = (int16_t)nmsed_classes[i * maxNumDets + j];
                queueIndex = -1;
                // in the constructor of each microservice, we map the class number to the corresponding queue index in 
                // `classToDntreamMap`.
                for (size_t k = 0; k < this->classToDnstreamMap.size(); ++k) {
                    if ((classToDnstreamMap.at(k).first == bboxClass) || (classToDnstreamMap.at(k).first == -1)) {
                        queueIndex = this->classToDnstreamMap.at(k).second; 
                        // Breaking is only appropriate if case we assume the downstream only wants to take one class
                        // TODO: More than class-of-interests for 1 queue
                        break;
                    }
                }
                // If this class number is not needed anywhere downstream
                if (queueIndex == -1) {
                    continue;
                }

                if (bboxClass == 0 || bboxClass == 2) {
                    saveGPUAsImg(singleImageBBoxList[j], "bbox_" + std::to_string(j) + ".jpg");
                }

                // Putting the bounding box into an `outReq` to be sent out
                bboxShape = {singleImageBBoxList[j].channels(), singleImageBBoxList[j].rows, singleImageBBoxList[j].cols};
                reqData = {
                    bboxShape,
                    singleImageBBoxList[j].clone()
                };
                outReqData.emplace_back(reqData);
                outReq = {
                    {currReq_genTime},
                    currReq.req_e2eSLOLatency,
                    {currReq_path},
                    1,
                    outReqData, //req_data
                    currReq.req_data // upstreamReq_data
                };
                msvc_OutQueue.at(queueIndex)->emplace(outReq);
                trace("{0:s} emplaced a bbox of class {1:d} to queue {2:d}.", msvc_name, bboxClass, queueIndex);
            }
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


/**
 * @brief Generate random bboxes
 * 
 * @param bboxList 
 */
void BaseBBoxCropper::generateRandomBBox(
    float *bboxList,
    const uint16_t height,
    const uint16_t width,
    const uint16_t numBboxes,
    const uint16_t seed
) {
    float x1, y1, x2, y2;

    std::mt19937 gen(seed);

    for (uint16_t j = 0; j < numBboxes; j++) {
        do {
            std::uniform_real_distribution<> x1_dis(0, width - 1);
            std::uniform_real_distribution<> y1_dis(0, height - 1);
            std::uniform_real_distribution<> width_dis(1, width / 2);
            std::uniform_real_distribution<> height_dis(1, height / 2);

            x1 = x1_dis(gen);
            y1 = y1_dis(gen);
            float width = width_dis(gen);
            float height = height_dis(gen);

            x2 = x1 + width;
            y2 = y1 + height;
        } while (x2 >= width || y2 >= height);
        *(bboxList + (j * 4) + 0) = x1;
        *(bboxList + (j * 4) + 1) = y1;
        *(bboxList + (j * 4) + 2) = x2;
        *(bboxList + (j * 4) + 3) = y2;
    }
}

void BaseBBoxCropper::cropProfiling() {
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

    // Random bboxes used for random cropping
    float *nmsed_randomBoxes;

    // To hold the inference time for each individual request
    uint64_t *inferenceTime;

    auto time_now = std::chrono::high_resolution_clock::now();

    //
    std::vector<RequestData<LocalCPUReqDataType>> inferTimeReportData;


    READY = true;

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

                infer_h = msvc_inferenceShape[0][1];
                infer_w = msvc_inferenceShape[0][2];
                
                maxNumDets = msvc_dataShape[2][0];

                delete num_detections;
                delete nmsed_boxes;
                delete nmsed_scores;
                delete nmsed_classes;

                num_detections = new int32_t[msvc_idealBatchSize];
                nmsed_boxes = new float[msvc_idealBatchSize * maxNumDets * 4];
                nmsed_scores = new float[msvc_idealBatchSize * maxNumDets];
                nmsed_classes = new float[msvc_idealBatchSize * maxNumDets];

                nmsed_randomBoxes = new float [msvc_idealBatchSize * maxNumDets * 4];
                for (uint16_t i = 0; i < msvc_idealBatchSize; i++) {
                    generateRandomBBox(nmsed_randomBoxes + i * 4, infer_h, infer_w, maxNumDets);
                }

                ptrList = {nmsed_boxes, nmsed_scores, nmsed_classes};

                inferenceTime = new uint64_t[msvc_idealBatchSize];

                outReqData.clear();
                singleImageBBoxList.clear();
                inferTimeReportData.clear();

                RELOADING = false;
                info("{0:s} is (RE)LOADED.", msvc_name);
            }
            //info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }

        // Processing the next incoming request
        currReq = msvc_InQueue.at(0)->pop2();
        msvc_inReqCount++;

        currReq_genTime = currReq.req_origGenTime[0];

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
            shape = currReq_data[i].shape;
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

        uint8_t numTimeStampPerReq = (uint8_t)(currReq.req_origGenTime.size() / currReq_batchSize);
        uint16_t insertPos = numTimeStampPerReq;

        // Doing post processing for the whole batch
        for (BatchSizeType i = 0; i < currReq_batchSize; ++i) {

            // If there is no object in frame, we don't have to do nothing.
            int numDetsInFrame = maxNumDets;

            // Otherwise, we need to do some cropping.
            orig_h = imageList[i].shape[1];
            orig_w = imageList[i].shape[2];

            crop(imageList[i].data, orig_h, orig_w, infer_h, infer_w, numDetsInFrame, nmsed_boxes + i * maxNumDets * 4, singleImageBBoxList);
            trace("{0:s} cropped {1:d} bboxes in image {2:d}", msvc_name, numDetsInFrame, i);

            // After cropping, we need to find the right queues to put the bounding boxes in
            for (int j = 0; j < numDetsInFrame; ++j) {
                bboxClass = -1;
                queueIndex = -1;
                // in the constructor of each microservice, we map the class number to the corresponding queue index in 
                // `classToDntreamMap`.
                for (size_t k = 0; k < this->classToDnstreamMap.size(); ++k) {
                    if ((classToDnstreamMap.at(k).first == bboxClass) || (classToDnstreamMap.at(k).first == -1)) {
                        queueIndex = this->classToDnstreamMap.at(k).second; 
                        // Breaking is only appropriate if case we assume the downstream only wants to take one class
                        // TODO: More than class-of-interests for 1 queue
                        break;
                    }
                }
                // // If this class number is not needed anywhere downstream
                // if (queueIndex == -1) {
                //     continue;
                // }

                // Make sure we always have output to stress test
                queueIndex = 0;

                // Putting the bounding box into an `outReq` to be sent out
                bboxShape = {singleImageBBoxList[j].channels(), singleImageBBoxList[j].rows, singleImageBBoxList[j].cols};
                reqData = {
                    bboxShape,
                    singleImageBBoxList[j].clone()
                };


                // msvc_OutQueue.at(queueIndex)->emplace(outReq);
                trace("{0:s} emplaced a bbox of class {1:d} to queue {2:d}.", msvc_name, bboxClass, queueIndex);
            }
            // // After cropping is done for this image in the batch, the image's cuda memory can be freed.
            // checkCudaErrorCode(cudaFree(imageList[i].data.cudaPtr()));
            // Clearing out data of the vector

            outReqData.clear();
            singleImageBBoxList.clear();

            // We don't need to send out anything. Just measure the time is enough.

            time_now = std::chrono::high_resolution_clock::now();
            inferenceTime[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(time_now - currReq.req_origGenTime[i]).count();

            if (insertPos < currReq.req_origGenTime.size()) {
                currReq.req_origGenTime.insert(currReq.req_origGenTime.begin() + insertPos, time_now);
                insertPos += numTimeStampPerReq + 1;
            } else if (insertPos == currReq.req_origGenTime.size()) {
                currReq.req_origGenTime.push_back(time_now);
            }
        }
        
        for (BatchSizeType i = 0; i < currReq.req_batchSize; i++) {
            inferTimeReportData.emplace_back(
                RequestData<LocalCPUReqDataType>{
                    {1}, 
                    cv::Mat{1, 1, CV_64F, &inferenceTime[i]}
                }
            );
        }        
        // END is in the travelPath of the last message meaning the profiling session is completed
        if (currReq.req_travelPath[currReq_batchSize - 1].find("PROFILE_ENDS") != std::string::npos) {
            // set this thread to pause to signal to the profiler that the current profiling session has been completed.
            this->pauseThread();
        }
        msvc_OutQueue[0]->emplace(
            Request<LocalCPUReqDataType>{
                currReq.req_origGenTime,
                {},
                currReq.req_travelPath,
                currReq.req_batchSize,
                inferTimeReportData
            }
        );
        inferTimeReportData.clear();
        // // Free all the output buffers of trtengine after cropping is done.
        // for (size_t i = 0; i < currReq_data.size(); i++) {
        //     checkCudaErrorCode(cudaFree(currReq_data.at(i).data.cudaPtr()));
        // }

        // If the current req batch is for warming up (signified by empty request paths), time is not calculated.
        
        trace("{0:s} sleeps for {1:d} millisecond", msvc_name, msvc_interReqTime);

        std::this_thread::sleep_for(std::chrono::milliseconds(this->msvc_interReqTime));
        // Synchronize the cuda stream
    }
    checkCudaErrorCode(cudaStreamDestroy(postProcStream), __func__);
}