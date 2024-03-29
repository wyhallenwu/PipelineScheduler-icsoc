#include "baseprocessor.h"

using namespace spdlog;

/**
 * @brief Scale the bounding box coordinates to the original aspect ratio of the image
 * 
 * @param orig_h Original height of the image
 * @param orig_w Original width of the image
 * @param infer_h Height of the image used for inference
 * @param infer_w Width of the image used for inference
 * @param bbox_coors [x1, y1, x2, y2]
 */
void scaleBBox(
    int orig_h,
    int orig_w,
    int infer_h,
    int infer_w,
    const float *infer_bboxCoors,
    int * orig_bboxCoors
) {
    float ratio = std::min(1.f * infer_h / orig_h, 1.f * infer_w / orig_w);
    infer_h = (int) (ratio * orig_h);
    infer_w = (int) (ratio * orig_w);

    // TO BE IMPLEMENTED
    float coor[4];
    for (uint8_t i = 0; i < 4; ++i) {
        coor[i] = (*(infer_bboxCoors + i));
    }

    float gain = std::min(1.f * infer_h / orig_h, 1.f * infer_w / orig_w);

    float pad_w = (1.f * infer_w - orig_w * gain) / 2.f;
    float pad_h = (1.f * infer_h - orig_h * gain) / 2.f;

    coor[0] -= pad_w;
    coor[1] -= pad_h;
    coor[2] -= pad_w;
    coor[3] -= pad_h;

    // if (scale_h > scale_w) {
    //     coor[1]= coor[1] / scale_w;
    //     coor[3]= coor[3] / scale_w;
    //     coor[0]= (coor[0] - (infer_h - scale_w * orig_h) / 2) / scale_w;
    //     coor[2]= (coor[2] - (infer_h - scale_w * orig_h) / 2) / scale_w;
    // } else {
    //     coor[1]= (coor[1] - (infer_w - scale_h * orig_w) / 2) / scale_h;
    //     coor[3]= (coor[3] - (infer_w - scale_h * orig_w) / 2) / scale_h;
    //     coor[0]= coor[0] / scale_h;
    //     coor[2]= coor[2] / scale_h;
    // }

    for (uint8_t i = 0; i < 4; ++i) {
        coor[i] /= gain;
        int maxcoor = (i % 2 == 0) ? orig_w : orig_h;
        if (coor[i] >= maxcoor) {
            coor[i] = maxcoor - 1;
        }
        if (coor[i] < 0) {
            coor[i] = 0;
        }
        *(orig_bboxCoors + i) = (int)coor[i];
    }
}

/**
 * @brief Cropping multiple boxes from 1 picture
 * 
 * @param image 
 * @param infer_h 
 * @param infer_w 
 * @param numDetections 
 * @param bbox_coorList 
 * @param croppedBBoxes 
 */
void crop(
    const cv::cuda::GpuMat &image,
    int orig_h,
    int orig_w,
    int infer_h,
    int infer_w,
    int numDetections,
    const float *bbox_coorList,
    std::vector<cv::cuda::GpuMat> &croppedBBoxes
) {
    int orig_bboxCoors[4];
    for (uint16_t i = 0; i < numDetections; ++i) {
        scaleBBox(orig_h, orig_w, infer_h, infer_w, bbox_coorList + i * 4, orig_bboxCoors);
        std::cout << (int)orig_bboxCoors[0] << " " << (int)orig_bboxCoors[1] << " " << (int)orig_bboxCoors[2] << " " << (int)orig_bboxCoors[3] << std::endl;
        cv::cuda::GpuMat croppedBBox = image(cv::Range((int)orig_bboxCoors[1], (int)orig_bboxCoors[3]), cv::Range((int)orig_bboxCoors[0], (int)orig_bboxCoors[2])).clone();
        croppedBBoxes.emplace_back(croppedBBox);
    }
}

/**
 * @brief Cropping 1 box from 1 picture
 * 
 * @param image 
 * @param infer_h 
 * @param infer_w 
 * @param numDetections 
 * @param bbox_coorList 
 * @param croppedBBoxes 
 */
void cropOneBox(
    const cv::cuda::GpuMat &image,
    int infer_h,
    int infer_w,
    int numDetections,
    const float *bbox_coorList,
    cv::cuda::GpuMat &croppedBBoxes
) {
    int orig_h, orig_w;
    orig_h = image.rows;
    orig_w = image.cols;
    int orig_bboxCoors[4];
    scaleBBox(orig_h, orig_w, infer_h, infer_w, bbox_coorList, orig_bboxCoors);
    cv::cuda::GpuMat croppedBBox = image(cv::Range((int)orig_bboxCoors[0], (int)orig_bboxCoors[2]), cv::Range((int)orig_bboxCoors[1], (int)orig_bboxCoors[3])).clone();
    croppedBBoxes = croppedBBox;
}

BaseBBoxCropper::BaseBBoxCropper(const BaseMicroserviceConfigs &configs) : Microservice(configs) {
    info("{0:s} is created.", msvc_name); 
}

void BaseBBoxCropper::cropping() {
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


    // Height and width of the image used for inference
    int orig_h, orig_w, infer_h, infer_w;

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
        trace("{0:s} popped a request of batch size {1:d}", msvc_name, currReq_batchSize);


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

        uint16_t maxNumDets = msvc_dataShape[2][0];

        std::vector<RequestData<LocalGPUReqDataType>> currReq_data = currReq.req_data;
        int32_t num_detections[currReq_batchSize];
        float nmsed_boxes[currReq_batchSize][maxNumDets][4];
        float nmsed_scores[currReq_batchSize][maxNumDets];
        float nmsed_classes[currReq_batchSize][maxNumDets];
        int32_t *numDetList = num_detections;
        float *nmsedBoxesList = &nmsed_boxes[0][0][0];
        float *nmsedScoresList = &nmsed_scores[0][0];
        float *nmsedClassesList = &nmsed_classes[0][0];

        // float numDetList[currReq_batchSize];
        // float nmsedBoxesList[currReq_batchSize][maxNumDets][4];
        // float nmsedScoresList[currReq_batchSize][maxNumDets];
        // float nmsedClassesList[currReq_batchSize][maxNumDets];
        std::vector<float *> ptrList{nmsedBoxesList, nmsedScoresList, nmsedClassesList};
        std::vector<size_t> bufferSizeList;

        for (std::size_t i = 0; i < (currReq_data.size() - 1); ++i) {
            size_t bufferSize = this->msvc_modelDataType * (size_t)currReq_batchSize;
            RequestDataShapeType shape = currReq_data[i].shape;
            for (uint8_t j = 0; j < shape.size(); ++j) {
                bufferSize *= shape[j];
            }
            bufferSizeList.emplace_back(bufferSize);
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

         /**
         * @brief TODOs:
         * Hardcoding because we hvent been able to properly carry the image to be cropped.
         * The cropping logic is ok though.
         * Need to figure out a way.
         */
        infer_h = currReq.req_data[4].shape[1];
        infer_w = currReq.req_data[4].shape[2];

        // List of images to be cropped from
        imageList = currReq.upstreamReq_data; 

        // class of the bounding box cropped from one the images in the image list
        int16_t bboxClass;
        // The index of the queue we are going to put data on based on the value of `bboxClass`
        NumQueuesType queueIndex;

        // Doing post processing for the whole batch
        for (BatchSizeType i = 0; i < currReq_batchSize; ++i) {

            // If there is no object in frame, we don't have to do nothing.
            int numDetsInFrame = (int)numDetList[i];
            if (numDetsInFrame <= 0) {
                outReqData.clear();
                continue;
            }

            // Otherwise, we need to do some cropping.
            orig_h = imageList[i].shape[1];
            orig_w = imageList[i].shape[2];

            crop(imageList[i].data, orig_h, orig_w, infer_h, infer_w, numDetsInFrame, nmsed_boxes[i][0], singleImageBBoxList);
            trace("{0:s} cropped {1:d} bboxes in image {2:d}", msvc_name, numDetsInFrame, i);

            // After cropping, we need to find the right queues to put the bounding boxes in
            for (int j = 0; j < numDetsInFrame; ++j) {
                bboxClass = (int16_t)nmsed_classes[i][j];
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

                if (bboxClass == 0) {
                    saveGPUAsImg(singleImageBBoxList[j], "bbox.jpg");
                }

                // Putting the bounding box into an `outReq` to be sent out
                bboxShape = {singleImageBBoxList[j].channels(), singleImageBBoxList[j].rows, singleImageBBoxList[j].cols};
                reqData = {
                    bboxShape,
                    singleImageBBoxList[j].clone()
                };
                outReqData.emplace_back(reqData);
                outReq = {
                    std::chrono::_V2::system_clock::now(),
                    currReq.req_e2eSLOLatency,
                    "",
                    1,
                    outReqData, //req_data
                    currReq.req_data // upstreamReq_data
                };
                // msvc_OutQueue.at(queueIndex)->emplace(outReq);
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
}