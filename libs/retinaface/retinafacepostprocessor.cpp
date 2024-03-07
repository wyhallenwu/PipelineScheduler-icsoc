#include <retinaface.h>

RetinaFacePostprocessor::RetinaFacePostprocessor(const BaseMicroserviceConfigs &config) : BasePostprocessor(config) {
}

void RetinaFacePostprocessor::postProcessing() {
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
         * @brief Each request to the postprocessing microservice of RetinaFace contains the buffers which are results of TRT inference 
         * as well as the images from which bounding boxes will be cropped.
         * `The buffers` are a vector of 2 small raw memory (float) buffers (since using `BatchedNMSPlugin` provided by TRT), which are
         * 1/ `num_detections` (Batch, 1): number of objects detected in this frame
         * 2/ `nmsed_boxes` (Batch, 1, 4): the only box remaining after nms is performed. In principle, there should be more than 1 boxes
         *                                  for each input picture. But in this case, the image is a cropped bounding box of human class,
         *                                  so it can have at most 1 face (if found).
         * We need to bring these buffers to CPU in order to process them.
         */

        std::vector<RequestData<LocalGPUReqDataType>> currReq_data = currReq.req_data;
        float num_detections[currReq_batchSize];
        float nmsed_boxes[currReq_batchSize][4];
        float *numDetList = &num_detections[0];
        float *nmsedBoxesList = &nmsed_boxes[0][0];

        std::vector<float *> ptrList{numDetList, nmsedBoxesList};
        std::vector<size_t> bufferSizeList;

        cudaStream_t postProcStream;
        checkCudaErrorCode(cudaStreamCreate(&postProcStream));
        for (std::size_t i = 0; i < currReq_data.size(); ++i) {
            size_t bufferSize = this->msvc_modelDataType * (size_t)currReq_batchSize;
            RequestShapeType shape = currReq_data[i].shape;
            for (uint8_t j = 0; j < shape.size(); ++j) {
                bufferSize *= shape[j];
            }
            bufferSizeList.emplace_back(bufferSize);
            checkCudaErrorCode(cudaMemcpyAsync(
                (void *) ptrList[i],
                currReq_data[i].data.cudaPtr(),
                bufferSize,
                cudaMemcpyDeviceToHost,
                postProcStream
            ));
        }

        // List of images to be cropped from
        // in this case, each image is a detected human bounding box
        imageList = currReq.upstreamReq_data; 
        NumQueuesType queueIndex;

        // Doing post processing for the whole batch
        for (BatchSizeType i = 0; i < currReq_batchSize; ++i) {
            // Height and width of the image used for inference
            int infer_h, infer_w;

            // If there is no face in frame, we don't have to do nothing.
            int numDetsInFrame = (int)numDetList[i];
            if (numDetsInFrame <= 0) {
                continue;
            }

            // The face found in each human class co
            cv::cuda::GpuMat foundFace;

            // Otherwise, we need to do some cropping.
            // Btw, we should have only 1 box to crop thus make singleImageBBoxList
            infer_h = imageList[i].shape[1];
            infer_w = imageList[i].shape[2];
            crop(imageList[i].data, infer_h, infer_w, numDetsInFrame, nmsed_boxes[i], foundFace);
            queueIndex = -1;

            // The face we found to the downstreams
            for (size_t k = 0; k < this->classToDnstreamMap.size(); ++k) {
                queueIndex = this->classToDnstreamMap.at(k).second;
                bboxShape = {foundFace.channels(), foundFace.rows, foundFace.cols};
                reqData = {
                    bboxShape,
                    foundFace.clone()
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
                msvc_OutQueue.at(queueIndex)->emplace(outReq);
                outReqData.clear();
            }
        }
        // Let the poor boy sleep for some time before he has to wake up and do postprocessing again.
        std::this_thread::sleep_for(std::chrono::milliseconds(this->msvc_interReqTime));
    }


}