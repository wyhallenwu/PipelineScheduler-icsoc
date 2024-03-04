#include <yolov5.h>

template<typename InType>
YoloV5Postprocessor<InType>::YoloV5Postprocessor(const BaseMicroserviceConfigs &config) : BasePostprocessor<InType>(config) {
}

template<typename InType>
void YoloV5Postprocessor<InType>::postProcessing() {
    // The time where the last request was generated.
    ClockTypeTemp lastReq_genTime;
    // The time where the current incoming request was generated.
    ClockTypeTemp currReq_genTime;
    // The time where the current incoming request arrives
    ClockTypeTemp currReq_recvTime;

    BatchSizeType currReq_batchSize;

    while (true) {
        if (!this->RUN_THREADS) {
            break;
        }
        // Processing the next incoming request
        InType currReq = this->InQueue.pop();
        this->msvc_inReqCount++;

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
         * @brief Each request to the postprocessing microservice of YOLOv5 contains the buffers which are results of TRT inference 
         * as well as the images from which bounding boxes will be cropped.
         * `The buffers` are a vector of 4 small raw memory (float) buffers (since using `BatchedNMSPlugin` provided by TRT), which are
         * 1/ `num_detections` (Batch, 1): number of objects detected in this frame
         * 2/ `nmsed_boxes` (Batch, TopK, 4): the boxes remaining after nms is performed.
         * 3/ `nmsed_scores` (Batch, TopK): the scores of these boxes.
         * 4/ `nmsed_classes` (Batch, TopK):
         * We need to bring these buffers to CPU in order to process them.
         */

        cudaStream_t postProcStream;
        uint16_t topDetResults = this->msvc_outReqShape[2][0];

        std::vector<Data<LocalGPUReqDataType>> currReq_data = currReq.req_data;
        // float num_detections[currReq_batchSize];
        // float nmsed_boxes[currReq_batchSize][topDetResults][4];
        // float nmsed_scores[currReq_batchSize][topDetResults];
        // float nmsed_classes[currReq_batchSize][topDetResults];
        // float *numDetList = &num_detections;
        // float *nmsedBoxesList = &nmsed_boxes;
        // float *nmsedScoresList = &nmsed_scores;
        // float *nmsedclassesList = &nmsed_classes;
        float numDetList[currReq_batchSize];
        float nmsedBoxesList[currReq_batchSize][topDetResults][4];
        float nmsedScoresList[currReq_batchSize][topDetResults];
        float nmsedClassesList[currReq_batchSize][topDetResults];
        std::vector<float *> ptrList = {&numDetList, &nmsedBoxesList, &nmsedScoresList, &nmsedClassesList};
        std::vector<size_t> bufferSizeList;

        
        for (std::size_t i = 0; i < currReq_data.size(); ++i) {
            size_t bufferSize = this->msvc_modelDataType * (size_t)currReq_batchSize;
            RequestShapeType shape = currReq_data[i].shape;
            for (uint8_t j = 0; j < shape.size(); ++j) {
                bufferSize *= shape[j];
            }
            bufferSizeList.emplace_back(bufferSize);
            cudaMemcpyAsync(
                (void *) ptrList[i],
                currReq_data[i].content.cudaPtr(),
                bufferSize,
                cudaMemcpyDeviceToHost,
                postProcStream
            );
        }
        std::vector<cv::cuda::GpuMat> singleImageBBoxList;
        for (BatchSizeType i = 0; i < currReq_batchSize; ++i) {
            
        }
    }


}