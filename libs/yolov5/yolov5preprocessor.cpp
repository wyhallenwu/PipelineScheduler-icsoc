#include <yolov5.h>

using namespace spdlog;

YoloV5Preprocessor::YoloV5Preprocessor(const BaseMicroserviceConfigs &configs) : BasePreprocessor(configs) {
    spdlog::info("{0:s} is created.", msvc_name); 
}

void YoloV5Preprocessor::batchRequests() {
    // The time where the last request was generated.
    ClockType lastReq_genTime;
    // The time where the current incoming request was generated.
    ClockType currReq_genTime;
    // The time where the current incoming request arrives
    ClockType currReq_recvTime;
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
    spdlog::info("{0:s} STARTS.", msvc_name); 
    cv::cuda::Stream preProcStream;
    while (true) {
        // Allowing this thread to naturally come to an end
        if (this->STOP_THREADS) {
            spdlog::info("{0:s} STOPS.", msvc_name);
            break;
        }
        else if (this->PAUSE_THREADS) {
            spdlog::info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }
        // Processing the next incoming request
        if (msvc_InQueue.at(0)->getActiveQueueIndex() != msvc_activeInQueueIndex.at(0)) {
            if (msvc_InQueue.at(0)->size(msvc_activeInQueueIndex.at(0)) == 0) {
                msvc_activeInQueueIndex.at(0) = msvc_InQueue.at(0)->getActiveQueueIndex();
                spdlog::trace("{0:s} Set current active queue index to {1:d}.", msvc_name, msvc_activeInQueueIndex.at(0));
            }
        }
        spdlog::trace("{0:s} Current active queue index {1:d}.", msvc_name, msvc_activeInQueueIndex.at(0));
        if (msvc_activeInQueueIndex.at(0) == 1) {
            currCPUReq = msvc_InQueue.at(0)->pop1();
            currReq = uploadReq(currCPUReq);
        } else if (msvc_activeInQueueIndex.at(0) == 2) {
            currReq = msvc_InQueue.at(0)->pop2();
        }
        msvc_inReqCount++;
        currReq_genTime = currReq.req_origGenTime;

        // We need to check if the next request is worth processing.
        // If it's too late, then we can drop and stop processing this request.
        if (!this->checkReqEligibility(currReq_genTime)) {
            continue;
        }
        // The generated time of this incoming request will be used to determine the rate with which the microservice should
        // check its incoming queue.
        currReq_recvTime = std::chrono::system_clock::now();
        if (this->msvc_inReqCount > 1) {
            this->updateReqRate(currReq_genTime);
        }
        currReq_batchSize = currReq.req_batchSize;
        trace("{0:s} popped a request of batch size {1:d}. In queue size is {2:d}.", msvc_name, currReq_batchSize, msvc_InQueue.at(0)->size());

        msvc_onBufferBatchSize++;
        // Resize the incoming request image the padd with the grey color
        // The resize image will be copied into a reserved buffer

        Stopwatch stopwatch;
        stopwatch.start();


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
            cv::Scalar(128, 128, 128),
            preProcStream
        );

        trace("{0:s} finished resizing a frame", msvc_name);
        data.shape = RequestShapeType({3, (this->msvc_outReqShape.at(0))[0][1], (this->msvc_outReqShape.at(0))[0][2]});
        bufferData.emplace_back(data);
        trace("{0:s} put an image into buffer. Current batch size is {1:d} ", msvc_name, msvc_onBufferBatchSize);

        stopwatch.stop();
        std::cout << "Time taken to preprocess a req is " << stopwatch.elapsed_seconds() << std::endl;
        // cudaFree(currReq.req_data[0].data.cudaPtr());
        // First we need to decide if this is an appropriate time to batch the buffered data or if we can wait a little more.
        // Waiting more means there is a higher chance the earliest request in the buffer will be late eventually.
        if (this->isTimeToBatch()) {
            // If true, copy the buffer data into the out queue
            outReq = {
                std::chrono::_V2::system_clock::now(),
                9999,
                "",
                msvc_onBufferBatchSize,
                bufferData,
                prevData
            };
            trace("{0:s} emplaced a request of batch size {1:d} ", msvc_name, msvc_onBufferBatchSize);
            msvc_OutQueue[0]->emplace(outReq);
            msvc_onBufferBatchSize = 0;
            bufferData.clear();
            prevData.clear();
        }
        trace("{0:s} sleeps for {1:d} millisecond", msvc_name, msvc_interReqTime);
        std::this_thread::sleep_for(std::chrono::milliseconds(this->msvc_interReqTime));
    }
}