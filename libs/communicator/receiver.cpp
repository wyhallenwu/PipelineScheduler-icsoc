#include "receiver.h"

Receiver::Receiver(const BaseMicroserviceConfigs &configs)
        : Microservice(configs) {
    if (msvc_RUNMODE == RUNMODE::PROFILING) {
        readConfigsFromJson(configs.msvc_appLvlConfigs);
        msvc_OutQueue[0]->setActiveQueueIndex(msvc_activeOutQueueIndex[0]);
    } else if (msvc_RUNMODE == RUNMODE::DEPLOYMENT) {
        grpc::EnableDefaultHealthCheckService(true);
        grpc::reflection::InitProtoReflectionServerBuilderPlugin();
        ServerBuilder builder;
        builder.AddListeningPort(configs.msvc_upstreamMicroservices.front().link[0], grpc::InsecureServerCredentials());
        builder.SetMaxSendMessageSize(1024 * 1024 * 1024);
        builder.SetMaxSendMessageSize(1024 * 1024 * 1024);
        builder.SetMaxMessageSize(1024 * 1024 * 1024);
        builder.SetMaxReceiveMessageSize(1024 * 1024 * 1024);

        builder.RegisterService(&service);
        cq = builder.AddCompletionQueue();
        server = builder.BuildAndStart();
        msvc_OutQueue[0]->setActiveQueueIndex(msvc_activeOutQueueIndex[0]);
        auto handler = std::thread(&Receiver::HandleRpcs, this);
        handler.detach();
    }
}

void Receiver::profileDataGenerator() {
    msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);
    setDevice();

    // Since we dont know the shape of data before hand, we would choose a few potential shapes and choose randomly amongst them
    // during profiling
    uint8_t randomShapeIndex;
    std::uniform_int_distribution<> dis(0, msvc_dataShape.size() - 1);
    uint16_t seed = 2024;
    std::mt19937 gen(2024);

    std::vector<RequestData<LocalCPUReqDataType>> requestData;
    RequestData<LocalCPUReqDataType> data;
    Request<LocalCPUReqDataType> request;
    RequestDataShapeType shape;
    cv::Mat img;
    std::string requestPath;
    if (msvc_OutQueue[0]->getActiveQueueIndex() != 1) msvc_OutQueue[0]->setActiveQueueIndex(1);
    msvc_OutQueue[0]->setQueueSize(1000);
    READY = true;

    Request<LocalCPUReqDataType> inferTimeReportReq;

    auto numWarmUpBatches = msvc_numWarmUpBatches;
    auto numProfileBatches = msvc_numProfileBatches;
    BatchSizeType batchSize = 1;
    BatchSizeType batchNum = 1;
    msvc_InQueue.at(0)->setActiveQueueIndex(1);

    while (true) {
        if (this->STOP_THREADS) {
            spdlog::info("{0:s} STOPS.", msvc_name);
            break;
        }
        else if (this->PAUSE_THREADS) {
            ///spdlog::info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }
        if (msvc_inReqCount > 0) {
            inferTimeReportReq = msvc_InQueue.at(0)->pop1();
            for (BatchSizeType i = 0; i < inferTimeReportReq.req_batchSize; i++) {
                std::cout << inferTimeReportReq.req_data[i].data.at<uint64_t>(0, 0) << std::endl;
            }
        }
        /**
         * @brief Warming up to avoid cold start effects.
         * During warming up, we use inference `numBatches` batches of requests.
         * 
         */
        if (numWarmUpBatches > 0) {
            for (BatchSizeType j = 0; j < msvc_idealBatchSize; ++j) {
                msvc_inReqCount++;
                randomShapeIndex = dis(gen);
                shape = msvc_dataShape[randomShapeIndex];
                img = cv::Mat(shape[1], shape[2], CV_8UC3);
                data = {
                    shape,
                    img
                };
                requestData.emplace_back(data);
                requestPath = "";
                request = {
                    {std::chrono::_V2::system_clock::now()},
                    {9999},
                    {requestPath},
                    1,
                    requestData
                };
                msvc_OutQueue[0]->emplace(request);
            }
            numWarmUpBatches--;
            requestData.clear();
        }
        /**
         * @brief For each model, we profile all batch size in range of [1, msvc_idealBatchSize] for `numProfileBatches` times.
         * 
         */
        else if (numWarmUpBatches == 0 && numProfileBatches > 0) {
            for (BatchSizeType i = 1; i <= batchSize; i++) { // Filling up the batch
                msvc_inReqCount++;

                // Choosing a random shape for a more generalized profiling results
                randomShapeIndex = dis(gen);
                shape = msvc_dataShape[randomShapeIndex];
                img = cv::Mat(shape[1], shape[2], CV_8UC3);
                data = {
                    shape,
                    img
                };
                requestData.emplace_back(data);

                // For bookkeeping, we add a certain pattern into the `requestPath` field.
                // [batchSize, batchNum, i]
                requestPath = std::to_string(batchSize) + "," + std::to_string(batchNum) + "," + std::to_string(i);

                // The very last batch of this profiling session is marked with "END" in the `requestPath` field.
                if ((batchSize == msvc_idealBatchSize) && (batchNum == msvc_numProfileBatches - 1)) {
                    requestPath = requestPath + "END";
                }
                request = {
                    {std::chrono::_V2::system_clock::now()},
                    {9999},
                    {requestPath},
                    1,
                    requestData
                };
                msvc_OutQueue[0]->emplace(request);
            }
            requestData.clear();
            
        }
        if (batchNum > msvc_numProfileBatches) {
            batchSize++;
            batchNum = 1;
        }
        if (batchSize > msvc_idealBatchSize) {
            this->pauseThread();
        }
    }
    msvc_logFile.close();
}

Receiver::GpuPointerRequestHandler::GpuPointerRequestHandler(
    DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
    ThreadSafeFixSizedDoubleQueue *out,
    uint64_t &msvc_inReqCount
) : RequestHandler(service, cq, out, msvc_inReqCount), responder(&ctx) {
    Proceed();
}

void Receiver::GpuPointerRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestGpuPointerTransfer(&ctx, &request, &responder, cq, cq,
                                           this);
    } else if (status == PROCESS) {
        if (OutQueue->getActiveQueueIndex() != 2) OutQueue->setActiveQueueIndex(2);
        new GpuPointerRequestHandler(service, cq, OutQueue, msvc_inReqCount);

        std::vector<RequestData<LocalGPUReqDataType>> elements = {};
        for (const auto &el: *request.mutable_elements()) {
            auto gpu_image = cv::cuda::GpuMat(el.height(), el.width(), CV_8UC3,
                                              (void *) (&el.data())).clone();
            elements.push_back({{gpu_image.channels(), el.height(), el.width()}, gpu_image});
        }
        Request<LocalGPUReqDataType> req = {
            {ClockType(std::chrono::nanoseconds(request.timestamp()))},
            {request.slo()},
            {request.path()},
            1,
            elements
        };
        OutQueue->emplace(req);

        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

Receiver::SharedMemoryRequestHandler::SharedMemoryRequestHandler(
    DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
    ThreadSafeFixSizedDoubleQueue *out,
    uint64_t &msvc_inReqCount
) : RequestHandler(service, cq, out, msvc_inReqCount), responder(&ctx) {
    Proceed();
}

void Receiver::SharedMemoryRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestSharedMemTransfer(&ctx, &request, &responder, cq, cq,
                                          this);
    } else if (status == PROCESS) {
        if (OutQueue->getActiveQueueIndex() != 1) OutQueue->setActiveQueueIndex(1);
        new SharedMemoryRequestHandler(service, cq, OutQueue, msvc_inReqCount);

        std::vector<RequestData<LocalCPUReqDataType>> elements = {};
        for (const auto &el: *request.mutable_elements()) {
            auto name = el.name().c_str();
            boost::interprocess::shared_memory_object shm{open_only, name, read_only};
            boost::interprocess::mapped_region region{shm, read_only};
            auto image = static_cast<cv::Mat *>(region.get_address());
            elements.push_back({{image->channels(), el.height(), el.width()}, *image});

            boost::interprocess::shared_memory_object::remove(name);
        }
        Request<LocalCPUReqDataType> req = {
            {ClockType(std::chrono::nanoseconds(request.timestamp()))},
            {request.slo()},
            {request.path()},
            1,
            elements
        };
        OutQueue->emplace(req);

        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

Receiver::SerializedDataRequestHandler::SerializedDataRequestHandler(
    DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
    ThreadSafeFixSizedDoubleQueue *out,
    uint64_t &msvc_inReqCount
) : RequestHandler(service, cq, out, msvc_inReqCount), responder(&ctx) {
    Proceed();
}

void Receiver::SerializedDataRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestSerializedDataTransfer(&ctx, &request, &responder, cq, cq,
                                               this);
    } else if (status == PROCESS) {
        if (OutQueue->getActiveQueueIndex() != 1) OutQueue->setActiveQueueIndex(1);
        new SerializedDataRequestHandler(service, cq, OutQueue, msvc_inReqCount);

        std::vector<RequestData<LocalCPUReqDataType>> elements = {};
        for (const auto &el: *request.mutable_elements()) {
            uint length = el.data().length();
            if (length != el.datalen()) {
                responder.Finish(reply, Status(grpc::INVALID_ARGUMENT, "Data length does not match"), this);
            }
            cv::Mat image = cv::Mat(el.height(), el.width(), CV_8UC3,
                                    const_cast<char *>(el.data().c_str())).clone();
            elements.push_back({{image.channels(), el.height(), el.width()}, image});
        }
        Request<LocalCPUReqDataType> req = {
            {std::chrono::high_resolution_clock::time_point(std::chrono::nanoseconds(request.timestamp()))},
            {request.slo()},
            {request.path()},
            1, 
            elements
        };
        OutQueue->emplace(req);

        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

// This can be run in multiple threads if needed.
void Receiver::HandleRpcs() {
    msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);
    setDevice();
    new GpuPointerRequestHandler(&service, cq.get(), msvc_OutQueue[0], msvc_inReqCount);
    new SharedMemoryRequestHandler(&service, cq.get(), msvc_OutQueue[0], msvc_inReqCount);
    new SerializedDataRequestHandler(&service, cq.get(), msvc_OutQueue[0], msvc_inReqCount);
    void *tag;  // uniquely identifies a request.
    bool ok;
    READY = true;
    while (true) {
        if (this->STOP_THREADS) {
            spdlog::info("{0:s} STOPS.", msvc_name);
            break;
        }
        else if (this->PAUSE_THREADS) {
            ///spdlog::info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }
        GPR_ASSERT(cq->Next(&tag, &ok));
        GPR_ASSERT(ok);
        static_cast<RequestHandler *>(tag)->Proceed();
    }
    msvc_logFile.close();
}