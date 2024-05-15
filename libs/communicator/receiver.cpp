#include "receiver.h"

ReceiverConfigs Receiver::loadConfigsFromJson(const json &jsonConfigs) {
    ReceiverConfigs configs;
    return configs;
}

void Receiver::loadConfigs(const json &jsonConfigs, bool isConstructing) {
    spdlog::trace("{0:s} is LOANDING configs...", __func__);

    if (!isConstructing) { // If this is not called from the constructor, then we are loading configs from a file for Microservice class
        Microservice::loadConfigs(jsonConfigs);
    }

    ReceiverConfigs configs = loadConfigsFromJson(jsonConfigs);

    if (msvc_RUNMODE == RUNMODE::PROFILING) {
        // readConfigsFromJson(configs.msvc_appLvlConfigs);
        msvc_OutQueue[0]->setActiveQueueIndex(msvc_activeOutQueueIndex[0]);
    } else if (msvc_RUNMODE == RUNMODE::DEPLOYMENT) {
        grpc::EnableDefaultHealthCheckService(true);
        grpc::reflection::InitProtoReflectionServerBuilderPlugin();
        ServerBuilder builder;
        builder.AddListeningPort(upstreamMicroserviceList.front().link[0], grpc::InsecureServerCredentials());
        builder.SetMaxSendMessageSize(1024 * 1024 * 1024);
        builder.SetMaxSendMessageSize(1024 * 1024 * 1024);
        builder.SetMaxMessageSize(1024 * 1024 * 1024);
        builder.SetMaxReceiveMessageSize(1024 * 1024 * 1024);
        builder.RegisterService(&service);
        cq = builder.AddCompletionQueue();
        server = builder.BuildAndStart();
        msvc_OutQueue[0]->setActiveQueueIndex(msvc_activeOutQueueIndex[0]);
        // or so
    }
    spdlog::trace("{0:s} FINISHED loading configs...", __func__);
}

Receiver::Receiver(const json &jsonConfigs) : Microservice(jsonConfigs) {
    loadConfigs(jsonConfigs, true);
    spdlog::info("{0:s} is created.", msvc_name); 
}

template<typename ReqDataType>
void Receiver::processInferTimeReport(Request<ReqDataType> &timeReport) {
    BatchSizeType batchSize = timeReport.req_batchSize;
    bool isProfileEnd = false;

    if (timeReport.req_travelPath[batchSize - 1].find("PROFILE_ENDS") != std::string::npos) {
        timeReport.req_travelPath[batchSize - 1] = removeSubstring(timeReport.req_travelPath[batchSize - 1], "PROFILE_ENDS");
        isProfileEnd = true;
    } else if (timeReport.req_travelPath[batchSize - 1].find("BATCH_ENDS") != std::string::npos) {
        timeReport.req_travelPath[batchSize - 1] = removeSubstring(timeReport.req_travelPath[batchSize - 1], "BATCH_ENDS");
    }
    BatchSizeType numTimeStamps = (BatchSizeType)(timeReport.req_origGenTime.size() / batchSize);
    for (BatchSizeType i = 0; i < batchSize; i++) {
        msvc_logFile << timeReport.req_travelPath[i] << ",";
        for (BatchSizeType j = 0; j < numTimeStamps - 1; j++) {
            msvc_logFile << timePointToEpochString(timeReport.req_origGenTime[i * numTimeStamps + j]) << ",";
        }
        msvc_logFile << timePointToEpochString(timeReport.req_origGenTime[i * numTimeStamps + numTimeStamps - 1]) << "|";

        for (BatchSizeType j = 1; j < numTimeStamps - 1; j++) {
            msvc_logFile << std::chrono::duration_cast<std::chrono::nanoseconds>(timeReport.req_origGenTime[i * numTimeStamps + j] - timeReport.req_origGenTime[i * numTimeStamps + j - 1]).count() << ",";
        }
        msvc_logFile << std::chrono::duration_cast<std::chrono::nanoseconds>(timeReport.req_origGenTime[(i + 1) * numTimeStamps - 1] - timeReport.req_origGenTime[(i + 1) * numTimeStamps - 2]).count() << std::endl;
    } 
    if (isProfileEnd) {
        this->pauseThread();
    }
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

        if (request.mutable_elements()->empty()) {
            responder.Finish(reply, Status(grpc::INVALID_ARGUMENT, "No valid data"), this);
            return;
        }

        std::vector<RequestData<LocalGPUReqDataType>> elements = {};
        for (const auto &el: *request.mutable_elements()) {
            void* data;
            cudaIpcMemHandle_t ipcHandle;
            memcpy(&ipcHandle, el.data().c_str(), sizeof(cudaIpcMemHandle_t));
            cudaError_t cudaStatus = cudaIpcOpenMemHandle(&data, ipcHandle, cudaIpcMemLazyEnablePeerAccess);
            if (cudaStatus != cudaSuccess) {
                std::cerr << "cudaIpcOpenMemHandle failed: " << cudaStatus << std::endl;
                continue;
            }
            auto gpu_image = cv::cuda::GpuMat(el.height(), el.width(), CV_8UC3, data).clone();
            elements = {{{gpu_image.channels(), el.height(), el.width()}, gpu_image}};

            cudaIpcCloseMemHandle(data);

            if (elements.empty()) continue;

            auto timestamps = std::vector<ClockType>();
            for (auto ts: el.timestamp()) {
                timestamps.push_back(std::chrono::time_point<std::chrono::system_clock>(std::chrono::nanoseconds(ts)));
            }
            timestamps.push_back(std::chrono::system_clock::now());

            Request<LocalGPUReqDataType> req = {
                    {timestamps},
                    {el.slo()},
                    {el.path()},
                    1,
                    elements
            };
            OutQueue->emplace(req);
        }

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
            auto name = el.data().c_str();
            boost::interprocess::shared_memory_object shm{open_only, name, read_only};
            boost::interprocess::mapped_region region{shm, read_only};
            auto image = static_cast<cv::Mat *>(region.get_address());
            elements = {{{image->channels(), el.height(), el.width()}, *image}};

            boost::interprocess::shared_memory_object::remove(name);

            auto timestamps = std::vector<ClockType>();
            for (auto ts: el.timestamp()) {
                timestamps.emplace_back(std::chrono::time_point<std::chrono::system_clock>(std::chrono::nanoseconds(ts)));
            }
            timestamps.push_back(std::chrono::system_clock::now());

            Request<LocalCPUReqDataType> req = {
                    {timestamps},
                    {el.slo()},
                    {el.path()},
                    1,
                    elements
            };
            OutQueue->emplace(req);
        }

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
            elements = {{{image.channels(), el.height(), el.width()}, image}};

            auto timestamps = std::vector<ClockType>();
            for (auto ts: el.timestamp()) {
                timestamps.emplace_back(std::chrono::time_point<std::chrono::system_clock>(std::chrono::nanoseconds(ts)));
            }
            timestamps.push_back(std::chrono::system_clock::now());

            Request<LocalCPUReqDataType> req = {
                    {timestamps},
                    {el.slo()},
                    {el.path()},
                    1,
                    elements
            };
            OutQueue->emplace(req);
        }

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
            if (RELOADING) {
                spdlog::trace("{0:s} is BEING (re)loaded...", msvc_name);
                setDevice();
                /*void* target;
                auto test = cv::cuda::GpuMat(1, 1, CV_8UC3);
                cudaIpcMemHandle_t ipcHandle;
                cudaIpcGetMemHandle(&ipcHandle, test.data);
                cudaError_t cudaStatus = cudaIpcOpenMemHandle(&target, ipcHandle, cudaIpcMemLazyEnablePeerAccess);
                cudaIpcCloseMemHandle(target);
                test.release();
                if (cudaStatus != cudaSuccess) {
                    std::cout << "cudaIpcOpenMemHandle failed: " << cudaStatus << std::endl;
                    setDevice();
                }*/
                RELOADING = false;
                spdlog::info("{0:s} is (RE)LOADED.", msvc_name);
            }
            ///spdlog::info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }
        GPR_ASSERT(cq->Next(&tag, &ok));
        GPR_ASSERT(ok);
        static_cast<RequestHandler *>(tag)->Proceed();
    }
    msvc_logFile.close();
}