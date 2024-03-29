#include "receiver.h"

Receiver::Receiver(const BaseMicroserviceConfigs &configs)
        : Microservice(configs) {
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

Receiver::GpuPointerRequestHandler::GpuPointerRequestHandler(DataTransferService::AsyncService *service,
                                                             ServerCompletionQueue *cq,
                                                             ThreadSafeFixSizedDoubleQueue *out)
        : RequestHandler(service, cq, out), responder(&ctx) {
    Proceed();
}

void Receiver::GpuPointerRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestGpuPointerTransfer(&ctx, &request, &responder, cq, cq,
                                           this);
    } else if (status == PROCESS) {
        if (OutQueue->getActiveQueueIndex() != 2) OutQueue->setActiveQueueIndex(2);
        new GpuPointerRequestHandler(service, cq, OutQueue);

        std::vector<RequestData<LocalGPUReqDataType>> elements = {};
        for (const auto &el: *request.mutable_elements()) {
            auto gpu_image = cv::cuda::GpuMat(el.height(), el.width(), CV_8UC3,
                                              (void *) (&el.data())).clone();
            elements.push_back({{gpu_image.channels(), el.height(), el.width()}, gpu_image});
        }
        Request<LocalGPUReqDataType> req = {
                std::chrono::high_resolution_clock::time_point(std::chrono::nanoseconds(request.timestamp())),
                request.slo(), request.path(), 1, elements};
        OutQueue->emplace(req);

        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

Receiver::SharedMemoryRequestHandler::SharedMemoryRequestHandler(DataTransferService::AsyncService *service,
                                                                 ServerCompletionQueue *cq,
                                                                 ThreadSafeFixSizedDoubleQueue *out)
        : RequestHandler(service, cq, out), responder(&ctx) {
    Proceed();
}

void Receiver::SharedMemoryRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestSharedMemTransfer(&ctx, &request, &responder, cq, cq,
                                          this);
    } else if (status == PROCESS) {
        if (OutQueue->getActiveQueueIndex() != 1) OutQueue->setActiveQueueIndex(1);
        new SharedMemoryRequestHandler(service, cq, OutQueue);

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
                std::chrono::high_resolution_clock::time_point(std::chrono::nanoseconds(request.timestamp())),
                request.slo(), request.path(), 1, elements};
        OutQueue->emplace(req);

        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

Receiver::SerializedDataRequestHandler::SerializedDataRequestHandler(DataTransferService::AsyncService *service,
                                                                     ServerCompletionQueue *cq,
                                                                     ThreadSafeFixSizedDoubleQueue *out)
        : RequestHandler(service, cq, out), responder(&ctx) {
    Proceed();
}

void Receiver::SerializedDataRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestSerializedDataTransfer(&ctx, &request, &responder, cq, cq,
                                               this);
    } else if (status == PROCESS) {
        if (OutQueue->getActiveQueueIndex() != 1) OutQueue->setActiveQueueIndex(1);
        new SerializedDataRequestHandler(service, cq, OutQueue);

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
                std::chrono::high_resolution_clock::time_point(std::chrono::nanoseconds(request.timestamp())),
                request.slo(), request.path(), 1, elements};
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
    new GpuPointerRequestHandler(&service, cq.get(), msvc_OutQueue[0]);
    new SharedMemoryRequestHandler(&service, cq.get(), msvc_OutQueue[0]);
    new SerializedDataRequestHandler(&service, cq.get(), msvc_OutQueue[0]);
    void *tag;  // uniquely identifies a request.
    bool ok;
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
}