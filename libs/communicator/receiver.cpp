#include "receiver.h"

GPULoader::GPULoader(const BaseMicroserviceConfigs &configs, ThreadSafeFixSizedDoubleQueue *out, const CommMethod &m)
        : Microservice(configs) {
    InQueue = new ThreadSafeFixSizedDoubleQueue();
    OutQueue = out;
    if (m == CommMethod::localGPU) {
        std::thread t(&GPULoader::Onloading, this);
        t.detach();
    } else if (m == CommMethod::localCPU) {
        std::thread t(&GPULoader::Offloading, this);
        t.detach();
    }
}

void GPULoader::Onloading() {
    while (true) {
        if (this->STOP_THREADS) {
                spdlog::info("{0:s} STOPS.", msvc_name);
                break;
        }
        else if (this->PAUSE_THREADS) {
            spdlog::info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }
        Stopwatch stopwatch;
        stopwatch.start();
        Request<LocalCPUReqDataType> req = InQueue->pop1();
        stopwatch.stop();
        std::cout << "Time to wait for a req is " << stopwatch.elapsed_seconds() << std::endl;
        // copy data to gpu using cuda
        std::vector<RequestData<LocalGPUReqDataType>> elements = {};
        for (const auto &el: req.req_data) {
            stopwatch.start();
            auto gpu_image = cv::cuda::GpuMat(el.shape[0], el.shape[1], CV_8UC3);
            stopwatch.stop();
            std::cout << "Time taken to allocate is " << stopwatch.elapsed_seconds() << std::endl;

            stopwatch.start();
            gpu_image.upload(el.data);
            elements.push_back({el.shape, gpu_image});
            stopwatch.stop();
            std::cout << "Time taken to upload is " << stopwatch.elapsed_seconds() << std::endl;
        }
        OutQueue->emplace(
                {req.req_origGenTime, req.req_e2eSLOLatency, req.req_travelPath, req.req_batchSize, elements});
    }
}

void GPULoader::Offloading() {
    while (true) {

        if (this->STOP_THREADS) {
                spdlog::info("{0:s} STOPS.", msvc_name);
                break;
        }
        else if (this->PAUSE_THREADS) {
            spdlog::info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }
        Request<LocalGPUReqDataType> req = InQueue->pop2();
        // copy data from gpu using cuda
        std::vector<RequestData<LocalCPUReqDataType>> elements = {};
        for (const auto &el: req.req_data) {
            cv::Mat image;
            el.data.download(image);
            elements.push_back({el.shape, image});
        }
        OutQueue->emplace(
                {req.req_origGenTime, req.req_e2eSLOLatency, req.req_travelPath, req.req_batchSize, elements});
    }
}

Receiver::Receiver(const BaseMicroserviceConfigs &configs, const CommMethod &m)
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
    LoadingQueue = (new GPULoader(configs, msvc_OutQueue[0], m))->getInQueue();
    cq = builder.AddCompletionQueue();
    server = builder.BuildAndStart();
    std::thread handler;
    if (m == CommMethod::localGPU) {
        msvc_OutQueue[0]->setActiveQueueIndex(2);
        handler = std::thread(&Receiver::HandleRpcsToGPU, this);
    } else if (m == CommMethod::localCPU) {
        msvc_OutQueue[0]->setActiveQueueIndex(1);
        handler = std::thread(&Receiver::HandleRpcsToCPU, this);
    }
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
        new GpuPointerRequestHandler(service, cq, OutQueue);

        std::vector<RequestData<LocalGPUReqDataType>> elements = {};
        for (const auto &el: *request.mutable_elements()) {
            auto gpu_image = cv::cuda::GpuMat(el.height(), el.width(), CV_8UC3,
                                              (void *) (&el.data())).clone();
            elements.push_back({{el.width(), el.height()}, gpu_image});
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
        new SharedMemoryRequestHandler(service, cq, OutQueue);

        std::vector<RequestData<LocalCPUReqDataType>> elements = {};
        for (const auto &el: *request.mutable_elements()) {
            auto name = el.name().c_str();
            boost::interprocess::shared_memory_object shm{open_only, name, read_only};
            boost::interprocess::mapped_region region{shm, read_only};
            auto image = static_cast<cv::Mat *>(region.get_address());
            elements.push_back({{el.width(), el.height()}, *image});

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
        new SerializedDataRequestHandler(service, cq, OutQueue);

        std::vector<RequestData<LocalCPUReqDataType>> elements = {};
        for (const auto &el: *request.mutable_elements()) {
            uint length = el.data().length();
            if (length != el.datalen()) {
                responder.Finish(reply, Status(grpc::INVALID_ARGUMENT, "Data length does not match"), this);
            }
            cv::Mat image = cv::Mat(el.height(), el.width(), CV_8UC3,
                                    const_cast<char *>(el.data().c_str())).clone();
            elements.push_back({{el.width(), el.height()}, image});
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
void Receiver::HandleRpcsToGPU() {
    new GpuPointerRequestHandler(&service, cq.get(), msvc_OutQueue[0]);
    new SharedMemoryRequestHandler(&service, cq.get(), LoadingQueue);
    new SerializedDataRequestHandler(&service, cq.get(), LoadingQueue);
    void *tag;  // uniquely identifies a request.
    bool ok;
    while (true) {
        GPR_ASSERT(cq->Next(&tag, &ok));
        GPR_ASSERT(ok);
        static_cast<RequestHandler *>(tag)->Proceed();
    }
}

void Receiver::HandleRpcsToCPU() {
    new GpuPointerRequestHandler(&service, cq.get(), LoadingQueue);
    new SharedMemoryRequestHandler(&service, cq.get(), msvc_OutQueue[0]);
    new SerializedDataRequestHandler(&service, cq.get(), msvc_OutQueue[0]);
    void *tag;
    bool ok;
    while (true) {
        GPR_ASSERT(cq->Next(&tag, &ok));
        GPR_ASSERT(ok);
        static_cast<RequestHandler *>(tag)->Proceed();
    }
}