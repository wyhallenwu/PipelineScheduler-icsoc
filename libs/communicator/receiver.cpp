#ifndef COMMUNICATOR_H
#define COMMUNICATOR_H

#include "communicator.h"

#endif

class GPULoader : public Microservice<DataRequest<LocalCPUDataType>> {
public:
    GPULoader(const BaseMicroserviceConfigs &configs, ThreadSafeFixSizedQueue<DataRequest<LocalGPUDataType>> *out)
            : Microservice(configs) {
        InQueue = new ThreadSafeFixSizedQueue<DataRequest<LocalCPUDataType>>();
        OutQueue = out;
    }

    void Schedule() override {
        DataRequest<LocalCPUDataType> req = InQueue->pop();
        // copy data to gpu using cuda
        std::vector<Data<LocalGPUDataType>> elements = {};
        for (const auto &el: req.req_data) {
            auto gpu_image = cv::cuda::GpuMat(req.req_dataShape[0], req.req_dataShape[1], CV_8UC3);
            gpu_image.upload(el.content);
            elements.push_back({req.req_dataShape, gpu_image});
        }
        OutQueue->emplace(
                {req.req_origGenTime, req.req_e2eSLOLatency, req.req_travelPath, elements});
    }

    ThreadSafeFixSizedQueue<DataRequest<LocalCPUDataType>> *getInQueue() {
        return InQueue;
    }

protected:
    ThreadSafeFixSizedQueue<DataRequest<LocalGPUDataType>> *OutQueue;
};

class Receiver : public GPUDataMicroservice<void> {
public:
    Receiver(const BaseMicroserviceConfigs &configs, const std::string &url, const uint16_t port)
            : GPUDataMicroservice<void>(configs) {
        std::string serveraddress = absl::StrFormat("%s:%d", url, port);
        grpc::EnableDefaultHealthCheckService(true);
        grpc::reflection::InitProtoReflectionServerBuilderPlugin();
        ServerBuilder builder;
        builder.AddListeningPort(serveraddress, grpc::InsecureServerCredentials());
        builder.RegisterService(&service);
        cq = builder.AddCompletionQueue();
        server = builder.BuildAndStart();
        LoadingQueue = GPULoader(BaseMicroserviceConfigs(), OutQueue).getInQueue();
        HandleRpcs();
    }

    ~Receiver() {
        server->Shutdown();
        cq->Shutdown();
    }

private:
    class RequestHandler {
    public:
        RequestHandler(DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
                       ThreadSafeFixSizedQueue<DataRequest<LocalCPUDataType>> *lq)
                : service(service), cq(cq), LoadingQueue(lq), status(CREATE) {};

        virtual ~RequestHandler() = default;

        virtual void Proceed() = 0;

    protected:
        enum CallStatus {
            CREATE, PROCESS, FINISH
        };

        DataTransferService::AsyncService *service;
        ServerCompletionQueue *cq;
        ServerContext ctx;
        ThreadSafeFixSizedQueue<DataRequest<LocalCPUDataType>> *LoadingQueue;
        CallStatus status;
    };

    class GpuPointerRequestHandler : public RequestHandler {
    public:
        GpuPointerRequestHandler(DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
                                 ThreadSafeFixSizedQueue<DataRequest<LocalCPUDataType>> *lq)
                : RequestHandler(service, cq, lq), responder(&ctx) {
            Proceed();
        }

        void Proceed() final {
            if (status == CREATE) {
                status = PROCESS;
                service->RequestGpuPointerTransfer(&ctx, &request, &responder, cq, cq,
                                                   this);
            } else if (status == PROCESS) {
                new GpuPointerRequestHandler(service, cq, LoadingQueue);

                std::vector<Data<LocalGPUDataType>> elements = {};
                for (const auto &el: *request.mutable_elements()) {
                    auto gpu_image = cv::cuda::GpuMat(el.height(), el.width(), CV_8UC3,
                                                      (void *) (&el.data()));
                    elements.push_back({{el.width(), el.height()}, gpu_image});
                }
                DataRequest<LocalGPUDataType> req = {request.timestamp(), request.slo(),
                                                     request.path(), elements};
                OutQueue->emplace(req);

                status = FINISH;
                responder.Finish(reply, Status::OK, this);
            } else {
                GPR_ASSERT(status == FINISH);
                delete this;
            }
        }

    private:
        GpuPointerPayload request;
        SimpleConfirm reply;
        grpc::ServerAsyncResponseWriter<SimpleConfirm> responder;
    };

    class SharedMemoryRequestHandler : public RequestHandler {
    public:
        SharedMemoryRequestHandler(DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
                                   ThreadSafeFixSizedQueue<DataRequest<LocalCPUDataType>> *lq)
                : RequestHandler(service, cq, lq), responder(&ctx) {
            Proceed();
        }

        void Proceed() final {
            if (status == CREATE) {
                status = PROCESS;
                service->RequestSharedMemTransfer(&ctx, &request, &responder, cq, cq,
                                                  this);
            } else if (status == PROCESS) {
                new SharedMemoryRequestHandler(service, cq, LoadingQueue);

                std::vector<Data<LocalCPUDataType>> elements = {};
                for (const auto &el: *request.mutable_elements()) {
                    auto name = el.name().c_str();
                    boost::interprocess::shared_memory_object shm{open_only, name, read_only};
                    boost::interprocess::mapped_region region{shm, read_only};
                    auto image = static_cast<cv::Mat *>(region.get_address());
                    elements.push_back({{el.width(), el.height()}, *image});

                    boost::interprocess::shared_memory_object::remove(name);
                }
                DataRequest<LocalCPUDataType> req = {request.timestamp(), request.slo(),
                                                     request.path(), elements};
                LoadingQueue->emplace(req);

                status = FINISH;
                responder.Finish(reply, Status::OK, this);
            } else {
                GPR_ASSERT(status == FINISH);
                delete this;
            }
        }

    private:
        SharedMemPayload request;
        SimpleConfirm reply;
        grpc::ServerAsyncResponseWriter<SimpleConfirm> responder;
    };

    class SerializedDataRequestHandler : public RequestHandler {
    public:
        SerializedDataRequestHandler(DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
                                     ThreadSafeFixSizedQueue<DataRequest<LocalCPUDataType>> *lq)
                : RequestHandler(service, cq, lq), responder(&ctx) {
            Proceed();
        }

        void Proceed() final {
            if (status == CREATE) {
                status = PROCESS;
                service->RequestSerializedDataTransfer(&ctx, &request, &responder, cq, cq,
                                                       this);
            } else if (status == PROCESS) {
                new SerializedDataRequestHandler(service, cq, LoadingQueue);

                std::vector<Data<LocalCPUDataType>> elements = {};
                for (const auto &el: *request.mutable_elements()) {
                    uint length = el.data().length();
                    if (length != el.datalen()) {
                        responder.Finish(reply, Status(grpc::INVALID_ARGUMENT, "Data length does not match"), this);
                    }
                    cv::Mat image = cv::Mat(el.height(), el.width(), CV_8UC3,
                                            const_cast<char *>(el.data().c_str())).clone();
                    elements.push_back({{el.width(), el.height()}, image});
                }
                DataRequest<LocalCPUDataType> req = {request.timestamp(), request.slo(),
                                                     request.path(), elements};
                LoadingQueue->emplace(req);

                status = FINISH;
                responder.Finish(reply, Status::OK, this);
            } else {
                GPR_ASSERT(status == FINISH);
                delete this;
            }
        }

    private:
        SerializedDataPayload request;
        SimpleConfirm reply;
        grpc::ServerAsyncResponseWriter<SimpleConfirm> responder;
    };

    // This can be run in multiple threads if needed.
    void HandleRpcs() {
        new GpuPointerRequestHandler(&service, cq.get(), LoadingQueue);
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

    std::unique_ptr<ServerCompletionQueue> cq;
    DataTransferService::AsyncService service;
    std::unique_ptr<Server> server;

    ThreadSafeFixSizedQueue<DataRequest<LocalCPUDataType>> *LoadingQueue;
};
