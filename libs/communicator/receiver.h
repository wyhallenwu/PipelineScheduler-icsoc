#ifndef PIPEPLUSPLUS_RECEIVER_H
#define PIPEPLUSPLUS_RECEIVER_H

#include "communicator.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using boost::interprocess::read_only;
using boost::interprocess::open_only;

class GPULoader : public Microservice<DataRequest<LocalCPUDataType>> {
public:
    GPULoader(const BaseMicroserviceConfigs &configs, ThreadSafeFixSizedQueue<DataRequest<LocalGPUReqDataType>> *out);

    void Schedule() override;

    ThreadSafeFixSizedQueue<DataRequest<LocalCPUDataType>> *getInQueue() {
        return InQueue;
    }

protected:
    ThreadSafeFixSizedQueue<DataRequest<LocalGPUReqDataType>> *OutQueue;
};

class Receiver : public GPUDataMicroservice<void> {
public:
    Receiver(const BaseMicroserviceConfigs &configs, const std::string &connection);

    ~Receiver() override {
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
                                 ThreadSafeFixSizedQueue<DataRequest<LocalCPUDataType>> *lq);

        void Proceed() final;

    private:
        GpuPointerPayload request;
        SimpleConfirm reply;
        grpc::ServerAsyncResponseWriter<SimpleConfirm> responder;
    };

    class SharedMemoryRequestHandler : public RequestHandler {
    public:
        SharedMemoryRequestHandler(DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
                                   ThreadSafeFixSizedQueue<DataRequest<LocalCPUDataType>> *lq);

        void Proceed() final;

    private:
        SharedMemPayload request;
        SimpleConfirm reply;
        grpc::ServerAsyncResponseWriter<SimpleConfirm> responder;
    };

    class SerializedDataRequestHandler : public RequestHandler {
    public:
        SerializedDataRequestHandler(DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
                                     ThreadSafeFixSizedQueue<DataRequest<LocalCPUDataType>> *lq);

        void Proceed() final;

    private:
        SerializedDataPayload request;
        SimpleConfirm reply;
        grpc::ServerAsyncResponseWriter<SimpleConfirm> responder;
    };

    // This can be run in multiple threads if needed.
    void HandleRpcs();

    std::unique_ptr<ServerCompletionQueue> cq;
    DataTransferService::AsyncService service;
    std::unique_ptr<Server> server;

    ThreadSafeFixSizedQueue<DataRequest<LocalCPUDataType>> *LoadingQueue;
};

#endif //PIPEPLUSPLUS_RECEIVER_H
