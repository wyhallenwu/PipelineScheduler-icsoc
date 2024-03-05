#ifndef PIPEPLUSPLUS_RECEIVER_H
#define PIPEPLUSPLUS_RECEIVER_H

#include "communicator.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using boost::interprocess::read_only;
using boost::interprocess::open_only;

class GPULoader : public Microservice {
public:
    GPULoader(const BaseMicroserviceConfigs &configs, ThreadSafeFixSizedDoubleQueue *out, const CommMethod &m);

    void Onloading();

    void Offloading();

    ThreadSafeFixSizedDoubleQueue *getInQueue() {
        return InQueue;
    }

protected:
    ThreadSafeFixSizedDoubleQueue *InQueue, *OutQueue;
};

class Receiver : public Microservice {
public:
    Receiver(const BaseMicroserviceConfigs &configs, const std::string &connection, const CommMethod &m);

    ~Receiver() override {
        server->Shutdown();
        cq->Shutdown();
    }

private:
    class RequestHandler {
    public:
        RequestHandler(DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
                       ThreadSafeFixSizedDoubleQueue *out)
                : service(service), cq(cq), OutQueue(out), status(CREATE) {};

        virtual ~RequestHandler() = default;

        virtual void Proceed() = 0;

    protected:
        enum CallStatus {
            CREATE, PROCESS, FINISH
        };

        DataTransferService::AsyncService *service;
        ServerCompletionQueue *cq;
        ServerContext ctx;
        ThreadSafeFixSizedDoubleQueue *OutQueue;
        CallStatus status;
    };

    class GpuPointerRequestHandler : public RequestHandler {
    public:
        GpuPointerRequestHandler(DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
                                 ThreadSafeFixSizedDoubleQueue *out);

        void Proceed() final;

    private:
        GpuPointerPayload request;
        SimpleConfirm reply;
        grpc::ServerAsyncResponseWriter<SimpleConfirm> responder;
    };

    class SharedMemoryRequestHandler : public RequestHandler {
    public:
        SharedMemoryRequestHandler(DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
                                   ThreadSafeFixSizedDoubleQueue *out);

        void Proceed() final;

    private:
        SharedMemPayload request;
        SimpleConfirm reply;
        grpc::ServerAsyncResponseWriter<SimpleConfirm> responder;
    };

    class SerializedDataRequestHandler : public RequestHandler {
    public:
        SerializedDataRequestHandler(DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
                                     ThreadSafeFixSizedDoubleQueue *out);

        void Proceed() final;

    private:
        SerializedDataPayload request;
        SimpleConfirm reply;
        grpc::ServerAsyncResponseWriter<SimpleConfirm> responder;
    };

    // This can be run in multiple threads if needed.
    void HandleRpcsToGPU();

    void HandleRpcsToCPU();

    std::unique_ptr<ServerCompletionQueue> cq;
    DataTransferService::AsyncService service;
    std::unique_ptr<Server> server;

    ThreadSafeFixSizedDoubleQueue *LoadingQueue;
};

#endif //PIPEPLUSPLUS_RECEIVER_H
