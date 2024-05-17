#ifndef PIPEPLUSPLUS_RECEIVER_H
#define PIPEPLUSPLUS_RECEIVER_H

#include "communicator.h"
#include <fstream>
#include <random>

using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using boost::interprocess::read_only;
using boost::interprocess::open_only;
using json = nlohmann::ordered_json;

struct ReceiverConfigs : BaseMicroserviceConfigs {
    uint16_t msvc_numWarmUpBatches;
    uint16_t msvc_numProfileBatches;
    uint8_t msvc_inputRandomizeScheme;
    std::string msvc_dataShape;
};

class Receiver : public Microservice {
public:
    Receiver(const json &jsonConfigs);

    ~Receiver() override {
        server->Shutdown();
        cq->Shutdown();
    }
    // Data generator for profiling
    void profileDataGenerator();

    template<typename ReqDataType>
    void processInferTimeReport(Request<ReqDataType> &timeReport);

    void dispatchThread() override {
        std::thread handler(&Receiver::HandleRpcs, this);
        handler.detach();
    }

    ReceiverConfigs loadConfigsFromJson(const json &jsonConfigs);

    void loadConfigs(const json &jsonConfigs, bool isConstructing = true);
    
private:
    class RequestHandler {
    public:
        RequestHandler(DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
                       ThreadSafeFixSizedDoubleQueue *out, uint64_t &msvc_inReqCount)
                : service(service), msvc_inReqCount(msvc_inReqCount), cq(cq), OutQueue(out), status(CREATE) {};

        virtual ~RequestHandler() = default;

        virtual void Proceed() = 0;

    protected:
        enum CallStatus {
            CREATE, PROCESS, FINISH
        };

        std::string containerName;
        DataTransferService::AsyncService *service;
        uint64_t &msvc_inReqCount;
        ServerCompletionQueue *cq;
        ServerContext ctx;
        ThreadSafeFixSizedDoubleQueue *OutQueue;
        CallStatus status;
    };

    class GpuPointerRequestHandler : public RequestHandler {
    public:
        GpuPointerRequestHandler(DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
                       ThreadSafeFixSizedDoubleQueue *out, uint64_t &msvc_inReqCount);

        void Proceed() final;

    private:
        ImageDataPayload request;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
    };

    class SharedMemoryRequestHandler : public RequestHandler {
    public:
        SharedMemoryRequestHandler(DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
                       ThreadSafeFixSizedDoubleQueue *out, uint64_t &msvc_inReqCount);

        void Proceed() final;

        void test() {
            ImageDataPayload request;
        }

    private:
        ImageDataPayload request;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
    };

    class SerializedDataRequestHandler : public RequestHandler {
    public:
        SerializedDataRequestHandler(DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
                       ThreadSafeFixSizedDoubleQueue *out, uint64_t &msvc_inReqCount);

        void Proceed() final;

    private:
        ImageDataPayload request;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
    };

    // This can be run in multiple threads if needed.
    void HandleRpcs();

    std::unique_ptr<ServerCompletionQueue> cq;
    DataTransferService::AsyncService service;
    std::unique_ptr<grpc::Server> server;
};

#endif //PIPEPLUSPLUS_RECEIVER_H
