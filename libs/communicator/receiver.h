#ifndef PIPEPLUSPLUS_RECEIVER_H
#define PIPEPLUSPLUS_RECEIVER_H

#include "communicator.h"
#include <fstream>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using boost::interprocess::read_only;
using boost::interprocess::open_only;
using json = nlohmann::json;

class Receiver : public Microservice {
public:
    Receiver(const BaseMicroserviceConfigs &configs);

    ~Receiver() override {
        server->Shutdown();
        cq->Shutdown();
    }

protected:
    void readConfigsFromJson(std::string cfgPath) {
        spdlog::trace("{0:s} attempts to parse Profiling configs from json file.", __func__);
        std::ifstream file(cfgPath);
        json j = json::parse(file);
        j.at("msvc_dataShape").get_to(msvc_dataShape);
        j.at("msvc_numWarmUpBatches").get_to(msvc_numWarmUpBatches);
        j.at("msvc_numProfileBatches").get_to(msvc_numProfileBatches);

        spdlog::trace("{0:s} finished parsing Config from file.", __func__);
    }

private:
    uint16_t msvc_numWarmUpBatches, msvc_numProfileBatches;
    uint8_t msvc_inputRandomizeScheme;
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
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
    };

    class SharedMemoryRequestHandler : public RequestHandler {
    public:
        SharedMemoryRequestHandler(DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
                                   ThreadSafeFixSizedDoubleQueue *out);

        void Proceed() final;

        void test() {
            SharedMemPayload request;
        }

    private:
        SharedMemPayload request;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
    };

    class SerializedDataRequestHandler : public RequestHandler {
    public:
        SerializedDataRequestHandler(DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
                                     ThreadSafeFixSizedDoubleQueue *out);

        void Proceed() final;

    private:
        SerializedDataPayload request;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
    };

    // This can be run in multiple threads if needed.
    void HandleRpcs();

    std::unique_ptr<ServerCompletionQueue> cq;
    DataTransferService::AsyncService service;
    std::unique_ptr<Server> server;
};

#endif //PIPEPLUSPLUS_RECEIVER_H
