#include <vector>
#include <thread>
#include "absl/strings/str_format.h"
#include <grpcpp/grpcpp.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/health_check_service_interface.h>

#include "../communicator/sender.cpp"
#include "../communicator/receiver.cpp"
#include "../protobufprotocols/indevicecommunication.grpc.pb.h"

using indevicecommunication::InDeviceCommunication;
using indevicecommunication::QueueSize;

enum TransferMethod {
    LocalCPU,
    RemoteCPU,
    GPU
};

struct ConnectionConfigs {
    std::string ip;
    int port;
};

class ContainerAgent {
public:
    ContainerAgent(const std::string &url, uint16_t device_port, uint16_t own_port,
                   std::vector<std::pair<BaseMicroserviceConfigs, TransferMethod>> &msvc_configs,
                   ConnectionConfigs &InConfigs, ConnectionConfigs &OutConfigs);

    ~ContainerAgent() {
        for (auto msvc: msvcs) {
            delete msvc;
        }
        server->Shutdown();
        server_cq->Shutdown();
        sender_cq->Shutdown();
    };

    [[nodiscard]] bool running() const {
        return run;
    }

    void SendQueueLengths();

private:
    class RequestHandler {
    public:
        RequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq)
                : service(service), cq(cq), status(CREATE) {};

        virtual ~RequestHandler() = default;
        virtual void Proceed() = 0;

    protected:
        enum CallStatus {
            CREATE, PROCESS, FINISH
        };

        InDeviceCommunication::AsyncService *service;
        ServerCompletionQueue *cq;
        ServerContext ctx;
        CallStatus status;
    };

    class StopRequestHandler : public RequestHandler {
    public:
        StopRequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq,
                           std::atomic<bool> *run)
                : RequestHandler(service, cq), responder(&ctx), run(run) {
            Proceed();
        }

        void Proceed() final;

    private:
        indevicecommunication::SimpleConfirm request;
        indevicecommunication::SimpleConfirm reply;
        grpc::ServerAsyncResponseWriter<indevicecommunication::SimpleConfirm> responder;
        std::atomic<bool> *run;
    };

    void HandleOutRpcs();

    std::vector<std::thread> msvc_threads;
    std::vector<Microservice<void> *> msvcs;
    std::unique_ptr<ServerCompletionQueue> server_cq;
    CompletionQueue *sender_cq;
    InDeviceCommunication::AsyncService service;
    std::unique_ptr<Server> server;
    std::unique_ptr<InDeviceCommunication::Stub> stub;
    std::atomic<bool> run{};
};