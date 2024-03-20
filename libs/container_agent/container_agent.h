#ifndef CONTAINER_AGENT_H
#define CONTAINER_AGENT_H

#include <vector>
#include <thread>
#include <fstream>
#include "absl/strings/str_format.h"
#include "absl/flags/parse.h"
#include "absl/flags/flag.h"
#include <grpcpp/grpcpp.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/health_check_service_interface.h>

#include "../json/json.h"
#include "microservice.h"
#include "indevicecommunication.grpc.pb.h"

ABSL_DECLARE_FLAG(std::string, name);
ABSL_DECLARE_FLAG(std::optional<std::string>, json);
ABSL_DECLARE_FLAG(std::optional<std::string>, json_path);
ABSL_DECLARE_FLAG(std::optional<std::string>, trt_json);
ABSL_DECLARE_FLAG(uint16_t, port);

using json = nlohmann::json;

using grpc::Status;
using grpc::CompletionQueue;
using grpc::ClientContext;
using grpc::ClientAsyncResponseReader;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using indevicecommunication::InDeviceCommunication;
using indevicecommunication::QueueSize;
using indevicecommunication::StaticConfirm;

enum TransferMethod {
    LocalCPU,
    RemoteCPU,
    GPU
};

namespace msvcconfigs {
    void from_json(const json &j, NeighborMicroserviceConfigs &val);

    void from_json(const json &j, BaseMicroserviceConfigs &val);

    std::vector<BaseMicroserviceConfigs> LoadFromJson();
}

class ContainerAgent {
public:
    ContainerAgent(const std::string &name, uint16_t own_port);

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

protected:
    void ReportStart();

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
        StaticConfirm request;
        StaticConfirm reply;
        grpc::ServerAsyncResponseWriter<StaticConfirm> responder;
        std::atomic<bool> *run;
    };

    void HandleRecvRpcs();

    std::string name;
    std::vector<Microservice*> msvcs;
    std::unique_ptr<ServerCompletionQueue> server_cq;
    CompletionQueue *sender_cq;
    InDeviceCommunication::AsyncService service;
    std::unique_ptr<Server> server;
    std::unique_ptr<InDeviceCommunication::Stub> stub;
    std::atomic<bool> run;
};

#endif