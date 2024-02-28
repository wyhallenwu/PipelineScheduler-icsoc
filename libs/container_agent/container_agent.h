#include <vector>
#include <thread>
#include "absl/strings/str_format.h"
#include "absl/flags/flag.h"
#include <grpcpp/grpcpp.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/health_check_service_interface.h>

#include "../json/json.h"
#include "../communicator/sender.cpp"
#include "../communicator/receiver.cpp"
#include "../protobufprotocols/indevicecommunication.grpc.pb.h"

ABSL_FLAG(std::string, name, "", "base name of container");
ABSL_FLAG(std::string, json, "", "configurations for microservices");
ABSL_FLAG(uint16_t, port, 0, "Server port for the service");

using json = nlohmann::json;

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

namespace msvcconfigs {
    void from_json(const json &j, NeighborMicroserviceConfigs &val)
    {
        j.at("name").get_to(val.name);
        j.at("comm").get_to(val.commMethod);
        j.at("link").get_to(val.link);
        j.at("qt").get_to(val.queueType);
        j.at("maxqs").get_to(val.maxQueueSize);
        j.at("coi").get_to(val.classOfInterest);
        j.at("shape").get_to(val.expectedShape);
    }

    void from_json(const json &j, BaseMicroserviceConfigs &val)
    {
        j.at("name").get_to(val.msvc_name);
        j.at("type").get_to(val.msvc_type);
        j.at("slo").get_to(val.msvc_svcLevelObjLatency);
        j.at("bs").get_to(val.msvc_idealBatchSize);
        j.at("ds").get_to(val.msvc_dataShape);
        j.at("upstrm").get_to(val.upstreamMicroservices);
        j.at("downstrm").get_to(val.dnstreamMicroservices);
    }
}

class ContainerAgent {
public:
    ContainerAgent(const std::string &name, const std::string &url, uint16_t device_port, uint16_t own_port,
                   std::vector<BaseMicroserviceConfigs> &msvc_configs);

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
    void ReportStart(int port);

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

    void HandleRecvRpcs();

    std::string name;
    std::vector<Microservice<void> *> msvcs;
    std::unique_ptr<ServerCompletionQueue> server_cq;
    CompletionQueue *sender_cq;
    InDeviceCommunication::AsyncService service;
    std::unique_ptr<Server> server;
    std::unique_ptr<InDeviceCommunication::Stub> stub;
    std::atomic<bool> run{};
};