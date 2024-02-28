#include "../container_agent/container_agent.h"
#include "../protobufprotocols/indevicecommunication.grpc.pb.h"
#include <cstdlib>

enum ContainerType {
    DataSource,
    Yolo5,
};

struct ContainerHandle {
    google::protobuf::RepeatedField<int32_t> queuelengths;
    std::unique_ptr<InDeviceCommunication::Stub> stub;
    CompletionQueue *cq;
};

namespace msvcconfigs {
    void to_json(json &j, const NeighborMicroserviceConfigs &val) {
        j["name"] = val.name;
        j["comm"] = val.commMethod;
        j["link"] = val.link;
        j["qt"] = val.queueType;
        j["maxqs"] = val.maxQueueSize;
        j["coi"] = val.classOfInterest;
        j["shape"] = val.expectedShape;
    }

    void to_json(json &j, const BaseMicroserviceConfigs &val) {
        j["name"] = val.msvc_name;
        j["type"] = val.msvc_type;
        j["slo"] = val.msvc_svcLevelObjLatency;
        j["bs"] = val.msvc_idealBatchSize;
        j["ds"] = val.msvc_dataShape;
        j["upstrm"] = val.upstreamMicroservices;
        j["downstrm"] = val.dnstreamMicroservices;
    }
}

class DeviceAgent {
public:
    DeviceAgent(const std::string &controller_url, uint16_t controller_port);

    ~DeviceAgent() {
        for (const auto &c: containers) {
            StopContainer(c.second);
        }
        server->Shutdown();
        server_cq->Shutdown();
    };

    void UpdateQueueLengths(const std::basic_string<char> &container_name,
                            const google::protobuf::RepeatedField<int32_t> &queuelengths) {
        containers[container_name].queuelengths = queuelengths;
    };

private:
    void CreateYolo5Container(int id, const NeighborMicroserviceConfigs &upstream,
                              const std::vector<NeighborMicroserviceConfigs> &downstreams, const MsvcSLOType &slo);

    json createConfigs(
            const std::vector<std::tuple<std::string, MicroserviceType, QueueType, std::vector<RequestShapeType>>> &data,
            const MsvcSLOType &slo, const NeighborMicroserviceConfigs &prev_msvc,
            const std::vector<NeighborMicroserviceConfigs> &next_msvc);

    void runDocker(const std::string &name, const std::string &start_string, const int &port) {
        system(absl::StrFormat(R"(docker run -p %i:%i pipeline-base-container --name="%s"--json="%s" --port=%i)", port,
                               port, name, start_string,port).c_str());
    };

    static void StopContainer(const ContainerHandle &container);

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

    class CounterUpdateRequestHandler : public RequestHandler {
    public:
        CounterUpdateRequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                    DeviceAgent *device)
                : RequestHandler(service, cq), responder(&ctx), device_agent(device) {
            Proceed();
        }

        void Proceed() final;

    private:
        QueueSize request;
        indevicecommunication::SimpleConfirm reply;
        grpc::ServerAsyncResponseWriter<indevicecommunication::SimpleConfirm> responder;
        DeviceAgent *device_agent;
    };

    class ReportStartRequestHandler : public RequestHandler {
    public:
        ReportStartRequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                  DeviceAgent *device)
                : RequestHandler(service, cq), responder(&ctx), device_agent(device) {
            Proceed();
        }

        void Proceed() final;

    private:
        indevicecommunication::ConnectionConfigs request;
        indevicecommunication::SimpleConfirm reply;
        grpc::ServerAsyncResponseWriter<indevicecommunication::SimpleConfirm> responder;
        DeviceAgent *device_agent;
    };

    void HandleRecvRpcs();

    std::map<std::string, ContainerHandle> containers;
    std::unique_ptr<ServerCompletionQueue> server_cq;
    std::unique_ptr<Server> server;
    CompletionQueue *sender_cq;
    std::unique_ptr<InDeviceCommunication::Stub> controller_stub;
    InDeviceCommunication::AsyncService service;
};