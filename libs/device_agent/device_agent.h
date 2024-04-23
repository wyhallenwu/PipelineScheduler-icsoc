#ifndef DEVICE_AGENT_H
#define DEVICE_AGENT_H

#include "container_agent.h"
#include "profiler.h"
#include "indevicecommunication.grpc.pb.h"
#include "controlcommunication.grpc.pb.h"
#include <cstdlib>
#include <misc.h>

using controlcommunication::ControlCommunication;
using controlcommunication::QueueSizes;
using controlcommunication::MicroserviceName;
using controlcommunication::MicroserviceConfig;

using trt::TRTConfigs;



typedef std::tuple<
    std::string, // name
    MicroserviceType, // type
    QueueLengthType, // queue length type
    int16_t, // class of interests
    std::vector<RequestDataShapeType>, //data shape
    QueueLengthType
> MsvcConfigTupleType;

enum ContainerType {
    DataSource,
    Yolo5,
};

struct ContainerHandle {
    google::protobuf::RepeatedField <int32_t> queuelengths;
    std::unique_ptr<InDeviceCommunication::Stub> stub;
    CompletionQueue *cq;
    unsigned int pid;
};

namespace msvcconfigs {
    void to_json(json &j, const NeighborMicroserviceConfigs &val);

    void to_json(json &j, const BaseMicroserviceConfigs &val);
}

class DeviceAgent {
public:
    DeviceAgent(const std::string &controller_url);

    ~DeviceAgent() {
        for (const auto &c: containers) {
            StopContainer(c.second);
        }
        device_server->Shutdown();
        device_cq->Shutdown();
    };

    void UpdateQueueLengths(const std::basic_string<char> &container_name,
                            const google::protobuf::RepeatedField <int32_t> &queuelengths) {
        containers[container_name].queuelengths = queuelengths;
    };

private:
    void CreateYolo5Container(
        int id,
        const NeighborMicroserviceConfigs &upstream,
        const std::vector<NeighborMicroserviceConfigs> &downstreams,
        const MsvcSLOType &slo,
        const BatchSizeType &batchSize,
        const std::string &logPath
    );

    void CreateDataSource(
        int id,
        const std::vector<NeighborMicroserviceConfigs> &downstreams,
        const MsvcSLOType &slo,
        const std::string &video_path,
        const std::string &logPath
    );

    static json createConfigs(
            const std::vector<MsvcConfigTupleType> &data,
            const MsvcSLOType &slo,
            const BatchSizeType &batchSize,
            const std::string &logPath,
            const NeighborMicroserviceConfigs &prev_msvc,
            const std::vector<NeighborMicroserviceConfigs> &next_msvc);

    void finishContainer(const std::string &executable, const std::string &name, const std::string &start_string,
                         const int &control_port, const int &data_port, const std::string &trt_config = "");

    static int runDocker(const std::string &executable, const std::string &name, const std::string &start_string,
                         const int &port, const std::string &trt_config) {
        if (trt_config.empty()) {
            std::cout << absl::StrFormat(
                    R"(docker run --network=host -d --gpus 1 pipeline-base-container %s --name="%s" --json='%s' --port=%i)",
                    executable, name, start_string, port).c_str() << std::endl;
            return system(absl::StrFormat(
                    R"(docker run --network=host -d --gpus 1 pipeline-base-container %s --name="%s" --json='%s' --port=%i)",
                    executable, name, start_string, port).c_str());
        } else {
            std::cout << absl::StrFormat(
                    R"(docker run --network=host -d --gpus 1 pipeline-base-container %s --name="%s" --json='%s' --port=%i --trt_json='%s')",
                    executable, name, start_string, port, trt_config).c_str() << std::endl;
            return system(absl::StrFormat(
                    R"(docker run --network=host -d --gpus 1 pipeline-base-container %s --name="%s" --json='%s' --port=%i --trt_json='%s')",
                    executable, name, start_string, port, trt_config).c_str());
        }
    };

    static void StopContainer(const ContainerHandle &container);

    class RequestHandler {
    public:
        RequestHandler(ServerCompletionQueue *cq) : cq(cq), status(CREATE) {}
        virtual ~RequestHandler() = default;
        virtual void Proceed() = 0;
    protected:
        enum CallStatus {
            CREATE, PROCESS, FINISH
        };
        ServerCompletionQueue *cq;
        ServerContext ctx;
        CallStatus status;
    };

    class DeviceRequestHandler : public RequestHandler {
    public:
        DeviceRequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq)
                : RequestHandler(cq), service(service) {};

    protected:
        InDeviceCommunication::AsyncService *service;
    };

    class ControlRequestHandler : public RequestHandler {
    public:
        ControlRequestHandler(ControlCommunication::AsyncService *service, ServerCompletionQueue *cq)
                : RequestHandler(cq), service(service) {};

    protected:
        ControlCommunication::AsyncService *service;
    };

    class CounterUpdateRequestHandler : public DeviceRequestHandler {
    public:
        CounterUpdateRequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                    DeviceAgent *device)
                : DeviceRequestHandler(service, cq), responder(&ctx), device_agent(device) {
            Proceed();
        }

        void Proceed() final;

    private:
        QueueSize request;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
        DeviceAgent *device_agent;
    };

    class ReportStartRequestHandler : public DeviceRequestHandler {
    public:
        ReportStartRequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                  DeviceAgent *device)
                : DeviceRequestHandler(service, cq), responder(&ctx), device_agent(device) {
            Proceed();
        }

        void Proceed() final;

    private:
        indevicecommunication::ConnectionConfigs request;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
        DeviceAgent *device_agent;
    };

    class StartMicroserviceRequestHandler : public ControlRequestHandler {
    public:
        StartMicroserviceRequestHandler(ControlCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                       DeviceAgent *device)
                : ControlRequestHandler(service, cq), responder(&ctx), device_agent(device) {
            Proceed();
        }

        void Proceed() final;

    private:
        MicroserviceConfig request;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
        DeviceAgent *device_agent;
    };

    class StopMicroserviceRequestHandler : public ControlRequestHandler {
    public:
        StopMicroserviceRequestHandler(ControlCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                  DeviceAgent *device)
                : ControlRequestHandler(service, cq), responder(&ctx), device_agent(device) {
            Proceed();
        }

        void Proceed() final;

    private:
        MicroserviceName request;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
        DeviceAgent *device_agent;
    };

    void HandleDeviceRecvRpcs();
    void HandleControlRecvRpcs();

    std::map<std::string, ContainerHandle> containers;
    std::unique_ptr<ServerCompletionQueue> device_cq;
    std::unique_ptr<Server> device_server;
    InDeviceCommunication::AsyncService device_service;
    std::unique_ptr<ControlCommunication::Stub> controller_stub;
    std::unique_ptr<ServerCompletionQueue> controller_cq;
    std::unique_ptr<Server> controller_server;
    CompletionQueue *controller_sending_cq;
    ControlCommunication::AsyncService controller_service;

    // This will be mounted into the container to easily collect all logs.
    std::string dev_logPath = "../logs";
};

#endif //DEVICE_AGENT_H