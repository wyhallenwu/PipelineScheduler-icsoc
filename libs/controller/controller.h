#ifndef PIPEPLUSPLUS_CONTROLLER_H
#define PIPEPLUSPLUS_CONTROLLER_H

#include "microservice.h"
#include <grpcpp/grpcpp.h>
#include "../json/json.h"
#include <thread>
#include "controlcommunication.grpc.pb.h"

using grpc::Status;
using grpc::CompletionQueue;
using grpc::ClientContext;
using grpc::ClientAsyncResponseReader;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using controlcommunication::ControlCommunication;
using controlcommunication::LightMetrics;
using controlcommunication::LightMetricsList;
using controlcommunication::FullMetrics;
using controlcommunication::FullMetricsList;
using controlcommunication::ConnectionConfigs;
using controlcommunication::Neighbor;
using controlcommunication::MicroserviceConfig;
using controlcommunication::MicroserviceLink;
using controlcommunication::MicroserviceSignal;
using EmptyMessage = google::protobuf::Empty;

enum DeviceType {
    Server,
    Edge
};

enum ModelType {
    DataSource,
    Yolov5,
    Yolov5Datasource,
    Arcface,
    Retinaface,
    Yolov5_Plate,
    Movenet,
    Emotionnet,
    Gender,
    Age
};

std::map<std::string, ModelType> MODEL_TYPES = {
        {":datasource",       DataSource},
        {":yolov5",           Yolov5},
        {":yolov5datasource", Yolov5Datasource},
        {":retinaface",       Retinaface},
        {":arcface",          Arcface},
        {":cartype",          Yolov5_Plate}, // Still needs to be finished
        {":plate",            Yolov5_Plate},
        {":gender",           Gender},
        {":age",              Age},
        {":movenet",          Movenet},
        {":emotion",          Emotionnet}
};

enum PipelineType {
    Traffic,
    Video_Call,
    Building_Security
};

struct Metrics {
    float requestRate = 0;
    double cpuUsage = 0;
    long memUsage = 0;
    unsigned int gpuUsage = 0;
    unsigned int gpuMemUsage = 0;
};

namespace TaskDescription {
    struct TaskStruct {
        std::string name;
        int slo;
        PipelineType type;
        std::string source;
        std::string device;
    };

    void to_json(nlohmann::json &j, const TaskStruct &val);

    void from_json(const nlohmann::json &j, TaskStruct &val);
}

class Controller {
public:
    Controller();

    ~Controller();

    void HandleRecvRpcs();

    void AddTask(const TaskDescription::TaskStruct &task);

    [[nodiscard]] bool isRunning() const { return running; };

    void Stop() { running = false; };

private:
    void UpdateLightMetrics(google::protobuf::RepeatedPtrField<LightMetrics> metrics);
    void UpdateFullMetrics(google::protobuf::RepeatedPtrField<FullMetrics> metrics);

    struct MicroserviceHandle;
    struct NodeHandle {
        std::string ip;
        std::shared_ptr<ControlCommunication::Stub> stub;
        CompletionQueue *cq;
        DeviceType type;
        int num_processors; // number of processing units, general cores for Edge or GPUs for server
        unsigned long mem_size; // memory size in MB
        std::map<std::string, MicroserviceHandle *> microservices;
        int next_free_port;
    };

    struct TaskHandle {
        int slo;
        PipelineType type;
        std::map<std::string, MicroserviceHandle *> subtasks;
    };

    struct MicroserviceHandle {
        std::string name;
        ModelType model;
        NodeHandle *device_agent;
        TaskHandle *task;
        int class_of_interest;
        int recv_port;
        Metrics metrics;
        google::protobuf::RepeatedField<int32_t> queue_lengths;
        std::vector<MicroserviceHandle *> upstreams;
        std::vector<MicroserviceHandle *> downstreams;
    };

    class RequestHandler {
    public:
        RequestHandler(ControlCommunication::AsyncService *service, ServerCompletionQueue *cq, Controller *c)
                : service(service), cq(cq), status(CREATE), controller(c), responder(&ctx) {}

        virtual ~RequestHandler() = default;

        virtual void Proceed() = 0;

    protected:
        enum CallStatus {
            CREATE, PROCESS, FINISH
        };
        ControlCommunication::AsyncService *service;
        ServerCompletionQueue *cq;
        ServerContext ctx;
        CallStatus status;
        Controller *controller;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
    };

    class LightMetricsRequestHandler : public RequestHandler {
    public:
        LightMetricsRequestHandler(ControlCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                   Controller *c)
                : RequestHandler(service, cq, c) {
            Proceed();
        }

        void Proceed() final;

    private:
        LightMetricsList request;
    };

    class FullMetricsRequestHandler : public RequestHandler {
    public:
        FullMetricsRequestHandler(ControlCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                  Controller *c)
                : RequestHandler(service, cq, c) {
            Proceed();
        }

        void Proceed() final;

    private:
        FullMetricsList request;
    };

    class DeviseAdvertisementHandler : public RequestHandler {
    public:
        DeviseAdvertisementHandler(ControlCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                   Controller *c)
                : RequestHandler(service, cq, c) {
            Proceed();
        }

        void Proceed() final;

    private:
        ConnectionConfigs request;
    };

    void StartMicroservice(std::pair<std::string, MicroserviceHandle *> &upstr, int slo, int batch_size,
                           std::string source = "");
    void MoveMicroservice(MicroserviceHandle *msvc, bool to_edge);
    static void AdjustUpstream(int port, MicroserviceHandle *msvc, NodeHandle *new_device);
    void StopMicroservice(std::string name, NodeHandle *device, bool forced = false);

    static std::vector<std::pair<std::string, std::vector<std::pair<std::string, int>>>>
    getModelsByPipelineType(PipelineType type);

    bool running;
    std::map<std::string, NodeHandle> devices;
    std::map<std::string, TaskHandle> tasks;
    std::map<std::string, MicroserviceHandle> microservices;

    ControlCommunication::AsyncService service;
    std::unique_ptr<grpc::Server> server;
    std::unique_ptr<ServerCompletionQueue> cq;
};


#endif //PIPEPLUSPLUS_CONTROLLER_H
