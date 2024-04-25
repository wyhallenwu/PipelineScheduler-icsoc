#ifndef PIPEPLUSPLUS_CONTROLLER_H
#define PIPEPLUSPLUS_CONTROLLER_H

#include "microservice.h"
#include <grpcpp/grpcpp.h>
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
using controlcommunication::MicroserviceConfig;
using controlcommunication::MicroserviceName;
using EmptyMessage = google::protobuf::Empty;

enum DeviceType {
    Server,
    Edge
};

enum ModelType {
    Yolov5,
    Arcface,
    Retinaface,
    Yolov5_Plate
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

class Controller {
public:
    Controller();

    ~Controller() = default;

    void HandleRecvRpcs();

    void AddTask(std::string name, int slo, PipelineType type, std::string source, std::string device);

    bool isRunning() { return running; };

    void Stop() { running = false; };

    void UpdateLightMetrics(google::protobuf::RepeatedPtrField<LightMetrics> metrics);

    void UpdateFullMetrics(google::protobuf::RepeatedPtrField<FullMetrics> metrics);

private:
    struct MicroserviceHandle;
    struct NodeHandle {
        std::shared_ptr<ControlCommunication::Stub> stub;
        CompletionQueue *cq;
        DeviceType type;
        int num_processors; // number of processing units, general cores for Edge or GPUs for server
        unsigned long mem_size; // memory size in MB
        std::vector<MicroserviceHandle *> microservices;
    };

    struct TaskHandle {
        int slo;
        PipelineType type;
        std::vector<MicroserviceHandle *> subtasks;
    };

    struct MicroserviceHandle {
        ModelType model;
        NodeHandle *device_agent;
        TaskHandle *task;
        google::protobuf::RepeatedField<int32_t> queue_lengths;
        Metrics metrics;
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
                : RequestHandler(service, cq, c){
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


    bool running;
    std::map<std::string, NodeHandle> devices;
    std::map<std::string, TaskHandle> tasks;
    std::map<std::string, MicroserviceHandle> microservices;

    ControlCommunication::AsyncService service;
    std::unique_ptr<grpc::Server> server;
    std::unique_ptr<ServerCompletionQueue> cq;
};


#endif //PIPEPLUSPLUS_CONTROLLER_H
