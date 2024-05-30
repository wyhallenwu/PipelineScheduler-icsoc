#ifndef PIPEPLUSPLUS_CONTROLLER_H
#define PIPEPLUSPLUS_CONTROLLER_H

#include "microservice.h"
#include <grpcpp/grpcpp.h>
#include "../json/json.h"
#include <thread>
#include "controlcommunication.grpc.pb.h"
#include <LightGBM/c_api.h>
#include <pqxx/pqxx>
#include "absl/strings/str_format.h"
#include "absl/flags/parse.h"
#include "absl/flags/flag.h"

using grpc::Status;
using grpc::CompletionQueue;
using grpc::ClientContext;
using grpc::ClientAsyncResponseReader;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using controlcommunication::ControlCommunication;
using controlcommunication::ConnectionConfigs;
using controlcommunication::Neighbor;
using controlcommunication::ContainerConfig;
using controlcommunication::ContainerLink;
using controlcommunication::ContainerInt;
using controlcommunication::ContainerSignal;
using EmptyMessage = google::protobuf::Empty;

enum SystemDeviceType {
    Server,
    Edge
};

enum ModelType {
    DataSource,
    Sink,
    Yolov5, // = Yolov5n
    Yolov5n320,
    Yolov5s,
    Yolov5m,
    Yolov5Datasource,
    Arcface,
    Retinaface,
    Yolov5_Plate,
    Movenet,
    Emotionnet,
    Gender,
    Age,
    CarBrand
};

extern std::map<ModelType, std::vector<std::string>> MODEL_INFO;

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

    void Scheduling();

    void AddTask(const TaskDescription::TaskStruct &task);

    [[nodiscard]] bool isRunning() const { return running; };

    void Stop() { running = false; };

private:
    struct ContainerHandle;
    struct NodeHandle {
        std::string ip;
        std::shared_ptr<ControlCommunication::Stub> stub;
        CompletionQueue *cq;
        SystemDeviceType type;
        int num_processors; // number of processing units, 1 for Edge or # GPUs for server
        std::vector<double> processors_utilization; // utilization per pu
        std::vector<unsigned long> mem_size; // memory size in MB
        std::vector<double> mem_utilization; // memory utilization per pu
        int next_free_port;
        std::map<std::string, ContainerHandle *> containers;
    };

    struct TaskHandle {
        int last_latency;
        int slo;
        PipelineType type;
        std::map<std::string, ContainerHandle *> subtasks;
    };

    struct ContainerHandle {
        std::string name;
        ModelType model;
        NodeHandle *device_agent;
        TaskHandle *task;
        int batch_size;
        int replicas;
        std::vector<int> cuda_device;
        int class_of_interest;
        int recv_port;
        Metrics metrics;
        google::protobuf::RepeatedField<int32_t> queue_lengths;
        std::vector<ContainerHandle *> upstreams;
        std::vector<ContainerHandle *> downstreams;
        // TODO: remove test code
        bool running;
    };

    float queryRequestRateInPeriod(const std::string &name, const uint32_t &period);

    void UpdateLightMetrics();

    void UpdateFullMetrics();

    double LoadTimeEstimator(const char *model_path, double input_mem_size);
    int InferTimeEstimator(ModelType model, int batch_size);


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

    void StartContainer(std::pair<std::string, ContainerHandle *> &upstr, int slo,
                        std::string source = "", int replica = 1);

    void FakeContainer(ContainerHandle* cont, int slo);
    void FakeStartContainer(std::pair<std::string, ContainerHandle *> &cont, int slo, int replica = 1);

    void MoveContainer(ContainerHandle *msvc, int cuda_device, bool to_edge, int replica = 1);

    static void AdjustUpstream(int port, ContainerHandle *msvc, NodeHandle *new_device, const std::string &dwnstr);

    void AdjustBatchSize(ContainerHandle *msvc, int new_bs);

    void StopContainer(std::string name, NodeHandle *device, bool forced = false);

    void optimizeBatchSizeStep(
            const std::vector<std::pair<ModelType, std::vector<std::pair<ModelType, int>>>> &models,
            std::map<ModelType, int> &batch_sizes, std::map<ModelType, int> &estimated_infer_times, int nObjects);

    std::map<ModelType, int> getInitialBatchSizes(
            const std::vector<std::pair<ModelType, std::vector<std::pair<ModelType, int>>>> &models, int slo,
            int nObjects);

    std::vector<std::pair<ModelType, std::vector<std::pair<ModelType, int>>>>
    getModelsByPipelineType(PipelineType type);

    bool running;
    std::map<std::string, NodeHandle> devices;
    std::map<std::string, TaskHandle> tasks;
    std::map<std::string, ContainerHandle> containers;

    ControlCommunication::AsyncService service;
    std::unique_ptr<grpc::Server> server;
    std::unique_ptr<ServerCompletionQueue> cq;

    std::unique_ptr<pqxx::connection> ctl_metricsServerConn = nullptr;
    MetricsServerConfigs ctl_metricsServerConfigs;
};


#endif //PIPEPLUSPLUS_CONTROLLER_H
