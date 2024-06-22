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

ABSL_DECLARE_FLAG(std::string, ctrl_configPath);
ABSL_DECLARE_FLAG(uint16_t, ctrl_verbose);
ABSL_DECLARE_FLAG(uint16_t, ctrl_loggingMode);

typedef std::vector<std::pair<ModelType, std::vector<std::pair<ModelType, int>>>> Pipeline;


struct HardwareMetrics {
    ClockType timestamp;
    CpuUtilType cpuUsage = 0;
    MemUsageType memUsage = 0;
    MemUsageType rssMemUsage = 0;
    GpuUtilType gpuUsage = 0;
    GpuMemUsageType gpuMemUsage = 0;
};

struct SummarizedHardwareMetrics {
    CpuUtilType cpuUsage = 0;
    MemUsageType memUsage = 0;
    MemUsageType rssMemUsage = 0;
    GpuUtilType gpuUsage = 0;
    GpuMemUsageType gpuMemUsage = 0;

    bool metricsAvailable = false;

    SummarizedHardwareMetrics& operator= (const SummarizedHardwareMetrics &metrics) {
        metricsAvailable = true;
        cpuUsage = std::max(metrics.cpuUsage, cpuUsage);
        memUsage = std::max(metrics.memUsage, memUsage);
        rssMemUsage = std::max(metrics.rssMemUsage, rssMemUsage);
        gpuUsage = std::max(metrics.gpuUsage, gpuUsage);
        gpuMemUsage = std::max(metrics.gpuMemUsage, gpuMemUsage);
        return *this;
    }

    void clear() {
        metricsAvailable = false;
        cpuUsage = 0;
        memUsage = 0;
        rssMemUsage = 0;
        gpuUsage = 0;
        gpuMemUsage = 0;
    }
};

namespace TaskDescription {
    struct TaskStruct {
        std::string name;
        int slo;
        PipelineType type;
        std::string source;
        std::string device;
    };

    void from_json(const nlohmann::json &j, TaskStruct &val);
}

class Controller {
public:
    Controller(int argc, char **argv);

    ~Controller();

    void HandleRecvRpcs();

    void Scheduling();

    void Init() { for (auto &t: initialTasks) AddTask(t); }

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
        std::string name;
        PipelineType type;
        std::string source;
        int slo;
        ClockType start_time;
        int last_latency;
        std::map<std::string, ContainerHandle *> subtasks;
    };

    struct ContainerHandle {
        std::string name;
        int class_of_interest;
        ModelType model;
        bool mergable;
        std::vector<int> dimensions;

        int replicas;
        std::vector<int> batch_size;
        std::vector<int> cuda_device;
        std::vector<int> recv_port;

        HardwareMetrics metrics;
        NodeHandle *device_agent;
        TaskHandle *task;
        std::vector<ContainerHandle *> upstreams;
        std::vector<ContainerHandle *> downstreams;
    };

    void readConfigFile(const std::string &config_path);

    double LoadTimeEstimator(const char *model_path, double input_mem_size);
    int InferTimeEstimator(ModelType model, int batch_size);
    std::map<ModelType, std::vector<int>> InitialRequestCount(const std::string &input, const Pipeline &models,
                                                              int fps = 30);


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
                        std::string source, int replica = 1, bool easy_allocation = true);

    void MoveContainer(ContainerHandle *msvc, bool to_edge, int cuda_device = 0, int replica = 1);

    static void AdjustUpstream(int port, ContainerHandle *msvc, NodeHandle *new_device, const std::string &dwnstr);

    static void SyncDatasource(Controller::ContainerHandle *prev, Controller::ContainerHandle *curr);

    void AdjustBatchSize(ContainerHandle *msvc, int new_bs, int replica = 1);

    void StopContainer(std::string name, NodeHandle *device, bool forced = false);

    void optimizeBatchSizeStep(
            const Pipeline &models,
            std::map<ModelType, int> &batch_sizes, std::map<ModelType, int> &estimated_infer_times, int nObjects);

    std::map<ModelType, int> getInitialBatchSizes(
            const Pipeline &models, int slo,
            int nObjects);

    Pipeline getModelsByPipelineType(PipelineType type);

    bool running;
    std::string ctrl_experimentName;
    std::string ctrl_systemName;
    std::vector<TaskDescription::TaskStruct> initialTasks;
    uint16_t ctrl_runtime;

    std::string ctrl_logPath;
    uint16_t ctrl_loggingMode;
    uint16_t ctrl_verbose;

    ContainerLibType ctrl_containerLib;
    std::map<std::string, NodeHandle> devices;
    std::map<std::string, TaskHandle> tasks;
    std::map<std::string, ContainerHandle> containers;

    ControlCommunication::AsyncService service;
    std::unique_ptr<grpc::Server> server;
    std::unique_ptr<ServerCompletionQueue> cq;

    std::unique_ptr<pqxx::connection> ctrl_metricsServerConn = nullptr;
    MetricsServerConfigs ctrl_metricsServerConfigs;

    std::vector<spdlog::sink_ptr> ctrl_loggerSinks = {};
    std::shared_ptr<spdlog::logger> ctrl_logger;
};


#endif //PIPEPLUSPLUS_CONTROLLER_H
