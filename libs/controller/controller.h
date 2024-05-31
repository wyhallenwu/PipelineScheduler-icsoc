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
    XavierNX,
    OrinNano
};

enum ModelType {
    DataSource,
    Sink,
    Yolov5,
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

struct ContainerHandle;

struct TaskHandle {
    int slo;
    PipelineType type;
    std::map<std::string, ContainerHandle *> subtasks;
};

struct NodeHandle {
    std::string name;
    std::string ip;
    std::shared_ptr<ControlCommunication::Stub> stub;
    CompletionQueue *cq;
    SystemDeviceType type;
    BandwidthType bandwidth;
    int num_processors; // number of processing units, 1 for Edge or # GPUs for server
    std::vector<double> processors_utilization; // utilization per pu
    std::vector<unsigned long> mem_size; // memory size in MB
    std::vector<double> mem_utilization; // memory utilization per pu
    int next_free_port;
    std::map<std::string, ContainerHandle *> containers;
};

struct ContainerHandle {
    std::string name;
    ModelType model;
    NodeHandle *device_agent;
    TaskHandle *task;
    int batch_size;
    // The deadline for the inference to be completed.
    // The amount of time from start of each cycle, in milliseconds.
    uint32_t inference_deadline;
    // Arrival rate of requests
    float arrival_rate;
    int replicas;
    std::vector<int> cuda_device;
    int class_of_interest;
    int recv_port;
    Metrics metrics;
    google::protobuf::RepeatedField<int32_t> queue_lengths;
    std::vector<ContainerHandle *> upstreams;
    std::vector<ContainerHandle *> downstreams;
};

struct ModelProfile {
    uint64_t preprocessorLatency;
    BatchLatencyProfileType inferenceLatency;
    uint64_t postprocessorLatency;
    // Average size of incoming queries
    int avgInputSize = 1; // bytes
    // Average total size of outgoing queries
    int avgOutputSize = 1; // bytes
};

struct PipelineModel {
    std::string device;
    // Whether the upstream is on another device
    bool isSplitPoint;
    // Max arrival rate of queries to this model over different periods (e.g., last 1 second, last 3 seconds, etc.)
    float arrivalRate;
    // Scale factors for different periods
    ScaleFactorType scaleFactors;
    // Latency profile of preprocessor, batch inferencer and postprocessor
    ModelProfile modelProfile;
    // The downstream models and their classes of interest
    std::vector<std::pair<ModelType, int>> downstreams;
    std::vector<std::pair<ModelType, int>> upstreams;
    // The batch size of the model
    BatchSizeType batchSize;
    // The number of replicas of the model
    uint8_t numReplicas;
    // Average latency to query to reach from the upstream
    uint64_t expectedTransmitLatency;
    // Average queueing latency, subjected to the arrival rate and processing rate of preprocessor
    uint64_t expectedQueueingLatency;
    // Average latency to process each query
    uint64_t expectedAvgPerQueryLatency;
    // Maximum latency to process each query as ones that come later have to wait to be processed in batch
    uint64_t expectedMaxProcessLatency;
    // Latency from the start of the pipeline until the end of this model
    uint64_t expectedStart2HereLatency = -1;
    // The estimated cost per query processed by this model
    uint64_t estimatedPerQueryCost = 0;
    // The estimated latency of the model
    uint64_t estimatedStart2HereCost = 0;
};

// Arrival rates during different periods (e.g., last 1 second, last 3 seconds, etc.)
typedef std::map<int, float> ArrivalRateType;
// Scale factors for different periods
typedef std::map<int, float> ScaleFactorType;

typedef std::map<BatchSizeType, uint64_t> BatchLatencyProfileType;

typedef int BandwidthType;

extern std::map<ModelType, std::vector<std::string>> MODEL_INFO;
extern std::map<SystemDeviceType, std::vector<std::string>> DEVICE_INFO;

enum PipelineType {
    Traffic,
    Video_Call,
    Building_Security
};

// Structure that whole information about the pipeline used for scheduling
typedef std::map<ModelType, PipelineModel> PipelineModelListType;

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
    void queryRequestRateInPeriod(const std::string &name, ArrivalRateType &periods);
    void queryScaleFactorInPeriod(const std::string &name, ScaleFactorType &periods);
    ModelProfile queryModelProfile(const std::string &name, const std::string &deviceName); 
    uint64_t queryTransmitLatency(const int packageSize, const std::string &sourceName, const std::string &destName);
    void incNumReplicas(PipelineModel &model);
    void decNumReplicas(PipelineModel &model);

    void estimateModelLatency(PipelineModel &model, const ModelType modelType);
    void estimatePipelineLatency(PipelineModelListType &pipeline, const ModelType &currModel, const uint64_t start2HereLatency);

    void getInitialBatchSizes(PipelineModelListType &models, uint64_t slo, int nObjects);
    void shiftModelToEdge(PipelineModelListType &models, const ModelType &currModel, uint64_t slo);

    PipelineModelListType getModelsByPipelineType(PipelineType type, const std::string &startDevice);

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

    void MoveContainer(ContainerHandle *msvc, int cuda_device, bool to_edge, int replica = 1);

    static void AdjustUpstream(int port, ContainerHandle *msvc, NodeHandle *new_device, const std::string &dwnstr);

    void AdjustBatchSize(ContainerHandle *msvc, int new_bs);

    void StopContainer(std::string name, NodeHandle *device, bool forced = false);

    void optimizeBatchSizeStep(
            const std::vector<std::pair<ModelType, std::vector<std::pair<ModelType, int>>>> &models,
            std::map<ModelType, int> &batch_sizes, std::map<ModelType, int> &estimated_infer_times, int nObjects);

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