#ifndef PIPEPLUSPLUS_CONTROLLER_H
#define PIPEPLUSPLUS_CONTROLLER_H

#include "microservice.h"
#include <grpcpp/grpcpp.h>
#include "../json/json.h"
#include <thread>
#include "controlcommunication.grpc.pb.h"
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
using controlcommunication::SystemInfo;
using controlcommunication::Neighbor;
using controlcommunication::LoopRange;
using controlcommunication::DummyMessage;
using controlcommunication::ContainerConfig;
using controlcommunication::ContainerLink;
using controlcommunication::ContainerInts;
using controlcommunication::ContainerSignal;
using EmptyMessage = google::protobuf::Empty;

ABSL_DECLARE_FLAG(std::string, ctrl_configPath);
ABSL_DECLARE_FLAG(uint16_t, ctrl_verbose);
ABSL_DECLARE_FLAG(uint16_t, ctrl_loggingMode);

// typedef std::vector<std::pair<ModelType, std::vector<std::pair<ModelType, int>>>> Pipeline;


struct ContainerHandle;

struct TaskHandle {
    std::string name;
    PipelineType type;
    std::string source;
    int slo;
    ClockType start_time;
    int last_latency;
    std::map<std::string, ContainerHandle *> subtasks;
};

struct NodeHandle {
    std::string name;
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
    // The latest network entries to determine the network conditions and latencies of transferring data
    std::map<std::string, NetworkEntryType> latestNetworkEntries = {};
    std::mutex nodeHandleMutex;

    NodeHandle() = default;

    NodeHandle(const std::string& name,
               const std::string& ip,
               std::shared_ptr<ControlCommunication::Stub> stub,
               grpc::CompletionQueue* cq,
               SystemDeviceType type,
               int num_processors,
               std::vector<double> processors_utilization,
               std::vector<unsigned long> mem_size,
               std::vector<double> mem_utilization,
               int next_free_port,
               std::map<std::string, ContainerHandle*> containers)
        : name(name),
          ip(ip),
          stub(std::move(stub)),
          cq(cq),
          type(type),
          num_processors(num_processors),
          processors_utilization(std::move(processors_utilization)),
          mem_size(std::move(mem_size)),
          mem_utilization(std::move(mem_utilization)),
          next_free_port(next_free_port),
          containers(std::move(containers)) {}

    NodeHandle(const NodeHandle &other) {
        name = other.name;
        ip = other.ip;
        stub = other.stub;
        cq = other.cq;
        type = other.type;
        num_processors = other.num_processors;
        processors_utilization = other.processors_utilization;
        mem_size = other.mem_size;
        mem_utilization = other.mem_utilization;
        next_free_port = other.next_free_port;
        containers = other.containers;
        latestNetworkEntries = other.latestNetworkEntries;
    }
};

struct ContainerHandle {
    std::string name;
    int class_of_interest;
    ModelType model;
    bool mergable;
    std::vector<int> dimensions;
    uint64_t inference_deadline;

    float arrival_rate;

    int num_replicas;
    std::vector<int> batch_size;
    std::vector<int> cuda_device;
    std::vector<int> recv_port;
    std::vector<std::string> model_file;

    HardwareMetrics metrics;
    NodeHandle *device_agent;
    TaskHandle *task;
    std::vector<ContainerHandle *> upstreams;
    std::vector<ContainerHandle *> downstreams;
    // Queue sizes of the model
    std::vector<QueueLengthType> queueSizes;

    // Flag to indicate whether the container is running
    // At the end of scheduling, all containerhandle marked with `running = false` object will be deleted
    bool running = false;

    // Number of microservices packed inside this container. A regular container has 5 namely
    // receiver, preprocessor, inferencer, postprocessor, sender
    uint8_t numMicroservices = 5;
    // Average latency to query to reach from the upstream
    uint64_t expectedTransferLatency = 0;
    // Average queueing latency, subjected to the arrival rate and processing rate of preprocessor
    uint64_t expectedQueueingLatency = 0;
    // Average latency to preprocess each query
    uint64_t expectedPreprocessLatency = 0;
    // Average latency to process each batch running at the specified batch size
    uint64_t expectedInferLatency = 0;
    // Average latency to postprocess each query
    uint64_t expectedPostprocessLatency = 0;
    // Expected throughput
    float expectedThroughput = 0;
};

struct PipelineModel {
    std::string device;
    std::string name;
    // Whether the upstream is on another device
    bool isSplitPoint;
    //
    ModelArrivalProfile arrivalProfiles;
    // Latency profile of preprocessor, batch inferencer and postprocessor
    PerDeviceModelProfileType processProfiles;
    // The downstream models and their classes of interest
    std::vector<std::pair<PipelineModel *, int>> downstreams;
    std::vector<std::pair<PipelineModel *, int>> upstreams;
    // The batch size of the model
    BatchSizeType batchSize;
    // The number of replicas of the model
    uint8_t numReplicas;
    // Average latency to query to reach from the upstream
    uint64_t expectedTransferLatency;
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

    std::string deviceTypeName;
};


// Structure that whole information about the pipeline used for scheduling
typedef std::vector<PipelineModel *> PipelineModelListType;



namespace TaskDescription {
    struct TaskStruct {
        // Name of the task (e.g., traffic, video_call, people, etc.)
        std::string name;
        // Full name to identify the task in the task list (which is a map)
        std::string fullName;
        int slo;
        PipelineType type;
        std::string source;
        std::string device;
        bool added = false;
    };

    void from_json(const nlohmann::json &j, TaskStruct &val);
}



class Controller {
public:
    Controller(int argc, char **argv);

    ~Controller();

    void HandleRecvRpcs();

    void Scheduling();

    void Init() { 
        bool allAdded = true;
        for (auto &t: initialTasks) {
            if (!t.added) {
                t.added = AddTask(t);
            }
            if (!t.added) {
                allAdded = false;
            }
            remainTasks.push_back(t);
        }
    }

    void InitRemain() {
        bool allAdded = true;
        for (auto &t: remainTasks) {
            if (!t.added) {
                t.added = AddTask(t);
            }
            if (!t.added) {
                allAdded = false;
                continue;
            }
            // Remove the task from the remain list
            remainTasks.erase(std::remove_if(remainTasks.begin(), remainTasks.end(),
                                             [&t](const TaskDescription::TaskStruct &task) {
                                                 return task.name == t.name;
                                             }), remainTasks.end());
        }
    }

    void addRemainTask(const TaskDescription::TaskStruct &task) {
        remainTasks.push_back(task);
    }

    bool AddTask(const TaskDescription::TaskStruct &task);

    [[nodiscard]] bool isRunning() const { return running; };

    void Stop() { running = false; };

private:

    struct Pipeline {
        PipelineModelListType pipelineModels;
        std::mutex pipelineMutex;

        Pipeline() = default;
        ~Pipeline() {
            for (auto &model: pipelineModels) {
                delete model;
            }
        }

        Pipeline(PipelineModelListType pipelineModels) {
            this->pipelineModels = pipelineModels;
        }

        Pipeline(const Pipeline &other) {
            pipelineModels = other.pipelineModels;
        }

        Pipeline& operator=(const Pipeline &other) {
            if (this != &other) {
                pipelineModels = other.pipelineModels;
            }
            return *this;
        }
    };

    std::vector<Pipeline> ctrl_unscheduledPipelines, ctrl_scheduledPipelines;

    NetworkEntryType initNetworkCheck(const NodeHandle &node, uint32_t minPacketSize = 1000, uint32_t maxPacketSize = 1228800, uint32_t numLoops = 20);
    uint8_t incNumReplicas(const PipelineModel *model);
    uint8_t decNumReplicas(const PipelineModel *model);

    void calculateQueueSizes(ContainerHandle &model, const ModelType modelType);
    uint64_t calculateQueuingLatency(const float &arrival_rate, const float &preprocess_rate);

    void estimateModelLatency(PipelineModel *currModel, const std::string& deviceName);
    void estimatePipelineLatency(PipelineModel *currModel, const uint64_t start2HereLatency);

    void getInitialBatchSizes(Pipeline &models, uint64_t slo, int nObjects);
    void shiftModelToEdge(Pipeline &models, const ModelType &currModel, uint64_t slo);

    PipelineModelListType getModelsByPipelineType(PipelineType type, const std::string &startDevice);

    void checkNetworkConditions();

    void readConfigFile(const std::string &config_path);

    // double LoadTimeEstimator(const char *model_path, double input_mem_size);
    int InferTimeEstimator(ModelType model, int batch_size);
    // std::map<ModelType, std::vector<int>> InitialRequestCount(const std::string &input, const Pipeline &models,
    //                                                           int fps = 30);

    void queryInDeviceNetworkEntries(NodeHandle *node);

    class RequestHandler {
    public:
        RequestHandler(ControlCommunication::AsyncService *service, ServerCompletionQueue *cq, Controller *c)
                : service(service), cq(cq), status(CREATE), controller(c) {}

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
    };

    class DeviseAdvertisementHandler : public RequestHandler {
    public:
        DeviseAdvertisementHandler(ControlCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                   Controller *c)
                : RequestHandler(service, cq, c), responder(&ctx) {
            Proceed();
        }

        void Proceed() final;

    private:
        ConnectionConfigs request;
        SystemInfo reply;
        grpc::ServerAsyncResponseWriter<SystemInfo> responder;
    };

    class DummyDataRequestHandler : public RequestHandler {
    public:
        DummyDataRequestHandler(ControlCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                   Controller *c)
                : RequestHandler(service, cq, c), responder(&ctx) {
            Proceed();
        }

        void Proceed() final;

    private:
        DummyMessage request;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
    };

    void StartContainer(std::pair<std::string, ContainerHandle *> &upstr, int slo,
                        std::string source, int replica = 1, bool easy_allocation = true);

    void MoveContainer(ContainerHandle *msvc, bool to_edge, int cuda_device = 0, int replica = 1);

    static void AdjustUpstream(int port, ContainerHandle *msvc, NodeHandle *new_device, const std::string &dwnstr);

    static void SyncDatasource(ContainerHandle *prev, ContainerHandle *curr);

    void AdjustBatchSize(ContainerHandle *msvc, int new_bs, int replica = 1);

    void AdjustResolution(ContainerHandle *msvc, std::vector<int> new_resolution, int replica = 1);

    void StopContainer(std::string name, NodeHandle *device, bool forced = false);

    // void optimizeBatchSizeStep(
    //         const Pipeline &models,
    //         std::map<ModelType, int> &batch_sizes, std::map<ModelType, int> &estimated_infer_times, int nObjects);

    bool running;
    std::string ctrl_experimentName;
    std::string ctrl_systemName;
    std::vector<TaskDescription::TaskStruct> initialTasks;
    std::vector<TaskDescription::TaskStruct> remainTasks;
    uint16_t ctrl_runtime;
    uint16_t ctrl_port_offset;

    std::string ctrl_logPath;
    uint16_t ctrl_loggingMode;
    uint16_t ctrl_verbose;

    ContainerLibType ctrl_containerLib;
    DeviceInfoType ctrl_sysDeviceInfo = {
        {Server, "server"},
        {AGXXavier, "agxavier"},
        {NXXavier, "nxavier"},
        {OrinNano, "orinano"}
    };

    std::map<std::string, NodeHandle> devices;
    std::map<std::string, TaskHandle> tasks;
    std::map<std::string, ContainerHandle> containers;
    std::map<std::string, NetworkEntryType> network_check_buffer;

    ControlCommunication::AsyncService service;
    std::unique_ptr<grpc::Server> server;
    std::unique_ptr<ServerCompletionQueue> cq;

    std::unique_ptr<pqxx::connection> ctrl_metricsServerConn = nullptr;
    MetricsServerConfigs ctrl_metricsServerConfigs;

    std::vector<spdlog::sink_ptr> ctrl_loggerSinks = {};
    std::shared_ptr<spdlog::logger> ctrl_logger;

    std::map<std::string, NetworkEntryType> ctrl_inDeviceNetworkEntries;
};


#endif //PIPEPLUSPLUS_CONTROLLER_H