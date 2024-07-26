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
struct PipelineModel;

struct GPUPortion;

struct GPULane {
    std::uint16_t gpuNum;
    std::uint16_t laneNum;
    std::uint64_t dutyCycle = 9999999999999999;
};

struct GPUPortion : GPULane {
    std::uint64_t start = 0;
    std::uint64_t end = 9999999999999999;
    GPULane * lane = nullptr;
    GPUPortion* next = nullptr;
    GPUPortion* prev = nullptr;
};

struct GPUPortionList {
    GPUPortion *head = nullptr;
    std::vector<GPUPortion *> list;
};

// Structure that whole information about the pipeline used for scheduling
typedef std::vector<PipelineModel *> PipelineModelListType;

struct TaskHandle;
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
    //
    uint8_t numGPULanes;
    //
    std::vector<GPULane *> gpuLanes;
    GPUPortionList freeGPUPortions;

    bool initialNetworkCheck = false;
    ClockType lastNetworkCheckTime;

    std::map<std::string, PipelineModel *> modelList;

    mutable std::mutex nodeHandleMutex;
    mutable std::mutex networkCheckMutex;

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
        std::lock(nodeHandleMutex, other.nodeHandleMutex);
        std::lock_guard<std::mutex> lock1(nodeHandleMutex, std::adopt_lock);
        std::lock_guard<std::mutex> lock2(other.nodeHandleMutex, std::adopt_lock);
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

    NodeHandle& operator=(const NodeHandle &other) {
        if (this != &other) {
            std::lock(nodeHandleMutex, other.nodeHandleMutex);
            std::lock_guard<std::mutex> lock1(nodeHandleMutex, std::adopt_lock);
            std::lock_guard<std::mutex> lock2(other.nodeHandleMutex, std::adopt_lock);
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
        return *this;
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

    int batch_size;
    int cuda_device;
    int recv_port;
    std::string model_file;

    NodeHandle *device_agent;
    TaskHandle *task;
    std::vector<ContainerHandle *> downstreams;
    std::vector<ContainerHandle *> upstreams;
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
    // Average in queueing latency, subjected to the arrival rate and processing rate of preprocessor
    uint64_t expectedQueueingLatency = 0;
    // Average batching latency, subjected to the preprocessing rate, batch size and processing rate of inferencer
    uint64_t expectedBatchingLatency = 0;
    // Average post queueing latency, subjected to the processing rate of postprocessor
    uint64_t expectedPostQueueingLatency = 0;
    // Average out queueing latency, subjected to the processing rate of sender
    uint64_t expectedOutQueueingLatency = 0;
    // Average latency to preprocess each query
    uint64_t expectedPreprocessLatency = 0;
    // Average latency to process each batch running at the specified batch size
    uint64_t expectedInferLatency = 0;
    // Average latency to postprocess each query
    uint64_t expectedPostprocessLatency = 0;
    // Expected throughput
    float expectedThroughput = 0;
    //
    uint64_t startTime;
    //
    uint64_t endTime;
    //
    GPUPortion *executionLane = nullptr;
    // points to the pipeline model that this container is part of
    PipelineModel *pipelineModel = nullptr;
    mutable std::mutex containerHandleMutex;

    ContainerHandle() = default;

        // Constructor
    ContainerHandle(const std::string& name,
                int class_of_interest,
                ModelType model,
                bool mergable,
                const std::vector<int>& dimensions = {},
                uint64_t inference_deadline = 0,
                float arrival_rate = 0.0f,
                const int batch_size = 0,
                const int cuda_device = 0,
                const int recv_port = 0,
                const std::string model_file = "",
                NodeHandle* device_agent = nullptr,
                TaskHandle* task = nullptr,
                PipelineModel* pipelineModel = nullptr,
                const std::vector<ContainerHandle*>& upstreams = {},
                const std::vector<ContainerHandle*>& downstreams = {},
                const std::vector<QueueLengthType>& queueSizes = {})
    : name(name),
      class_of_interest(class_of_interest),
      model(model),
      mergable(mergable),
      dimensions(dimensions),
      inference_deadline(inference_deadline),
      arrival_rate(arrival_rate),
      batch_size(batch_size),
      cuda_device(cuda_device),
      recv_port(recv_port),
      model_file(model_file),
      device_agent(device_agent),
      task(task),
      pipelineModel(pipelineModel),
      upstreams(upstreams),
      downstreams(downstreams),
      queueSizes(queueSizes) {}
    
    // Copy constructor
    ContainerHandle(const ContainerHandle& other) {
        std::lock(containerHandleMutex, other.containerHandleMutex);
        std::lock_guard<std::mutex> lock1(containerHandleMutex, std::adopt_lock);
        std::lock_guard<std::mutex> lock2(other.containerHandleMutex, std::adopt_lock);

        name = other.name;
        class_of_interest = other.class_of_interest;
        model = other.model;
        mergable = other.mergable;
        dimensions = other.dimensions;
        inference_deadline = other.inference_deadline;
        arrival_rate = other.arrival_rate;
        batch_size = other.batch_size;
        cuda_device = other.cuda_device;
        recv_port = other.recv_port;
        model_file = other.model_file;
        device_agent = other.device_agent;
        task = other.task;
        upstreams = other.upstreams;
        downstreams = other.downstreams;
        queueSizes = other.queueSizes;
        running = other.running;
        numMicroservices = other.numMicroservices;
        expectedTransferLatency = other.expectedTransferLatency;
        expectedQueueingLatency = other.expectedQueueingLatency;
        expectedBatchingLatency = other.expectedBatchingLatency;
        expectedPostQueueingLatency = other.expectedPostQueueingLatency;
        expectedOutQueueingLatency = other.expectedOutQueueingLatency;
        expectedPreprocessLatency = other.expectedPreprocessLatency;
        expectedInferLatency = other.expectedInferLatency;
        expectedPostprocessLatency = other.expectedPostprocessLatency;
        expectedThroughput = other.expectedThroughput;
        startTime = other.startTime;
        endTime = other.endTime;
        executionLane = other.executionLane;
        pipelineModel = other.pipelineModel;
    }

    // Copy assignment operator
    ContainerHandle& operator=(const ContainerHandle& other) {
        if (this != &other) {
            std::lock(containerHandleMutex, other.containerHandleMutex);
            std::lock_guard<std::mutex> lock1(containerHandleMutex, std::adopt_lock);
            std::lock_guard<std::mutex> lock2(other.containerHandleMutex, std::adopt_lock);
            name = other.name;
            class_of_interest = other.class_of_interest;
            model = other.model;
            mergable = other.mergable;
            dimensions = other.dimensions;
            inference_deadline = other.inference_deadline;
            arrival_rate = other.arrival_rate;
            batch_size = other.batch_size;
            cuda_device = other.cuda_device;
            recv_port = other.recv_port;
            model_file = other.model_file;
            device_agent = other.device_agent;
            task = other.task;
            upstreams = other.upstreams;
            downstreams = other.downstreams;
            queueSizes = other.queueSizes;
            running = other.running;
            numMicroservices = other.numMicroservices;
            expectedTransferLatency = other.expectedTransferLatency;
            expectedQueueingLatency = other.expectedQueueingLatency;
            expectedBatchingLatency = other.expectedBatchingLatency;
            expectedPostQueueingLatency = other.expectedPostQueueingLatency;
            expectedOutQueueingLatency = other.expectedOutQueueingLatency;
            expectedPreprocessLatency = other.expectedPreprocessLatency;
            expectedInferLatency = other.expectedInferLatency;
            expectedPostprocessLatency = other.expectedPostprocessLatency;
            expectedThroughput = other.expectedThroughput;
            startTime = other.startTime;
            endTime = other.endTime;
            executionLane = other.executionLane;
            pipelineModel = other.pipelineModel;
        }
        return *this;
    }
};

struct PipelineModel {
    std::string name;
    TaskHandle *task;
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
    uint8_t numReplicas = -1;
    // The assigned cuda device for each replica
    std::vector<uint8_t> cudaDevices;
    // Average latency to query to reach from the upstream
    uint64_t expectedTransferLatency = 0;
    // Average queueing latency, subjected to the arrival rate and processing rate of preprocessor
    uint64_t expectedQueueingLatency = 0;
    // Average batching latency, subjected to the preprocessing rate, batch size and processing rate of inferencer
    uint64_t expectedBatchingLatency = 0;
    // Average post queueing latency, subjected to the processing rate of postprocessor
    uint64_t expectedPostQueueingLatency = 0;
    // Average out queueing latency, subjected to the processing rate of sender
    uint64_t expectedOutQueueingLatency = 0;
    // Average latency to process each query
    uint64_t expectedAvgPerQueryLatency = 0;
    // Maximum latency to process each query as ones that come later have to wait to be processed in batch
    uint64_t expectedMaxProcessLatency = 0;
    // Latency from the start of the pipeline until the end of this model
    uint64_t expectedStart2HereLatency = -1;
    // The estimated cost per query processed by this model
    uint64_t estimatedPerQueryCost = 0;
    // The estimated latency of the model
    uint64_t estimatedStart2HereCost = 0;
    // Batching deadline
    uint64_t batchingDeadline = 9999999999;

    std::vector<int> dimensions = {-1, -1};

    std::string device;
    std::string deviceTypeName;
    NodeHandle *deviceAgent;

    bool merged = false;
    bool toBeRun = true;

    std::vector<std::string> possibleDevices;
    // Manifestations are the list of containers that will be created for this model
    std::vector<ContainerHandle *> manifestations;

    // Source
    std::string datasourceName;

    mutable std::mutex pipelineModelMutex;

        // Constructor with default parameters
    PipelineModel(const std::string& device = "",
                  const std::string& name = "",
                  TaskHandle *task = nullptr,
                  bool isSplitPoint = false,
                  const ModelArrivalProfile& arrivalProfiles = ModelArrivalProfile(),
                  const PerDeviceModelProfileType& processProfiles = PerDeviceModelProfileType(),
                  const std::vector<std::pair<PipelineModel*, int>>& downstreams = {},
                  const std::vector<std::pair<PipelineModel*, int>>& upstreams = {},
                  const BatchSizeType& batchSize = BatchSizeType(),
                  uint8_t numReplicas = -1,
                  std::vector<uint8_t> cudaDevices = {},
                  uint64_t expectedTransferLatency = 0,
                  uint64_t expectedQueueingLatency = 0,
                  uint64_t expectedAvgPerQueryLatency = 0,
                  uint64_t expectedMaxProcessLatency = 0,
                  const std::string& deviceTypeName = "",
                  bool merged = false,
                  bool toBeRun = true,
                  const std::vector<std::string>& possibleDevices = {})
        : device(device),
          name(name),
          task(task),
          isSplitPoint(isSplitPoint),
          arrivalProfiles(arrivalProfiles),
          processProfiles(processProfiles),
          downstreams(downstreams),
          upstreams(upstreams),
          batchSize(batchSize),
          numReplicas(numReplicas),
          cudaDevices(cudaDevices),
          expectedTransferLatency(expectedTransferLatency),
          expectedQueueingLatency(expectedQueueingLatency),
          expectedAvgPerQueryLatency(expectedAvgPerQueryLatency),
          expectedMaxProcessLatency(expectedMaxProcessLatency),
          deviceTypeName(deviceTypeName),
          merged(merged),
          toBeRun(toBeRun),
          possibleDevices(possibleDevices) {}

    // Copy constructor
    PipelineModel(const PipelineModel& other) {
        std::lock(pipelineModelMutex, other.pipelineModelMutex);
        std::lock_guard<std::mutex> lock1(other.pipelineModelMutex, std::adopt_lock);
        std::lock_guard<std::mutex> lock2(pipelineModelMutex, std::adopt_lock);
        device = other.device;
        name = other.name;
        task = other.task;
        isSplitPoint = other.isSplitPoint;
        arrivalProfiles = other.arrivalProfiles;
        processProfiles = other.processProfiles;
        downstreams = other.downstreams;
        upstreams = other.upstreams;
        batchSize = other.batchSize;
        numReplicas = other.numReplicas;
        cudaDevices = other.cudaDevices;
        expectedTransferLatency = other.expectedTransferLatency;
        expectedQueueingLatency = other.expectedQueueingLatency;
        expectedBatchingLatency = other.expectedBatchingLatency;
        expectedPostQueueingLatency = other.expectedPostQueueingLatency;
        expectedOutQueueingLatency = other.expectedOutQueueingLatency;
        expectedAvgPerQueryLatency = other.expectedAvgPerQueryLatency;
        expectedMaxProcessLatency = other.expectedMaxProcessLatency;
        expectedStart2HereLatency = other.expectedStart2HereLatency;
        estimatedPerQueryCost = other.estimatedPerQueryCost;
        estimatedStart2HereCost = other.estimatedStart2HereCost;
        batchingDeadline = other.batchingDeadline;
        deviceTypeName = other.deviceTypeName;
        merged = other.merged;
        toBeRun = other.toBeRun;
        possibleDevices = other.possibleDevices;
        dimensions = other.dimensions;
        manifestations = {};
        for (auto& container : other.manifestations) {
            manifestations.push_back(new ContainerHandle(*container));
        }
        deviceAgent = other.deviceAgent;
        datasourceName = other.datasourceName;
    }

    // Assignment operator
    PipelineModel& operator=(const PipelineModel& other) {
        if (this != &other) {
            std::lock(pipelineModelMutex, other.pipelineModelMutex);
            std::lock_guard<std::mutex> lock1(pipelineModelMutex, std::adopt_lock);
            std::lock_guard<std::mutex> lock2(other.pipelineModelMutex, std::adopt_lock);
            device = other.device;
            name = other.name;
            task = other.task;
            isSplitPoint = other.isSplitPoint;
            arrivalProfiles = other.arrivalProfiles;
            processProfiles = other.processProfiles;
            downstreams = other.downstreams;
            upstreams = other.upstreams;
            batchSize = other.batchSize;
            numReplicas = other.numReplicas;
            cudaDevices = other.cudaDevices;
            expectedTransferLatency = other.expectedTransferLatency;
            expectedQueueingLatency = other.expectedQueueingLatency;
            expectedBatchingLatency = other.expectedBatchingLatency;
            expectedPostQueueingLatency = other.expectedPostQueueingLatency;
            expectedOutQueueingLatency = other.expectedOutQueueingLatency;
            expectedAvgPerQueryLatency = other.expectedAvgPerQueryLatency;
            expectedMaxProcessLatency = other.expectedMaxProcessLatency;
            expectedStart2HereLatency = other.expectedStart2HereLatency;
            estimatedPerQueryCost = other.estimatedPerQueryCost;
            estimatedStart2HereCost = other.estimatedStart2HereCost;
            batchingDeadline = other.batchingDeadline;
            deviceTypeName = other.deviceTypeName;
            merged = other.merged;
            toBeRun = other.toBeRun;
            possibleDevices = other.possibleDevices;
            dimensions = other.dimensions;
            manifestations = {};
            for (auto& container : other.manifestations) {
                manifestations.push_back(new ContainerHandle(*container));
            }
            deviceAgent = other.deviceAgent;
            datasourceName = other.datasourceName;
        }
        return *this;
    }
};

PipelineModelListType deepCopyPipelineModelList(const PipelineModelListType& original);

struct TaskHandle {
    std::string tk_name;
    std::string tk_fullName;
    PipelineType tk_type;
    std::string tk_source;
    std::string tk_src_device;
    int tk_slo;
    ClockType tk_startTime;
    int tk_lastLatency;
    std::map<std::string, std::vector<ContainerHandle*>> tk_subTasks;
    PipelineModelListType tk_pipelineModels;
    mutable std::mutex tk_mutex;

    bool tk_newlyAdded = true;

    TaskHandle() = default;

    ~TaskHandle() {
        // Ensure no other threads are using this object
        std::lock_guard<std::mutex> lock(tk_mutex);
        for (auto& model : tk_pipelineModels) {
            delete model;
        }
    }

    TaskHandle(const std::string& tk_name,
               const std::string& tk_fullName,
               PipelineType tk_type,
               const std::string& tk_source,
               const std::string& tk_src_device,
               int tk_slo,
               ClockType tk_startTime,
               int tk_lastLatency)
    : tk_name(tk_name),
      tk_type(tk_type),
      tk_source(tk_source),
      tk_src_device(tk_src_device),
      tk_slo(tk_slo),
      tk_startTime(tk_startTime),
      tk_lastLatency(tk_lastLatency) {}

    TaskHandle(const TaskHandle& other) {
        std::lock(tk_mutex, other.tk_mutex);
        std::lock_guard<std::mutex> lock1(other.tk_mutex, std::adopt_lock);
        std::lock_guard<std::mutex> lock2(tk_mutex, std::adopt_lock);
        tk_name = other.tk_name;
        tk_fullName = other.tk_fullName;
        tk_type = other.tk_type;
        tk_source = other.tk_source;
        tk_src_device = other.tk_src_device;
        tk_slo = other.tk_slo;
        tk_startTime = other.tk_startTime;
        tk_lastLatency = other.tk_lastLatency;
        tk_subTasks = other.tk_subTasks;
        tk_pipelineModels = {};
        for (auto& model : other.tk_pipelineModels) {
            tk_pipelineModels.push_back(new PipelineModel(*model));
            tk_pipelineModels.back()->task = this;
        }
        for (auto& model : this->tk_pipelineModels) {
            for (auto& downstream : model->downstreams) {
                for (auto& model2 : tk_pipelineModels) {
                    if (model2->name != downstream.first->name || model2->device != downstream.first->device) {
                        continue;
                    }
                    downstream.first = model2;
                }
            }
            for (auto& upstream : model->upstreams) {
                for (auto& model2 : tk_pipelineModels) {
                    if (model2->name != upstream.first->name || model2->device != upstream.first->device) {
                        continue;
                    }
                    upstream.first = model2;
                }
            }
        }
        tk_newlyAdded = other.tk_newlyAdded;
    }

    TaskHandle& operator=(const TaskHandle& other) {
        if (this != &other) {
            std::lock(tk_mutex, other.tk_mutex);
            std::lock_guard<std::mutex> lock1(tk_mutex, std::adopt_lock);
            std::lock_guard<std::mutex> lock2(other.tk_mutex, std::adopt_lock);
            tk_name = other.tk_name;
            tk_fullName = other.tk_fullName;
            tk_type = other.tk_type;
            tk_source = other.tk_source;
            tk_src_device = other.tk_src_device;
            tk_slo = other.tk_slo;
            tk_startTime = other.tk_startTime;
            tk_lastLatency = other.tk_lastLatency;
            tk_subTasks = other.tk_subTasks;
            tk_pipelineModels = {};
            for (auto& model : other.tk_pipelineModels) {
                tk_pipelineModels.push_back(new PipelineModel(*model));
                tk_pipelineModels.back()->task = this;
            }
            for (auto& model : this->tk_pipelineModels) {
                for (auto& downstream : model->downstreams) {
                    for (auto& model2 : tk_pipelineModels) {
                        if (model2->name != downstream.first->name || model2->device != downstream.first->device) {
                            continue;
                        }
                        downstream.first = model2;
                    }
                }
                for (auto& upstream : model->upstreams) {
                    for (auto& model2 : tk_pipelineModels) {
                        if (model2->name != upstream.first->name || model2->device != upstream.first->device) {
                            continue;
                        }
                        upstream.first = model2;
                    }
                }
            }
            tk_newlyAdded = other.tk_newlyAdded;
        }
        return *this;
    }
};



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
        for (auto &t: initialTasks) {
            if (!t.added) {
                t.added = AddTask(t);
            }
            if (!t.added) {
                remainTasks.push_back(t);
            }
        }
    }

    void InitRemain() {
        for (auto &t: remainTasks) {
            if (!t.added) {
                t.added = AddTask(t);
            }
            if (t.added) {
                // Remove the task from the remain list
                remainTasks.erase(std::remove_if(remainTasks.begin(), remainTasks.end(),
                                                [&t](const TaskDescription::TaskStruct &task) {
                                                    return task.name == t.name;
                                                }), remainTasks.end());
            }
        }
    }

    void addRemainTask(const TaskDescription::TaskStruct &task) {
        remainTasks.push_back(task);
    }

    bool AddTask(const TaskDescription::TaskStruct &task);

    ContainerHandle *TranslateToContainer(PipelineModel *model, NodeHandle *device, unsigned int i);

    void ApplyScheduling();

    [[nodiscard]] bool isRunning() const { return running; };

    void Stop() { running = false; };

    void readInitialObjectCount(
        const std::string& path 
    );

private:
    void initiateGPULanes(NodeHandle &node);

    NetworkEntryType initNetworkCheck(NodeHandle &node, uint32_t minPacketSize = 1000, uint32_t maxPacketSize = 1228800, uint32_t numLoops = 20);
    uint8_t incNumReplicas(const PipelineModel *model);
    uint8_t decNumReplicas(const PipelineModel *model);

    void calculateQueueSizes(ContainerHandle &model, const ModelType modelType);
    uint64_t calculateQueuingLatency(const float &arrival_rate, const float &preprocess_rate);

    void queryingProfiles(TaskHandle *task);

    void estimateModelLatency(PipelineModel *currModel);
    void estimateModelNetworkLatency(PipelineModel *currModel);
    void estimatePipelineLatency(PipelineModel *currModel, const uint64_t start2HereLatency);

    void getInitialBatchSizes(TaskHandle *task, uint64_t slo);
    void shiftModelToEdge(PipelineModelListType &pipeline, PipelineModel *currModel, uint64_t slo, const std::string& edgeDevice);

    bool mergeArrivalProfiles(ModelArrivalProfile &mergedProfile, const ModelArrivalProfile &toBeMergedProfile);
    bool mergeProcessProfiles(PerDeviceModelProfileType &mergedProfile, const PerDeviceModelProfileType &toBeMergedProfile);
    bool mergeModels(PipelineModel *mergedModel, PipelineModel *tobeMergedModel);
    TaskHandle mergePipelines(const std::string& taskName);
    void mergePipelines();

    bool containerTemporalScheduling(ContainerHandle *container);
    bool modelTemporalScheduling(PipelineModel *pipelineModel);
    void temporalScheduling();

    PipelineModelListType getModelsByPipelineType(PipelineType type, const std::string &startDevice, const std::string &pipelineName = "", const std::string &streamName = "");

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

    void StartContainer(ContainerHandle *container, bool easy_allocation = true);

    void MoveContainer(ContainerHandle *container, NodeHandle *new_device);

    static void AdjustUpstream(int port, ContainerHandle *msvc, NodeHandle *new_device, const std::string &dwnstr);

    static void SyncDatasource(ContainerHandle *prev, ContainerHandle *curr);

    void AdjustBatchSize(ContainerHandle *msvc, int new_bs);

    void AdjustCudaDevice(ContainerHandle *msvc, unsigned int new_device);

    void AdjustResolution(ContainerHandle *msvc, std::vector<int> new_resolution);

    void StopContainer(ContainerHandle *container, NodeHandle *device, bool forced = false);

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

    struct Devices {
    public:
        void addDevice(const std::string &name, NodeHandle *node) {
            std::lock_guard<std::mutex> lock(devicesMutex);
            list[name] = node;
        }

        void removeDevice(const std::string &name) {
            std::lock_guard<std::mutex> lock(devicesMutex);
            list.erase(name);
        }

        NodeHandle *getDevice(const std::string &name) {
            std::lock_guard<std::mutex> lock(devicesMutex);
            return list[name];
        }

        std::vector<NodeHandle *> getList() {
            std::lock_guard<std::mutex> lock(devicesMutex);
            std::vector<NodeHandle *> devices;
            for (auto &d: list) {
                devices.push_back(d.second);
            }
            return devices;
        }

        std::map<std::string, NodeHandle*> getMap() {
            std::lock_guard<std::mutex> lock(devicesMutex);
            return list;
        }

        bool hasDevice(const std::string &name) {
            std::lock_guard<std::mutex> lock(devicesMutex);
            return list.find(name) != list.end();
        }
    // TODO: MAKE THIS PRIVATE TO AVOID NON-THREADSAFE ACCESS
    public:
        std::map<std::string, NodeHandle*> list = {};
        std::mutex devicesMutex;
    };
    
    Devices devices;

    struct Tasks {
    public:
        void addTask(const std::string &name, TaskHandle *task) {
            std::lock_guard<std::mutex> lock(tasksMutex);
            list[name] = task;
        }

        void removeTask(const std::string &name) {
            std::lock_guard<std::mutex> lock(tasksMutex);
            list.erase(name);
        }

        TaskHandle *getTask(const std::string &name) {
            std::lock_guard<std::mutex> lock(tasksMutex);
            return list[name];
        }

        std::vector<TaskHandle *> getList() {
            std::lock_guard<std::mutex> lock(tasksMutex);
            std::vector<TaskHandle *> tasks;
            for (auto &t: list) {
                tasks.push_back(t.second);
            }
            return tasks;
        }

        std::map<std::string, TaskHandle*> getMap() {
            std::lock_guard<std::mutex> lock(tasksMutex);
            return list;
        }

        bool hasTask(const std::string &name) {
            std::lock_guard<std::mutex> lock(tasksMutex);
            return list.find(name) != list.end();
        }

        Tasks() = default;

        // Copy constructor
        Tasks(const Tasks &other) {
            std::lock(tasksMutex, other.tasksMutex);
            std::lock_guard<std::mutex> lock1(tasksMutex, std::adopt_lock);
            std::lock_guard<std::mutex> lock2(other.tasksMutex, std::adopt_lock);
            list = {};
            for (auto &t: other.list) {
                list[t.first] = new TaskHandle(*t.second);
            }
        }

        Tasks& operator=(const Tasks &other) {
            if (this != &other) {
                std::lock(tasksMutex, other.tasksMutex);
                std::lock_guard<std::mutex> lock1(tasksMutex, std::adopt_lock);
                std::lock_guard<std::mutex> lock2(other.tasksMutex, std::adopt_lock);
                list = {};
                for (auto &t: other.list) {
                    list[t.first] = new TaskHandle(*t.second);
                }
            }
            return *this;
        }

    // TODO: MAKE THIS PRIVATE TO AVOID NON-THREADSAFE ACCESS
    public:
        std::map<std::string, TaskHandle*> list = {};
        mutable std::mutex tasksMutex;
    };
    Tasks ctrl_unscheduledPipelines, ctrl_savedUnscheduledPipelines, ctrl_scheduledPipelines, ctrl_pastScheduledPipelines;

    void deepCopyTasks(Tasks& source, Tasks& destination); //add by distream



    struct Containers {
    public:
        void addContainer(const std::string &name, ContainerHandle *container) {
            std::lock_guard<std::mutex> lock(containersMutex);
            list[name] = container;
        }

        void removeContainer(const std::string &name) {
            std::lock_guard<std::mutex> lock(containersMutex);
            list.erase(name);
        }

        ContainerHandle *getContainer(const std::string &name) {
            std::lock_guard<std::mutex> lock(containersMutex);
            return list[name];
        }

        std::vector<ContainerHandle *> getList() {
            std::lock_guard<std::mutex> lock(containersMutex);
            std::vector<ContainerHandle *> containers;
            for (auto &c: list) {
                containers.push_back(c.second);
            }
            return containers;
        }

        std::map<std::string, ContainerHandle *> getMap() {
            std::lock_guard<std::mutex> lock(containersMutex);
            return list;
        }

        bool hasContainer(const std::string &name) {
            std::lock_guard<std::mutex> lock(containersMutex);
            return list.find(name) != list.end();
        }
    //TODO: MAKE THIS PRIVATE TO AVOID NON-THREADSAFE ACCESS
    public:
        std::map<std::string, ContainerHandle*> list = {};
        std::mutex containersMutex;
    };
    Containers containers;

    std::map<std::string, NetworkEntryType> network_check_buffer;

    ControlCommunication::AsyncService service;
    std::unique_ptr<grpc::Server> server;
    std::unique_ptr<ServerCompletionQueue> cq;

    std::unique_ptr<pqxx::connection> ctrl_metricsServerConn = nullptr;
    MetricsServerConfigs ctrl_metricsServerConfigs;

    std::vector<spdlog::sink_ptr> ctrl_loggerSinks = {};
    std::shared_ptr<spdlog::logger> ctrl_logger;

    std::map<std::string, NetworkEntryType> ctrl_inDeviceNetworkEntries;

    // TODO: Read from config file
    std::uint64_t ctrl_schedulingIntervalSec = 10;//600;
    ClockType ctrl_nextSchedulingTime = std::chrono::system_clock::now();

    std::map<std::string, std::map<std::string, float>> ctrl_initialRequestRates;

    uint16_t ctrl_systemFPS;
    
//////////////////////////////////////////////////distream_add//////////////////////////////////////////////////////
    std::mutex nodeHandleMutex;
    struct Partitioner
    {
        // NodeHandle& edge;
        // NodeHandle& server;
        // need server here
        float BaseParPoint;
        float FineGrainedOffset;
    };

    struct Partitioner;
    std::vector<NodeHandle> nodes;
    std::pair<std::vector<NodeHandle>, std::vector<NodeHandle>> categorizeNodes(const std::vector<NodeHandle> &nodes);
    double calculateTotalprocessedRate(const PipelineModel *model, const std::vector<NodeHandle> &nodes, bool is_edge);
    int calculateTotalQueue(const std::vector<NodeHandle> &nodes, bool is_edge);
    double getMaxTP(const PipelineModel *model, std::vector<NodeHandle> nodes, bool is_edge);
    void scheduleBaseParPointLoop(const PipelineModel *model, Partitioner *partitioner, std::vector<NodeHandle> nodes);
    float ComputeAveragedNormalizedWorkload(const std::vector<NodeHandle> &nodes, bool is_edge);
    void scheduleFineGrainedParPointLoop(Partitioner *partitioner, const std::vector<NodeHandle> &nodes);
    void DecideAndMoveContainer(const PipelineModel *model, std::vector<NodeHandle> &nodes, Partitioner *partitioner, int cuda_device);
    float calculateRatio(const std::vector<NodeHandle> &nodes);
};


#endif //PIPEPLUSPLUS_CONTROLLER_H