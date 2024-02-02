#include <string>
#include <chrono>
#include <queue>
#include <list>
#include <opencv4/opencv2/opencv.hpp>

typedef uint16_t NumQueuesType;
typedef uint16_t QueueLengthType;
typedef uint32_t MsvcSLOType;
typedef uint16_t NumMscvType;
typedef cv::Mat CPUReqDataType;
typedef std::string ShmReqDataType;
//typedef std::chrono::high_resolution_clock::time_point ClockType;
typedef uint16_t ClockType;
const uint8_t CUDA_IPC_HANDLE_LENGTH = 64; // bytes
typedef char GPUReqDataType;
typedef std::vector<int32_t> RequestShapeType;
/**
 * @brief 
 * 
 * @tparam RequestData
 */
struct MetaRequest {
    // The moment this request was generated at the begining of the pipeline.
    ClockType req_origGenTime;
    // The end-to-end service level latency objective to which this request is subject
    MsvcSLOType req_e2eSLOLatency;
    // Shape of data contained in the request. Helps interpret the data.
    RequestShapeType req_dataShape;
    // The path that this request and its ancestors have travelled through.
    // Template `[microserviceID_reqNumber][microserviceID_reqNumber][microserviceID_reqNumberWhenItIsSentOut]`
    // For instance, `[YOLOv5Prep-01_05][YOLOv5s_05][YOLOv5post_07]`
    std::string req_travelPath;

    MetaRequest(
        ClockType genTime,
        MsvcSLOType latency, 
        RequestShapeType shape,
        std::string path
    ) : req_origGenTime(genTime), req_e2eSLOLatency(latency), req_dataShape(shape), req_travelPath(path) {}
};

/**
 * @brief 
 * 
 */
struct GPUDataRequest : MetaRequest {
    // The data of that this request carries.
    // There are several types of data a request can carry.
    GPUReqDataType req_data[CUDA_IPC_HANDLE_LENGTH];
    GPUDataRequest(
        ClockType genTime,
        MsvcSLOType latency,
        RequestShapeType shape,
        std::string path,
        GPUReqDataType data[CUDA_IPC_HANDLE_LENGTH]
    ) : MetaRequest(genTime, latency, shape, path) {
        strcpy(req_data, data);
    };
};

/**
 * @brief 
 * 
 * @tparam DataType 
 */
template <typename DataType>
struct DataRequest : MetaRequest {
    DataType req_data;

    DataRequest<DataType>(
        ClockType genTime,
        MsvcSLOType latency,
        RequestShapeType shape,
        std::string path,
        DataType data
    ) : MetaRequest(genTime, latency, shape, path), req_data(data) {};
};

/**
 * @brief 
 * 
 */
enum class CommMethod {
    sharedMemory,
    gRPCLocal,
    gRPC,
    localQueue,
};

/**
 * @brief 
 * 
 */
enum class QueueType {
    gpuDataQueue,
    shmDataQueue,
    cpuDataQueue,
};

/**
 * @brief Descriptions of up and downstream microservices neighboring this current microservice.
 * 
 * 
 */
struct NeighborMicroserviceConfigs {
    // Name of the up/downstream microservice
    std::string name;
    // The communication method for the microservice to 
    CommMethod commMethod;
    //
    std::string link;
    //
    QueueType queueType;
    //
    QueueLengthType maxQueueSize;
};

/**
 * @brief
 * 
 */
enum class MicroserviceType {
    Receiver,
    Regular,
    Sender,
};

/**
 * @brief 
 * 
 */
struct BaseMicroserviceConfigs {
    // Name of the microservice
    std::string msvc_name;
    // Type of microservice data receiver, data processor, or data sender
    MicroserviceType msvc_type;
    // The acceptable latency for each individual request processed by this microservice, in `ms`
    MsvcSLOType msvc_svcLevelObjLatency;
    // Shape of data produced by this microservice
    std::vector<uint16_t> msvc_dataShape;
    // List of upstream microservices
    std::list<NeighborMicroserviceConfigs> upstreamMicroservices;
    std::list<NeighborMicroserviceConfigs> dnstreamMicroservices;
};



/**
 * @brief 
 * 
 */
class Microservice {
public:
    // Constructor that loads a struct args
    Microservice(const BaseMicroserviceConfigs& configs);
    ~Microservice();
    // Name Identifier assigned to the microservice in the format of `type_of_msvc-number`.
    // For instance, an object detector could be named `YOLOv5s-01`.
    // Another example is the 
    std::string msvc_name;
protected:
    struct NeighborMicroservice : NeighborMicroserviceConfigs {
        NumQueuesType queueNum;
        NeighborMicroservice(const NeighborMicroserviceConfigs& configs) {
            name = configs.name;
            commMethod = configs.commMethod;
            link = configs.link;
            queueType = configs.queueType;
            maxQueueSize = configs.maxQueueSize;
        }
    };

    MsvcSLOType msvc_svcLevelObjLatency;
    NumMscvType numUpstreamMicroservices = 0;
    NumMscvType numShmUpstreamMicroservices = 0;
    NumMscvType numGRPCUpstreamMicroservices = 0;
    NumMscvType numQueueUpstreamMicroservices = 0;

    NumMscvType numDnstreamMicroservices = 0;
    NumMscvType numShmDnstreamMicroservices = 0;
    NumMscvType numGRPCDnstreamMicroservices = 0;
    NumMscvType numQueueDnstreamMicroservices = 0;

    std::vector<NeighborMicroservice> upstreamMicroserviceList;
    std::vector<NeighborMicroservice> dnstreamMicroserviceList;


    NumQueuesType numGPUOutQueues = 0;
    NumQueuesType numShmOutQueues = 0;
    NumQueuesType numCPUOutQueues = 0;

    NumQueuesType numGPUInQueues = 0;
    NumQueuesType numShmInQueues = 0;
    NumQueuesType numCPUInQueues = 0;

    
    std::vector<std::queue<GPUDataRequest>> gpuInQueueList;
    std::vector<std::queue<GPUDataRequest>> gpuOutQueueList;
    std::vector<std::queue<DataRequest<ShmReqDataType>>> shmInQueueList;
    std::vector<std::queue<DataRequest<ShmReqDataType>>> shmOutQueueList;
    std::vector<std::queue<DataRequest<CPUReqDataType>>> cpuInQueueList;
    std::vector<std::queue<DataRequest<CPUReqDataType>>> cpuOutQueueList;

    NumQueuesType generateQueue(const QueueType queueType, bool isInQueue);

};
