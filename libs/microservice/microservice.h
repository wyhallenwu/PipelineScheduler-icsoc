#include <string>
#include <chrono>
#include <queue>
#include <deque>
#include <list>
#include <opencv4/opencv2/opencv.hpp>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <opencv2/opencv.hpp>

typedef uint16_t NumQueuesType;
typedef uint16_t QueueLengthType;
typedef uint32_t MsvcSLOType;
typedef uint16_t NumMscvType;
typedef cv::Mat CPUReqDataType;
typedef std::string ShmReqDataType;
//typedef std::chrono::high_resolution_clock::time_point ClockType;
typedef int64_t ClockType;
const uint8_t CUDA_IPC_HANDLE_LENGTH = 64; // bytes
typedef const char * GPUReqDataType;
typedef std::vector<int32_t> RequestShapeType;
typedef std::vector<cv::cuda::GpuMat> LocalGPUDataType;
typedef std::vector<cv::Mat> LocalCPUDataType;

template <typename InType, int MaxSize = 100>
class ThreadSafeFixSizedQueue {
private:
    std::queue<InType> queue;
    std::mutex q_mutex;
    std::condition_variable q_condition;

public:
    void emplace(InType request) {
        std::unique_lock<std::mutex> lock(q_mutex);
        if (queue.size() == MaxSize) {
            queue.pop();
        }
        queue.emplace(request);
        q_condition.notify_one();
    }

    InType pop() {
        std::unique_lock<std::mutex> lock(q_mutex);
        q_condition.wait(
            lock,
            [this]() {return !queue.empty();}
        );
        InType request = queue.front();
        queue.pop();
        return request;
    }

    int32_t size() {
        return queue.size();
    }
};

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
    GPUReqDataType req_data;
    GPUDataRequest(
        ClockType genTime,
        MsvcSLOType latency,
        RequestShapeType shape,
        std::string path,
        GPUReqDataType data
    ) : MetaRequest(genTime, latency, shape, path), req_data(data) {
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
    localGPUDataQueue,
    localCPUDataQueue,
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
    // For a Downstream Microservice, this is the data class (defined by the current microservice's model) to be sent this neighbor.
    // For instance, if the model is trained on coco and this neighbor microservice expects coco human, then the value is `0`.
    // Value `-1` denotes all classes.
    // Value `-2` denotes Upstream Microservice.
    int16_t classOfInterest;
    // The shape of data this neighbor microservice expects from the current microservice.
    RequestShapeType expectedShape;
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
    RequestShapeType msvc_dataShape;
    // List of upstream microservices
    std::list<NeighborMicroserviceConfigs> upstreamMicroservices;
    std::list<NeighborMicroserviceConfigs> dnstreamMicroservices;
};



/**
 * @brief 
 * 
 */
template <typename InType>
class Microservice {
public:
    // Constructor that loads a struct args
    Microservice(const BaseMicroserviceConfigs& configs);
    ~Microservice();
    // Name Identifier assigned to the microservice in the format of `type_of_msvc-number`.
    // For instance, an object detector could be named `YOLOv5s-01`.
    // Another example is the
    std::string msvc_name;

    virtual void SetInQueue(ThreadSafeFixSizedQueue<InType> *queue) {
        InQueue = queue;
    };
    virtual void Schedule();

protected:
    struct NeighborMicroservice : NeighborMicroserviceConfigs {
        NumQueuesType queueNum;
        NeighborMicroservice(const NeighborMicroserviceConfigs& configs, NumQueuesType queueNum) 
            :NeighborMicroserviceConfigs(configs), 
             queueNum(queueNum) {}
    };

    MsvcSLOType msvc_svcLevelObjLatency;
    MsvcSLOType msvc_interReqTime;

    uint32_t msvc_inReqCount = 0;
    uint32_t msvc_outReqCount = 0;


    NumMscvType numUpstreamMicroservices = 0;
    NumMscvType numDnstreamMicroservices = 0;

    std::vector<NeighborMicroservice> upstreamMicroserviceList;
    std::vector<NeighborMicroservice> dnstreamMicroserviceList;
    std::vector<std::tuple<uint32_t, uint32_t>> classToDnstreamMap;

    ThreadSafeFixSizedQueue<InType>* InQueue;

    virtual bool isTimeToBatch();
    virtual bool checkReqEligibility();
};

template <typename InType>
class GPUDataMicroservice : public Microservice<InType> {
public:
    GPUDataMicroservice(const BaseMicroserviceConfigs &configs);
    ~GPUDataMicroservice();

    ThreadSafeFixSizedQueue<DataRequest<LocalGPUDataType>>* getOutQueue () {
        return OutQueue;
    }
    void Schedule() override;

protected:
    static ThreadSafeFixSizedQueue<DataRequest<LocalGPUDataType>> *OutQueue;
};

template <typename InType>
class SerDataMicroservice : public Microservice<InType> {
public:
    SerDataMicroservice(const BaseMicroserviceConfigs &configs);
    ~SerDataMicroservice();

    ThreadSafeFixSizedQueue<DataRequest<CPUReqDataType>>* getOutQueue () {
        return OutQueue;
    }
    void Schedule() override;

protected:
    ThreadSafeFixSizedQueue<DataRequest<CPUReqDataType>> *OutQueue;
};

template <typename InType>
class LocalGPUDataMicroservice : public Microservice<InType> {
public:
    LocalGPUDataMicroservice(const BaseMicroserviceConfigs &configs);
    ~LocalGPUDataMicroservice();

    ThreadSafeFixSizedQueue<DataRequest<LocalGPUDataType>>* getOutQueue () {
        return OutQueue;
    }
    void Schedule() override;

protected:
    ThreadSafeFixSizedQueue<DataRequest<LocalGPUDataType>> *OutQueue;
};

// template <typename InType>
// class LocalCPUDataMicroservice : public Microservice<InType> {
// public:
//     LocalCPUDataMicroservice(const BaseMicroserviceConfigs &configs);
//     ~LocalCPUDataMicroservice();

//     ThreadSafeFixSizedQueue<DataRequest<LocalCPUDataType>>* getOutQueue () {
//         return OutQueue;
//     }
//     void Schedule() override;

// protected:
//     ThreadSafeFixSizedQueue<DataRequest<LocalCPUDataType>> *OutQueue;
// };

// template <typename InType>
// class DualLocalDataMicroservice : public Microservice<InType> {
// public:
//     DualLocalDataMicroservice(const BaseMicroserviceConfigs &configs);
//     ~DualLocalDataMicroservice();

//     ThreadSafeFixSizedQueue<DataRequest<LocalGPUDataType>>* getGPUOutQueue () {
//         return LocalGPUOutQueue;
//     }
//     ThreadSafeFixSizedQueue<DataRequest<LocalCPUDataType>>* getCPUOutQueue () {
//         return LocalCPUOutQueue;
//     }
//     void Schedule() override;

// protected:
//     ThreadSafeFixSizedQueue<DataRequest<LocalGPUDataType>> *LocalGPUOutQueue;
//     ThreadSafeFixSizedQueue<DataRequest<LocalCPUDataType>> *LocalCPUOutQueue;
// };