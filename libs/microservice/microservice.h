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
#include <utility>

#ifndef MICROSERVICE_H
#define MICROSERVICE_H

typedef uint16_t NumQueuesType;
typedef uint16_t QueueLengthType;
typedef uint32_t MsvcSLOType;
typedef uint16_t NumMscvType;
typedef cv::Mat InterConCPUReqDataType;
typedef std::string ShmReqDataType;
typedef std::chrono::high_resolution_clock::time_point ClockTypeTemp;
typedef int64_t ClockType;
const uint8_t CUDA_IPC_HANDLE_LENGTH = 64; // bytes
typedef const char *InterConGPUReqDataType;
typedef std::vector<int32_t> RequestShapeType;
typedef cv::cuda::GpuMat LocalGPUReqDataType;
typedef cv::Mat LocalCPUDataType;
typedef uint16_t BatchSizeType;

template<typename InType, int MaxSize = 100>
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
                [this]() { return !queue.empty(); }
        );
        InType request = queue.front();
        queue.pop();
        return request;
    }

    int32_t size() {
        return queue.size();
    }
};

template<typename T, int MaxSize = 100>
class FixSizedQueue {
private:
    std::queue<T> queue;
public:
    void emplace(T elem) {
        if (queue.size() == MaxSize) {
            queue.pop();
        }
        queue.emplace(elem);
    }

    T pop() {
        T out = queue.front();
        queue.pop();
        return out;
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

    // Batch size
    BatchSizeType req_batchSize;

    MetaRequest(
            ClockType genTime,
            MsvcSLOType latency,
            std::string path,
            BatchSizeType batchSize
    ) : req_origGenTime(genTime), req_e2eSLOLatency(latency), req_travelPath(std::move(path)),
        req_batchSize(batchSize) {}
};

struct GPUData {
    RequestShapeType shape;
    InterConGPUReqDataType data;
};

/**
 * @brief Sending CUDA Handle
 * 
 */
struct GPUDataRequest : MetaRequest {
    // The GPU data of that this request carries.
    std::vector<GPUData> req_data;

    GPUDataRequest(
            ClockType genTime,
            MsvcSLOType latency,
            std::string path,
            BatchSizeType batchSize,
            std::vector<GPUData> data
    ) : MetaRequest(genTime, latency, std::move(path), batchSize), req_data(std::move(data)) {
    };
};

template<typename Type>
struct Data {
    RequestShapeType shape;
    Type content;
};

/**
 * @brief 
 * 
 * @tparam DataType 
 */
template<typename DataType>
struct DataRequest : MetaRequest {
    std::vector<Data<DataType>> req_data;

    DataRequest<DataType>(
            ClockType genTime,
            MsvcSLOType latency,
            std::string path,
            BatchSizeType batchSize,
            std::vector<Data<DataType>> data
    ) : MetaRequest(genTime, latency, path, batchSize), req_data(data) {};
};

/**
 * @brief 
 * 
 */
enum class CommMethod {
    sharedMemory,
    gRPCLocal, // gRPCLocal = GPU
    gRPC,
    localQueue,
};

/**
 * @brief 
 * 
 */
enum class QueueType {
    none,
    localGPUDataQueue,
    localCPUDataQueue,
    gpuDataQueue,
    shmDataQueue,
    cpuDataQueue,
};

enum class NeighborType {
    Upstream,
    Downstream,
};

namespace msvcconfigs {
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
        std::vector<std::string> link;
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
        std::vector<RequestShapeType> expectedShape;
    };

    /**
     * @brief
     *
     */
    enum class MicroserviceType {
        Receiver,
        Preprocessor,
        Inference,
        Postprocessor,
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
        // Ideal batch size for this microservice, runtime batch size could be smaller though
        BatchSizeType msvc_idealBatchSize;
        // Shape of data produced by this microservice
        std::vector<RequestShapeType> msvc_dataShape;
        // List of upstream microservices
        std::list<NeighborMicroserviceConfigs> upstreamMicroservices;
        std::list<NeighborMicroserviceConfigs> dnstreamMicroservices;
    };
}

using msvcconfigs::NeighborMicroserviceConfigs;
using msvcconfigs::BaseMicroserviceConfigs;
using msvcconfigs::MicroserviceType;


/**
 * @brief 
 * 
 */
template<typename InType>
class Microservice {
public:
    // Constructor that loads a struct args
    explicit Microservice(const BaseMicroserviceConfigs &configs);

    virtual ~Microservice() = default;

    // Name Identifier assigned to the microservice in the format of `type_of_msvc-number`.
    // For instance, an object detector could be named `YOLOv5s-01`.
    // Another example is the
    std::string msvc_name;

    virtual void SetInQueue(ThreadSafeFixSizedQueue<InType> *queue) {
        InQueue = queue;
    };

    virtual QueueLengthType GetOutQueueSize();

    virtual void Schedule();

protected:
    // Used to signal to thread when not to run and to bring thread to a natural end.
    bool RUN_THREADS = false;

    /**
     * @brief 
     * 
     */
    struct NeighborMicroservice : NeighborMicroserviceConfigs {
        NumQueuesType queueNum;

        NeighborMicroservice(const NeighborMicroserviceConfigs &configs, NumQueuesType queueNum)
                : NeighborMicroserviceConfigs(configs),
                  queueNum(queueNum) {}
    };

    //
    MsvcSLOType msvc_svcLevelObjLatency;
    //
    MsvcSLOType msvc_interReqTime;

    //
    uint32_t msvc_inReqCount = 0;
    //
    uint32_t msvc_outReqCount = 0;

    //
    NumMscvType numUpstreamMicroservices = 0;
    //
    NumMscvType numDnstreamMicroservices = 0;

    //
    std::vector<RequestShapeType> msvc_outReqShape;

    // Ideal batch size for this microservice, runtime batch size could be smaller though
    BatchSizeType msvc_idealBatchSize;

    //
    std::vector<NeighborMicroservice> upstreamMicroserviceList;
    //
    std::vector<NeighborMicroservice> dnstreamMicroserviceList;
    //
    std::vector<std::tuple<uint32_t, uint32_t>> classToDnstreamMap;

    //
    ThreadSafeFixSizedQueue<InType> *InQueue;

    //
    virtual bool isTimeToBatch();

    //
    virtual bool checkReqEligibility(ClockTypeTemp currReq_genTime);

    //
    virtual void updateReqRate(ClockTypeTemp lastInterReqDuration);
};

template<typename InType>
class GPUDataMicroservice : public Microservice<InType> {
public:
    explicit GPUDataMicroservice(const BaseMicroserviceConfigs &configs);

    ThreadSafeFixSizedQueue<DataRequest<LocalGPUReqDataType>> *getOutQueue() {
        return OutQueue;
    }

    QueueLengthType GetOutQueueSize() {
        return OutQueue->size();
    }

protected:
    static ThreadSafeFixSizedQueue<DataRequest<LocalGPUReqDataType>> *OutQueue;
};

template<typename InType>
class SerDataMicroservice : public Microservice<InType> {
public:
    explicit SerDataMicroservice(const BaseMicroserviceConfigs &configs);

    ThreadSafeFixSizedQueue<DataRequest<InterConCPUReqDataType>> *getOutQueue() {
        return OutQueue;
    }

    QueueLengthType GetOutQueueSize() {
        return OutQueue->size();
    }

protected:
    ThreadSafeFixSizedQueue<DataRequest<InterConCPUReqDataType>> *OutQueue;
};

template<typename InType>
class LocalGPUDataMicroservice : public Microservice<InType> {
public:
    explicit LocalGPUDataMicroservice(const BaseMicroserviceConfigs &configs);

    ThreadSafeFixSizedQueue<DataRequest<LocalGPUReqDataType>> *getOutQueue() {
        return OutQueue;
    }

    QueueLengthType GetOutQueueSize() {
        return OutQueue->size();
    }

protected:
    ThreadSafeFixSizedQueue<DataRequest<LocalGPUReqDataType>> *OutQueue;
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

//     ThreadSafeFixSizedQueue<DataRequest<LocalGPUReqDataType>>* getGPUOutQueue () {
//         return LocalGPUOutQueue;
//     }
//     ThreadSafeFixSizedQueue<DataRequest<LocalCPUDataType>>* getCPUOutQueue () {
//         return LocalCPUOutQueue;
//     }
//     void Schedule() override;

// protected:
//     ThreadSafeFixSizedQueue<DataRequest<LocalGPUReqDataType>> *LocalGPUOutQueue;
//     ThreadSafeFixSizedQueue<DataRequest<LocalCPUDataType>> *LocalCPUOutQueue;
// };
#endif