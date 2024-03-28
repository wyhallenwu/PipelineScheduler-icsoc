#include <chrono>
#include <queue>
#include <deque>
#include <list>
#include <opencv4/opencv2/opencv.hpp>
#include <mutex>
#include <misc.h>
#include <condition_variable>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <utility>
#include <spdlog/spdlog.h>

#ifndef MICROSERVICE_H
#define MICROSERVICE_H

typedef uint16_t NumQueuesType;
typedef uint16_t QueueLengthType;
typedef uint32_t MsvcSLOType;
typedef uint16_t NumMscvType;
typedef std::chrono::high_resolution_clock::time_point ClockType;
const uint8_t CUDA_IPC_HANDLE_LENGTH = 64; // bytes
typedef const char *InterConGPUReqDataType;
typedef std::vector<int32_t> RequestShapeType;
typedef cv::cuda::GpuMat LocalGPUReqDataType;
typedef cv::Mat LocalCPUReqDataType;
typedef uint16_t BatchSizeType;


template<typename DataType>
struct RequestData {
    RequestShapeType shape;
    DataType data;

    RequestData(RequestShapeType s, DataType d) : data(d) {
        shape = s;
    }

    RequestData() {}
    ~RequestData() {
        data.release();
        shape.clear();
    }
};

/**
 * @brief 
 * 
 * @tparam RequestData
 */
template<typename DataType>
struct Request {
    // The moment this request was generated at the begining of the pipeline.
    ClockType req_origGenTime = std::chrono::_V2::system_clock::now();
    // The end-to-end service level latency objective to which this request is subject
    MsvcSLOType req_e2eSLOLatency = 0;
    // The path that this request and its ancestors have travelled through.
    // Template `[microserviceID_reqNumber][microserviceID_reqNumber][microserviceID_reqNumberWhenItIsSentOut]`
    // For instance, `[YOLOv5Prep-01_05][YOLOv5s_05][YOLOv5post_07]`
    std::string req_travelPath = "";

    // Batch size
    BatchSizeType req_batchSize = 0;

    // The Inter-container GPU data of that this request carries.
    std::vector<RequestData<DataType>> req_data = {};
    // To carry the data of the upstream microservice in case we need them for further processing.
    // For instance, for cropping we need both the original image (`upstreamReq_data`) and the output
    // of the inference engine, which is a result of `req_data`.
    // If there is nothing to carry, it is a blank vector.
    std::vector<RequestData<DataType>> upstreamReq_data = {};

    Request() {};

    Request(
        ClockType genTime,
        MsvcSLOType latency,
        std::string path,
        BatchSizeType batchSize,
        std::vector<RequestData<DataType>> data,
        std::vector<RequestData<DataType>> upstream_data

    ) : req_origGenTime(genTime),
        req_e2eSLOLatency(latency),
        req_travelPath(std::move(path)),
        req_batchSize(batchSize) {
            req_data = data;
            upstreamReq_data = upstream_data;
    }
    
    // df
    Request(
        ClockType genTime,
        MsvcSLOType latency,
        std::string path,
        BatchSizeType batchSize,
        std::vector<RequestData<DataType>> data
    ) : req_origGenTime(genTime),
        req_e2eSLOLatency(latency),
        req_travelPath(std::move(path)),
        req_batchSize(batchSize) {
            req_data = data;
    }

    /**
     * @brief making our request 
     * 
     * @param other 
     * @return Request& 
     */
    Request& operator=(const Request& other) {
        if (this != &other) {
            req_origGenTime = other.req_origGenTime;
            req_e2eSLOLatency = other.req_e2eSLOLatency;
            req_travelPath = other.req_travelPath;
            req_batchSize = other.req_batchSize;
            req_data = other.req_data;
            upstreamReq_data = other.upstreamReq_data;
        }
        return *this;
    }

    ~Request() {
        req_data.clear();
        upstreamReq_data.clear();
    }
};

//template<int MaxSize=100>
class ThreadSafeFixSizedDoubleQueue {
private:
    std::queue<Request<LocalCPUReqDataType>> cpuQueue;
    std::queue<Request<LocalGPUReqDataType>> gpuQueue;
    std::mutex q_mutex;
    std::condition_variable q_condition;
    std::uint8_t activeQueueIndex;
    size_t MaxSize = 100;

public:
    /**
     * @brief Emplacing Type 1 requests
     * 
     * @param request 
     */
    void emplace(Request<LocalCPUReqDataType> request) {
        std::unique_lock<std::mutex> lock(q_mutex);
        if (cpuQueue.size() == MaxSize) {
            cpuQueue.pop();
        }
        cpuQueue.emplace(request);
        q_condition.notify_one();
        q_mutex.unlock();
    }

    /**
     * @brief Emplacing Type 2 requests
     * 
     * @param request 
     */
    void emplace(Request<LocalGPUReqDataType> request) {
        std::unique_lock<std::mutex> lock(q_mutex);
        if (gpuQueue.size() == MaxSize) {
            gpuQueue.pop();
        }
        gpuQueue.emplace(request);
        q_condition.notify_one();
        q_mutex.unlock();
    }

    /**
     * @brief poping Type 1 requests
     * 
     * @param request 
     */
    Request<LocalCPUReqDataType> pop1() {
        std::unique_lock<std::mutex> lock(q_mutex);
        q_condition.wait(
                lock,
                [this]() { return !cpuQueue.empty(); }
        );
        Request<LocalCPUReqDataType> request = cpuQueue.front();
        cpuQueue.pop();
        q_mutex.unlock();
        return request;
    }

    /**
     * @brief popping Type 2 requests
     * 
     * @param request 
     */
    Request<LocalGPUReqDataType> pop2() {
        std::unique_lock<std::mutex> lock(q_mutex);
        q_condition.wait(
                lock,
                [this]() { return !gpuQueue.empty(); }
        );
        Request<LocalGPUReqDataType> request = gpuQueue.front();
        gpuQueue.pop();
        q_mutex.unlock();
        return request;
    }

    int32_t size() {
        if (activeQueueIndex == 1) {
            return cpuQueue.size();
        } //else if (activeQueueIndex == 2) {
        return gpuQueue.size();
        //}
    }

    int32_t size(uint8_t queueIndex) {
        if (queueIndex == 1) {
            return cpuQueue.size();
        } //else if (activeQueueIndex == 2) {
        return gpuQueue.size();
        //}
    }

    void setActiveQueueIndex(uint8_t index) {
        activeQueueIndex = index;
    }

    ~ThreadSafeFixSizedDoubleQueue() {
        std::queue<Request<LocalGPUReqDataType>>().swap(gpuQueue);
        std::queue<Request<LocalCPUReqDataType>>().swap(cpuQueue);
    }
    uint8_t getActiveQueueIndex() {
        return activeQueueIndex;
    }
};

/**
 * @brief 
 * 
 */
enum class CommMethod {
    sharedMemory,
    GpuAddress,
    serialized,
    localGPU,
    localCPU,
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
        std::list<NeighborMicroserviceConfigs> msvc_upstreamMicroservices;
        std::list<NeighborMicroserviceConfigs> msvc_dnstreamMicroservices;
    };
}

using msvcconfigs::NeighborMicroserviceConfigs;
using msvcconfigs::BaseMicroserviceConfigs;
using msvcconfigs::MicroserviceType;


/**
 * @brief 
 * 
 */
class Microservice {
public:
    // Constructor that loads a struct args
    explicit Microservice(const BaseMicroserviceConfigs &configs);

    virtual ~Microservice() = default;

    // Name Identifier assigned to the microservice in the format of `type_of_msvc-number`.
    // For instance, an object detector could be named `YOLOv5s-01`.
    // Another example is the
    std::string msvc_name;

    void SetInQueue(std::vector<ThreadSafeFixSizedDoubleQueue*> queue) {
        msvc_InQueue = std::move(queue);
    };

    std::vector<ThreadSafeFixSizedDoubleQueue*> GetOutQueue() {
        return msvc_OutQueue;
    };

    virtual QueueLengthType GetOutQueueSize(int i) {return msvc_OutQueue[i]->size();};

protected:
    std::vector<ThreadSafeFixSizedDoubleQueue*> msvc_InQueue, msvc_OutQueue;
    //
    std::vector<uint8_t> msvc_activeInQueueIndex = {}, msvc_activeOutQueueIndex = {};

    // Used to signal to thread when not to run and to bring thread to a natural end.
    bool STOP_THREADS = false;
    bool PAUSE_THREADS = false;

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
    MsvcSLOType msvc_interReqTime = 1;

    //
    uint32_t msvc_inReqCount = 0;
    //
    uint32_t msvc_outReqCount = 0;

    //
    NumMscvType nummsvc_upstreamMicroservices = 0;
    //
    NumMscvType nummsvc_dnstreamMicroservices = 0;

    // The expected shape of the data for the next microservice
    std::vector<std::vector<RequestShapeType>> msvc_outReqShape;
    // The shape of the data to be processed by this microservice
    std::vector<RequestShapeType> msvc_dataShape;

    // Ideal batch size for this microservice, runtime batch size could be smaller though
    BatchSizeType msvc_idealBatchSize;

    //
    MODEL_DATA_TYPE msvc_modelDataType = MODEL_DATA_TYPE::fp32;

    //
    std::vector<NeighborMicroservice> upstreamMicroserviceList;
    //
    std::vector<NeighborMicroservice> dnstreamMicroserviceList;
    //
    std::vector<std::pair<int16_t, NumQueuesType>> classToDnstreamMap;

    //
    virtual bool isTimeToBatch() {return true;};

    //
    virtual bool checkReqEligibility(ClockType currReq_genTime) {return true;};

    //
    virtual void updateReqRate(ClockType lastInterReqDuration);


};


#endif