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
#include <fstream>
#include <iostream>
#include "../json/json.h"
#include <vector>

#include <chrono>
#include <ctime>

#ifndef MICROSERVICE_H
#define MICROSERVICE_H

using json = nlohmann::ordered_json;


template<typename DataType>
struct RequestData {
    RequestDataShapeType shape;
    DataType data;

    RequestData(RequestDataShapeType s, DataType d) : data(d) {
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
    BatchTimeType req_origGenTime;
    // The end-to-end service level latency objective to which this request is subject
    RequestSLOType req_e2eSLOLatency;
    // The path that this request and its ancestors have travelled through.
    // Template `[microserviceID_reqNumber][microserviceID_reqNumber][microserviceID_reqNumberWhenItIsSentOut]`
    // For instance, `[YOLOv5_01_05][retinaface_02_09]`
    RequestPathType req_travelPath;

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
            BatchTimeType genTime,
            RequestSLOType latency,
            RequestPathType path,
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
            BatchTimeType genTime,
            RequestSLOType latency,
            RequestPathType path,
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
    Request &operator=(const Request &other) {
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
    std::string q_name;
    std::queue<Request<LocalCPUReqDataType>> q_cpuQueue;
    std::queue<Request<LocalGPUReqDataType>> q_gpuQueue;
    std::mutex q_mutex;
    std::condition_variable q_condition;
    std::uint8_t activeQueueIndex;
    QueueLengthType q_MaxSize = 100;
    std::int16_t class_of_interest;
    bool isEmpty;

public:
    ThreadSafeFixSizedDoubleQueue(QueueLengthType size, int16_t coi) : q_MaxSize(size), class_of_interest(coi) {}

    ~ThreadSafeFixSizedDoubleQueue() {
        std::queue<Request<LocalGPUReqDataType>>().swap(q_gpuQueue);
        std::queue<Request<LocalCPUReqDataType>>().swap(q_cpuQueue);
    }

    /**
     * @brief Emplacing Type 1 requests
     * 
     * @param request 
     */
    void emplace(Request<LocalCPUReqDataType> request) {
        std::unique_lock<std::mutex> lock(q_mutex);
        //if (q_cpuQueue.size() == q_MaxSize) {
        //    q_cpuQueue.pop();
        //}
        q_cpuQueue.emplace(request);
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
        //if (q_gpuQueue.size() == q_MaxSize) {
        //    q_gpuQueue.pop();
        //}
        q_gpuQueue.emplace(request);
        q_condition.notify_one();
        q_mutex.unlock();
    }

    /**
     * @brief poping Type 1 requests
     * 
     * @param request 
     */
    Request<LocalCPUReqDataType> pop1(uint16_t timeout = 100) {
        std::unique_lock<std::mutex> lock(q_mutex);

        Request<LocalCPUReqDataType> request;
        isEmpty = !q_condition.wait_for(
                lock,
                std::chrono::milliseconds(timeout),
                [this]() { return !q_cpuQueue.empty(); }
        );
        if (!isEmpty) {
            request = q_cpuQueue.front();
            q_cpuQueue.pop();
        } else {
            request.req_travelPath = {"empty"};
        }
        q_mutex.unlock();
        return request;
    }

    /**
     * @brief popping Type 2 requests
     * 
     * @param request 
     */
    Request<LocalGPUReqDataType> pop2(uint16_t timeout = 100) {
        std::unique_lock<std::mutex> lock(q_mutex);
        Request<LocalGPUReqDataType> request;
        isEmpty = !q_condition.wait_for(
                lock,
                std::chrono::milliseconds(timeout),
                [this]() { return !q_gpuQueue.empty(); }
        );
        if (!isEmpty) {
            request = q_gpuQueue.front();
            q_gpuQueue.pop();
            q_mutex.unlock();
            return request;
        } else {
            request.req_travelPath = {"empty"};
        }
        return request;
    }

    void setQueueSize(uint32_t queueSize) {
        q_MaxSize = queueSize;
    }

    int32_t size() {
        if (activeQueueIndex == 1) {
            return q_cpuQueue.size();
        } //else if (activeQueueIndex == 2) {
        return q_gpuQueue.size();
        //}
    }

    int32_t size(uint8_t queueIndex) {
        if (queueIndex == 1) {
            return q_cpuQueue.size();
        } //else if (activeQueueIndex == 2) {
        return q_gpuQueue.size();
        //}
    }

    void setActiveQueueIndex(uint8_t index) {
        activeQueueIndex = index;
    }

    uint8_t getActiveQueueIndex() {
        return activeQueueIndex;
    }

    void setClassOfInterest(int16_t classOfInterest) {
        class_of_interest = classOfInterest;
    }

    int16_t getClassOfInterest() {
        return class_of_interest;
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

enum RUNMODE {
    DEPLOYMENT,
    PROFILING
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
        std::vector<RequestDataShapeType> expectedShape;
    };

    /**
     * @brief Have to match the the type here and the type in the json config file
     *
     */
    enum class MicroserviceType {
        // Receiver should have number smaller than 500
        Receiver = 0,
        // DataProcessor should have number between 500 and 1000
        DataSource = 500,
        ProfileGenerator = 501,
        DataSink = 502,
        // Preprocessor should have number between 1000 and 2000
        PreprocessBatcher = 1000,
        // Inferencer should have number larger than 2000
        TRTInferencer = 2000,
        // Postprocessor should have number larger than 3000
        Postprocessor = 3000,
        PostprocessorBBoxCropper = 3001,
        PostProcessorClassifer = 3002,
        PostProcessorSMClassifier = 3003,
        PostProcessorBBoxCropperVerifier = 3004,
        PostProcessorKPointExtractor = 3005,
        // Sender should have number larger than 4000
        Sender = 4000,
        LocalCPUSender = 4001,
        SerializedCPUSender = 4002,
        GPUSender = 4003
    };

    /**
     * @brief
     *
     */
    struct BaseMicroserviceConfigs {
        // Name of the container
        std::string msvc_contName;
        // Name of the microservice
        std::string msvc_name;
        // Type of microservice data receiver, data processor, or data sender
        MicroserviceType msvc_type;
        // Application level configs
        std::string msvc_appLvlConfigs = "";
        // The acceptable latency for each individual request processed by this microservice, in `ms`
        MsvcSLOType msvc_svcLevelObjLatency;
        // 
        QueueLengthType msvc_maxQueueSize;
        // Ideal batch size for this microservice, runtime batch size could be smaller though
        BatchSizeType msvc_idealBatchSize;
        // Shape of data produced by this microservice
        std::vector<RequestDataShapeType> msvc_dataShape;
        // GPU index, -1 means CPU
        int8_t msvc_deviceIndex = 0;
        // Log dir
        std::string msvc_containerLogPath;
        // Run mode
        RUNMODE msvc_RUNMODE = RUNMODE::DEPLOYMENT;
        // List of upstream microservices
        std::list<NeighborMicroserviceConfigs> msvc_upstreamMicroservices;
        std::list<NeighborMicroserviceConfigs> msvc_dnstreamMicroservices;
    };


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


    void from_json(const json &j, NeighborMicroserviceConfigs &val);

    void from_json(const json &j, BaseMicroserviceConfigs &val);

    void to_json(json &j, const NeighborMicroserviceConfigs &val);

    void to_json(json &j, const BaseMicroserviceConfigs &val);
}

using msvcconfigs::NeighborMicroserviceConfigs;
using msvcconfigs::BaseMicroserviceConfigs;
using msvcconfigs::MicroserviceType;

class arrivalReqRecords {
public:
    arrivalReqRecords(uint64_t keepLength = 60000) {
        this->keepLength = std::chrono::milliseconds(keepLength);
    }
    ~arrivalReqRecords() = default;


    /**
     * @brief Add a new arrival to the records. There are 3 timestamps to keep be kept.
     * 1. The time the request is processed by the upstream postprocessor and placed onto the outqueue.
     * 2. The time the request is sent out by upstream sender.
     * 3. The time the request is placed onto the outqueue of receiver.
     *
     * @param timestamps
     */
    void addRecord(RequestTimeType timestamps, uint64_t reqNumber) {
        std::unique_lock<std::mutex> lock(mutex);
        records.push_back(std::make_tuple(timestamps[0], timestamps[1], timestamps[2], reqNumber));
        currNumEntries++;
        totalNumEntries++;
        clearOldRecords();
        mutex.unlock();
    }

    void clearOldRecords() {
        std::chrono::milliseconds timePassed;
        auto timeNow = std::chrono::high_resolution_clock::now();
        auto it = records.begin();
        while (it != records.end()) {
            timePassed = std::chrono::duration_cast<std::chrono::milliseconds>(timeNow - std::get<2>(*it));
            if (timePassed > keepLength) {
                it = records.erase(it);
                currNumEntries--;
            } else {
                break;
            }
        }
    }

    ArrivalRecordType getRecords() {
        std::unique_lock<std::mutex> lock(mutex);
        ArrivalRecordType temp = records;
        records.clear();
        currNumEntries = 0;
        mutex.unlock();
        return temp;
    }
    void setKeepLength(uint64_t keepLength) {
        this->keepLength = std::chrono::milliseconds(keepLength);
    }

private:
    std::mutex mutex;
    ArrivalRecordType records;
    std::chrono::milliseconds keepLength;
    uint64_t totalNumEntries = 0, currNumEntries = 0;
};

/**
 * @brief 
 * 
 */
class Microservice {
public:
    // Constructor that loads a struct args
    explicit Microservice(const json &jsonConfigs);

    virtual ~Microservice() = default;

    // Name Identifier assigned to the microservice in the format of `type_of_msvc-number`.
    // For instance, an object detector could be named `YOLOv5s-01`.
    // Another example is the
    std::string msvc_name;

    // Name of the contianer that holds this microservice
    std::string msvc_containerName;


    void SetInQueue(std::vector<ThreadSafeFixSizedDoubleQueue *> queue) {
        msvc_InQueue = std::move(queue);
    };

    std::vector<ThreadSafeFixSizedDoubleQueue *> GetOutQueue() {
        return msvc_OutQueue;
    };

    ThreadSafeFixSizedDoubleQueue *GetOutQueue(int coi) {
        for (auto &queue: msvc_OutQueue) {
            if (queue->getClassOfInterest() == coi) {
                return queue;
            }
        }
        return nullptr;
    };

    MicroserviceType getMsvcType() {
        return msvc_type;
    }

    virtual QueueLengthType GetOutQueueSize(int i) { return msvc_OutQueue[i]->size(); };

    int GetDroppedReqCount() const { return droppedReqCount; };

    int GetArrivalRate() const { return msvc_interReqTime; };

    void stopThread() {
        STOP_THREADS = true;
    }

    void pauseThread() {
        PAUSE_THREADS = true;
        READY = false;
    }

    void unpauseThread() {
        PAUSE_THREADS = false;
    }

    bool checkReady() {
        return READY;
    }

    bool checkPause() {
        return PAUSE_THREADS;
    }

    RUNMODE checkMode() {
        return msvc_RUNMODE;
    }

    void setRELOAD() {
        RELOADING = true;
    }

    /**
     * @brief Set the Device index
     * should be called at least once for each thread
     * 
     */
    void setDevice() {
        setDevice(msvc_deviceIndex);
    }

    void setInferenceShape(RequestShapeType shape) {
        msvc_inferenceShape = shape;
    }

    virtual void setProfileConfigs(const json &profileConfigs) {};

    /**
     * @brief Set the Device index
     * should be called at least once for each thread (except when the above function is already called)
     * 
     */
    void setDevice(int8_t deviceIndex) {
        if (deviceIndex >= 0) {
            int currentDevice = cv::cuda::getDevice();
            if (currentDevice != deviceIndex) {
                cv::cuda::resetDevice();
                cv::cuda::setDevice(deviceIndex);
                checkCudaErrorCode(cudaSetDevice(deviceIndex), __func__);
            }
            cudaFree(0);
        }
    }

    void setDeviceIndex(int8_t deviceIndex) {
        msvc_deviceIndex = deviceIndex;
    }

    void setContainerLogPath(std::string dirPath) {
        msvc_microserviceLogPath = dirPath + "/" + msvc_name;
    }

    virtual void dispatchThread() {};

    virtual void loadConfigs(const json &jsonConfigs, bool isConstructing = false);

    virtual ArrivalRecordType getArrivalRecords() {
        return {};
    }

    bool RELOADING = true;

    std::ofstream msvc_logFile;

    bool PAUSE_THREADS = false;

protected:
    std::vector<ThreadSafeFixSizedDoubleQueue *> msvc_InQueue, msvc_OutQueue;
    //
    std::vector<uint8_t> msvc_activeInQueueIndex = {}, msvc_activeOutQueueIndex = {};

    // Used to signal to thread when not to run and to bring thread to a natural end.
    bool STOP_THREADS = false;
    bool READY = false;

    /**
     * @brief Running mode of the container, globally set for all microservices inside the container
     * Default to be deployment.
     */
    RUNMODE msvc_RUNMODE = RUNMODE::DEPLOYMENT;

    //Path to specific Application configurations for this microservice
    std::string msvc_appLvlConfigs = "";

    // GPU index, -1 means CPU
    int8_t msvc_deviceIndex = -1;

    //type
    MicroserviceType msvc_type;

    //
    MsvcSLOType msvc_svcLevelObjLatency;
    //
    MsvcSLOType msvc_interReqTime = 1;

    //
    uint64_t msvc_inReqCount = 0;
    //
    uint64_t msvc_outReqCount = 0;

    //
    NumMscvType nummsvc_upstreamMicroservices = 0;
    //
    NumMscvType nummsvc_dnstreamMicroservices = 0;

    // The expected shape of the data for the next microservice
    std::vector<std::vector<RequestDataShapeType>> msvc_outReqShape;
    // The shape of the data to be processed by this microservice
    std::vector<RequestDataShapeType> msvc_dataShape;

    RequestShapeType msvc_inferenceShape;

    // Ideal batch size for this microservice, runtime batch size could be smaller though
    BatchSizeType msvc_idealBatchSize;

    //
    MODEL_DATA_TYPE msvc_modelDataType = MODEL_DATA_TYPE::fp32;

    //
    std::vector<msvcconfigs::NeighborMicroservice> upstreamMicroserviceList;
    //
    std::vector<msvcconfigs::NeighborMicroservice> dnstreamMicroserviceList;
    //
    std::vector<std::pair<int16_t, NumQueuesType>> classToDnstreamMap;

    //
    virtual bool isTimeToBatch() { return true; };

    //
    virtual bool checkReqEligibility(std::vector<ClockType> &currReq_time) { return true; };

    //
    virtual void updateReqRate(ClockType lastInterReqDuration);

    // Logging file path, where each microservice is supposed to log in running metrics
    std::string msvc_microserviceLogPath;

    int droppedReqCount;
};


RequestData<LocalGPUReqDataType> uploadReqData(
        const RequestData<LocalCPUReqDataType> &cpuData,
        void *cudaPtr = NULL,
        cv::cuda::Stream &stream = cv::cuda::Stream::Null()
);

Request<LocalGPUReqDataType> uploadReq(
        const Request<LocalCPUReqDataType> &cpuReq,
        std::vector<void *> cudaPtr = {},
        cv::cuda::Stream &stream = cv::cuda::Stream::Null()
);


#endif