#include<microservice.h>
#include<iostream>
#include<opencv2/opencv.hpp>

/**
 * @brief 
 * 
 * @param queueType 
 * @param isInQueue 
 * @return QueueLengthType 
 */
//template<typename InType>
//NumQueuesType GPUDataMicroservice<InType>::generateQueue(const QueueType queueType, bool isInQueue) {
//    QueueLengthType queueNum = -1;
//    if (isInQueue) {
//        std::queue<InType> newQueue;
//        numGPUInQueues++;
//        InQueue = newQueue;
//        queueNum += 1;
//    } else {
//        std::queue<GPUDataRequest> newQueue;
//        numGPUOutQueues++;
//        OutQueue = newQueue;
//        queueNum = OutQueue.size() - 1;
//    }
//    return queueNum;
//}
//
//template<typename InType>
//NumQueuesType ShMemMicroservice<ShmReqDataType>::generateQueue(const QueueType queueType, bool isInQueue) {
//    QueueLengthType queueNum = -1;
//    if (isInQueue) {
//        std::queue<InType> newQueue;
//        numGPUInQueues++;
//        InQueue = newQueue;
//        queueNum += 1;
//    } else {
//        std::queue<DataRequest<ShmReqDataType>> newQueue;
//        numGPUOutQueues++;
//        OutQueue = newQueue;
//        queueNum = OutQueue.size() - 1;
//    }
//    return queueNum;
//}
//
//template<typename InType>
//NumQueuesType SerDataMicroservice<InType>::generateQueue(const QueueType queueType, bool isInQueue) {
//    QueueLengthType queueNum = -1;
//    std::queue<DataRequest<CPUReqDataType>> *newQueue;
//    const cv::dnn::dnn4_v20230620::MatShape size = {1, 3, 640, 640};
//    cv::Mat mat = cv::Mat(size, CV_32F, 3);
//    ClockType time = 1;
//    DataRequest<CPUReqDataType> req = DataRequest<CPUReqDataType>(time, 1, size, "", mat);
//    newQueue->emplace(req);
//    if (isInQueue) {
//        numGPUInQueues++;
//        InQueue = *newQueue;
//        queueNum += 1;
//    } else {
//        numGPUOutQueues++;
//        OutQueue = *newQueue;
//        queueNum = OutQueue.size() - 1;
//    }
//    return queueNum;
//}

template<typename InType>
Microservice<InType>::Microservice(const BaseMicroserviceConfigs &configs) {
    msvc_name = configs.msvc_name;
    msvc_svcLevelObjLatency = configs.msvc_svcLevelObjLatency;


    std::list<NeighborMicroserviceConfigs>::const_iterator it;
    for (it = configs.upstreamMicroservices.begin(); it != configs.upstreamMicroservices.end(); ++it) {
        numDnstreamMicroservices++;
        // NeighborMicroservice
        // For each upstream microservice, a
        // switch (it->commMethod) {
        // case CommMethod::gRPC:
        //     break;

        // case CommMethod::gRPCLocal:
        //     break;

        // case CommMethod::sharedMemory:
        //     break;
        // case CommMethod::localQueue:
        //     // In this case, we don't necessarily as each microservice would communicate with it upstream neighbors through
        //     // local thread-safe queues
        //     break;
        // default:
        //     break;
        // }
        // // If the communication method is specified as `localQueue` then there is no need to set up the input queue for this
        // // microservice as its in queue is the upstream microservice's out queue.
        // // Otherwise, we would need to set up the queue
    }

    // for (it = configs.dnstreamMicroservices.begin(); it != configs.upstreamMicroservices.end(); ++it) {
    //     numUpstreamMicroservices++;
    //     switch (it->commMethod) {
    //     case CommMethod::gRPC:
    //         break;

    //     case CommMethod::gRPCLocal:
    //         break;

    //     case CommMethod::sharedMemory:
    //         break;
    //     case CommMethod::localQueue:
    //         // In this case, we don't necessarily as each microservice would communicate with it upstream neighbors through
    //         // local thread-safe queues
    //         break;
    //     default:
    //         break;
    //     }
    // }
}

template<typename InType>
GPUDataMicroservice<InType>::GPUDataMicroservice(const BaseMicroserviceConfigs &configs)
        :Microservice<InType>(configs) {
    OutQueue = std::queue<GPUDataRequest>();
}

template<typename InType>
ShMemMicroservice<InType>::ShMemMicroservice(const BaseMicroserviceConfigs &configs)
        :Microservice<InType>(configs) {
    OutQueue = std::queue<DataRequest<ShmReqDataType>>();
}

template<typename InType>
SerDataMicroservice<InType>::SerDataMicroservice(const BaseMicroserviceConfigs &configs)
        :Microservice<InType>(configs) {
    OutQueue = std::queue<DataRequest<CPUReqDataType>>();
}

template<typename InType>
void Microservice<InType>::Schedule() {
    if (InQueue->empty()) {
        return;
    }
    InType data = InQueue->front();
    InQueue->pop();
    // process data
    // No out queue as this is only used for final job
}

template<typename InType>
void GPUDataMicroservice<InType>::Schedule() {
    if (Microservice<InType>::InQueue->empty()) {
        return;
    }
    InType data = Microservice<InType>::InQueue->front();
    Microservice<InType>::InQueue->pop();
    // process data
    OutQueue.emplace(data);
}

template<typename InType>
void ShMemMicroservice<InType>::Schedule() {
    if (Microservice<InType>::InQueue->empty()) {
        return;
    }
    InType data = Microservice<InType>::InQueue->front();
    Microservice<InType>::InQueue->pop();
    // process data
    OutQueue.emplace(data);
}

template<typename InType>
void SerDataMicroservice<InType>::Schedule() {
    if (Microservice<InType>::InQueue->empty()) {
        return;
    }
    InType data = Microservice<InType>::InQueue->front();
    Microservice<InType>::InQueue->pop();
    // process data
    OutQueue.emplace(data);
}

