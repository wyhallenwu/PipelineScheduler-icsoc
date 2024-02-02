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
NumQueuesType Microservice::generateQueue(const QueueType queueType, bool isInQueue) {
    QueueLengthType queueNum = -1;
    switch (queueType) {
        case QueueType::gpuDataQueue: {
            std::queue<GPUDataRequest> newQueue;
            if (isInQueue) {
                numGPUInQueues++;
                gpuInQueueList.emplace_back(newQueue);
                queueNum = gpuInQueueList.size() - 1;
            } else {
                numGPUOutQueues++;
                gpuOutQueueList.emplace_back(newQueue);
                queueNum = gpuOutQueueList.size() - 1;
            }
            break;
        }
        case QueueType::shmDataQueue: {
            std::queue<DataRequest<ShmReqDataType>> newQueue;
            if (isInQueue) {
                numShmInQueues++;
                shmInQueueList.emplace_back(newQueue);
                queueNum = shmInQueueList.size() - 1;
            } else {
                numShmOutQueues++;
                shmOutQueueList.emplace_back(newQueue);
                queueNum = shmOutQueueList.size() - 1;
            }
            break;
        }
        case QueueType::cpuDataQueue: {
            std::queue<DataRequest<CPUReqDataType>> *newQueue;
            std::vector<void*> newQueueList;
            newQueueList.emplace_back(newQueue)
            const cv::dnn::dnn4_v20230620::MatShape size = {1, 3, 640, 640};
            cv::Mat mat = cv::Mat(size, CV_32F, 3);
            ClockType time = 1;
            DataRequest<CPUReqDataType> req = DataRequest<CPUReqDataType>(time, 1, size, "", mat);
            &newQueueList[0].emplace_back(req);
            std::cout << &newQueueList[0].size();
            if (isInQueue) {
                numCPUInQueues++;
                cpuInQueueList.emplace_back(newQueue);
                queueNum = cpuInQueueList.size() - 1;
            } else {
                numCPUOutQueues++;
                cpuOutQueueList.emplace_back(newQueue);
                queueNum = cpuOutQueueList.size() - 1;
            }
            break;
        }
        default:
            break;
    }
    return queueNum;
}

Microservice::Microservice(const BaseMicroserviceConfigs& configs) {
    msvc_name = configs.msvc_name;
    msvc_svcLevelObjLatency = configs.msvc_svcLevelObjLatency;


    std::list<NeighborMicroserviceConfigs>::const_iterator it;
    for (it = configs.upstreamMicroservices.begin();it != configs.upstreamMicroservices.end(); ++it) {
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
