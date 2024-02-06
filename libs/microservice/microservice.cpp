#include<microservice.h>
#include<iostream>
#include<opencv2/opencv.hpp>

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
    OutQueue = new ThreadSafeFixSizedQueue<DataRequest<LocalGPUDataType>>;
}

template<typename InType>
SerDataMicroservice<InType>::SerDataMicroservice(const BaseMicroserviceConfigs &configs)
        :Microservice<InType>(configs) {
    OutQueue = new ThreadSafeFixSizedQueue<DataRequest<CPUReqDataType>>();
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
    OutQueue->emplace(data);
}

template<typename InType>
void SerDataMicroservice<InType>::Schedule() {
    if (Microservice<InType>::InQueue->empty()) {
        return;
    }
    InType data = Microservice<InType>::InQueue->front();
    Microservice<InType>::InQueue->pop();
    // process data
    OutQueue->emplace(data);
}

