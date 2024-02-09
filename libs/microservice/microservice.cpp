#include<microservice.h>
#include<iostream>
#include<opencv2/opencv.hpp>

/**
 * @brief Construct a new Microservice< In Type>:: Microservice object
 * 
 * @tparam InType 
 * @param configs 
 */
template<typename InType>
Microservice<InType>::Microservice(const BaseMicroserviceConfigs &configs) {
    msvc_name = configs.msvc_name;
    msvc_svcLevelObjLatency = configs.msvc_svcLevelObjLatency;

    std::list<NeighborMicroserviceConfigs>::const_iterator it;
    for (it = configs.dnstreamMicroservices.begin(); it != configs.upstreamMicroservices.end(); ++it) {
        NeighborMicroservice dnStreamMsvc = NeighborMicroservice(configs, numDnstreamMicroservices);
        dnstreamMicroserviceList.emplace_back(dnStreamMsvc);
        classToDnstreamMap.emplace_back({dnStreamMsvc.classOfInterest, numDnstreamMicroservices++});
    }

    for (it = configs.upstreamMicroservices.begin(); it != configs.upstreamMicroservices.end(); ++it) {
        NeighborMicroservice upStreamMsvc = NeighborMicroservice(configs, numUpstreamMicroservices++);
    }
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

