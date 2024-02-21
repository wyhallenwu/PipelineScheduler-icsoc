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
        // Create downstream neigbor config and push that into a list for information later
        // Local microservice supposedly has only 1 downstream but `sender` microservices could have multiple.
        NeighborMicroservice dnStreamMsvc = NeighborMicroservice(configs, numDnstreamMicroservices);
        dnstreamMicroserviceList.emplace_back(dnStreamMsvc);
        // This maps the data class to be send to this downstream microservice and the microservice's index.
        classToDnstreamMap.emplace_back({dnStreamMsvc.classOfInterest, numDnstreamMicroservices++});
    }

    for (it = configs.upstreamMicroservices.begin(); it != configs.upstreamMicroservices.end(); ++it) {
        NeighborMicroservice upStreamMsvc = NeighborMicroservice(configs, numUpstreamMicroservices++);
    }
}

/**
 * @brief 
 * 
 * @tparam InType 
 * @param lastInterReqDuration 
 */
template<typename InType>
void Microservice<InType>::updateReqRate(ClockTypeTemp lastInterReqDuration) {
    msvc_interReqTime = 0.0001;
}

template<typename InType>
GPUDataMicroservice<InType>::GPUDataMicroservice(const BaseMicroserviceConfigs &configs)
        :Microservice<InType>(configs) {
    OutQueue = new ThreadSafeFixSizedQueue<DataRequest<LocalGPUReqDataType>>();
}

template<typename InType>
LocalGPUDataMicroservice<InType>::LocalGPUDataMicroservice(const BaseMicroserviceConfigs &configs)
        :Microservice<InType>(configs) {
    OutQueue = new ThreadSafeFixSizedQueue<DataRequest<LocalGPUReqDataType>>();
}

template<typename InType>
SerDataMicroservice<InType>::SerDataMicroservice(const BaseMicroserviceConfigs &configs)
        :Microservice<InType>(configs) {
    OutQueue = new ThreadSafeFixSizedQueue<DataRequest<InterConCPUReqDataType>>();
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
