#include<microservice.h>
#include<iostream>
#include<opencv2/opencv.hpp>

/**
 * @brief Construct a new Microservice< In Type>:: Microservice object
 * 
 * @tparam InType 
 * @param configs 
 */
Microservice::Microservice(const BaseMicroserviceConfigs &configs) {
    msvc_name = configs.msvc_name;
    msvc_svcLevelObjLatency = configs.msvc_svcLevelObjLatency;
    msvc_InQueue = {};
    msvc_OutQueue = {};

    std::list<NeighborMicroserviceConfigs>::const_iterator it;
    for (it = configs.dnstreamMicroservices.begin(); it != configs.dnstreamMicroservices.end(); ++it) {
        msvc_OutQueue.emplace_back(new ThreadSafeFixSizedDoubleQueue());
        // Create downstream neigbor config and push that into a list for information later
        // Local microservice supposedly has only 1 downstream but `sender` microservices could have multiple.
        NeighborMicroservice dnStreamMsvc = NeighborMicroservice(*it, numDnstreamMicroservices);
        dnstreamMicroserviceList.emplace_back(dnStreamMsvc);
        // This maps the data class to be sent to this downstream microservice and the microservice's index.
        std::tuple<uint16_t, uint16_t> map = {dnStreamMsvc.classOfInterest, numDnstreamMicroservices++};
        classToDnstreamMap.emplace_back(map);
    }

    for (it = configs.upstreamMicroservices.begin(); it != configs.upstreamMicroservices.end(); ++it) {
        NeighborMicroservice upStreamMsvc = NeighborMicroservice(*it, numUpstreamMicroservices++);
    }
}

/**
 * @brief 
 * 
 * @tparam InType 
 * @param lastInterReqDuration 
 */
void Microservice::updateReqRate(ClockType lastInterReqDuration) {
    msvc_interReqTime = 0.0001;
}
