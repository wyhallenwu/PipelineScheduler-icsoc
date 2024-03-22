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
    msvc_dataShape = configs.msvc_dataShape;
    msvc_name = configs.msvc_name;
    msvc_svcLevelObjLatency = configs.msvc_svcLevelObjLatency;
    msvc_InQueue = {};
    msvc_OutQueue = {};
    msvc_outReqShape = {};

    std::list<NeighborMicroserviceConfigs>::const_iterator it;
    for (it = configs.msvc_dnstreamMicroservices.begin(); it != configs.msvc_dnstreamMicroservices.end(); ++it) {
        msvc_OutQueue.emplace_back(new ThreadSafeFixSizedDoubleQueue());
        // Create downstream neigbor config and push that into a list for information later
        // Local microservice supposedly has only 1 downstream but `sender` microservices could have multiple.
        NeighborMicroservice dnStreamMsvc = NeighborMicroservice(*it, nummsvc_dnstreamMicroservices);
        dnstreamMicroserviceList.emplace_back(dnStreamMsvc);
        // This maps the data class to be sent to this downstream microservice and the microservice's index.
        std::pair<int16_t, uint16_t> map = {dnStreamMsvc.classOfInterest, nummsvc_dnstreamMicroservices++};
        classToDnstreamMap.emplace_back(map);
        msvc_outReqShape.emplace_back(it->expectedShape); // This is a dummy value for now
    }

    for (it = configs.msvc_upstreamMicroservices.begin(); it != configs.msvc_upstreamMicroservices.end(); ++it) {
        NeighborMicroservice upStreamMsvc = NeighborMicroservice(*it, nummsvc_upstreamMicroservices++);
    }
}

/**
 * @brief 
 * 
 * @tparam InType 
 * @param lastInterReqDuration 
 */
void Microservice::updateReqRate(ClockType lastInterReqDuration) {
    msvc_interReqTime = 1;
}
