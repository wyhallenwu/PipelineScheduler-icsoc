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
    for (it = configs.dnstreamMicroservices.begin(); it != configs.dnstreamMicroservices.end(); ++it) {
        msvc_OutQueue.emplace_back(new ThreadSafeFixSizedDoubleQueue());
        // Create downstream neigbor config and push that into a list for information later
        // Local microservice supposedly has only 1 downstream but `sender` microservices could have multiple.
        NeighborMicroservice dnStreamMsvc = NeighborMicroservice(*it, numDnstreamMicroservices);
        dnstreamMicroserviceList.emplace_back(dnStreamMsvc);
        // This maps the data class to be sent to this downstream microservice and the microservice's index.
        std::pair<uint16_t, uint16_t> map = {dnStreamMsvc.classOfInterest, numDnstreamMicroservices++};
        classToDnstreamMap.emplace_back(map);
        msvc_outReqShape.emplace_back(it->expectedShape[0]); // This is a dummy value for now
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

[{"bs":1,"downstrm":[{"coi":-1,"comm":3,"link":[""],"maxqs":10,"name":"datasource_0::sender","shape":[[0,0]]}],"ds":[[0,0]],"name":"datasource_0::data_reader","slo":1,"type":3,"upstrm":[{"coi":-2,"comm":4,"link":["./test.mp4"],"maxqs":0,"name":"video_source","shape":[[0,0]]}]},
 {"bs":1,"downstrm":[{"coi":-1,"comm":0,"link":["172.17.0.1:55000"],"maxqs":10,"name":"yolov5_0","shape":[[0,0]]}],"ds":[[0,0]],"name":"datasource_0::sender","slo":1,"type":4,"upstrm":[{"coi":-2,"comm":4,"link":["./test.mp4"],"maxqs":0,"name":"video_source","shape":[[0,0]]}]}]