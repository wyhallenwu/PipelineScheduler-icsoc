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
    msvc_idealBatchSize = configs.msvc_idealBatchSize;
    msvc_dataShape = configs.msvc_dataShape;
    msvc_name = configs.msvc_name;
    msvc_svcLevelObjLatency = configs.msvc_svcLevelObjLatency;
    msvc_InQueue = {};
    msvc_OutQueue = {};
    msvc_outReqShape = {};
    msvc_type = configs.msvc_type;
    PAUSE_THREADS = true;
    msvc_appLvlConfigs = configs.msvc_appLvlConfigs;
    msvc_deviceIndex = configs.msvc_deviceIndex;
    msvc_RUNMODE = configs.msvc_RUNMODE;

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
        if (it->commMethod == CommMethod::localCPU) {
            msvc_activeOutQueueIndex.emplace_back(1);
        } else if (it->commMethod == CommMethod::localGPU) {
            msvc_activeOutQueueIndex.emplace_back(2);
        }
    }

    for (it = configs.msvc_upstreamMicroservices.begin(); it != configs.msvc_upstreamMicroservices.end(); ++it) {
        NeighborMicroservice upStreamMsvc = NeighborMicroservice(*it, nummsvc_upstreamMicroservices++);
        if (it->commMethod == CommMethod::localCPU) {
            msvc_activeInQueueIndex.emplace_back(1);
        } else if (it->commMethod == CommMethod::localGPU) {
            msvc_activeInQueueIndex.emplace_back(2);
        }
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

RequestData<LocalGPUReqDataType> uploadReqData(
    const RequestData<LocalCPUReqDataType>& cpuData,
    void * cudaPtr,
    cv::cuda::Stream &stream
) {
    RequestData<LocalGPUReqDataType> gpuData;

    gpuData.shape = cpuData.shape;
    if (cudaPtr != NULL) {
        gpuData.data = cv::cuda::GpuMat(
            cpuData.data.rows,
            cpuData.data.cols,
            cpuData.data.type(),
            cudaPtr
        );
    } else {
        gpuData.data = cv::cuda::GpuMat(
            cpuData.data.rows,
            cpuData.data.cols,
            cpuData.data.type()
        );
    }

    gpuData.data.upload(cpuData.data);

    return gpuData;
}


Request<LocalGPUReqDataType> uploadReq(
    
    const Request<LocalCPUReqDataType>& cpuReq,
    std::vector<void *> cudaPtr,
    cv::cuda::Stream &stream
) {
    Request<LocalGPUReqDataType> gpuReq;
    gpuReq.req_origGenTime = cpuReq.req_origGenTime;
    gpuReq.req_e2eSLOLatency = cpuReq.req_e2eSLOLatency;
    gpuReq.req_travelPath = cpuReq.req_travelPath;
    gpuReq.req_batchSize = cpuReq.req_batchSize;

    for (uint16_t i = 0; i < cpuReq.req_data.size(); i++) {
        RequestData<LocalGPUReqDataType> req_data;
        if (cudaPtr.size() > 0) {
            req_data = uploadReqData(cpuReq.req_data[i], cudaPtr[i]);
        } else {
            req_data = uploadReqData(cpuReq.req_data[i]);
        }
        gpuReq.req_data.emplace_back(req_data);
    }
    return gpuReq;
}
