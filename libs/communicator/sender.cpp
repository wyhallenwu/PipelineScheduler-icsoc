#include "sender.h"

SenderConfigs Sender::loadConfigsFromJson(const json &jsonConfigs) {
    SenderConfigs configs;
    configs.msvc_name = jsonConfigs["msvc_name"];
    return configs;
}

void Sender::loadConfigs(const json &jsonConfigs, bool isConstructing) {

    if (!isConstructing) { //If this is not called from the constructor, we need to load the configs for Sender's base, Micrsoservice class
        Microservice::loadConfigs(jsonConfigs);
    }

    SenderConfigs configs = loadConfigsFromJson(jsonConfigs);

    stubs = std::vector<std::unique_ptr<DataTransferService::Stub>>();
    stubs.push_back(
            DataTransferService::NewStub(
                    grpc::CreateChannel(dnstreamMicroserviceList.front().link[0], grpc::InsecureChannelCredentials())));
    multipleStubs = false;
    READY = true;
}

Sender::Sender(const json &jsonConfigs) : Microservice(jsonConfigs) {
    loadConfigs(jsonConfigs, true);

}


std::string Sender::HandleRpcs(std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> &rpc, CompletionQueue &cq,
                               EmptyMessage &reply, Status &status) {
    rpc->Finish(&reply, &status, (void *) 1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(cq.Next(&got_tag, &ok));
    GPR_ASSERT(ok);

    if (status.ok()) {
        return "Complete";
    } else {
        std::cout << status.error_code() << ": " << status.error_message() << std::endl;
        return "RPC failed";
    }
}

GPUSender::GPUSender(const json &jsonConfigs) : Sender(jsonConfigs) {
    addToName("sender", "GPU");
    tagToGpuPointer = std::map<void *, std::vector<std::vector<RequestData<LocalGPUReqDataType>>> *>();
    spdlog::trace("{0:s} GPUSender is created.", msvc_name);
}

void GPUSender::Process() {
    msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);
    std::vector<std::vector<RequestData<LocalGPUReqDataType>>> elements;
    std::vector<RequestTimeType> timestamp;
    std::vector<std::string> path;
    std::vector<uint32_t> slo;
    while (READY) {
        if (this->STOP_THREADS) {
            spdlog::info("{0:s} STOPS.", msvc_name);
            break;
        } else if (this->PAUSE_THREADS) {
            if (RELOADING) {
                spdlog::trace("{0:s} is BEING (re)loaded...", msvc_name);
                setDevice();
                RELOADING = false;
                spdlog::info("{0:s} is (RE)LOADED.", msvc_name);
            }
            ///spdlog::info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }
        auto request = msvc_InQueue[0]->pop2();
        // Meaning the the timeout in pop() has been reached and no request was actually popped
        if (strcmp(request.req_travelPath[0].c_str(), "empty") == 0) {
            continue;

        /**
         * @brief ONLY IN PROFILING MODE
         * Check if the profiling is to be stopped, if true, then send a signal to the downstream microservice to stop profiling
         */
        } else if (strcmp(request.req_travelPath[0].c_str(), "STOP_PROFILING") == 0) {
            STOP_THREADS = true;
            msvc_OutQueue[0]->emplace(request);
            continue;
        }
        int size = msvc_InQueue[0]->size();
        elements = {request.req_data};
        /**
         * @brief An outgoing request should contain exactly 3 timestamps:
         * 1. The time when the request was generated at the very beginning of the pipeline, this timestamp is always at the front.
         * 2. The time when the request was putin the out queue of the previous microservice, which is either a postprocessor (regular container) or a data reader (data source).
         * 3. The time this request is sent, which is right about now().
         */
        timestamp = {{request.req_origGenTime[0].front(), request.req_origGenTime[0].back()}};
        timestamp[0].emplace_back(std::chrono::system_clock::now());
        path = {request.req_travelPath[0]};
        slo = {request.req_e2eSLOLatency[0]};
        while (size-- > 0 && elements.size() < 10) {
            request = msvc_InQueue[0]->pop2();
            elements.push_back(request.req_data);
            timestamp = {{request.req_origGenTime[0].front(), request.req_origGenTime[0].back()}};
            timestamp[0].emplace_back(std::chrono::system_clock::now());
            path.push_back(request.req_travelPath[0]);
            slo.push_back(request.req_e2eSLOLatency[0]);
        }

        SendGpuPointer(
                elements,
                timestamp,
                path,
                slo
        );
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    msvc_logFile.close();
}

std::string GPUSender::SendGpuPointer(
        std::vector<std::vector<RequestData<LocalGPUReqDataType>>> &elements,
        std::vector<RequestTimeType> &timestamp, std::vector<std::string> &path, std::vector<uint32_t> &slo) {
    CompletionQueue cq;

    ImageDataPayload request;
    for (unsigned int i = 0; i < elements.size(); i++) {
        cudaIpcMemHandle_t ipcHandle;
        char *serializedData[sizeof(cudaIpcMemHandle_t)];
        cudaError_t cudaStatus = cudaIpcGetMemHandle(&ipcHandle, elements[i][0].data.ptr<uchar>());
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaIpcGetMemHandle failed: " << cudaStatus << std::endl;
            continue;
        }
        memcpy(&serializedData, &ipcHandle, sizeof(cudaIpcMemHandle_t));

        ImageData* ref = request.add_elements();
        ref->set_data(serializedData, sizeof(cudaIpcMemHandle_t));
        ref->set_height(elements[i][0].shape[1]);
        ref->set_width(elements[i][0].shape[2]);
        for (auto ts: timestamp[0]) {
            ref->add_timestamp(std::chrono::duration_cast<TimePrecisionType>(ts.time_since_epoch()).count());
        }
        ref->set_path(path[i]);
        ref->set_slo(slo[i]);
    }

    if (request.elements_size() == 0) {
        return "No elements to send";
    }
    EmptyMessage reply;
    ClientContext context;
    Status status;

    auto tag = (void *) (uintptr_t) (rand_int(0, 1000));
    while (tagToGpuPointer.find(tag) != tagToGpuPointer.end()) {
        tag = (void *) (uintptr_t) (rand_int(0, 1000));
    }
    tagToGpuPointer[tag] = &elements;

    if (!multipleStubs) {
        std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
                stubs[0]->AsyncGpuPointerTransfer(&context, request, &cq));
        return HandleRpcs(rpc, cq, reply, status, tag);
    }

    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            stubs[rand_int(0, stubs.size() - 1)]->AsyncGpuPointerTransfer(&context, request, &cq));
    return HandleRpcs(rpc, cq, reply, status, tag);
}

std::string GPUSender::HandleRpcs(std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> &rpc, CompletionQueue &cq,
                                  EmptyMessage &reply, Status &status, void *tag) {
    rpc->Finish(&reply, &status, tag);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(cq.Next(&got_tag, &ok));
    GPR_ASSERT(ok);
    if (status.ok()) {
        if (got_tag == tag) {
            tagToGpuPointer.erase(tag);
        } else {
            return "Complete but Wrong Tag Received";
        }
        return "Complete";
    } else {
        std::cout << status.error_code() << ": " << status.error_message() << std::endl;
        return "RPC failed";
    }
}

LocalCPUSender::LocalCPUSender(const json &jsonConfigs) : Sender(jsonConfigs) {
    addToName("sender", "LocalCPU");
    spdlog::trace("{0:s} LocalCPUSender is created.", msvc_name);
}

void LocalCPUSender::Process() {
    msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);
    std::vector<std::vector<RequestData<LocalCPUReqDataType>>> elements;
    std::vector<RequestTimeType> timestamp;
    std::vector<std::string> path;
    std::vector<uint32_t> slo;
    while (READY) {
        if (this->STOP_THREADS) {
            spdlog::info("{0:s} STOPS.", msvc_name);
            break;
        } else if (this->PAUSE_THREADS) {
            if (RELOADING) {
                spdlog::trace("{0:s} is BEING (re)loaded...", msvc_name);
                setDevice();
                RELOADING = false;
                spdlog::info("{0:s} is (RE)LOADED.", msvc_name);
            }
            ///spdlog::info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }
        auto request = msvc_InQueue[0]->pop1();
        // Meaning the the timeout in pop() has been reached and no request was actually popped
        if (strcmp(request.req_travelPath[0].c_str(), "empty") == 0) {
            continue;
        /**
         * @brief ONLY IN PROFILING MODE
         * Check if the profiling is to be stopped, if true, then send a signal to the downstream microservice to stop profiling
         */
        } else if (strcmp(request.req_travelPath[0].c_str(), "STOP_PROFILING") == 0) {
            STOP_THREADS = true;
            msvc_OutQueue[0]->emplace(request);
            continue;
        }
        int size = msvc_InQueue[0]->size();
        elements = {request.req_data};
        /**
         * @brief An outgoing request should contain exactly 3 timestamps:
         * 1. The time when the request was generated at the very beginning of the pipeline, this timestamp is always at the front.
         * 2. The time when the request was putin the out queue of the previous microservice, which is either a postprocessor (regular container) or a data reader (data source).
         * 3. The time this request is sent, which is right about now().
         */
        timestamp = {{request.req_origGenTime[0].front(), request.req_origGenTime[0].back()}};
        timestamp[0].emplace_back(std::chrono::system_clock::now());
        path = {request.req_travelPath[0]};
        slo = {request.req_e2eSLOLatency[0]};
        while (size-- > 0  && elements.size() < 10) {
            request = msvc_InQueue[0]->pop1();
            elements.push_back(request.req_data);
            timestamp = {{request.req_origGenTime[0].front(), request.req_origGenTime[0].back()}};
            timestamp[0].emplace_back(std::chrono::system_clock::now());
            path.push_back(request.req_travelPath[0]);
            slo.push_back(request.req_e2eSLOLatency[0]);
        }

        SendSharedMemory(
                elements,
                timestamp,
                path,
                slo
        );
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    msvc_logFile.close();
}

std::string LocalCPUSender::SendSharedMemory(
        std::vector<std::vector<RequestData<LocalCPUReqDataType>>> &elements,
        std::vector<RequestTimeType> &timestamp, std::vector<std::string> &path, std::vector<uint32_t> &slo) {
    CompletionQueue cq;
    ImageDataPayload request;
    char *name;
    for (unsigned int i = 0; i < elements.size(); i++) {
        auto ref = request.add_elements();
        sprintf(name, "shared %d", rand_int(0, 1000));
        boost::interprocess::shared_memory_object shm{create_only, name, read_write};
        shm.truncate(elements[i][0].data.total() * elements[i][0].data.elemSize());
        boost::interprocess::mapped_region region{shm, read_write};
        std::memcpy(region.get_address(), elements[i][0].data.data, elements[i][0].data.total() * elements[i][0].data.elemSize());

        ref->set_data(name);
        ref->set_height(elements[i][0].shape[1]);
        ref->set_width(elements[i][0].shape[2]);
        for (auto ts: timestamp[0]) {
            ref->add_timestamp(std::chrono::duration_cast<TimePrecisionType>(ts.time_since_epoch()).count());
        }
        ref->set_path(path[i]);
        ref->set_slo(slo[i]);
    }
    EmptyMessage reply;
    ClientContext context;
    Status status;

    if (!multipleStubs) {
        std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
                stubs[0]->AsyncSharedMemTransfer(&context, request, &cq));
        return HandleRpcs(rpc, cq, reply, status);
    }

    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            stubs[rand_int(0, stubs.size() - 1)]->AsyncSharedMemTransfer(&context, request, &cq));
    return HandleRpcs(rpc, cq, reply, status);
}

RemoteCPUSender::RemoteCPUSender(const json &jsonConfigs) : Sender(jsonConfigs) {
    addToName("sender", "RemoteCPU");
    spdlog::trace("{0:s} RemoteCPUSender is created.", msvc_name);
}

void RemoteCPUSender::Process() {
    msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);
    std::vector<std::vector<RequestData<LocalCPUReqDataType>>> elements;
    std::vector<RequestTimeType> timestamp;
    std::vector<std::string> path;
    std::vector<uint32_t> slo;
    while (READY) {
        if (this->STOP_THREADS) {
            spdlog::info("{0:s} STOPS.", msvc_name);
            break;
        } else if (this->PAUSE_THREADS) {
            if (RELOADING) {
                spdlog::trace("{0:s} is BEING (re)loaded...", msvc_name);
                setDevice();
                RELOADING = false;
                spdlog::info("{0:s} is (RE)LOADED.", msvc_name);
            }
            ///spdlog::info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }
        auto request = msvc_InQueue[0]->pop1();
        // Meaning the the timeout in pop() has been reached and no request was actually popped
        if (strcmp(request.req_travelPath[0].c_str(), "empty") == 0) {
            continue;
        
        /**
         * @brief ONLY IN PROFILING MODE
         * Check if the profiling is to be stopped, if true, then send a signal to the downstream microservice to stop profiling
         */
        } else if (strcmp(request.req_travelPath[0].c_str(), "STOP_PROFILING") == 0) {
            STOP_THREADS = true;
            msvc_OutQueue[0]->emplace(request);
            continue;
        }
        int size = msvc_InQueue[0]->size();
        elements = {request.req_data};
        /**
         * @brief An outgoing request should contain exactly 3 timestamps:
         * 1. The time when the request was generated at the very beginning of the pipeline, this timestamp is always at the front.
         * 2. The time when the request was putin the out queue of the previous microservice, which is either a postprocessor (regular container) or a data reader (data source).
         * 3. The time this request is sent, which is right about now().
         */
        timestamp = {{request.req_origGenTime[0].front(), request.req_origGenTime[0].back()}};
        timestamp[0].emplace_back(std::chrono::system_clock::now());
        path = {request.req_travelPath[0]};
        slo = {request.req_e2eSLOLatency[0]};
        while (size-- > 0  && elements.size() < msvc_idealBatchSize) {
            request = msvc_InQueue[0]->pop1();
            elements.push_back(request.req_data);
            timestamp = {{request.req_origGenTime[0].front(), request.req_origGenTime[0].back()}};
            timestamp[0].emplace_back(std::chrono::system_clock::now());
            path.push_back(request.req_travelPath[0]);
            slo.push_back(request.req_e2eSLOLatency[0]);
        }

        SendSerializedData(
                elements,
                timestamp,
                path,
                slo
        );
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    msvc_logFile.close();
}

std::string RemoteCPUSender::SendSerializedData(
        std::vector<std::vector<RequestData<LocalCPUReqDataType>>> &elements,
        std::vector<RequestTimeType> &timestamp, std::vector<std::string> &path, std::vector<uint32_t> &slo) { // We use unix time encoded to int64
    CompletionQueue cq;

    ImageDataPayload request;
    for (unsigned int i = 0; i < elements.size(); i++) {
        auto ref = request.add_elements();
        ref->set_data(elements[i][0].data.data, elements[i][0].data.total() * elements[i][0].data.elemSize());

        //Metadata meta;
        ref->set_height(elements[i][0].shape[1]);
        ref->set_width(elements[i][0].shape[2]);
        ref->set_height(elements[i][0].shape[1]);
        ref->set_width(elements[i][0].shape[2]);
        for (auto ts: timestamp[0]) {
            ref->add_timestamp(std::chrono::duration_cast<TimePrecisionType>(ts.time_since_epoch()).count());
        }
        ref->set_path(path[i]);
        ref->set_slo(slo[i]);
        ref->set_datalen(elements[i][0].data.total() * elements[i][0].data.elemSize());
        //request.set_allocated_metadata(&meta);
    }
    EmptyMessage reply;
    ClientContext context;
    Status status;

    if (!multipleStubs) {
        std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
                stubs[0]->AsyncSerializedDataTransfer(&context, request, &cq));
        return HandleRpcs(rpc, cq, reply, status);
    }

    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            stubs[rand_int(0, stubs.size() - 1)]->AsyncSerializedDataTransfer(&context, request, &cq));
    return HandleRpcs(rpc, cq, reply, status);
}