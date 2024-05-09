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
            DataTransferService::NewStub(grpc::CreateChannel(dnstreamMicroserviceList.front().link[0], grpc::InsecureChannelCredentials())));
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
    tagToGpuPointer = std::map<void *, std::vector<RequestData<LocalGPUReqDataType>> *>();
    spdlog::trace("{0:s} GPUSender is created.", msvc_name);
}

void GPUSender::Process() {
    msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);
    while (READY) {
        if (this->STOP_THREADS) {
            spdlog::info("{0:s} STOPS.", msvc_name);
            break;
        }
        else if (this->PAUSE_THREADS) {
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
        }

        SendGpuPointer(
            request.req_data,
            request.req_origGenTime[0],
            request.req_travelPath[0],
            request.req_e2eSLOLatency[0]
        );
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    msvc_logFile.close();
}

std::string GPUSender::SendGpuPointer(
        std::vector<RequestData<LocalGPUReqDataType>> &elements,
        const RequestTimeType &timestamp, const std::string &path, const uint32_t &slo) {
    CompletionQueue cq;

    GpuPointerPayload request;
    for (auto ts: timestamp) {
        request.add_timestamp(std::chrono::duration_cast<std::chrono::nanoseconds>(ts.time_since_epoch()).count());
    }
    request.set_path(path);
    request.set_slo(slo);
    for (RequestData<LocalGPUReqDataType> el: elements) {
        cudaIpcMemHandle_t ipcHandle;
        char * serializedData[sizeof(cudaIpcMemHandle_t)];
        cudaError_t cudaStatus = cudaIpcGetMemHandle(&ipcHandle, el.data.ptr<uchar>());
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaIpcGetMemHandle failed: " << cudaStatus << std::endl;
            continue;
        }
        memcpy(&serializedData, &ipcHandle, sizeof(cudaIpcMemHandle_t));

        auto ref = request.add_elements();
        ref->set_data(serializedData, sizeof(cudaIpcMemHandle_t));
        ref->set_height(el.shape[1]);
        ref->set_width(el.shape[2]);
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
            //for (RequestData<LocalGPUReqDataType> el: *tagToGpuPointer[tag]) {
                //el.data.release();
            //}
            //delete tagToGpuPointer[tag];
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
    while (READY) {
        if (this->STOP_THREADS) {
            spdlog::info("{0:s} STOPS.", msvc_name);
            break;
        }
        else if (this->PAUSE_THREADS) {
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
        }

        SendSharedMemory(request.req_data, request.req_origGenTime[0], request.req_travelPath[0], request.req_e2eSLOLatency[0]);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    msvc_logFile.close();
}

std::string LocalCPUSender::SendSharedMemory(
    const std::vector<RequestData<LocalCPUReqDataType>> &elements,
    const RequestTimeType &timestamp,
    const std::string &path,
    const uint32_t &slo) {
    CompletionQueue cq;
    SharedMemPayload request;
    for (auto ts: timestamp) {
        request.add_timestamp(std::chrono::duration_cast<std::chrono::nanoseconds>(ts.time_since_epoch()).count());
    }
    request.set_path(path);
    request.set_slo(slo);
    char* name;
    for (RequestData<LocalCPUReqDataType> el: elements) {
        auto ref = request.add_elements();
        sprintf(name, "shared %d", rand_int(0, 1000));
        boost::interprocess::shared_memory_object shm{create_only, name, read_write};
        shm.truncate(el.data.total() * el.data.elemSize());
        boost::interprocess::mapped_region region{shm, read_write};
        std::memcpy(region.get_address(), el.data.data, el.data.total() * el.data.elemSize());
        ref->set_name(name);
        ref->set_height(el.shape[1]);
        ref->set_width(el.shape[2]);
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
    while (READY) {
        if (this->STOP_THREADS) {
            spdlog::info("{0:s} STOPS.", msvc_name);
            break;
        }
        else if (this->PAUSE_THREADS) {
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
        }

        SendSerializedData(request.req_data, request.req_origGenTime[0], request.req_travelPath[0], request.req_e2eSLOLatency[0]);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    msvc_logFile.close();
}

std::string RemoteCPUSender::SendSerializedData(
        const std::vector<RequestData<LocalCPUReqDataType>> &elements, 
        const RequestTimeType &timestamp, const std::string &path,
        const uint32_t &slo) { // We use unix time encoded to int64
    CompletionQueue cq;

    SerializedDataPayload request;
    for (auto ts: timestamp) {
        request.add_timestamp(std::chrono::duration_cast<std::chrono::nanoseconds>(ts.time_since_epoch()).count());
    }
    request.set_path(path);
    request.set_slo(slo);
    for (RequestData<LocalCPUReqDataType> el: elements) {
        auto ref = request.add_elements();
        ref->set_data(el.data.data, el.data.total() * el.data.elemSize());
        ref->set_height(el.shape[1]);
        ref->set_width(el.shape[2]);
        ref->set_datalen(el.data.total() * el.data.elemSize());
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