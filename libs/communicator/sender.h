#ifndef PIPEPLUSPLUS_SENDER_H
#define PIPEPLUSPLUS_SENDER_H

#include "communicator.h"

using grpc::ClientContext;
using grpc::ClientAsyncResponseReader;
using grpc::CompletionQueue;
using boost::interprocess::read_write;
using boost::interprocess::create_only;
using json = nlohmann::ordered_json;


struct SenderConfigs : BaseMicroserviceConfigs {
    // Empty for now
    uint8_t dummy;
};

class Sender : public Microservice {
public:
    Sender(const json &jsonConfigs);

    virtual void Process() = 0;

    SenderConfigs loadConfigsFromJson(const json &jsonConfigs);

    virtual void loadConfigs(const json &jsonConfigs, bool isConstructing = false) override;

protected:
    static inline std::mt19937 &generator() {
        // the generator will only be seeded once (per thread) since it's static
        static thread_local std::mt19937 gen(std::random_device{}());
        return gen;
    }

    static int rand_int(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(generator());
    }

    static std::string HandleRpcs(std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> &rpc, CompletionQueue &cq,
                                  EmptyMessage &reply, Status &status);

    void addToName(const std::string substring, const std::string strToAdd) {
        msvc_name.replace(msvc_name.find(substring), substring.length(), strToAdd + substring);
        msvc_microserviceLogPath.replace(msvc_microserviceLogPath.find(substring), substring.length(), strToAdd + substring);
    }

    std::vector<std::unique_ptr<DataTransferService::Stub>> stubs;
    bool multipleStubs;
    std::atomic<bool> run{};
};

class GPUSender : public Sender {
public:
    explicit GPUSender(const json &jsonConfigs);

    void Process() final;

    void dispatchThread() {
        std::thread sender(&GPUSender::Process, this);
        sender.detach();
    }

    std::string SendGpuPointer(
            std::vector<std::vector<RequestData<LocalGPUReqDataType>>> &elements,
            std::vector<RequestTimeType> &timestamp, std::vector<std::string> &path, std::vector<uint32_t> &slo);

private:
    std::string HandleRpcs(std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> &rpc, CompletionQueue &cq,
                           EmptyMessage &reply, Status &status, void *tag);

    void serializeIpcMemHandle(const cudaIpcMemHandle_t& handle, char* buffer) {
        memcpy(buffer, &handle, sizeof(cudaIpcMemHandle_t));
    }

    std::map<void *, std::vector<std::vector<RequestData<LocalGPUReqDataType>>> *> tagToGpuPointer;
};

class LocalCPUSender : public Sender {
public:
    LocalCPUSender(const json &jsonConfigs);

    void Process() final;

    void dispatchThread() final {
        std::thread sender(&LocalCPUSender::Process, this);
        sender.detach();
    }

    std::string
    SendSharedMemory(
            std::vector<std::vector<RequestData<LocalCPUReqDataType>>> &elements,
            std::vector<RequestTimeType> &timestamp, std::vector<std::string> &path, std::vector<uint32_t> &slo);
};

class RemoteCPUSender : public Sender {
public:
    RemoteCPUSender(const json &jsonConfigs);

    void Process() final;

    void dispatchThread() final {
        std::thread sender(&RemoteCPUSender::Process, this);
        sender.detach();
    }

    std::string SendSerializedData(
            std::vector<std::vector<RequestData<LocalCPUReqDataType>>> &elements,
            std::vector<RequestTimeType> &timestamp, std::vector<std::string> &path, std::vector<uint32_t> &slo);
};

#endif //PIPEPLUSPLUS_SENDER_H
