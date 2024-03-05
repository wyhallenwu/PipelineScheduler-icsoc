#ifndef PIPEPLUSPLUS_SENDER_H
#define PIPEPLUSPLUS_SENDER_H

#include "communicator.h"

using grpc::ClientContext;
using grpc::ClientAsyncResponseReader;
using grpc::CompletionQueue;
using boost::interprocess::read_write;
using boost::interprocess::create_only;

class Sender : public Microservice {
public:
    Sender(const BaseMicroserviceConfigs &configs, const std::string &connection);

    virtual void Process() = 0;

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

    static std::string HandleRpcs(std::unique_ptr<ClientAsyncResponseReader<SimpleConfirm>> &rpc, CompletionQueue &cq,
                                  SimpleConfirm &reply, Status &status);

    std::vector<std::unique_ptr<DataTransferService::Stub>> stubs;
    bool multipleStubs;
    std::atomic<bool> run{};
};

class GPUSender : public Sender {
public:
    explicit GPUSender(const BaseMicroserviceConfigs &configs, const std::string &connection);

    void Process() final;

    std::string SendGpuPointer(
            std::vector<RequestData<LocalGPUReqDataType>> &elements,
            const ClockType &timestamp, const std::string &path, const uint32_t &slo);

private:
    static std::string HandleRpcs(std::unique_ptr<ClientAsyncResponseReader<SimpleConfirm>> &rpc, CompletionQueue &cq,
                                  SimpleConfirm &reply, Status &status, void *tag);

    static std::map<void *, std::vector<RequestData<LocalGPUReqDataType>> *> tagToGpuPointer;
};

class LocalCPUSender : public Sender {
public:
    LocalCPUSender(const BaseMicroserviceConfigs &configs, const std::string &connection);

    void Process() final;

    std::string
    SendSharedMemory(const std::vector<RequestData<LocalCPUReqDataType>> &elements, const ClockType &timestamp, const std::string &path,
                     const uint32_t &slo);
};

class RemoteCPUSender : public Sender {
public:
    RemoteCPUSender(const BaseMicroserviceConfigs &configs, const std::string &connection);

    void Process() final;

    std::string SendSerializedData(
            const std::vector<RequestData<LocalCPUReqDataType>> &elements, const ClockType &timestamp, const std::string &path,
            const uint32_t &slo);
};

#endif //PIPEPLUSPLUS_SENDER_H
