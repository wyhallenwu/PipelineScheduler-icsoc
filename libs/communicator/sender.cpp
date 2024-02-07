#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include <grpcpp/grpcpp.h>
#include <random>
#include <boost/interprocess/shared_memory_object.hpp>
#include <cuda_runtime.h>

#include "pipelinescheduler.grpc.pb.h"
#include "microservice.h"

ABSL_FLAG(std::string, target, "localhost:50051", "Server address");

using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientAsyncResponseReader;
using grpc::Status;
using grpc::CompletionQueue;
using boost::interprocess::read_write;
using boost::interprocess::create_only;
using pipelinescheduler::DataTransferService;
using pipelinescheduler::GpuPointerPayload;
using pipelinescheduler::SharedMemPayload;
using pipelinescheduler::SerializedDataPayload;
using pipelinescheduler::SimpleConfirm;

template<typename InType>
class Sender : public Microservice<InType> {
public:
    Sender(const BaseMicroserviceConfigs &configs, const std::string &target_str) : Microservice<InType>(configs) {
        stub_ = DataTransferService::NewStub(grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));
    }
    ~Sender() {}

protected:
    static std::string HandleRpcs(std::unique_ptr<ClientAsyncResponseReader<SimpleConfirm>> &rpc, CompletionQueue &cq,
                                  SimpleConfirm &reply, Status &status) {
        rpc->Finish(&reply, &status, (void*) 1);
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

    std::unique_ptr<DataTransferService::Stub> stub_;
};

class GPUSender : public Sender<DataRequest<LocalGPUDataType>> {
public:
    explicit GPUSender(const BaseMicroserviceConfigs &configs, const std::string &target) : Sender(configs, target) {
        tagToGpuPointer = std::map<void*, void*>();
    }

    std::string SendGpuPointer(
            void* pointer, const std::pair<int32_t, int32_t> &dims,
            const int64_t timestamp, const std::string &path, const uint32_t &slo) {
        CompletionQueue cq;

        GpuPointerPayload request;
        request.set_timestamp(timestamp);
        request.set_width(dims.first);
        request.set_height(dims.second);
        request.set_path(path);
        request.set_slo(slo);
        request.set_pointer(&pointer, sizeof(pointer));
        SimpleConfirm reply;
        ClientContext context;
        Status status;

        auto tag = (void*)(uintptr_t)(rand_tag(0, 1000));
        while (tagToGpuPointer.find(tag) != tagToGpuPointer.end()) {
            tag = (void*)(uintptr_t)(rand_tag(0, 1000));
        }
        std::unique_ptr<ClientAsyncResponseReader<SimpleConfirm>> rpc(
                stub_->AsyncGpuPointerTransfer(&context, request, &cq));
        tagToGpuPointer[tag] = pointer;
        return HandleRpcs(rpc, cq, reply, status, tag);
    }

private:
    static inline std::mt19937& generator() {
        // the generator will only be seeded once (per thread) since it's static
        static thread_local std::mt19937 gen(std::random_device{}());
        return gen;
    }
    template<typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
    static T rand_tag(T min, T max) {
        std::uniform_int_distribution<T> dist(min, max);
        return dist(generator());
    }

    static std::string HandleRpcs(std::unique_ptr<ClientAsyncResponseReader<SimpleConfirm>> &rpc, CompletionQueue &cq,
                                  SimpleConfirm &reply, Status &status, void* tag) {
        rpc->Finish(&reply, &status, tag);
        void *got_tag;
        bool ok = false;
        GPR_ASSERT(cq.Next(&got_tag, &ok));
        GPR_ASSERT(ok);
        if (status.ok()) {
            if (got_tag == tag) {
                cudaFree(tagToGpuPointer[tag]);
                tagToGpuPointer.erase(tag);
            }
            else {
                return "Complete but Wrong Tag Received";
            }
            return "Complete";
        } else {
            std::cout << status.error_code() << ": " << status.error_message() << std::endl;
            return "RPC failed";
        }
    }

    static std::map<void*, void*> tagToGpuPointer;
};

class LocalCPUSender : public Sender<DataRequest<LocalCPUDataType>> {
public:
    LocalCPUSender(const BaseMicroserviceConfigs &configs, const std::string &target) : Sender(configs, target) {}

    std::string
    SendSharedMemory(const std::string &name, const int64_t timestamp, const std::string &path, const uint32_t &slo) {
        CompletionQueue cq;

        SharedMemPayload request;
        request.set_timestamp(timestamp);
        request.set_path(path);
        request.set_slo(slo);
        request.set_name(name);
        SimpleConfirm reply;
        ClientContext context;
        Status status;

        std::unique_ptr<ClientAsyncResponseReader<SimpleConfirm>> rpc(
                stub_->AsyncSharedMemTransfer(&context, request, &cq));
        return HandleRpcs(rpc, cq, reply, status);
    }
};

class RemoteCPUSender : public Sender<DataRequest<LocalCPUDataType>> {
public:
    RemoteCPUSender(const BaseMicroserviceConfigs &configs, const std::string &target) : Sender(configs, target) {}

    std::string SendSerializedData(
            const std::string &data, const uint32_t &data_size, const std::pair<int32_t, int32_t> &dims,
            const int64_t timestamp, const std::string &path,
            const uint32_t &slo) { // We use unix time encoded to int64
        CompletionQueue cq;

        SerializedDataPayload request;
        request.set_timestamp(timestamp);
        request.set_datalen(data_size);
        request.set_width(dims.first);
        request.set_height(dims.second);
        request.set_path(path);
        request.set_slo(slo);
        request.set_data(data);
        SimpleConfirm reply;
        ClientContext context;
        Status status;

        std::unique_ptr<ClientAsyncResponseReader<SimpleConfirm>> rpc(
                stub_->AsyncSerializedDataTransfer(&context, request, &cq));
        return HandleRpcs(rpc, cq, reply, status);
    }
};