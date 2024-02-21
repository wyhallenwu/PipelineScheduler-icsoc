#include "absl/flags/parse.h"
#include <random>
#include <cuda_runtime.h>

#include "microservice.h"
#include "sender.h"

template<typename InType>
class Sender : public Microservice<InType> {
public:
    Sender(const BaseMicroserviceConfigs &configs, const std::string &target_str) : Microservice<InType>(configs) {
        stubs = std::vector<std::unique_ptr<DataTransferService::Stub>>();
        stubs.push_back(
                DataTransferService::NewStub(grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials())));
        multipleStubs = false;
    }

    ~Sender() {}

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
                                  SimpleConfirm &reply, Status &status) {
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

    std::vector<std::unique_ptr<DataTransferService::Stub>> stubs;
    bool multipleStubs;
};

class GPUSender : public Sender<DataRequest<LocalGPUReqDataType>> {
public:
    explicit GPUSender(const BaseMicroserviceConfigs &configs, const std::string &target) : Sender(configs, target) {
        tagToGpuPointer = std::map<void *, std::vector<ImageData> *>();
    }

    std::string SendGpuPointer(
            std::vector<ImageData> &elements,
            const int64_t timestamp, const std::string &path, const uint32_t &slo) {
        CompletionQueue cq;

        GpuPointerPayload request;
        request.set_timestamp(timestamp);
        request.set_path(path);
        request.set_slo(slo);
        for (ImageData el: elements) {
            auto ref = request.add_elements();
            ref->set_data(&el.data, sizeof(el.data));
            ref->set_width(el.dims.first);
            ref->set_height(el.dims.second);
        }
        SimpleConfirm reply;
        ClientContext context;
        Status status;

        auto tag = (void *) (uintptr_t) (rand_int(0, 1000));
        while (tagToGpuPointer.find(tag) != tagToGpuPointer.end()) {
            tag = (void *) (uintptr_t) (rand_int(0, 1000));
        }
        tagToGpuPointer[tag] = &elements;

        if (!multipleStubs) {
            std::unique_ptr<ClientAsyncResponseReader<SimpleConfirm>> rpc(
                    stubs[0]->AsyncGpuPointerTransfer(&context, request, &cq));
            return HandleRpcs(rpc, cq, reply, status, tag);
        }

        std::unique_ptr<ClientAsyncResponseReader<SimpleConfirm>> rpc(
                stubs[rand_int(0, stubs.size() - 1)]->AsyncGpuPointerTransfer(&context, request, &cq));
        return HandleRpcs(rpc, cq, reply, status, tag);
    }

private:
    static std::string HandleRpcs(std::unique_ptr<ClientAsyncResponseReader<SimpleConfirm>> &rpc, CompletionQueue &cq,
                                  SimpleConfirm &reply, Status &status, void *tag) {
        rpc->Finish(&reply, &status, tag);
        void *got_tag;
        bool ok = false;
        GPR_ASSERT(cq.Next(&got_tag, &ok));
        GPR_ASSERT(ok);
        if (status.ok()) {
            if (got_tag == tag) {
                for (ImageData el: *tagToGpuPointer[tag]) {
                    cudaFree(el.data);
                }
                delete tagToGpuPointer[tag];
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

    static std::map<void *, std::vector<ImageData> *> tagToGpuPointer;
};

class LocalCPUSender : public Sender<DataRequest<LocalCPUDataType>> {
public:
    LocalCPUSender(const BaseMicroserviceConfigs &configs, const std::string &target) : Sender(configs, target) {}

    std::string
    SendSharedMemory(const std::vector<MemoryImageData> &elements, const int64_t timestamp, const std::string &path,
                     const uint32_t &slo) {
        CompletionQueue cq;

        SharedMemPayload request;
        request.set_timestamp(timestamp);
        request.set_path(path);
        request.set_slo(slo);
        for (MemoryImageData el: elements) {
            auto ref = request.add_elements();
            ref->set_name(el.name);
            ref->set_width(el.dims.first);
            ref->set_height(el.dims.second);
        }
        SimpleConfirm reply;
        ClientContext context;
        Status status;

        if (!multipleStubs) {
            std::unique_ptr<ClientAsyncResponseReader<SimpleConfirm>> rpc(
                    stubs[0]->AsyncSharedMemTransfer(&context, request, &cq));
            return HandleRpcs(rpc, cq, reply, status);
        }

        std::unique_ptr<ClientAsyncResponseReader<SimpleConfirm>> rpc(
                stubs[rand_int(0, stubs.size() - 1)]->AsyncSharedMemTransfer(&context, request, &cq));
        return HandleRpcs(rpc, cq, reply, status);
    }
};

class RemoteCPUSender : public Sender<DataRequest<LocalCPUDataType>> {
public:
    RemoteCPUSender(const BaseMicroserviceConfigs &configs, const std::string &target) : Sender(configs, target) {}

    std::string SendSerializedData(
            const std::vector<SerialImageData> &elements, const int64_t timestamp, const std::string &path,
            const uint32_t &slo) { // We use unix time encoded to int64
        CompletionQueue cq;

        SerializedDataPayload request;
        request.set_timestamp(timestamp);
        request.set_path(path);
        request.set_slo(slo);
        for (SerialImageData el: elements) {
            auto ref = request.add_elements();
            ref->set_data(el.data);
            ref->set_width(el.dims.first);
            ref->set_height(el.dims.second);
            ref->set_datalen(el.size);
        }
        SimpleConfirm reply;
        ClientContext context;
        Status status;

        if (!multipleStubs) {
            std::unique_ptr<ClientAsyncResponseReader<SimpleConfirm>> rpc(
                    stubs[0]->AsyncSerializedDataTransfer(&context, request, &cq));
            return HandleRpcs(rpc, cq, reply, status);
        }

        std::unique_ptr<ClientAsyncResponseReader<SimpleConfirm>> rpc(
                stubs[rand_int(0, stubs.size() - 1)]->AsyncSerializedDataTransfer(&context, request, &cq));
        return HandleRpcs(rpc, cq, reply, status);
    }
};