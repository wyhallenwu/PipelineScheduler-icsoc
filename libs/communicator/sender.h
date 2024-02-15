#include <cstdint>
#include <utility>
#include <string>
#include <grpcpp/grpcpp.h>
#include <boost/interprocess/shared_memory_object.hpp>

#include "pipelinescheduler.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientAsyncResponseReader;
using grpc::Status;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using grpc::CompletionQueue;
using boost::interprocess::read_write;
using boost::interprocess::create_only;
using boost::interprocess::read_only;
using boost::interprocess::open_only;
using pipelinescheduler::DataTransferService;
using pipelinescheduler::GpuPointerPayload;
using pipelinescheduler::SharedMemPayload;
using pipelinescheduler::SerializedDataPayload;
using pipelinescheduler::SimpleConfirm;

struct ImageData {
    void *data;
    std::pair<int32_t, int32_t> dims;
};

struct MemoryImageData {
    std::string name;
    std::pair<int32_t, int32_t> dims;
};

struct SerialImageData {
    std::string data;
    std::pair<int32_t, int32_t> dims;
    uint32_t size;
};
