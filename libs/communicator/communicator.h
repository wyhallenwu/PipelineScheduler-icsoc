#ifndef PIPEPLUSPLUS_COMMUNICATOR_H
#define PIPEPLUSPLUS_COMMUNICATOR_H

#include <cstdint>
#include <utility>
#include <string>
#include <random>
#include "absl/strings/str_format.h"
#include "absl/flags/parse.h"
#include <grpcpp/grpcpp.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/health_check_service_interface.h>
#include <google/protobuf/empty.pb.h>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <cuda_runtime.h>
#include <thread>

#include "microservice.h"
#include "pipelinescheduler.grpc.pb.h"

using grpc::Channel;
using grpc::Status;
using pipelinescheduler::DataTransferService;
using pipelinescheduler::GpuPointerPayload;
using pipelinescheduler::SharedMemPayload;
using pipelinescheduler::SerializedDataPayload;
using EmptyMessage = google::protobuf::Empty;

#endif //PIPEPLUSPLUS_COMMUNICATOR_H