#include "device_agent.h"

DeviceAgent::DeviceAgent(const std::string &controller_url, uint16_t controller_port) {
    std::string server_address = absl::StrFormat("%s:%d", "localhost", 2000);
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    server_cq = builder.AddCompletionQueue();
    server = builder.BuildAndStart();

    std::string target_str = absl::StrFormat("%s:%d", controller_url, controller_port);
    controller_stub = InDeviceCommunication::NewStub(
            grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));
    sender_cq = new CompletionQueue();

    containers = std::map<std::string, ContainerHandle>();

    HandleRecvRpcs();
}

void DeviceAgent::StopContainer(const ContainerHandle &container) {
    StaticConfirm request;
    StaticConfirm reply;
    ClientContext context;
    Status status;
    std::unique_ptr<ClientAsyncResponseReader<StaticConfirm>> rpc(
            container.stub->AsyncStopExecution(&context, request, container.cq));
    rpc->Finish(&reply, &status, (void *) 1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(container.cq->Next(&got_tag, &ok));
    GPR_ASSERT(ok);
    if (!status.ok()) {
        std::cout << "Stop RPC failed" << status.error_code() << ": " << status.error_message() << std::endl;
    }
}

void DeviceAgent::CreateYolo5Container(int id, const NeighborMicroserviceConfigs &upstream,
                                       const std::vector<NeighborMicroserviceConfigs> &downstreams,
                                       const MsvcSLOType &slo) {
    std::string name = "yolov5_" + std::to_string(id);
    json j = createConfigs(
            {{name + "::receiver",      MicroserviceType::Receiver,      QueueType::localGPUDataQueue,10, -1, {}},
             {name + "::preprocessor",  MicroserviceType::Preprocessor,  QueueType::localGPUDataQueue,10, -1, {}},
             {name + "::inference",     MicroserviceType::Inference,     QueueType::localGPUDataQueue,10, -1, {}},
             {name + "::postprocessor", MicroserviceType::Postprocessor, QueueType::none,             10, -1, {}},
             {name + "::sender",        MicroserviceType::Sender,        QueueType::localCPUDataQueue,10, -1, {}}},
            slo, upstream, downstreams
    );
    std::thread container(&DeviceAgent::runDocker, this, name, to_string(j), 49152 + containers.size());
    container.detach();
}

void DeviceAgent::CreateDataSource(int id, const std::vector<NeighborMicroserviceConfigs> &downstreams,
                                       const MsvcSLOType &slo, const std::string &video_path) {
    std::string name = "data_source_" + std::to_string(id);
    json j = createConfigs(
            {{name + "::source",  MicroserviceType::Postprocessor,  QueueType::cpuDataQueue,30, -1, {}},
             {name + "::sender",  MicroserviceType::Sender,         QueueType::cpuDataQueue,30, -1, {}}},
            slo, {"video", CommMethod::localQueue, {video_path}, QueueType::none, 0, -2, }, downstreams
    );
    std::thread container(&DeviceAgent::runDocker, this, name, to_string(j), 49152 + containers.size());
    container.detach();
}

json DeviceAgent::createConfigs(
        const std::vector<std::tuple<std::string, MicroserviceType, QueueType, QueueLengthType, int16_t, std::vector<RequestShapeType>>> &data,
        const MsvcSLOType &slo, const NeighborMicroserviceConfigs &prev_msvc,
        const std::vector<NeighborMicroserviceConfigs> &next_msvc) {
    int i = 0, j = next_msvc.size() + 1;
    std::vector<BaseMicroserviceConfigs> configs;
    NeighborMicroserviceConfigs upstream;
    for (auto &msvc: data) {
        if (i == 0) {
            upstream = prev_msvc;
        }
        std::list<NeighborMicroserviceConfigs> downstream;
        if (std::get<1>(msvc) == MicroserviceType::Postprocessor) {
            while (--j > 0) {
                downstream.push_back(
                        {std::get<0>(data[i + j]), CommMethod::localQueue, {""}, std::get<2>(data[i + j]), std::get<3>(data[i + j]), std::get<4>(data[i + j]),
                         std::get<5>(data[i + j])});
            }
        } else if (std::get<1>(msvc) == MicroserviceType::Sender) {
            downstream.push_back(next_msvc[j++]);
        } else {
            downstream.push_back({std::get<0>(data[++i]), CommMethod::localQueue, {""}, std::get<2>(msvc), std::get<3>(msvc), std::get<4>(msvc),
                                  std::get<5>(msvc)});
        }
        configs.push_back({std::get<0>(msvc), std::get<1>(msvc), slo, 1, std::get<5>(msvc), {upstream}, downstream});
        //current mvsc becomes upstream for next msvc
        upstream = {std::get<0>(msvc), CommMethod::localQueue, {""}, std::get<2>(msvc), std::get<3>(msvc), -2, std::get<5>(msvc)};
    }
    return json(configs);
}

void DeviceAgent::HandleRecvRpcs() {
    while (true) {
        void *tag;
        bool ok;
        if (!server_cq->Next(&tag, &ok)) {
            break;
        }
        static_cast<RequestHandler *>(tag)->Proceed();
    }
}

void DeviceAgent::CounterUpdateRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestSendQueueSize(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new CounterUpdateRequestHandler(service, cq, device_agent);
        device_agent->UpdateQueueLengths(request.name(), request.size());
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void DeviceAgent::ReportStartRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestReportMsvcStart(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new ReportStartRequestHandler(service, cq, device_agent);
        std::string target_str = absl::StrFormat("%s:%d", "localhost", request.port());
        device_agent->containers[request.msvc_name()] = {{},
                                                         InDeviceCommunication::NewStub(grpc::CreateChannel(target_str,
                                                                                                            grpc::InsecureChannelCredentials())),
                                                         new CompletionQueue()};
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

int main() {
    DeviceAgent agent("localhost", 1999);
}