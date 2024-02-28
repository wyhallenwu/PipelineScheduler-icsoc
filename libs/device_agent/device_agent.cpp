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
    indevicecommunication::SimpleConfirm request;
    indevicecommunication::SimpleConfirm reply;
    ClientContext context;
    Status status;
    std::unique_ptr<ClientAsyncResponseReader<indevicecommunication::SimpleConfirm>> rpc(
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
    std::string name = "yolo5_" + std::to_string(id);
    json j = createConfigs(
            {{name + "_receiver",      MicroserviceType::Receiver,      QueueType::localGPUDataQueue, {}},
             {name + "_preprocessor",  MicroserviceType::Preprocessor,  QueueType::localGPUDataQueue, {}},
             {name + "_inference",     MicroserviceType::Inference,     QueueType::localGPUDataQueue, {}},
             {name + "_postprocessor", MicroserviceType::Postprocessor, QueueType::none,              {}},
             {name + "_sender",        MicroserviceType::Sender,        QueueType::localCPUDataQueue, {}}},
            slo, upstream, downstreams
    );
    std::thread container(&DeviceAgent::runDocker, this, name, to_string(j), 5050 + containers.size());
    container.detach();
}

json DeviceAgent::createConfigs(
        const std::vector<std::tuple<std::string, MicroserviceType, QueueType, std::vector<RequestShapeType>>> &data,
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
                        {std::get<0>(data[i + j]), CommMethod::localQueue, {""}, std::get<2>(data[i + j]), 30, -1,
                         std::get<3>(data[i + j])});
            }
        } else if (std::get<1>(msvc) == MicroserviceType::Receiver) {
            downstream.push_back(next_msvc[j++]);
        } else {
            downstream.push_back({std::get<0>(data[++i]), CommMethod::localQueue, {""}, std::get<2>(msvc), 30, -1,
                                  std::get<3>(msvc)});
        }
        configs.push_back({std::get<0>(msvc), std::get<1>(msvc), slo, 1, std::get<3>(msvc), {upstream}, downstream});
        //current mvsc becomes upstream for next msvc
        upstream = {std::get<0>(msvc), CommMethod::localQueue, {""}, std::get<2>(msvc), 30, -2, std::get<3>(msvc)};
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