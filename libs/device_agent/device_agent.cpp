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


    CreateDataSource(0, {{"yolov5_0", CommMethod::sharedMemory, {"localhost:55000"}, 10, -1, {}}}, 1, "path/to/video");
    //CreateYolo5Container(0, {"datasource_0", CommMethod::sharedMemory, {"localhost:55000"}, 10, -2, {}}, {}, 1);

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
            {{name + "::receiver",      MicroserviceType::Receiver,      10, -1, {}},
             {name + "::preprocessor",  MicroserviceType::Preprocessor,  10, -1, {}},
             {name + "::inference",     MicroserviceType::Inference,     10, -1, {}},
             {name + "::postprocessor", MicroserviceType::Postprocessor, 10, -1, {}},
             {name + "::sender",        MicroserviceType::Sender,        10, -1, {}}},
            slo, upstream, downstreams
    );
    finishContainer(name, to_string(j), 49152 + containers.size());
}

void DeviceAgent::CreateDataSource(int id, const std::vector<NeighborMicroserviceConfigs> &downstreams,
                                   const MsvcSLOType &slo, const std::string &video_path) {
    std::string name = "datasource_" + std::to_string(id);
    NeighborMicroserviceConfigs upstream = {"video_source", CommMethod::localCPU, {video_path}, 0, -2, {}};
    json j = createConfigs({
                                   {name + "::data_reader", MicroserviceType::Postprocessor, 10, -1, {}},
                                   {name + "::sender",      MicroserviceType::Sender,        10, -1, {}}},
                           slo, upstream, downstreams
    );
    finishContainer("./Container_Datasource", name, to_string(j), 49152 + containers.size());
}

void DeviceAgent::finishContainer(const std::string &executable, const std::string &name, const std::string &start_string, const int &port) {
    //std::thread container(&DeviceAgent::runDocker, name, start_string, port);
    //container.detach();
    runDocker(executable, name, start_string, port);
    std::string target = absl::StrFormat("%s:%d", "localhost", port);
    containers[name] = {{},
                        InDeviceCommunication::NewStub(grpc::CreateChannel(target, grpc::InsecureChannelCredentials())),
                        new CompletionQueue()};
}

json DeviceAgent::createConfigs(
        const std::vector<std::tuple<std::string, MicroserviceType, QueueLengthType, int16_t, std::vector<RequestShapeType>>> &data,
        const MsvcSLOType &slo, const NeighborMicroserviceConfigs &prev_msvc,
        const std::vector<NeighborMicroserviceConfigs> &next_msvc) {
    int i = 0, j = next_msvc.size() + 1;
    std::vector<BaseMicroserviceConfigs> configs;
    NeighborMicroserviceConfigs upstream = {std::get<0>(data[0]), CommMethod::localGPU, {""}, std::get<2>(data[0]), -2, std::get<4>(data[0])};
    for (auto &msvc: data) {
        if (i == 0) {
            upstream = prev_msvc;
        }
        std::list<NeighborMicroserviceConfigs> downstream;
        if (std::get<1>(msvc) == MicroserviceType::Postprocessor) {
            while (--j > 0) {
                downstream.push_back(
                        {std::get<0>(data[i + j]), CommMethod::localGPU, {""}, std::get<2>(data[i + j]),
                         std::get<3>(data[i + j]), std::get<4>(data[i + j])});
            }
        } else if (std::get<1>(msvc) == MicroserviceType::Sender) {
            downstream.push_back(next_msvc[j++]);
        } else {
            downstream.push_back(
                    {std::get<0>(data[++i]), CommMethod::localGPU, {""}, std::get<2>(msvc), std::get<3>(msvc),
                     std::get<4>(msvc)});
        }
        configs.push_back({std::get<0>(msvc), std::get<1>(msvc), slo, 1, std::get<4>(msvc), {upstream}, downstream});
        //current mvsc becomes upstream for next msvc
        upstream = {std::get<0>(msvc), CommMethod::localGPU, {""}, std::get<2>(msvc), -2, std::get<4>(msvc)};
    }
    return json(configs);
}

void DeviceAgent::HandleRecvRpcs() {
    new CounterUpdateRequestHandler(&service, server_cq.get(), this);
    new ReportStartRequestHandler(&service, server_cq.get(), this);
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
        std::cout << "Received start report from " << request.msvc_name() << std::endl;
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