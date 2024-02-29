#include "container_agent.h"

ABSL_FLAG(std::string, name, "", "base name of container");
ABSL_FLAG(std::string, json, "", "configurations for microservices");
ABSL_FLAG(uint16_t, port, 0, "Server port for the service");

void msvcconfigs::from_json(const json &j, msvcconfigs::NeighborMicroserviceConfigs &val) {
    j.at("name").get_to(val.name);
    j.at("comm").get_to(val.commMethod);
    j.at("link").get_to(val.link);
    j.at("qt").get_to(val.queueType);
    j.at("maxqs").get_to(val.maxQueueSize);
    j.at("coi").get_to(val.classOfInterest);
    j.at("shape").get_to(val.expectedShape);
}

void msvcconfigs::from_json(const json &j, msvcconfigs::BaseMicroserviceConfigs &val) {
    j.at("name").get_to(val.msvc_name);
    j.at("type").get_to(val.msvc_type);
    j.at("slo").get_to(val.msvc_svcLevelObjLatency);
    j.at("bs").get_to(val.msvc_idealBatchSize);
    j.at("ds").get_to(val.msvc_dataShape);
    j.at("upstrm").get_to(val.upstreamMicroservices);
    j.at("downstrm").get_to(val.dnstreamMicroservices);
}

ContainerAgent::ContainerAgent(const std::string &name, uint16_t device_port, uint16_t own_port) : name(name) {
    std::string server_address = absl::StrFormat("%s:%d", "localhost", own_port);
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    server_cq = builder.AddCompletionQueue();
    server = builder.BuildAndStart();

    std::string target_str = absl::StrFormat("%s:%d", "localhost", device_port);
    stub = InDeviceCommunication::NewStub(grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));
    sender_cq = new CompletionQueue();

    run = true;
    std::thread receiver(&ContainerAgent::HandleRecvRpcs, this);
    receiver.detach();
    ReportStart(own_port);
}

void ContainerAgent::ReportStart(int port) {
    indevicecommunication::ConnectionConfigs request;
    request.set_ip("localhost");
    request.set_port(port);
    StaticConfirm reply;
    ClientContext context;
    std::unique_ptr<ClientAsyncResponseReader<StaticConfirm>> rpc(
            stub->AsyncReportMsvcStart(&context, request, sender_cq));
    Status status;
    rpc->Finish(&reply, &status, (void *) 1);
}

void ContainerAgent::SendQueueLengths() {
    QueueSize request;
    for (auto msvc: msvcs) {
        request.add_size(msvc->GetOutQueueSize());
    }
    StaticConfirm reply;
    ClientContext context;
    std::unique_ptr<ClientAsyncResponseReader<StaticConfirm>> rpc(
            stub->AsyncSendQueueSize(&context, request, sender_cq));
    Status status;
    rpc->Finish(&reply, &status, (void *) 1);
}

void ContainerAgent::HandleRecvRpcs() {
    new StopRequestHandler(&service, server_cq.get(), &run);
    void *tag;
    bool ok;
    while (run) {
        GPR_ASSERT(server_cq->Next(&tag, &ok));
        GPR_ASSERT(ok);
        static_cast<RequestHandler *>(tag)->Proceed();
    }
}

void ContainerAgent::StopRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestStopExecution(&ctx, &request, &responder, cq, cq,
                                      this);
    } else if (status == PROCESS) {
        *run = false;
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

Yolo5ContainerAgent::Yolo5ContainerAgent(const std::string &name, uint16_t device_port,
                                         uint16_t own_port, std::vector<BaseMicroserviceConfigs> &msvc_configs)
        : ContainerAgent(name, device_port, own_port) {
    for (auto &config: msvc_configs) {
        if (config.msvc_type == MicroserviceType::Sender) {
            if (config.dnstreamMicroservices.front().commMethod == CommMethod::sharedMemory)
                msvcs.push_back(
                        reinterpret_cast<Microservice<void> *const>(new LocalCPUSender(config,
                                                                                       config.dnstreamMicroservices.front().link[0])));
            else if (config.dnstreamMicroservices.front().commMethod == CommMethod::gRPC)
                msvcs.push_back(
                        reinterpret_cast<Microservice<void> *const>(new RemoteCPUSender(config,
                                                                                        config.dnstreamMicroservices.front().link[0])));
            else if (config.dnstreamMicroservices.front().commMethod == CommMethod::gRPCLocal) // gRPCLocal = GPU
                msvcs.push_back(
                        reinterpret_cast<Microservice<void> *const>(new GPUSender(config,
                                                                                  config.dnstreamMicroservices.front().link[0])));
        } else if (config.msvc_type == MicroserviceType::Receiver) {
            msvcs.push_back(
                    reinterpret_cast<Microservice<void> *const>(new Receiver(config,
                                                                             config.upstreamMicroservices.front().link[0])));
        } else if (config.msvc_type == MicroserviceType::Preprocessor) {
            msvcs.push_back(
                    reinterpret_cast<Microservice<void> *const>(new YoloV5Preprocessor<LocalGPUReqDataType>(config)));
        } else if (config.msvc_type == MicroserviceType::Inference) {
            msvcs.push_back(
                    reinterpret_cast<Microservice<void> *const>(new YoloV5Inference<LocalGPUReqDataType>(config,
                                                                                                         TRTConfigs())));
        }
    }
}

DataSourceAgent::DataSourceAgent(const std::string &name, uint16_t device_port, uint16_t own_port,
                                 std::vector<BaseMicroserviceConfigs> &msvc_configs)
        : ContainerAgent(name, device_port, own_port) {
    msvcs.push_back(reinterpret_cast<Microservice<void> *const>(new DataReader(msvc_configs[0],
                                                                               msvc_configs[0].upstreamMicroservices.front().link[0])));
    msvcs.push_back(reinterpret_cast<Microservice<void> *const>(new LocalCPUSender(msvc_configs[1],
                                                                                   msvc_configs[1].dnstreamMicroservices.front().link[0])));
}

//int main(int argc, char **argv) {
//    absl::ParseCommandLine(argc, argv);
//    auto msvc_configs = json::parse(absl::GetFlag(FLAGS_json)).get<std::vector<BaseMicroserviceConfigs>>();
//    std::string name = absl::GetFlag(FLAGS_name);
//    ContainerAgent *agent;
//    if (name.find("yolov5") != std::string::npos) {
//        agent = new Yolo5ContainerAgent(name, 2000, absl::GetFlag(FLAGS_port), msvc_configs);
//    } else if (name.find("data_source") != std::string::npos) {
//        agent = new DataSourceAgent(name, 2000, absl::GetFlag(FLAGS_port), msvc_configs);
//    } else if (name.find("base") != std::string::npos) {
//        agent = new ContainerAgent(name, 2000, absl::GetFlag(FLAGS_port));
//    } else {
//        return 1;
//    }
//    while (agent->running()) {
//        std::this_thread::sleep_for(std::chrono::seconds(10));
//        agent->SendQueueLengths();
//    }
//    delete agent;
//    return 0;
//}

