#include "container_agent.h"

ABSL_FLAG(std::string, name, "", "base name of container");
ABSL_FLAG(std::optional<std::string>, json, std::nullopt, "configurations for microservices as json");
ABSL_FLAG(std::optional<std::string>, json_path, std::nullopt, "json for configuration inside a file");
ABSL_FLAG(std::optional<std::string>, trt_json, std::nullopt, "optional json for TRTConfiguration");
ABSL_FLAG(std::optional<std::string>, trt_json_path, std::nullopt, "json for TRTConfiguration");
ABSL_FLAG(uint16_t, port, 0, "server port for the service");
ABSL_FLAG(uint16_t, verbose, 2, "verbose level 0:trace, 1:debug, 2:info, 3:warn, 4:error, 5:critical, 6:off");

void msvcconfigs::from_json(const json &j, msvcconfigs::NeighborMicroserviceConfigs &val) {
    j.at("nb_name").get_to(val.name);
    j.at("nb_commMethod").get_to(val.commMethod);
    j.at("nb_link").get_to(val.link);
    j.at("nb_maxQueueSize").get_to(val.maxQueueSize);
    j.at("nb_classOfInterest").get_to(val.classOfInterest);
    j.at("nb_expectedShape").get_to(val.expectedShape);
}

void msvcconfigs::from_json(const json &j, msvcconfigs::BaseMicroserviceConfigs &val) {
    j.at("msvc_name").get_to(val.msvc_name);
    j.at("msvc_type").get_to(val.msvc_type);
    j.at("msvc_svcLevelObjLatency").get_to(val.msvc_svcLevelObjLatency);
    j.at("msvc_idealBatchSize").get_to(val.msvc_idealBatchSize);
    j.at("msvc_dataShape").get_to(val.msvc_dataShape);
    j.at("msvc_upstreamMicroservices").get_to(val.msvc_upstreamMicroservices);
    j.at("msvc_dnstreamMicroservices").get_to(val.msvc_dnstreamMicroservices);
}

std::vector<BaseMicroserviceConfigs> msvcconfigs::LoadFromJson() {
    if (!absl::GetFlag(FLAGS_json).has_value()) {
        spdlog::trace("{0:s} attempts to parse Microservice Configs from command line.", __func__);
        if (absl::GetFlag(FLAGS_json_path).has_value()) {
            std::ifstream file(absl::GetFlag(FLAGS_json_path).value());
            spdlog::trace("{0:s} finished parsing Microservice Configs from command line.", __func__);
            return json::parse(file).get<std::vector<BaseMicroserviceConfigs>>();
        } else {
            spdlog::error("No Configurations found. Please provide configuration either as json or file.");
            exit(1);
        }
    } else {
        spdlog::trace("{0:s} attempts to parse Microservice Configs from file.", __func__);
        if (absl::GetFlag(FLAGS_json_path).has_value()) {
            spdlog::error("No Configurations found. Please provide configuration either as json or file.");
            exit(1);
        } else {
            spdlog::trace("{0:s} finished parsing Microservice Configs from file.", __func__);
            return json::parse(absl::GetFlag(FLAGS_json).value()).get<std::vector<BaseMicroserviceConfigs>>();
        }
    }
}

ContainerAgent::ContainerAgent(const std::string &name, uint16_t own_port) : name(name) {
    std::string server_address = absl::StrFormat("%s:%d", "localhost", own_port);
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    server_cq = builder.AddCompletionQueue();
    server = builder.BuildAndStart();

    stub = InDeviceCommunication::NewStub(grpc::CreateChannel("localhost:2000", grpc::InsecureChannelCredentials()));
    sender_cq = new CompletionQueue();

    run = true;
    std::thread receiver(&ContainerAgent::HandleRecvRpcs, this);
    receiver.detach();
    ReportStart();
}

void ContainerAgent::ReportStart() {
    indevicecommunication::ConnectionConfigs request;
    request.set_msvc_name(name);
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
        request.add_size(msvc->GetOutQueueSize(0));
        spdlog::info("{0:s} Length of queue is {1:d}", msvc->msvc_name, msvc->GetOutQueueSize(0));
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
