#include "container_agent.h"

ABSL_FLAG(std::string, name, "", "base name of container");
ABSL_FLAG(std::string, json, "", "configurations for microservices");
ABSL_FLAG(uint16_t, port, 0, "Server port for the service");

void msvcconfigs::from_json(const json &j, msvcconfigs::NeighborMicroserviceConfigs &val) {
    j.at("name").get_to(val.name);
    j.at("comm").get_to(val.commMethod);
    j.at("link").get_to(val.link);
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
