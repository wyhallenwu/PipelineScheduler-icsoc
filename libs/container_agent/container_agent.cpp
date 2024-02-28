#include "container_agent.h"


ContainerAgent::ContainerAgent(const std::string &name, const std::string &url, uint16_t device_port, uint16_t own_port,
                               std::vector<BaseMicroserviceConfigs> &msvc_configs) {
    std::string server_address = absl::StrFormat("%s:%d", url, own_port);
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

    for (auto &config: msvc_configs) {
        if (config.msvc_type == MicroserviceType::Sender) {
            if (config.dnstreamMicroservices.front().commMethod == CommMethod::sharedMemory)
                msvcs.push_back(
                        reinterpret_cast<Microservice<void> *const>(new LocalCPUSender(config, config.dnstreamMicroservices.front().link[0])));
            else if (config.dnstreamMicroservices.front().commMethod == CommMethod::gRPC)
                msvcs.push_back(
                        reinterpret_cast<Microservice<void> *const>(new RemoteCPUSender(config, config.dnstreamMicroservices.front().link[0])));
            else if (config.dnstreamMicroservices.front().commMethod == CommMethod::gRPCLocal) // gRPCLocal = GPU
                msvcs.push_back(
                        reinterpret_cast<Microservice<void> *const>(new GPUSender(config, config.dnstreamMicroservices.front().link[0])));
        } else if (config.msvc_type == MicroserviceType::Receiver) {
            msvcs.push_back(
                    reinterpret_cast<Microservice<void> *const>(new Receiver(config, config.upstreamMicroservices.front().link[0])));
        }
    }

    run = true;
    std::thread receiver(&ContainerAgent::HandleRecvRpcs, this);
    receiver.detach();
    ReportStart(own_port);
}

void ContainerAgent::ReportStart(int port) {
    indevicecommunication::ConnectionConfigs request;
    request.set_ip("localhost");
    request.set_port(port);
    indevicecommunication::SimpleConfirm reply;
    ClientContext context;
    std::unique_ptr<ClientAsyncResponseReader<indevicecommunication::SimpleConfirm>> rpc(
            stub->AsyncReportMsvcStart(&context, request, sender_cq));
    Status status;
    rpc->Finish(&reply, &status, (void *) 1);
}

void ContainerAgent::SendQueueLengths() {
    QueueSize request;
    for (auto msvc: msvcs) {
        request.add_size(msvc->GetOutQueueSize());
    }
    indevicecommunication::SimpleConfirm reply;
    ClientContext context;
    std::unique_ptr<ClientAsyncResponseReader<indevicecommunication::SimpleConfirm>> rpc(
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

int main(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);
    auto msvc_configs = json::parse(absl::GetFlag(FLAGS_json)).get<std::vector<BaseMicroserviceConfigs>>();
    auto agent = new ContainerAgent(absl::GetFlag(FLAGS_name), "localhost", 2000, absl::GetFlag(FLAGS_port),
                                    msvc_configs);

    while (agent->running()) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        agent->SendQueueLengths();
    }
    delete agent;
    return 0;
}

