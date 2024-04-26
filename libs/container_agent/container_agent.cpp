#include "container_agent.h"

ABSL_FLAG(std::string, name, "", "base name of container");
ABSL_FLAG(std::optional<std::string>, json, std::nullopt, "configurations for microservices as json");
ABSL_FLAG(std::optional<std::string>, json_path, std::nullopt, "json for configuration inside a file");
ABSL_FLAG(std::optional<std::string>, trt_json, std::nullopt, "optional json for TRTConfiguration");
ABSL_FLAG(std::optional<std::string>, trt_json_path, std::nullopt, "json for TRTConfiguration");
ABSL_FLAG(uint16_t, port, 0, "server port for the service");
ABSL_FLAG(int16_t, device, 0, "Index of GPU device");
ABSL_FLAG(uint16_t, verbose, 2, "verbose level 0:trace, 1:debug, 2:info, 3:warn, 4:error, 5:critical, 6:off");
ABSL_FLAG(std::string, log_dir, "../logs", "Log path for the container");
ABSL_FLAG(std::string, profiling_configs, "", "flag to make the model running in profiling mode.");


contRunArgs loadRunArgs(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);

    std::string name = absl::GetFlag(FLAGS_name);
    int8_t device = absl::GetFlag(FLAGS_device);
    uint16_t logLevel = absl::GetFlag(FLAGS_verbose);
    std::string logPath = absl::GetFlag(FLAGS_log_dir);
    std::string profiling_configs = absl::GetFlag(FLAGS_profiling_configs);

    RUNMODE runmode = profiling_configs.empty() ? RUNMODE::DEPLOYMENT : RUNMODE::PROFILING;

    spdlog::set_pattern("[%C-%m-%d %H:%M:%S.%f] [%l] %v");
    spdlog::set_level(spdlog::level::level_enum(logLevel));

    std::tuple<json, json> configs = msvcconfigs::loadJson();
    json pipeConfigs = std::get<0>(configs);
    json profilingConfigs = std::get<1>(configs);

    for (auto& cfg : pipeConfigs) {
        cfg["msvc_contName"] = name;
        cfg["msvc_deviceIndex"] = device;
        cfg["msvc_containerLogPath"] = logPath + "/" + name;
        cfg["msvc_RUNMODE"] = runmode;
    }

    checkCudaErrorCode(cudaSetDevice(device), __func__);

    return {name, absl::GetFlag(FLAGS_port), device, logPath, runmode, pipeConfigs, profilingConfigs};
};

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

std::tuple<json, json> msvcconfigs::loadJson() {
    json pipeConfigs, profilingConfigs;
    if (!absl::GetFlag(FLAGS_json).has_value()) {
        spdlog::trace("{0:s} attempts to load Json Configs from command line.", __func__);
        if (absl::GetFlag(FLAGS_json_path).has_value()) {
            std::ifstream file(absl::GetFlag(FLAGS_json_path).value());
            pipeConfigs = json::parse(file).at("pipeline");
            try {
                profilingConfigs = json::parse(file).at("profiling");
            } catch (json::parse_error &e) {
                spdlog::trace("{0:s} No profiling configurations found.", __func__);
            }
            spdlog::trace("{0:s} finished loading Json Configs from command line.", __func__);
            return std::make_tuple(pipeConfigs, profilingConfigs);
        } else {
            spdlog::error("No Configurations found. Please provide configuration either as json or file.");
            exit(1);
        }
    } else {
        spdlog::trace("{0:s} attempts to load Json Configs from file.", __func__);
        if (absl::GetFlag(FLAGS_json_path).has_value()) {
            spdlog::error("No Configurations found. Please provide configuration either as json or file.");
            exit(1);
        } else {
            spdlog::trace("{0:s} finished loading Json Configs from file.", __func__);
            std::ifstream file(absl::GetFlag(FLAGS_json_path).value());
            pipeConfigs = json::parse(file).at("pipeline");
            try {
                profilingConfigs = json::parse(file).at("profiling");
            } catch (json::out_of_range &e) {
                spdlog::trace("{0:s} No profiling configurations found.", __func__);
            }
            spdlog::trace("{0:s} finished loading Json Configs from command line.", __func__);
            return std::make_tuple(pipeConfigs, profilingConfigs);
        }
    }
}

ContainerAgent::ContainerAgent(
    const std::string &name,
    uint16_t own_port,
    int8_t devIndex,
    const std::string &logPath,
    RUNMODE runmode,
    const json &profiling_configs
) : ContainerAgent(name, own_port, devIndex, logPath) {

    cont_RUNMODE = runmode;
    if (cont_RUNMODE == RUNMODE::PROFILING) {
        profiling_configs.at("profile_minBatch").get_to(cont_profilingConfigs.minBatch);
        profiling_configs.at("profile_maxBatch").get_to(cont_profilingConfigs.maxBatch);
        profiling_configs.at("profile_stepMode").get_to(cont_profilingConfigs.stepMode);
        profiling_configs.at("profile_step").get_to(cont_profilingConfigs.step);
        profiling_configs.at("profile_templateModelPath").get_to(cont_profilingConfigs.templateModelPath);
    }
    

}

ContainerAgent::ContainerAgent(
    const std::string &name,
    uint16_t own_port,
    int8_t devIndex,
    const std::string &logPath
) : name(name) {
    arrivalRate = 0;

    // Create the logDir for this container
    cont_logDir = logPath + "/" + name;
    std::filesystem::create_directory(
        std::filesystem::path(cont_logDir)
    );

    std::string server_address = absl::StrFormat("%s:%d", "localhost", own_port);
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    server_cq = builder.AddCompletionQueue();
    server = builder.BuildAndStart();

    stub = InDeviceCommunication::NewStub(grpc::CreateChannel("localhost:60003", grpc::InsecureChannelCredentials()));
    sender_cq = new CompletionQueue();

    run = true;
    std::thread receiver(&ContainerAgent::HandleRecvRpcs, this);
    receiver.detach();
}

void ContainerAgent::ReportStart() {
    indevicecommunication::ConnectionConfigs request;
    request.set_msvc_name(name);
    request.set_pid(getpid());
    EmptyMessage reply;
    ClientContext context;
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            stub->AsyncReportMsvcStart(&context, request, sender_cq));
    Status status;
    rpc->Finish(&reply, &status, (void *) 1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(sender_cq->Next(&got_tag, &ok));
    GPR_ASSERT(ok);
}

void ContainerAgent::SendState() {
    State request;
    request.set_name(name);
    request.set_arrival_rate(arrivalRate);
    for (auto msvc: msvcs) {
        request.add_queue_size(msvc->GetOutQueueSize(0));
        spdlog::info("{0:s} Length of queue is {1:d}", msvc->msvc_name, msvc->GetOutQueueSize(0));
    }
    EmptyMessage reply;
    ClientContext context;
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            stub->AsyncSendState(&context, request, sender_cq));
    Status status;
    rpc->Finish(&reply, &status, (void *) 1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(sender_cq->Next(&got_tag, &ok));
    GPR_ASSERT(ok);
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

void ContainerAgent::checkReady() {
    // ReportStart(); // RIGHTNOW
    bool ready = false;
    while (!ready) {
        ready = true;
        
        spdlog::info("{0:s} waiting for all microservices to be ready.", __func__);
        for (auto msvc : msvcs) {
            if (!msvc->checkReady()) {
                ready = false;
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
    START();
}
