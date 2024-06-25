#include "device_agent.h"

ABSL_FLAG(uint16_t, dev_verbose, 0, "Verbosity level of the Device Agent.");
ABSL_FLAG(uint16_t, dev_loggingMode, 0, "Logging mode of the Device Agent. 0:stdout, 1:file, 2:both");
ABSL_FLAG(std::string, dev_logPath, "../logs", "Path to the log dir for the Device Agent.");

ABSL_FLAG(std::string, device_type, "", "string that identifies the device type");
ABSL_FLAG(std::string, controller_url, "", "string that identifies the controller url without port!");
ABSL_FLAG(uint16_t, dev_port_offset, 0, "port offset for starting the control communication");

const int CONTAINER_BASE_PORT = 50001;
const int CONTROLLER_BASE_PORT = 60001;
const int DEVICE_CONTROL_PORT = 60002;
const int INDEVICE_CONTROL_PORT = 60003;

std::string getHostIP() {
    struct ifaddrs *ifAddrStruct = nullptr;
    struct ifaddrs *ifa = nullptr;
    void *tmpAddrPtr = nullptr;

    getifaddrs(&ifAddrStruct);

    for (ifa = ifAddrStruct; ifa != nullptr; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr) {
            continue;
        }
        // check it is IP4
        if (ifa->ifa_addr->sa_family == AF_INET) {
            tmpAddrPtr = &((struct sockaddr_in *) ifa->ifa_addr)->sin_addr;
            char addressBuffer[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, tmpAddrPtr, addressBuffer, INET_ADDRSTRLEN);
            if (std::strcmp(ifa->ifa_name, "lo") != 0) { // exclude loopback
                freeifaddrs(ifAddrStruct);
                return {addressBuffer};
            }
        }
    }
    if (ifAddrStruct != nullptr) freeifaddrs(ifAddrStruct);
    return "";
}

DeviceAgent::DeviceAgent(const std::string &controller_url, const std::string n, SystemDeviceType type) {
    dev_name = n;
    containers = std::map<std::string, DevContainerHandle>();

    dev_port_offset = absl::GetFlag(FLAGS_dev_port_offset);
    dev_loggingMode = absl::GetFlag(FLAGS_dev_loggingMode);
    dev_verbose = absl::GetFlag(FLAGS_dev_verbose);
    dev_logPath = absl::GetFlag(FLAGS_dev_logPath);

    setupLogger(
        dev_logPath,
        "device_agent",
        dev_loggingMode,
        dev_verbose,
        dev_loggerSinks,
        dev_logger
    );

    dev_containerLib = getContainerLib(abbreviate(SystemDeviceTypeList[type]));

    dev_metricsServerConfigs.from_json(json::parse(std::ifstream("../jsons/metricsserver.json")));
    dev_metricsServerConfigs.user = "device_agent";
    dev_metricsServerConfigs.password = "agent";
    dev_metricsServerConn = connectToMetricsServer(dev_metricsServerConfigs, "Device_agent");

    std::string server_address = absl::StrFormat("%s:%d", "0.0.0.0", INDEVICE_CONTROL_PORT + dev_port_offset);
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder device_builder;
    device_builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    device_builder.RegisterService(&device_service);
    device_cq = device_builder.AddCompletionQueue();
    device_server = device_builder.BuildAndStart();

    server_address = absl::StrFormat("%s:%d", "0.0.0.0", DEVICE_CONTROL_PORT + dev_port_offset);
    ServerBuilder controller_builder;
    controller_builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    controller_builder.RegisterService(&controller_service);
    controller_cq = controller_builder.AddCompletionQueue();
    controller_server = controller_builder.BuildAndStart();
    std::string target_str = absl::StrFormat("%s:%d", controller_url, CONTROLLER_BASE_PORT + dev_port_offset);
    controller_stub = ControlCommunication::NewStub(
            grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));
    controller_sending_cq = new CompletionQueue();

    Ready(dev_name, getHostIP(), type);

    dev_logPath += "/" + dev_experiment_name;
    std::filesystem::create_directories(
            std::filesystem::path(dev_logPath)
    );

    dev_logPath += "/" + dev_system_name;
    std::filesystem::create_directories(
            std::filesystem::path(dev_logPath)
    );

    dev_profiler = new Profiler({});

    running = true;
    threads = std::vector<std::thread>();
    threads.emplace_back(&DeviceAgent::HandleDeviceRecvRpcs, this);
    threads.emplace_back(&DeviceAgent::HandleControlRecvRpcs, this);
    for (auto &thread: threads) {
        thread.detach();
    }
}

void DeviceAgent::collectRuntimeMetrics() {
    std::string sql;
    while (running) {
        /*auto metricsStopwatch = Stopwatch();
        metricsStopwatch.start();
        auto startTime = metricsStopwatch.getStartTime();
        uint64_t scrapeLatencyMillisec = 0;
        uint64_t timeDiff;
        std::vector<Profiler::sysStats> stats = dev_profiler->reportDeviceStats();
        for (int i = 0; i < stats.size(); i++) {
            dev_runtimeMetrics[i].gpuUsage = stats[i].gpuUtilization;
            dev_runtimeMetrics[i].gpuMemUsage = stats[i].gpuMemoryUsage;
        }
        for (auto &container: containers) {
            if (container.second.pid > 0 && timePointCastMillisecond(startTime) >=
                timePointCastMillisecond(dev_metricsServerConfigs.nextHwMetricsScrapeTime) && container.second.pid > 0) {
                Profiler::sysStats stats = dev_profiler->reportAtRuntime(container.second.pid, container.second.pid);
                container.second.hwMetrics = {stats.cpuUsage, stats.memoryUsage, stats.rssMemory, stats.gpuUtilization,
                                  stats.gpuMemoryUsage};
                metricsStopwatch.stop();
                scrapeLatencyMillisec = (uint64_t) std::ceil(metricsStopwatch.elapsed_microseconds() / 1000.f);
                dev_metricsServerConfigs.nextHwMetricsScrapeTime = std::chrono::high_resolution_clock::now() +
                                                                    std::chrono::milliseconds(
                                                                            dev_metricsServerConfigs.hwMetricsScrapeIntervalMillisec -
                                                                            scrapeLatencyMillisec);
                spdlog::get("container_agent")->trace("{0:s} SCRAPE hardware metrics. Latency {1:d}ms.",
                                                      dev_name,
                                                      scrapeLatencyMillisec);
                metricsStopwatch.start();
            }
        }

        startTime = std::chrono::high_resolution_clock::now();


        //TODO: @Tung save the metrics to the database as desired

        metricsStopwatch.stop();
        auto reportLatencyMillisec = (uint64_t) std::ceil(metricsStopwatch.elapsed_microseconds() / 1000.f);
        ClockType nextTime;
        nextTime = std::min(dev_metricsServerConfigs.nextMetricsReportTime,
                                dev_metricsServerConfigs.nextHwMetricsScrapeTime);
        timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(nextTime - std::chrono::high_resolution_clock::now()).count();
        std::chrono::milliseconds sleepPeriod(timeDiff - (reportLatencyMillisec) + 2);
        spdlog::get("container_agent")->trace("{0:s} Container Agent's Metric Reporter sleeps for {1:d} milliseconds.", dev_name, sleepPeriod.count());
        std::this_thread::sleep_for(sleepPeriod);*/
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
}

void DeviceAgent::testNetwork(float min_size, float max_size, int num_loops) {
    spdlog::get("container_agent")->info("Testing network with min size: {}, max size: {}, num loops: {}",
                                         min_size, max_size, num_loops);
    ClockType timestamp;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist = std::normal_distribution<float>((min_size + max_size) / 2, (max_size - min_size) / 6);
    for (int i = 0; i < num_loops; i++) {
        DummyMessage request;
        EmptyMessage reply;
        ClientContext context;
        Status status;
        int size = (int) dist(gen);
        std::vector<char> data(size, 'a');
        timestamp = std::chrono::high_resolution_clock::now();
        request.set_origin_name(dev_name);
        request.set_gen_time(std::chrono::duration_cast<TimePrecisionType>(timestamp.time_since_epoch()).count());
        request.set_data(data.data(), size);
        std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
                controller_stub->AsyncSendDummyData(&context, request, controller_sending_cq));
        finishGrpc(rpc, reply, status, controller_sending_cq);
    }
    spdlog::get("container_agent")->trace("Network test completed");
}

bool DeviceAgent::CreateContainer(
        ModelType model,
        std::string pipe_name,
        BatchSizeType batch_size,
        std::vector<int> input_dims,
        int replica_id,
        int allocation_mode,
        int device,
        const MsvcSLOType &slo,
        const google::protobuf::RepeatedPtrField<Neighbor> &upstreams,
        const google::protobuf::RepeatedPtrField<Neighbor> &downstreams
) {
    std::string modelName = getContainerName(dev_type, model);
    try {
        std::string cont_name = abbreviate(pipe_name + "_" + dev_containerLib[modelName].taskName + "_" + std::to_string(replica_id));
        std::cout << "Creating container: " << cont_name << std::endl;
        std::string executable = dev_containerLib[modelName].runCommand;
        json start_config;
        if (model == ModelType::Sink) {
            start_config["experimentName"] = dev_experiment_name;
            start_config["systemName"] = dev_system_name;
            start_config["pipelineName"] = pipe_name;
            runDocker(executable, cont_name, to_string(start_config), device, 0);
            return true;
        }

        start_config = dev_containerLib[modelName].templateConfig;

        // adjust container configs
        start_config["container"]["cont_experimentName"] = dev_experiment_name;
        start_config["container"]["cont_systemName"] = dev_system_name;
        start_config["container"]["cont_pipeName"] = pipe_name;
        start_config["container"]["cont_hostDevice"] = dev_name;
        start_config["container"]["cont_name"] = cont_name;
        start_config["container"]["cont_allocationMode"] = allocation_mode;

        json base_config = start_config["container"]["cont_pipeline"];

        // adjust pipeline configs
        for (auto &j: base_config) {
            j["msvc_idealBatchSize"] = batch_size;
            j["msvc_svcLevelObjLatency"] = slo;
        }
        if (model == ModelType::DataSource) {
            base_config[0]["msvc_dataShape"] = {input_dims};
        } else if (model == ModelType::Yolov5nDsrc || model == ModelType::RetinafaceDsrc) {
            base_config[0]["msvc_dataShape"] = {input_dims};
            base_config[0]["msvc_type"] = 500;
        } else {
            base_config[1]["msvc_dnstreamMicroservices"][0]["nb_expectedShape"] = {input_dims};
        }


        // adjust receiver upstreams
        base_config[0]["msvc_upstreamMicroservices"][0]["nb_name"] = upstreams.at(0).name();
        base_config[0]["msvc_upstreamMicroservices"][0]["nb_link"] = {upstreams.at(0).ip()};
        if (upstreams.at(0).gpu_connection()) {
            base_config[0]["msvc_dnstreamMicroservices"][0]["nb_commMethod"] = CommMethod::localGPU;
        } else {
            base_config[0]["msvc_dnstreamMicroservices"][0]["nb_commMethod"] = CommMethod::serialized;
        }

        // adjust sender downstreams
        json sender = base_config.back();
        json *postprocessor = &base_config[base_config.size() - 2];
        json post_down = base_config[base_config.size() - 2]["msvc_dnstreamMicroservices"][0];
        base_config[base_config.size() - 2]["msvc_dnstreamMicroservices"] = json::array();
        base_config.erase(base_config.size() - 1);
        int i = 1;
        for (auto &d: downstreams) {
            sender["msvc_name"] = sender["msvc_name"].get<std::string>() + std::to_string(i);
            sender["msvc_dnstreamMicroservices"][0]["nb_name"] = d.name();
            sender["msvc_dnstreamMicroservices"][0]["nb_link"] = {d.ip()};
            post_down["nb_name"] = sender["msvc_name"];
            if (d.gpu_connection()) {
                post_down["nb_commMethod"] = CommMethod::localGPU;
                sender["msvc_dnstreamMicroservices"][0]["nb_commMethod"] = CommMethod::localGPU;
            } else {
                post_down["nb_commMethod"] = CommMethod::localCPU;
                sender["msvc_dnstreamMicroservices"][0]["nb_commMethod"] = CommMethod::serialized;
            }
            post_down["nb_classOfInterest"] = d.class_of_interest();

            postprocessor->at("msvc_dnstreamMicroservices").push_back(post_down);
            base_config.push_back(sender);
        }

        // start container
        start_config["container"]["cont_pipeline"] = base_config;
        unsigned int control_port = CONTAINER_BASE_PORT + dev_port_offset + containers.size();
        runDocker(executable, cont_name, to_string(start_config), device, control_port);
        std::string target = absl::StrFormat("%s:%d", "localhost", control_port);
        containers[cont_name] = {InDeviceCommunication::NewStub(
                grpc::CreateChannel(target, grpc::InsecureChannelCredentials())),
                                 new CompletionQueue(), control_port, 0, {}};
        return true;
    } catch (std::exception &e) {
        spdlog::get("container_agent")->error("Error creating container: {}", e.what());
        return false;
    }
}

void DeviceAgent::StopContainer(const DevContainerHandle &container, bool forced) {
    indevicecommunication::Signal request;
    EmptyMessage reply;
    ClientContext context;
    Status status;
    request.set_forced(forced);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            container.stub->AsyncStopExecution(&context, request, container.cq));
    finishGrpc(rpc, reply, status, container.cq);
}

void DeviceAgent::UpdateContainerSender(const std::string &cont_name, const std::string &dwnstr, const std::string &ip,
                                        const int &port) {
    Connection request;
    EmptyMessage reply;
    ClientContext context;
    Status status;
    request.set_name(dwnstr);
    request.set_ip(ip);
    request.set_port(port);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            containers[cont_name].stub->AsyncUpdateSender(&context, request, containers[cont_name].cq));
    finishGrpc(rpc, reply, status, containers[cont_name].cq);
}

void DeviceAgent::SyncDatasources(const std::string &cont_name, const std::string &dsrc) {
    indevicecommunication::Int32 request;
    EmptyMessage reply;
    ClientContext context;
    Status status;
    request.set_value(containers[dsrc].port);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            containers[cont_name].stub->AsyncSyncDatasources(&context, request, containers[cont_name].cq));
    finishGrpc(rpc, reply, status, containers[cont_name].cq);
}

void DeviceAgent::Ready(const std::string &cont_name, const std::string &ip, SystemDeviceType type) {
    ConnectionConfigs request;
    SystemInfo reply;
    ClientContext context;
    Status status;
    int processing_units;
    request.set_device_name(cont_name);
    request.set_device_type(type);
    request.set_ip_address(ip);
    Profiler *profiler = new Profiler({});
    if (type == SystemDeviceType::Server) {
        processing_units = profiler->getGpuCount();
        request.set_processors(processing_units);
        for (auto &mem: profiler->getGpuMemory(processing_units)) {
            request.add_memory(mem);
        }
    } else {
        struct sysinfo sys_info;
        if (sysinfo(&sys_info) != 0) {
            spdlog::get("container_agent")->error("sysinfo call failed!");
            exit(1);
        }
        processing_units = 1;
        request.set_processors(processing_units);
        request.add_memory(sys_info.totalram * sys_info.mem_unit / 1000000);
    }

    dev_runtimeMetrics = std::vector<SummarizedHardwareMetrics>(processing_units);
    std::unique_ptr<ClientAsyncResponseReader<SystemInfo>> rpc(
            controller_stub->AsyncAdvertiseToController(&context, request, controller_sending_cq));
    finishGrpc(rpc, reply, status, controller_sending_cq);
    if (!status.ok()) {
        spdlog::error("Ready RPC failed with code: {} and message: {}", status.error_code(), status.error_message());
        exit(1);
    }
    dev_system_name = reply.name();
    dev_experiment_name = reply.experiment();
}

void DeviceAgent::HandleDeviceRecvRpcs() {
    new ReportStartRequestHandler(&device_service, device_cq.get(), this);
    void *tag;
    bool ok;
    while (running) {
        if (!device_cq->Next(&tag, &ok)) {
            break;
        }
        GPR_ASSERT(ok);
        static_cast<RequestHandler *>(tag)->Proceed();
    }
}

void DeviceAgent::HandleControlRecvRpcs() {
    new ExecuteNetworkTestRequestHandler(&controller_service, controller_cq.get(), this);
    new StartContainerRequestHandler(&controller_service, controller_cq.get(), this);
    new UpdateDownstreamRequestHandler(&controller_service, controller_cq.get(), this);
    new UpdateBatchsizeRequestHandler(&controller_service, controller_cq.get(), this);
    new StopContainerRequestHandler(&controller_service, controller_cq.get(), this);
    void *tag;
    bool ok;
    while (running) {
        if (!controller_cq->Next(&tag, &ok)) {
            break;
        }
        GPR_ASSERT(ok);
        static_cast<RequestHandler *>(tag)->Proceed();
    }
}

int getContainerProcessPid(std::string container_name_or_id) {
    std::string cmd = "docker inspect --format '{{.State.Pid}}' " + container_name_or_id;
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    try {
        return std::stoi(result);
    } catch (std::exception &e) {
        return 0;
    }
}

void DeviceAgent::ReportStartRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestReportMsvcStart(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new ReportStartRequestHandler(service, cq, device_agent);

        int pid = getContainerProcessPid(device_agent->dev_system_name + "_" + request.msvc_name());
        device_agent->containers[request.msvc_name()].pid = pid;
        device_agent->dev_profiler->addPid(pid);
        spdlog::get("container_agent")->info("Received start report from {} with pid: {}", request.msvc_name(), pid);
        reply.set_pid(pid);
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void DeviceAgent::ExecuteNetworkTestRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestExecuteNetworkTest(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new ExecuteNetworkTestRequestHandler(service, cq, device_agent);
        device_agent->testNetwork((float) request.min(), (float) request.max(), request.repetitions());
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void DeviceAgent::StartContainerRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestStartContainer(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new StartContainerRequestHandler(service, cq, device_agent);
        std::vector<int> input_dims;
        for (auto &dim: request.input_dimensions()) {
            input_dims.push_back(dim);
        }
        bool success = device_agent->CreateContainer(static_cast<ModelType>(request.model()), request.pipeline_name(),
                                                     request.batch_size(), input_dims, request.replica_id(),
                                                     request.allocation_mode(), request.device(),
                                                     request.slo(), request.upstream(), request.downstream());
        if (!success) {
            status = FINISH;
            responder.Finish(reply, Status::CANCELLED, this);
        } else {
            status = FINISH;
            responder.Finish(reply, Status::OK, this);
        }
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void DeviceAgent::StopContainerRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestStopContainer(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new StopContainerRequestHandler(service, cq, device_agent);
        if (device_agent->containers.find(request.name()) == device_agent->containers.end()) {
            status = FINISH;
            responder.Finish(reply, Status::CANCELLED, this);
            return;
        }
        device_agent->StopContainer(device_agent->containers[request.name()], request.forced());
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void DeviceAgent::UpdateDownstreamRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestUpdateDownstream(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new UpdateDownstreamRequestHandler(service, cq, device_agent);
        device_agent->UpdateContainerSender(request.name(), request.downstream_name(), request.ip(), request.port());
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void DeviceAgent::SyncDatasourceRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestSyncDatasource(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new SyncDatasourceRequestHandler(service, cq, device_agent);
        device_agent->SyncDatasources(request.name(), request.downstream_name());
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void DeviceAgent::UpdateBatchsizeRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestUpdateBatchSize(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new UpdateBatchsizeRequestHandler(service, cq, device_agent);
        ClientContext context;
        Status state;
        indevicecommunication::Int32 bs;
        bs.set_value(request.value());
        std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
                device_agent->containers[request.name()].stub->AsyncUpdateBatchSize(&context, bs,
                                                                                    device_agent->containers[request.name()].cq));
        rpc->Finish(&reply, &state, (void *) 1);
        void *got_tag;
        bool ok = false;
        GPR_ASSERT(device_agent->containers[request.name()].cq->Next(&got_tag, &ok));
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}