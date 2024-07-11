#include "controller.h"

ABSL_FLAG(std::string, ctrl_configPath, "../jsons/experiments/base-experiment.json",
          "Path to the configuration file for this experiment.");
ABSL_FLAG(uint16_t, ctrl_verbose, 0, "Verbosity level of the controller.");
ABSL_FLAG(uint16_t, ctrl_loggingMode, 0, "Logging mode of the controller. 0:stdout, 1:file, 2:both");
ABSL_FLAG(std::string, ctrl_logPath, "../logs", "Path to the log dir for the controller.");

const int DATA_BASE_PORT = 55001;
const int CONTROLLER_BASE_PORT = 60001;
const int DEVICE_CONTROL_PORT = 60002;

// ============================================================ Configurations ============================================================ //
// ======================================================================================================================================== //
// ======================================================================================================================================== //
// ======================================================================================================================================== //

void Controller::readInitialObjectCount(const std::string &path)
{
    std::ifstream file(path);
    json j = json::parse(file);
    std::map<std::string, std::map<std::string, std::map<int, float>>> initialPerSecondRate;
    for (auto &item : j.items())
    {
        std::string streamName = item.key();
        initialPerSecondRate[streamName] = {};
        for (auto &object : item.value().items())
        {
            std::string objectName = object.key();
            initialPerSecondRate[streamName][objectName] = {};
            std::vector<int> perFrameObjCount = object.value().get<std::vector<int>>();
            int numFrames = perFrameObjCount.size();
            int totalNumObjs = 0;
            for (auto i = 0; i < numFrames; i++)
            {
                totalNumObjs += perFrameObjCount[i];
                if ((i + 1) % 30 != 0)
                {
                    continue;
                }
                int seconds = (i + 1) / 30;
                initialPerSecondRate[streamName][objectName][seconds] = totalNumObjs * 1.f / seconds;
            }
        }
        float skipRate = ctrl_systemFPS / 30.f;
        std::map<std::string, float> *stream = &(ctrl_initialRequestRates[streamName]);
        float maxPersonRate = 1.2 * std::max_element(initialPerSecondRate[streamName]["person"].begin(), initialPerSecondRate[streamName]["person"].end())->second * skipRate;
        maxPersonRate = std::max(maxPersonRate, ctrl_systemFPS * 1.f);
        float maxCarRate = 1.2 * std::max_element(initialPerSecondRate[streamName]["car"].begin(), initialPerSecondRate[streamName]["car"].end())->second * skipRate;
        maxCarRate = std::max(maxCarRate, ctrl_systemFPS * 1.f);
        if (streamName.find("traffic") != std::string::npos)
        {
            stream->insert({"yolov5n", ctrl_systemFPS});

            stream->insert({"retina1face", std::ceil(maxPersonRate)});
            stream->insert({"arcface", std::ceil(maxPersonRate * 0.6)});
            stream->insert({"carbrand", std::ceil(maxCarRate)});
            stream->insert({"platedet", std::ceil(maxCarRate)});
        }
        else if (streamName.find("people") != std::string::npos)
        {
            stream->insert({"yolov5n", ctrl_systemFPS});
            stream->insert({"retina1face", std::ceil(maxPersonRate)});
            stream->insert({"age", std::ceil(maxPersonRate) * 0.6});
            stream->insert({"gender", std::ceil(maxPersonRate) * 0.6});
            stream->insert({"movenet", std::ceil(maxPersonRate)});
        }
        else if (streamName.find("zoom") != std::string::npos)
        {
            stream->insert({"retinaface", ctrl_systemFPS});
            stream->insert({"arcface", std::ceil(maxPersonRate)});
            stream->insert({"age", std::ceil(maxPersonRate)});
            stream->insert({"gender", std::ceil(maxPersonRate)});
            stream->insert({"emotionnet", std::ceil(maxPersonRate)});
        }
    }
}

void Controller::readConfigFile(const std::string &path)
{
    std::ifstream file(path);
    json j = json::parse(file);

    ctrl_experimentName = j["expName"];
    ctrl_systemName = j["systemName"];
    ctrl_runtime = j["runtime"];
    ctrl_port_offset = j["port_offset"];
    ctrl_systemFPS = j["system_fps"];
    initialTasks = j["initial_pipelines"];
}

void TaskDescription::from_json(const nlohmann::json &j, TaskDescription::TaskStruct &val)
{
    j.at("pipeline_name").get_to(val.name);
    j.at("pipeline_target_slo").get_to(val.slo);
    j.at("pipeline_type").get_to(val.type);
    j.at("video_source").get_to(val.source);
    j.at("pipeline_source_device").get_to(val.device);
    val.fullName = val.name + "_" + val.device;
}

// ============================================================================================================================================ //
// ============================================================================================================================================ //
// ============================================================================================================================================ //

// ============================================================= Con/Desstructors ============================================================= //
// ============================================================================================================================================ //
// ============================================================================================================================================ //
// ============================================================================================================================================ //

Controller::Controller(int argc, char **argv)
{
    absl::ParseCommandLine(argc, argv);
    readConfigFile(absl::GetFlag(FLAGS_ctrl_configPath));
    readInitialObjectCount("../jsons/object_count.json");

    ctrl_logPath = absl::GetFlag(FLAGS_ctrl_logPath);
    ctrl_logPath += "/" + ctrl_experimentName;
    std::filesystem::create_directories(
        std::filesystem::path(ctrl_logPath));
    ctrl_logPath += "/" + ctrl_systemName;
    std::filesystem::create_directories(
        std::filesystem::path(ctrl_logPath));
    ctrl_verbose = absl::GetFlag(FLAGS_ctrl_verbose);
    ctrl_loggingMode = absl::GetFlag(FLAGS_ctrl_loggingMode);

    setupLogger(
        ctrl_logPath,
        "controller",
        ctrl_loggingMode,
        ctrl_verbose,
        ctrl_loggerSinks,
        ctrl_logger);

    ctrl_containerLib = getContainerLib("all");

    json metricsCfgs = json::parse(std::ifstream("../jsons/metricsserver.json"));
    ctrl_metricsServerConfigs.from_json(metricsCfgs);
    ctrl_metricsServerConfigs.schema = abbreviate(ctrl_experimentName + "_" + ctrl_systemName);
    ctrl_metricsServerConfigs.user = "controller";
    ctrl_metricsServerConfigs.password = "agent";
    ctrl_metricsServerConn = connectToMetricsServer(ctrl_metricsServerConfigs, "controller");

    std::string sql = "CREATE SCHEMA IF NOT EXISTS " + ctrl_metricsServerConfigs.schema + ";";
    pushSQL(*ctrl_metricsServerConn, sql);
    sql = "GRANT USAGE ON SCHEMA " + ctrl_metricsServerConfigs.schema + " TO device_agent, container_agent;";
    pushSQL(*ctrl_metricsServerConn, sql);
    sql = "GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA " + ctrl_metricsServerConfigs.schema + " TO device_agent, container_agent;";
    pushSQL(*ctrl_metricsServerConn, sql);
    sql = "GRANT CREATE ON SCHEMA " + ctrl_metricsServerConfigs.schema + " TO device_agent, container_agent;";
    pushSQL(*ctrl_metricsServerConn, sql);
    sql = "ALTER DEFAULT PRIVILEGES IN SCHEMA " + ctrl_metricsServerConfigs.schema + " GRANT SELECT, INSERT ON TABLES TO device_agent, container_agent;";
    pushSQL(*ctrl_metricsServerConn, sql);

    std::thread networkCheckThread(&Controller::checkNetworkConditions, this);
    networkCheckThread.detach();

    running = true;

    std::string server_address = absl::StrFormat("%s:%d", "0.0.0.0", CONTROLLER_BASE_PORT + ctrl_port_offset);
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    cq = builder.AddCompletionQueue();
    server = builder.BuildAndStart();

    ctrl_nextSchedulingTime = std::chrono::system_clock::now();
}

Controller::~Controller()
{
    std::unique_lock<std::mutex> lock(containers.containersMutex);
    for (auto &msvc : containers.list)
    {
        StopContainer(msvc.second, msvc.second->device_agent, true);
    }

    std::unique_lock<std::mutex> lock2(devices.devicesMutex);
    for (auto &device : devices.list)
    {
        device.second->cq->Shutdown();
        void *got_tag;
        bool ok = false;
        while (device.second->cq->Next(&got_tag, &ok))
            ;
    }
    server->Shutdown();
    cq->Shutdown();
}

// ============================================================================================================================================ //
// ============================================================================================================================================ //
// ============================================================================================================================================ //

// ============================================================ Excutor/Maintainers ============================================================ //
// ============================================================================================================================================= //
// ============================================================================================================================================= //
// ============================================================================================================================================= //

bool Controller::AddTask(const TaskDescription::TaskStruct &t)
{
    std::cout << "Adding task: " << t.name << std::endl;
    TaskHandle *task = new TaskHandle{t.name, t.fullName, t.type, t.source, t.device, t.slo, {}, 0};

    std::map<std::string, NodeHandle *> deviceList = devices.getMap();

    if (deviceList.find(t.device) == deviceList.end())
    {
        spdlog::error("Device {0:s} is not connected", t.device);
        return false;
    }

    while (!deviceList.at(t.device)->initialNetworkCheck)
    {
        spdlog::get("container_agent")->info("Waiting for device {0:s} to finish network check", t.device);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    task->tk_src_device = t.device;

    task->tk_pipelineModels = getModelsByPipelineType(t.type, t.device, t.name);
    std::unique_lock<std::mutex> lock2(ctrl_unscheduledPipelines.tasksMutex);
    ctrl_unscheduledPipelines.list.insert({task->tk_name, task});

    lock2.unlock();
    return true;
}

/**
 * @brief call this method after the pipeline models have been added to scheduled
 *
 */
void Controller::ApplyScheduling()
{
    // collect all running containers by device and model name
    std::vector<ContainerHandle *> new_containers;
    std::unique_lock lock_devices(devices.devicesMutex);
    std::unique_lock lock_pipelines(ctrl_scheduledPipelines.tasksMutex);
    std::unique_lock lock_containers(containers.containersMutex);

    for (auto &pipe : ctrl_scheduledPipelines.list)
    {
        for (auto &model : pipe.second->tk_pipelineModels)
        {
            std::unique_lock lock_model(model->pipelineModelMutex);
            std::vector<ContainerHandle *> candidates = model->task->tk_subTasks[model->name];
            // make sure enough containers are running with the right configurations
            if (candidates.size() < model->numReplicas)
            {
                // start additional containers
                for (unsigned int i = candidates.size(); i < model->numReplicas; i++)
                {
                    ContainerHandle *container = TranslateToContainer(model, devices.list[model->device], i);
                    new_containers.push_back(container);
                }
            }
            else if (candidates.size() > model->numReplicas)
            {
                // remove the extra containers
                for (unsigned int i = model->numReplicas; i < candidates.size(); i++)
                {
                    StopContainer(candidates[i], candidates[i]->device_agent);
                    model->task->tk_subTasks[model->name].erase(
                        std::remove(model->task->tk_subTasks[model->name].begin(),
                                    model->task->tk_subTasks[model->name].end(), candidates[i]),
                        model->task->tk_subTasks[model->name].end());
                    candidates.erase(candidates.begin() + i);
                }
            }

            // ensure right configurations of all containers
            int i = 0;
            for (auto *candidate : candidates)
            {
                if (candidate->device_agent->name != model->device)
                {
                    candidate->batch_size = model->batchSize;
                    candidate->cuda_device = model->cudaDevices[i++];
                    MoveContainer(candidate, devices.list[model->device]);
                    continue;
                }
                if (candidate->batch_size != model->batchSize)
                    AdjustBatchSize(candidate, model->batchSize);
                if (candidate->cuda_device != model->cudaDevices[i++])
                    AdjustCudaDevice(candidate, model->cudaDevices[i - 1]);
            }
        }
    }

    for (auto container : new_containers)
    {
        StartContainer(container);
        containers.list.insert({container->name, container});
    }
}

bool CheckMergable(const std::string &m)
{
    return m == "datasource" || m == "yolov5n" || m == "retina1face" || m == "yolov5ndsrc" || m == "retina1facedsrc";
}

ContainerHandle *Controller::TranslateToContainer(PipelineModel *model, NodeHandle *device, unsigned int i)
{
    auto *container = new ContainerHandle{abbreviate(model->task->tk_name + "_" + model->name),
                                          model->upstreams[0].second,
                                          ModelTypeReverseList[model->name],
                                          CheckMergable(model->name),
                                          {0},
                                          model->estimatedStart2HereCost,
                                          0.0,
                                          model->batchSize,
                                          model->cudaDevices[i],
                                          device->next_free_port++,
                                          ctrl_containerLib[model->name].modelPath,
                                          device,
                                          model->task};
    if (model->name == "datasource" || model->name == "yolov5ndsrc" || model->name == "retina1facedsrc")
    {
        container->dimensions = ctrl_containerLib[model->name].templateConfig["container"]["cont_pipeline"][0]["msvc_dataShape"][0].get<std::vector<int>>();
    }
    else if (model->name != "sink")
    {
        container->dimensions = ctrl_containerLib[model->name].templateConfig["container"]["cont_pipeline"][1]["msvc_dnstreamMicroservices"][0]["nb_expectedShape"][0].get<std::vector<int>>();
    }
    model->task->tk_subTasks[model->name].push_back(container);

    for (auto &downstream : model->downstreams)
    {
        for (auto &downstreamContainer : downstream.first->task->tk_subTasks[downstream.first->name])
        {
            if (downstreamContainer->device_agent == device)
            {
                container->downstreams.push_back(downstreamContainer);
                downstreamContainer->upstreams.push_back(container);
            }
        }
    }
    for (auto &upstream : model->upstreams)
    {
        for (auto &upstreamContainer : upstream.first->task->tk_subTasks[upstream.first->name])
        {
            if (upstreamContainer->device_agent == device)
            {
                container->upstreams.push_back(upstreamContainer);
                upstreamContainer->downstreams.push_back(container);
            }
        }
    }
    return container;
}

void Controller::StartContainer(ContainerHandle *container, bool easy_allocation)
{
    std::cout << "Starting container: " << container->name << std::endl;
    ContainerConfig request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_pipeline_name(container->task->tk_name);
    request.set_model(container->model);
    request.set_model_file(container->model_file);
    request.set_batch_size(container->batch_size);
    request.set_allocation_mode(easy_allocation);
    request.set_device(container->cuda_device);
    request.set_slo(container->inference_deadline);
    for (auto dim : container->dimensions)
    {
        request.add_input_dimensions(dim);
    }
    for (auto dwnstr : container->downstreams)
    {
        Neighbor *dwn = request.add_downstream();
        dwn->set_name(dwnstr->name);
        dwn->set_ip(absl::StrFormat("%s:%d", dwnstr->device_agent->ip, dwnstr->recv_port));
        dwn->set_class_of_interest(dwnstr->class_of_interest);
        if (dwnstr->model == Sink)
        {
            dwn->set_gpu_connection(false);
        }
        else
        {
            dwn->set_gpu_connection((container->device_agent == dwnstr->device_agent) &&
                                    (container->cuda_device == dwnstr->cuda_device));
        }
    }
    if (request.downstream_size() == 0)
    {
        Neighbor *dwn = request.add_downstream();
        dwn->set_name("video_sink");
        dwn->set_ip("./out.log"); // output log file
        dwn->set_class_of_interest(-1);
        dwn->set_gpu_connection(false);
    }
    if (container->model == DataSource || container->model == Yolov5nDsrc || container->model == RetinafaceDsrc)
    {
        Neighbor *up = request.add_upstream();
        up->set_name("video_source");
        up->set_ip(container->task->tk_source);
        up->set_class_of_interest(-1);
        up->set_gpu_connection(false);
    }
    else
    {
        for (auto upstr : container->upstreams)
        {
            Neighbor *up = request.add_upstream();
            up->set_name(upstr->name);
            up->set_ip(absl::StrFormat("0.0.0.0:%d", container->recv_port));
            up->set_class_of_interest(-2);
            up->set_gpu_connection((container->device_agent == upstr->device_agent) &&
                                   (container->cuda_device == upstr->cuda_device));
        }
    }
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
        container->device_agent->stub->AsyncStartContainer(&context, request,
                                                           container->device_agent->cq));
    finishGrpc(rpc, reply, status, container->device_agent->cq);
    if (!status.ok())
    {
        std::cout << status.error_code() << ": An error occured while sending the request" << std::endl;
    }
}

void Controller::MoveContainer(ContainerHandle *container, NodeHandle *device)
{
    NodeHandle *old_device = container->device_agent;
    bool start_dsrc = false, merge_dsrc = false;
    if (device->name != "server")
    {
        if (container->mergable)
        {
            merge_dsrc = true;
            if (container->model == Yolov5n)
            {
                container->model = Yolov5nDsrc;
            }
            else if (container->model == Retinaface)
            {
                container->model = RetinafaceDsrc;
            }
        }
    }
    else
    {
        if (container->mergable)
        {
            start_dsrc = true;
            if (container->model == Yolov5nDsrc)
            {
                container->model = Yolov5n;
            }
            else if (container->model == RetinafaceDsrc)
            {
                container->model = Retinaface;
            }
        }
    }
    container->device_agent = device;
    container->recv_port = device->next_free_port++;
    device->containers.insert({container->name, container});
    container->cuda_device = container->cuda_device;
    StartContainer(container, !(start_dsrc || merge_dsrc));
    for (auto upstr : container->upstreams)
    {
        if (start_dsrc)
        {
            StartContainer(upstr, false);
            SyncDatasource(container, upstr);
        }
        else if (merge_dsrc)
        {
            SyncDatasource(upstr, container);
            StopContainer(upstr, old_device);
        }
        else
        {
            AdjustUpstream(container->recv_port, upstr, device, container->name);
        }
    }
    StopContainer(container, old_device);
    old_device->containers.erase(container->name);
}

void Controller::AdjustUpstream(int port, ContainerHandle *upstr, NodeHandle *new_device,
                                const std::string &dwnstr)
{
    ContainerLink request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_name(upstr->name);
    request.set_downstream_name(dwnstr);
    request.set_ip(new_device->ip);
    request.set_port(port);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
        upstr->device_agent->stub->AsyncUpdateDownstream(&context, request, upstr->device_agent->cq));
    finishGrpc(rpc, reply, status, upstr->device_agent->cq);
}

void Controller::SyncDatasource(ContainerHandle *prev, ContainerHandle *curr)
{
    ContainerLink request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_name(prev->name);
    request.set_downstream_name(curr->name);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
        curr->device_agent->stub->AsyncSyncDatasource(&context, request, curr->device_agent->cq));
    finishGrpc(rpc, reply, status, curr->device_agent->cq);
}

void Controller::AdjustBatchSize(ContainerHandle *msvc, int new_bs)
{
    msvc->batch_size = new_bs;
    ContainerInts request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_name(msvc->name);
    request.add_value(new_bs);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
        msvc->device_agent->stub->AsyncUpdateBatchSize(&context, request, msvc->device_agent->cq));
    finishGrpc(rpc, reply, status, msvc->device_agent->cq);
}

void Controller::AdjustCudaDevice(ContainerHandle *msvc, unsigned int new_device)
{
    msvc->cuda_device = new_device;
    // TODO: also adjust actual running container
}

void Controller::AdjustResolution(ContainerHandle *msvc, std::vector<int> new_resolution)
{
    msvc->dimensions = new_resolution;
    ContainerInts request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_name(msvc->name);
    request.add_value(new_resolution[0]);
    request.add_value(new_resolution[1]);
    request.add_value(new_resolution[2]);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
        msvc->device_agent->stub->AsyncUpdateResolution(&context, request, msvc->device_agent->cq));
    finishGrpc(rpc, reply, status, msvc->device_agent->cq);
}

void Controller::StopContainer(ContainerHandle *container, NodeHandle *device, bool forced)
{
    ContainerSignal request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_name(container->name);
    request.set_forced(forced);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
        device->stub->AsyncStopContainer(&context, request, containers.list[container->name]->device_agent->cq));
    finishGrpc(rpc, reply, status, device->cq);
    containers.list.erase(container->name);
    container->device_agent->containers.erase(container->name);
    for (auto upstr : container->upstreams)
    {
        upstr->downstreams.erase(std::remove(upstr->downstreams.begin(), upstr->downstreams.end(), container), upstr->downstreams.end());
    }
    for (auto dwnstr : container->downstreams)
    {
        dwnstr->upstreams.erase(std::remove(dwnstr->upstreams.begin(), dwnstr->upstreams.end(), container), dwnstr->upstreams.end());
    }
}

/**
 * @brief
 *
 * @param node
 */
void Controller::queryInDeviceNetworkEntries(NodeHandle *node)
{
    std::string deviceTypeName = SystemDeviceTypeList[node->type];
    std::string deviceTypeNameAbbr = abbreviate(deviceTypeName);
    if (ctrl_inDeviceNetworkEntries.find(deviceTypeName) == ctrl_inDeviceNetworkEntries.end())
    {
        std::string tableName = "prof_" + deviceTypeNameAbbr + "_netw";
        std::string sql = absl::StrFormat("SELECT p95_transfer_duration_us, p95_total_package_size_b "
                                          "FROM %s ",
                                          tableName);
        pqxx::result res = pullSQL(*ctrl_metricsServerConn, sql);
        if (res.empty())
        {
            spdlog::get("container_agent")->error("No in-device network entries found for device type {}.", deviceTypeName);
            return;
        }
        for (pqxx::result::const_iterator row = res.begin(); row != res.end(); ++row)
        {
            std::pair<uint32_t, uint64_t> entry = {row["p95_total_package_size_b"].as<uint32_t>(), row["p95_transfer_duration_us"].as<uint64_t>()};
            ctrl_inDeviceNetworkEntries[deviceTypeName].emplace_back(entry);
        }
        spdlog::get("container_agent")->info("Finished querying in-device network entries for device type {}.", deviceTypeName);
    }
    std::unique_lock lock(node->nodeHandleMutex);
    node->latestNetworkEntries[deviceTypeName] = aggregateNetworkEntries(ctrl_inDeviceNetworkEntries[deviceTypeName]);
    std::cout << node->latestNetworkEntries[deviceTypeName].size() << std::endl;
}

/**
 * @brief
 *
 * @param container calculating queue sizes for the container before its official deployment.
 * @param modelType
 */
void Controller::calculateQueueSizes(ContainerHandle &container, const ModelType modelType)
{
    float preprocessRate = 1000000.f / container.expectedPreprocessLatency;                // queries per second
    float postprocessRate = 1000000.f / container.expectedPostprocessLatency;              // qps
    float inferRate = 1000000.f / (container.expectedInferLatency * container.batch_size); // batch per second

    QueueLengthType minimumQueueSize = 30;

    // Receiver to Preprocessor
    // Utilization of preprocessor
    float preprocess_rho = container.arrival_rate / preprocessRate;
    QueueLengthType preprocess_inQueueSize = std::max((QueueLengthType)std::ceil(preprocess_rho * preprocess_rho / (2 * (1 - preprocess_rho))), minimumQueueSize);
    float preprocess_thrpt = std::min(preprocessRate, container.arrival_rate);

    // Preprocessor to Inferencer
    // Utilization of inferencer
    float infer_rho = preprocess_thrpt / container.batch_size / inferRate;
    QueueLengthType infer_inQueueSize = std::max((QueueLengthType)std::ceil(infer_rho * infer_rho / (2 * (1 - infer_rho))), minimumQueueSize);
    float infer_thrpt = std::min(inferRate, preprocess_thrpt / container.batch_size); // batch per second

    float postprocess_rho = (infer_thrpt * container.batch_size) / postprocessRate;
    QueueLengthType postprocess_inQueueSize = std::max((QueueLengthType)std::ceil(postprocess_rho * postprocess_rho / (2 * (1 - postprocess_rho))), minimumQueueSize);
    float postprocess_thrpt = std::min(postprocessRate, infer_thrpt * container.batch_size);

    QueueLengthType sender_inQueueSize = postprocess_inQueueSize * container.batch_size;

    container.queueSizes = {preprocess_inQueueSize, infer_inQueueSize, postprocess_inQueueSize, sender_inQueueSize};

    container.expectedThroughput = postprocess_thrpt;
}

// ============================================================================================================================================ //
// ============================================================================================================================================ //
// ============================================================================================================================================ //

// ============================================================ Communication Handlers ============================================================ //
// ================================================================================================================================================ //
// ================================================================================================================================================ //
// ================================================================================================================================================ //

void Controller::HandleRecvRpcs()
{
    new DeviseAdvertisementHandler(&service, cq.get(), this);
    new DummyDataRequestHandler(&service, cq.get(), this);
    void *tag;
    bool ok;
    while (running)
    {
        if (!cq->Next(&tag, &ok))
        {
            break;
        }
        GPR_ASSERT(ok);
        static_cast<RequestHandler *>(tag)->Proceed();
    }
}

void Controller::DeviseAdvertisementHandler::Proceed()
{
    if (status == CREATE)
    {
        status = PROCESS;
        service->RequestAdvertiseToController(&ctx, &request, &responder, cq, cq, this);
    }
    else if (status == PROCESS)
    {
        new DeviseAdvertisementHandler(service, cq, controller);
        std::string target_str = absl::StrFormat("%s:%d", request.ip_address(), DEVICE_CONTROL_PORT + controller->ctrl_port_offset);
        std::string deviceName = request.device_name();
        NodeHandle *node = new NodeHandle{deviceName,
                                          request.ip_address(),
                                          ControlCommunication::NewStub(
                                              grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials())),
                                          new CompletionQueue(),
                                          static_cast<SystemDeviceType>(request.device_type()),
                                          request.processors(),
                                          std::vector<double>(request.processors(), 0.0),
                                          std::vector<unsigned long>(request.memory().begin(), request.memory().end()),
                                          std::vector<double>(request.processors(), 0.0),
                                          DATA_BASE_PORT + controller->ctrl_port_offset,
                                          {}};
        reply.set_name(controller->ctrl_systemName);
        reply.set_experiment(controller->ctrl_experimentName);
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
        controller->devices.addDevice(deviceName, node);
        spdlog::get("container_agent")->info("Device {} is connected to the system", request.device_name());
        controller->queryInDeviceNetworkEntries(controller->devices.list.at(deviceName));

        if (deviceName != "server")
        {
            std::thread networkCheck(&Controller::initNetworkCheck, controller, std::ref(*(controller->devices.list[deviceName])), 1000, 1200000, 30);
            networkCheck.detach();
        }
    }
    else
    {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void Controller::DummyDataRequestHandler::Proceed()
{
    if (status == CREATE)
    {
        status = PROCESS;
        service->RequestSendDummyData(&ctx, &request, &responder, cq, cq, this);
    }
    else if (status == PROCESS)
    {
        new DummyDataRequestHandler(service, cq, controller);
        ClockType now = std::chrono::system_clock::now();
        unsigned long diff = std::chrono::duration_cast<TimePrecisionType>(
                                 now - std::chrono::time_point<std::chrono::system_clock>(TimePrecisionType(request.gen_time())))
                                 .count();
        unsigned int size = request.data().size();
        controller->network_check_buffer[request.origin_name()].push_back({size, diff});
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    }
    else
    {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

std::string DeviceNameToType(std::string name)
{
    if (name == "server")
    {
        return "server";
    }
    else
    {
        return name.substr(0, name.size() - 1);
    }
}

// ============================================================================================================================================ //
// ============================================================================================================================================ //
// ============================================================================================================================================ //

// ============================================================ Network Conditions ============================================================ //

// void Controller::optimizeBatchSizeStep(
//         const Pipeline &models,
//         std::map<ModelType, int> &batch_sizes, std::map<ModelType, int> &estimated_infer_times, int nObjects) {
//     ModelType candidate;
//     int max_saving = 0;
//     std::vector<ModelType> blacklist;
//     for (const auto &m: models) {
//         int saving;
//         if (max_saving == 0) {
//             saving =
//                     estimated_infer_times[m.first] - InferTimeEstimator(m.first, batch_sizes[m.first] * 2);
//         } else {
//             if (batch_sizes[m.first] == 64 ||
//                 std::find(blacklist.begin(), blacklist.end(), m.first) != blacklist.end()) {
//                 continue;
//             }
//             for (const auto &d: m.second) {
//                 if (batch_sizes[d.first] > batch_sizes[m.first]) {
//                     blacklist.push_back(d.first);
//                 }
//             }
//             saving = estimated_infer_times[m.first] -
//                      (InferTimeEstimator(m.first, batch_sizes[m.first] * 2) * (nObjects / batch_sizes[m.first] * 2));
//         }
//         if (saving > max_saving) {
//             max_saving = saving;
//             candidate = m.first;
//         }
//     }
//     batch_sizes[candidate] *= 2;
//     estimated_infer_times[candidate] -= max_saving;
// }

// double Controller::LoadTimeEstimator(const char *model_path, double input_mem_size) {
//     // Load the pre-trained model
//     BoosterHandle booster;
//     int num_iterations = 1;
//     int ret = LGBM_BoosterCreateFromModelfile(model_path, &num_iterations, &booster);

//     // Prepare the input data
//     std::vector<double> input_data = {input_mem_size};

//     // Perform inference
//     int64_t out_len;
//     std::vector<double> out_result(1);
//     ret = LGBM_BoosterPredictForMat(booster,
//                                     input_data.data(),
//                                     C_API_DTYPE_FLOAT64,
//                                     1,  // Number of rows
//                                     1,  // Number of columns
//                                     1,  // Is row major
//                                     C_API_PREDICT_NORMAL,  // Predict type
//                                     0,  // Start iteration
//                                     -1,  // Number of iterations, -1 means use all
//                                     "",  // Parameter
//                                     &out_len,
//                                     out_result.data());
//     if (ret != 0) {
//         std::cout << "Failed to perform inference!" << std::endl;
//         exit(ret);
//     }

//     // Print the predicted value
//     std::cout << "Predicted value: " << out_result[0] << std::endl;

//     // Free the booster handle
//     LGBM_BoosterFree(booster);

//     return out_result[0];
// }

/**
 * @brief
 *
 * @param model to specify model
 * @param batch_size for targeted batch size (binary)
 * @return int for inference time per full batch in nanoseconds
 */
int Controller::InferTimeEstimator(ModelType model, int batch_size)
{
    return 0;
}

// std::map<ModelType, std::vector<int>> Controller::InitialRequestCount(const std::string &input, const Pipeline &models,
//                                                                       int fps) {
//     std::map<ModelType, std::vector<int>> request_counts = {};
//     std::vector<int> fps_values = {fps, fps * 3, fps * 7, fps * 15, fps * 30, fps * 60};

//     request_counts[models[0].first] = fps_values;
//     json objectCount = json::parse(std::ifstream("../jsons/object_count.json"))[input];

//     for (const auto &m: models) {
//         if (m.first == ModelType::Sink) {
//             request_counts[m.first] = std::vector<int>(6, 0);
//             continue;
//         }

//         for (const auto &d: m.second) {
//             if (d.second == -1) {
//                 request_counts[d.first] = request_counts[m.first];
//             } else {
//                 std::vector<int> objects = (d.second == 0 ? objectCount["person"]
//                                                           : objectCount["car"]).get<std::vector<int>>();

//                 for (int j: fps_values) {
//                     int count = std::accumulate(objects.begin(), objects.begin() + j, 0);
//                     request_counts[d.first].push_back(request_counts[m.first][0] * count);
//                 }
//             }
//         }
//     }
//     return request_counts;
// }

/**
 * @brief '
 *
 * @param node
 * @param minPacketSize bytes
 * @param maxPacketSize bytes
 * @param numLoops
 * @return NetworkEntryType
 */
NetworkEntryType Controller::initNetworkCheck(NodeHandle &node, uint32_t minPacketSize, uint32_t maxPacketSize, uint32_t numLoops)
{
    LoopRange request;
    EmptyMessage reply;
    ClientContext context;
    Status status;
    request.set_min(minPacketSize);
    request.set_max(maxPacketSize);
    request.set_repetitions(numLoops);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
        node.stub->AsyncExecuteNetworkTest(&context, request, node.cq));
    finishGrpc(rpc, reply, status, node.cq);

    while (network_check_buffer[node.name].size() < numLoops)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    NetworkEntryType entries = network_check_buffer[node.name];
    entries = aggregateNetworkEntries(entries);
    network_check_buffer[node.name].clear();
    spdlog::get("container_agent")->info("Finished network check for device {}.", node.name);
    std::lock_guard lock(node.nodeHandleMutex);
    node.initialNetworkCheck = true;
    node.latestNetworkEntries["server"] = entries;
    node.lastNetworkCheckTime = std::chrono::system_clock::now();
    return entries;
};

/**
 * @brief Query the latest network entries for each device to determine the network conditions.
 * If no such entries exists, send to each device a request for network testing.
 *
 */
void Controller::checkNetworkConditions()
{
    std::this_thread::sleep_for(TimePrecisionType(5 * 1000000));
    while (running)
    {
        Stopwatch stopwatch;
        stopwatch.start();
        std::map<std::string, NetworkEntryType> networkEntries = {};

        for (auto [deviceName, nodeHandle] : devices.getMap())
        {
            std::unique_lock<std::mutex> lock(nodeHandle->nodeHandleMutex);
            bool initialNetworkCheck = nodeHandle->initialNetworkCheck;
            uint64_t timeSinceLastCheck = std::chrono::duration_cast<TimePrecisionType>(
                                              std::chrono::system_clock::now() - nodeHandle->lastNetworkCheckTime)
                                              .count() /
                                          1000000;
            lock.unlock();
            if (deviceName == "server" || (initialNetworkCheck && timeSinceLastCheck < 60))
            {
                spdlog::get("container_agent")->info("Skipping network check for device {}.", deviceName);
                continue;
            }
            initNetworkCheck(*nodeHandle, 1000, 1200000, 30);
        }
        // std::string tableName = ctrl_metricsServerConfigs.schema + "." + abbreviate(ctrl_experimentName) + "_serv_netw";
        // std::string query = absl::StrFormat("SELECT sender_host, p95_transfer_duration_us, p95_total_package_size_b "
        //                     "FROM %s ", tableName);

        // pqxx::result res = pullSQL(*ctrl_metricsServerConn, query);
        // //Getting the latest network entries into the networkEntries map
        // for (pqxx::result::const_iterator row = res.begin(); row != res.end(); ++row) {
        //     std::string sender_host = row["sender_host"].as<std::string>();
        //     if (sender_host == "server" || sender_host == "serv") {
        //         continue;
        //     }
        //     std::pair<uint32_t, uint64_t> entry = {row["p95_total_package_size_b"].as<uint32_t>(), row["p95_transfer_duration_us"].as<uint64_t>()};
        //     networkEntries[sender_host].emplace_back(entry);
        // }

        // // Updating NodeHandle object with the latest network entries
        // for (auto &[deviceName, entries] : networkEntries) {
        //     // If entry belongs to a device that is not in the list of devices, ignore it
        //     if (devices.list.find(deviceName) == devices.list.end() || deviceName != "server") {
        //         continue;
        //     }
        //     std::lock_guard<std::mutex> lock(devices.list[deviceName].nodeHandleMutex);
        //     devices.list[deviceName].latestNetworkEntries["server"] = aggregateNetworkEntries(entries);
        // }

        // // If no network entries exist for a device, send a request to the device to perform network testing
        // for (auto &[deviceName, nodeHandle] : devices.list) {
        //     if (nodeHandle.latestNetworkEntries.size() == 0) {
        //         // TODO: Send a request to the device to perform network testing

        //     }
        // }

        stopwatch.stop();
        uint64_t sleepTimeUs = 60 * 1000000 - stopwatch.elapsed_microseconds();
        std::this_thread::sleep_for(TimePrecisionType(sleepTimeUs));
    }
}

// ============================================================================================================================================ //
// ============================================================================================================================================ //
// ============================================================================================================================================ //

PipelineModelListType Controller::getModelsByPipelineType(PipelineType type, const std::string &startDevice, const std::string &pipelineName)
{
    switch (type)
    {
    case PipelineType::Traffic:
    {
        auto *datasource = new PipelineModel{startDevice, "datasource", {}, true, {}, {}};
        datasource->possibleDevices = {startDevice};

        auto *yolov5n = new PipelineModel{
            "server",
            "yolov5n",
            {},
            true,
            {},
            {},
            {},
            {{datasource, -1}}};
        yolov5n->possibleDevices = {startDevice, "server"};
        datasource->downstreams.push_back({yolov5n, -1});

        auto *retina1face = new PipelineModel{
            "server",
            "retina1face",
            {},
            false,
            {},
            {},
            {},
            {{yolov5n, 0}}};
        retina1face->possibleDevices = {startDevice, "server"};
        yolov5n->downstreams.push_back({retina1face, 0});

        auto *arcface = new PipelineModel{
            "server",
            "arcface",
            {},
            false,
            {},
            {},
            {},
            {{retina1face, -1}}};
        arcface->possibleDevices = {"server"};
        retina1face->downstreams.push_back({arcface, -1});

        auto *carbrand = new PipelineModel{
            "server",
            "carbrand",
            {},
            false,
            {},
            {},
            {},
            {{yolov5n, 2}}};
        carbrand->possibleDevices = {"server"};
        yolov5n->downstreams.push_back({carbrand, 2});

        auto *platedet = new PipelineModel{
            "server",
            "platedet",
            {},
            false,
            {},
            {},
            {},
            {{yolov5n, 2}}};
        platedet->possibleDevices = {"server"};
        yolov5n->downstreams.push_back({platedet, 2});

        auto *sink = new PipelineModel{
            "server",
            "sink",
            {},
            false,
            {},
            {},
            {},
            {{retina1face, -1}, {carbrand, -1}, {platedet, -1}}};
        sink->possibleDevices = {"server"};
        retina1face->downstreams.push_back({sink, -1});
        carbrand->downstreams.push_back({sink, -1});
        platedet->downstreams.push_back({sink, -1});

        if (!pipelineName.empty())
        {
            yolov5n->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[pipelineName][yolov5n->name];
            retina1face->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[pipelineName][retina1face->name];
            arcface->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[pipelineName][arcface->name];
            carbrand->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[pipelineName][carbrand->name];
            platedet->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[pipelineName][platedet->name];
        }

        return {datasource, yolov5n, retina1face, arcface, carbrand, platedet, sink};
    }
    case PipelineType::Building_Security:
    {
        auto *datasource = new PipelineModel{startDevice, "datasource", {}, true, {}, {}};
        datasource->possibleDevices = {startDevice};
        auto *yolov5n = new PipelineModel{
            "server",
            "yolov5n",
            {},
            true,
            {},
            {},
            {},
            {{datasource, -1}}};
        yolov5n->possibleDevices = {startDevice, "server"};
        datasource->downstreams.push_back({yolov5n, -1});

        auto *retina1face = new PipelineModel{
            "server",
            "retina1face",
            {},
            false,
            {},
            {},
            {},
            {{yolov5n, 0}}};
        retina1face->possibleDevices = {startDevice, "server"};
        yolov5n->downstreams.push_back({retina1face, 0});

        auto *movenet = new PipelineModel{
            "server",
            "movenet",
            {},
            false,
            {},
            {},
            {},
            {{yolov5n, 0}}};
        movenet->possibleDevices = {"server"};
        yolov5n->downstreams.push_back({movenet, 0});

        auto *gender = new PipelineModel{
            "server",
            "gender",
            {},
            false,
            {},
            {},
            {},
            {{retina1face, -1}}};
        gender->possibleDevices = {"server"};
        retina1face->downstreams.push_back({gender, -1});

        auto *age = new PipelineModel{
            "server",
            "age",
            {},
            false,
            {},
            {},
            {},
            {{retina1face, -1}}};
        age->possibleDevices = {"server"};
        retina1face->downstreams.push_back({age, -1});

        auto *sink = new PipelineModel{
            "server",
            "sink",
            {},
            false,
            {},
            {},
            {},
            {{gender, -1}, {age, -1}, {movenet, -1}}};
        sink->possibleDevices = {"server"};
        gender->downstreams.push_back({sink, -1});
        age->downstreams.push_back({sink, -1});
        movenet->downstreams.push_back({sink, -1});

        if (!pipelineName.empty())
        {
            yolov5n->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[pipelineName][yolov5n->name];
            retina1face->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[pipelineName][retina1face->name];
            movenet->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[pipelineName][movenet->name];
            gender->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[pipelineName][movenet->name];
            age->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[pipelineName][age->name];
        }

        return {datasource, yolov5n, retina1face, movenet, gender, age, sink};
    }
    case PipelineType::Video_Call:
    {
        auto *datasource = new PipelineModel{startDevice, "datasource", {}, true, {}, {}};
        datasource->possibleDevices = {startDevice};
        auto *retina1face = new PipelineModel{
            "server",
            "retina1face",
            {},
            true,
            {},
            {},
            {},
            {{datasource, -1}}};
        retina1face->possibleDevices = {startDevice, "server"};
        datasource->downstreams.push_back({retina1face, -1});

        auto *emotionnet = new PipelineModel{
            "server",
            "emotionnet",
            {},
            false,
            {},
            {},
            {},
            {{retina1face, -1}}};
        emotionnet->possibleDevices = {"server"};
        retina1face->downstreams.push_back({emotionnet, -1});

        auto *age = new PipelineModel{
            "server",
            "age",
            {},
            false,
            {},
            {},
            {},
            {{retina1face, -1}}};
        age->possibleDevices = {startDevice, "server"};
        retina1face->downstreams.push_back({age, -1});

        auto *gender = new PipelineModel{
            "server",
            "gender",
            {},
            false,
            {},
            {},
            {},
            {{retina1face, -1}}};
        gender->possibleDevices = {startDevice, "server"};
        retina1face->downstreams.push_back({gender, -1});

        auto *arcface = new PipelineModel{
            "server",
            "arcface",
            {},
            false,
            {},
            {},
            {},
            {{retina1face, -1}}};
        arcface->possibleDevices = {"server"};
        retina1face->downstreams.push_back({arcface, -1});

        auto *sink = new PipelineModel{
            "server",
            "sink",
            {},
            false,
            {},
            {},
            {},
            {{emotionnet, -1}, {age, -1}, {gender, -1}, {arcface, -1}}};
        sink->possibleDevices = {"server"};
        emotionnet->downstreams.push_back({sink, -1});
        age->downstreams.push_back({sink, -1});
        gender->downstreams.push_back({sink, -1});
        arcface->downstreams.push_back({sink, -1});

        if (!pipelineName.empty())
        {
            retina1face->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[pipelineName][retina1face->name];
            emotionnet->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[pipelineName][emotionnet->name];
            age->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[pipelineName][age->name];
            gender->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[pipelineName][gender->name];
            arcface->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[pipelineName][arcface->name];
        }

        return {datasource, retina1face, emotionnet, age, gender, arcface, sink};
    }
    default:
        return {};
    }
}

PipelineModelListType deepCopyPipelineModelList(const PipelineModelListType &original)
{
    PipelineModelListType newList;
    newList.reserve(original.size());
    for (const auto *model : original)
    {
        newList.push_back(new PipelineModel(*model));
    }
    return newList;
}

// ----------------------------------------------------------------------------------------------------------------
//                                            jellyfish implementations
// ----------------------------------------------------------------------------------------------------------------

bool ModelSetCompare::operator()(
    const std::tuple<std::string, float> &lhs,
    const std::tuple<std::string, float> &rhs) const
{
    return std::get<1>(lhs) < std::get<1>(rhs);
}

/**
 * @brief add profiled information of model
 *
 * @param model_type
 * @param accuracy
 * @param batch_size
 * @param inference_latency
 * @param throughput
 */
void ModelProfilesJF::add(std::string name, float accuracy, int batch_size, float inference_latency, int width, int height,
                          PipelineModel *m)
{
    auto key = std::tuple<std::string, float>{name, accuracy};
    m->batchSize = batch_size;
    m->expectedMaxProcessLatency = inference_latency;
    m->throughput = (int(batch_size / (inference_latency * 1e-6)) / 10) * 10;
    m->width = width;
    m->height = height;
    m->name = name;
    m->accuracy = accuracy;
    infos[key].push_back(m);
}

void ModelProfilesJF::add(PipelineModel *model_info)
{
    auto key =
        std::tuple<std::string, float>{model_info->name, model_info->accuracy};
    infos[key].push_back(model_info);
}

void ModelProfilesJF::debugging()
{
    std::cout << "======================ModelProfiles Debugging=======================" << std::endl;
    for (auto it = infos.begin(); it != infos.end(); ++it)
    {
        auto key = it->first;
        auto profilings = it->second;
        std::cout << "*********************************************" << std::endl;
        std::cout << "Model: " << std::get<0>(key) << ", Accuracy: " << std::get<1>(key) << std::endl;
        for (const auto &model_info : profilings)
        {
            std::cout << "batch size: " << model_info->batchSize << ", latency: " << model_info->expectedMaxProcessLatency
                      << ", width: " << model_info->width << ", height: " << model_info->height << std::endl;
        }
        std::cout << "*********************************************" << std::endl;
    }
}

// -------------------------------------------------------------------------------------------
//                               implementation of ClientProfilesJF
// -------------------------------------------------------------------------------------------

/**
 * @brief sort the budget which equals (SLO - networking time)
 *
 * @param clients
 */
void ClientProfilesJF::sortBudgetDescending(std::vector<PipelineModel *> &clients)
{
    std::sort(clients.begin(), clients.end(),
              [](const PipelineModel *a, const PipelineModel *b)
              {
                  return a->task->tk_slo - a->expectedTransferLatency > b->task->tk_slo - b->expectedTransferLatency;
              });
}

void ClientProfilesJF::add(PipelineModel *m)
{
    models.push_back(m);
}

void ClientProfilesJF::debugging()
{
    std::cout << "===================================ClientProfiles Debugging==========================" << std::endl;
    for (const auto &client_model : models)
    {
        std::cout << "Unique id: " << client_model->device << ", buget: " << client_model->task->tk_slo << ", req_rate: "
                  << client_model->arrivalProfiles.arrivalRates << std::endl;
    }
}

// -------------------------------------------------------------------------------------------
//                               implementation of scheduling algorithms
// -------------------------------------------------------------------------------------------

void Controller::Scheduling()
{
    while (running)
    {
        // update the networking time for each client-server pair
        // there is only one server in JellyFish, so get the server device from any recorded yolo model.
        // note: it's not allowed to be empty here, or it will cause the UB
        if (this->model_profiles_jf.infos.empty())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(5000));
            continue;
        }

        PipelineModel *model = this->model_profiles_jf.infos.begin()->second[0];
        std::unique_lock<std::mutex> lock(model->pipelineModelMutex);
        std::string server_device = model->device;
        lock.unlock();

        for (auto &client_model : client_profiles_jf.models)
        {
            std::unique_lock<std::mutex> client_lock(client_model->pipelineModelMutex);
            std::string client_device = client_model->device;
            client_lock.unlock();
            NetworkProfile network_proflie = queryNetworkProfile(
                *ctrl_metricsServerConn,
                ctrl_experimentName,
                ctrl_systemName,
                client_model->task->tk_name,
                client_model->task->tk_source,
                ctrl_containerLib[client_model->device].taskName,
                ctrl_containerLib[client_model->device].modelName,
                client_device,
                DeviceNameToType(client_device),
                server_device,
                DeviceNameToType(server_device),
                client_model->possibleNetworkEntryPairs[std::make_pair(client_device, server_device)]);
            auto lat = network_proflie.p95TransferDuration;
            client_model->expectedTransferLatency = lat;
        }

        // start scheduling

        auto mappings = mapClient(this->client_profiles_jf, this->model_profiles_jf);

        for (auto &mapping : mappings)
        {
            // retrieve the mapping for one model and its paired clients
            auto model_info = std::get<0>(mapping);
            auto selected_clients = std::get<1>(mapping);
            int batch_size = std::get<2>(mapping);

            // find the PipelineModel* of that model
            PipelineModel *m = this->model_profiles_jf.infos[model_info][0];
            for (auto &model : this->model_profiles_jf.infos[model_info])
            {
                if (model->batchSize == batch_size)
                {
                    // note: if occurs core dump, it's possible that there is no matchable pointer
                    // and the p is null
                    m = model;
                    break;
                }
            }
            // clear the upstream of that model
            std::unique_lock<std::mutex> model_lock(m->pipelineModelMutex);
            m->upstreams.clear();

            // TODO: leave another function to translate the changing of upstream, downstream to ContainerHandle

            // adjust downstream, upstream and resolution
            // CHECKME: vaildate the class of interest here, default to 1 for simplicity
            for (auto &client : selected_clients)
            {
                m->upstreams.push_back(std::make_pair(client, -1));
                std::unique_lock<std::mutex> client_lock(client->pipelineModelMutex);
                client->downstreams.clear();
                client->downstreams.push_back(std::make_pair(m, -1));

                // retrieve new resolution
                int width = m->width;
                int height = m->height;

                client_lock.unlock();

                std::unique_lock<std::mutex> container_lock(this->containers.containersMutex);
                for (auto it = this->containers.list.begin(); it != this->containers.list.end(); ++it)
                {
                    if (it->first == client->device)
                    {
                        std::vector<int> rs = {3, height, width};
                        AdjustResolution(it->second, rs);
                    }
                }
                container_lock.unlock();
            }
        }
        // use list of devices, tasks and containers to schedule depending on your algorithm
        // put helper functions as a private member function of the controller and write them at the bottom of this file.
        std::this_thread::sleep_for(std::chrono::milliseconds(
            5000)); // sleep time can be adjusted to your algorithm or just left at 5 seconds for now
    }
}

std::vector<PipelineModel *> findOptimalClients(std::vector<PipelineModel *> &models,
                                                std::vector<PipelineModel *> &clients)
{
    // sort clients
    ClientProfilesJF::sortBudgetDescending(clients);
    std::cout << "findOptimal start" << std::endl;
    std::cout << "available sorted clients: " << std::endl;
    for (auto &client : clients)
    {
        std::cout << client->device << " " << client->task->tk_slo - client->expectedTransferLatency << " " << client->arrivalProfiles.arrivalRates
                  << std::endl;
    }
    std::cout << "available models: " << std::endl;
    for (auto &model : models)
    {
        std::cout << model->name << " " << model->accuracy << " " << model->batchSize << " " << model->throughput << " "
                  << model->expectedMaxProcessLatency << std::endl;
    }
    std::tuple<int, int> best_cell;
    int best_value = 0;

    // dp
    auto [max_batch_size, max_index] = findMaxBatchSize(models, clients[0]);

    std::cout << "max batch size: " << max_batch_size
              << " and index: " << max_index << std::endl;

    assert(max_batch_size > 0);

    // construct the dp matrix
    int rows = clients.size() + 1;
    int h = 10; // assume gcd of all clients' req rate
    // find max throughput
    int max_throughput = 0;
    for (auto &model : models)
    {
        if (model->throughput > max_throughput)
        {
            max_throughput = model->throughput;
        }
    }
    // init matrix
    int cols = max_throughput / h + 1;
    std::cout << "max_throughput: " << max_throughput << std::endl;
    std::cout << "row: " << rows << " cols: " << cols << std::endl;
    std::vector<std::vector<int>> dp_mat(rows, std::vector<int>(cols, 0));
    // iterating
    for (int client_index = 1; client_index < clients.size(); client_index++)
    {
        auto &client = clients[client_index];
        auto result = findMaxBatchSize(models, client, max_batch_size);
        max_batch_size = std::get<0>(result);
        max_index = std::get<1>(result);
        std::cout << "client ip: " << client->device << ", max_batch_size: " << max_batch_size << ", max_index: "
                  << max_index << std::endl;
        if (max_batch_size <= 0)
        {
            break;
        }
        int cols_upperbound = int(models[max_index]->throughput / h);
        int lambda_i = client->arrivalProfiles.arrivalRates;
        int v_i = client->arrivalProfiles.arrivalRates;
        std::cout << "cols_up " << cols_upperbound << ", req " << lambda_i
                  << std::endl;
        for (int k = 1; k <= cols_upperbound; k++)
        {

            int w_k = k * h;
            if (lambda_i <= w_k)
            {
                int k_prime = (w_k - lambda_i) / h;
                int v = v_i + dp_mat[client_index - 1][k_prime];
                if (v > dp_mat[client_index - 1][k])
                {
                    dp_mat[client_index][k] = v;
                }
                if (v > best_value)
                {
                    best_cell = std::make_tuple(client_index, k);
                    best_value = v;
                }
            }
            else
            {
                dp_mat[client_index][k] = dp_mat[client_index - 1][k];
            }
        }
    }

    std::cout << "updated dp_mat" << std::endl;
    for (auto &row : dp_mat)
    {
        for (auto &v : row)
        {
            std::cout << v << " ";
        }
        std::cout << std::endl;
    }

    // perform backtracing from (row, col)
    // using dp_mat, best_cell, best_value

    std::vector<PipelineModel *> selected_clients;

    auto [row, col] = best_cell;

    std::cout << "best cell: " << row << " " << col << std::endl;
    int w = dp_mat[row][col];
    while (row > 0 && col > 0)
    {
        std::cout << row << " " << col << std::endl;
        if (dp_mat[row][col] == dp_mat[row - 1][col])
        {
            row--;
        }
        else
        {
            auto c = clients[row - 1];
            int w_i = c->arrivalProfiles.arrivalRates;
            row = row - 1;
            col = int((w - w_i) / h);
            w = col * h;
            assert(w == dp_mat[row][col]);
            selected_clients.push_back(c);
        }
    }

    std::cout << "findOptimal end" << std::endl;
    std::cout << "selected clients" << std::endl;
    for (auto sc : selected_clients)
    {
        std::cout << sc->device << " " << sc->task->tk_slo << " " << sc->arrivalProfiles.arrivalRates << std::endl;
    }

    return selected_clients;
}

/**
 * @brief client dnn mapping algorithm strictly following the paper jellyfish's Algo1
 *
 * @param client_profile
 * @param model_profiles
 * @return a vector of [ (model_name, accuracy), vec[clients], batch_size ]
 */
std::vector<
    std::tuple<std::tuple<std::string, float>, std::vector<PipelineModel *>, int>>
mapClient(ClientProfilesJF client_profile, ModelProfilesJF model_profiles)
{
    std::cout << " ======================= mapClient ==========================" << std::endl;

    std::vector<
        std::tuple<std::tuple<std::string, float>, std::vector<PipelineModel *>, int>>
        mappings;
    std::vector<PipelineModel *> clients = client_profile.models;

    int map_size = model_profiles.infos.size();
    int key_index = 0;
    for (auto it = model_profiles.infos.begin(); it != model_profiles.infos.end();
         ++it)
    {
        key_index++;
        std::cout << "before filtering" << std::endl;
        for (auto &c : clients)
        {
            std::cout << c->device << " " << c->task->tk_slo << " " << c->arrivalProfiles.arrivalRates << std::endl;
        }

        auto selected_clients = findOptimalClients(it->second, clients);

        // tradeoff:
        // assign all left clients to the last available model
        if (key_index == map_size)
        {
            std::cout << "assign all rest clients" << std::endl;
            selected_clients = clients;
            clients.clear();
            std::cout << "selected clients assgined" << std::endl;
            for (auto &c : selected_clients)
            {
                std::cout << c->device << " " << c->task->tk_slo << " " << c->arrivalProfiles.arrivalRates << std::endl;
            }
            assert(clients.size() == 0);
        }

        int batch_size = check_and_assign(it->second, selected_clients);

        std::cout << "model throughput: " << it->second[0]->throughput << std::endl;
        std::cout << "batch size: " << batch_size << std::endl;

        mappings.push_back(
            std::make_tuple(it->first, selected_clients, batch_size));
        std::cout << "start removing collected clients" << std::endl;
        for (auto &sc : selected_clients)
        {
            clients.erase(std::remove_if(clients.begin(), clients.end(),
                                         [&sc](const PipelineModel *c)
                                         {
                                             return c->device == sc->device;
                                         }),
                          clients.end());
        }
        std::cout << "after filtering" << std::endl;
        for (auto &c : clients)
        {
            std::cout << c->device << " " << c->task->tk_slo << " " << c->arrivalProfiles.arrivalRates << std::endl;
        }
        if (clients.size() == 0)
        {
            break;
        }
    }

    std::cout << "mapping relation" << std::endl;
    for (auto &t : mappings)
    {
        std::cout << "======================" << std::endl;
        auto [model_info, clients, batch_size] = t;
        std::cout << std::get<0>(model_info) << " " << std::get<1>(model_info)
                  << " " << batch_size << std::endl;
        for (auto &client : clients)
        {
            std::cout << "client name: " << client->device << ", req rate: " << client->arrivalProfiles.arrivalRates << ", budget-lat: "
                      << client->task->tk_slo << std::endl;
        }
        std::cout << "======================" << std::endl;
    }
    std::cout << "======================= End mapClient =======================" << std::endl;
    return mappings;
}

/**
 * @brief find the max available batch size for the associated clients of
 * corresponding model
 *
 * @param model
 * @param selected_clients
 * @return int
 */
int check_and_assign(std::vector<PipelineModel> &model,
                     std::vector<PipelineModel> &selected_clients)
{
    int total_req_rate = 0;
    // sum all selected req rate
    for (auto &client : selected_clients)
    {
        total_req_rate += client.arrivalProfiles.arrivalRates;
    }
    int max_batch_size = 1;

    for (auto &model_info : model)
    {
        if (model_info.throughput > total_req_rate &&
            max_batch_size < model_info.batchSize)
        {
            max_batch_size = model_info.batchSize;
        }
    }
    return max_batch_size;
}

/**
 * @brief find the max available batch size for the associated clients of
 * corresponding model
 *
 * @param model
 * @param selected_clients
 * @return int
 */
int check_and_assign(std::vector<PipelineModel *> &model,
                     std::vector<PipelineModel *> &selected_clients)
{
    int total_req_rate = 0;
    // sum all selected req rate
    for (auto &client : selected_clients)
    {
        total_req_rate += client->arrivalProfiles.arrivalRates;
    }
    int max_batch_size = 1;

    for (auto model_info : model)
    {
        if (model_info->throughput > total_req_rate &&
            max_batch_size < model_info->batchSize)
        {
            max_batch_size = model_info->batchSize;
        }
    }
    return max_batch_size;
}

// ====================== helper functions implementation ============================

/**
 * @brief find the maximum batch size for the client, the model vector is the set of model only different in batch size
 *
 * @param models
 * @param budget
 * @return max_batch_size, index
 */
std::tuple<int, int> findMaxBatchSize(const std::vector<PipelineModel *> &models,
                                      const PipelineModel *client, int max_available_batch_size)
{
    int max_batch_size = 0;
    int index = 0;
    int max_index = 0;
    for (const auto &model : models)
    {
        // CHECKME: the inference time should be limited by (budget - transmission time)
        if (model->expectedMaxProcessLatency * 2.0 < client->task->tk_slo - client->expectedTransferLatency &&
            model->batchSize > max_batch_size && model->batchSize <= max_available_batch_size)
        {
            max_batch_size = model->batchSize;
            max_index = index;
        }
        index++;
    }
    return std::make_tuple(max_batch_size, max_index);
}

// -------------------------------------------------------------------------------------------
//                                  end of implementations
// -------------------------------------------------------------------------------------------