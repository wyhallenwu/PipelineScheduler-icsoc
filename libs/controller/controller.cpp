#include "controller.h"

ABSL_FLAG(std::string, ctrl_configPath, "../jsons/experiments/base-experiment.json",
          "Path to the configuration file for this experiment.");
ABSL_FLAG(uint16_t, ctrl_verbose, 0, "Verbosity level of the controller.");
ABSL_FLAG(uint16_t, ctrl_loggingMode, 0, "Logging mode of the controller. 0:stdout, 1:file, 2:both");
ABSL_FLAG(std::string, ctrl_logPath, "../logs", "Path to the log dir for the controller.");

const int DATA_BASE_PORT = 55001;
const int CONTROLLER_BASE_PORT = 60001;
const int DEVICE_CONTROL_PORT = 60002;

void Controller::readConfigFile(const std::string &path)
{
    std::ifstream file(path);
    json j = json::parse(file);

    ctrl_experimentName = j["expName"];
    ctrl_systemName = j["systemName"];
    ctrl_runtime = j["runtime"];
    ctrl_port_offset = j["port_offset"];
    initialTasks = j["initial_pipelines"];
}

void TaskDescription::from_json(const nlohmann::json &j, TaskDescription::TaskStruct &val)
{
    j.at("pipeline_name").get_to(val.name);
    val.fullName = val.name + "_" + val.device;
    j.at("pipeline_target_slo").get_to(val.slo);
    j.at("pipeline_type").get_to(val.type);
    j.at("video_source").get_to(val.source);
    j.at("pipeline_source_device").get_to(val.device);
}

Controller::Controller(int argc, char **argv)
{
    absl::ParseCommandLine(argc, argv);
    readConfigFile(absl::GetFlag(FLAGS_ctrl_configPath));

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
}

Controller::~Controller()
{
    std::unique_lock<std::mutex> lock(containers.containersMutex);
    for (auto &msvc : containers.list)
    {
        StopContainer(msvc.first, msvc.second.device_agent, true);
    }

    std::unique_lock<std::mutex> lock2(devices.devicesMutex);
    for (auto &device : devices.list)
    {
        device.second.cq->Shutdown();
        void *got_tag;
        bool ok = false;
        while (device.second.cq->Next(&got_tag, &ok))
            ;
    }
    server->Shutdown();
    cq->Shutdown();
}

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

void Controller::Scheduling()
{
    // TODO: please out your scheduling loop inside of here
    while (running)
    {
        // use list of devices, tasks and containers to schedule depending on your algorithm
        // put helper functions as a private member function of the controller and write them at the bottom of this file.
        std::this_thread::sleep_for(std::chrono::milliseconds(
            5000)); // sleep time can be adjusted to your algorithm or just left at 5 seconds for now

        // update the networking time for each client-server pair
        // there is only one server in JellyFish, so get the server device from any recorded yolo model.
        // note: it's not allowed to be empty here, or it will cause the UB

        // CHECKME: lock correctness here
        auto model = this->model_profiles_jf.infos.begin()->second[0].model;
        std::unique_lock<std::mutex> lock = std::lock(model->pipelineModelMutex);
        auto server_device = model->device;
        lock.unlock();

        for (auto &client_info : client_profiles_jf.infos)
        {
            // CHECKME: lock correctness here
            auto client_model = client_info.model;
            std::unique_lock<std::mutex> client_lock = std::lock(client_model->pipelineModelMutex);
            auto client_device = client_info.model->device;
            client_lock.unlock();
            NetworkProfile network_proflie = queryNetworkProfile(
                *ctrl_metricsServerConn,
                ctrl_experimentName,
                ctrl_systemName,
                client_info.task_name,
                client_info.task_source,
                ctrl_containerLib[client_info.ip].taskName,
                ctrl_containerLib[client_info.ip].modelName,
                client_device,
                server_device,
                client_info.network_pairs[std::make_pair(client_device, server_device)]);
            auto lat = network_proflie.p95TransferDuration;
            client_info.set_transmission_latency(lat);
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
            ModelInfoJF m = this->model_profiles_jf.infos[model_info][0];
            for (auto &model : this->model_profiles_jf.infos[model_info])
            {
                if (model.batch_size == batch_size)
                {
                    // note: if occurs core dump, it's possible that there is no matchable pointer
                    // and the p is null
                    m = model;
                    break;
                }
            }
            // clear the upstream of that model
            // CHECKME: lock correctness here
            std::unique_lock<std::mutex> model_lock = std::lock(m.model->pipelineModelMutex);
            m.model->upstreams.clear();

            // TODO: leave another function to translate the changing of upstream, downstream to ContainerHandle

            // adjust downstream, upstream and resolution
            // CHECKME: vaildate the class of interest here, default to 1 for simplicity
            for (auto &client : selected_clients)
            {
                m.model->upstreams.push_back(std::make_pair(client.model, 1));
                std::unique_lock<std::mutex> client_lock = std::lock(client.model->pipelineModelMutex);
                client.model->downstreams.clear();
                client.model->downstreams.push_back(std::make_pair(m.model, 1));

                // retrieve new resolution
                int width = m.width;
                int height = m.height;

                client_lock.unlock();

                std::unique_lock<std::mutex> container_lock = std::lock(this->containers.containersMutex);
                for (auto it = this->containers.begin(); it != this->containers.end(); ++it)
                {
                    if (it->first == client.ip)
                    {
                        // CHECKME: excute resolution adjustment
                        std::vector<int> rs = {width, height, 3};
                        AdjustResolution(&(it->second), rs, 1);
                    }
                }
                container_lock.unlock();
            }
            model_lock.unlock();
        }
    }

    void Controller::queryInDeviceNetworkEntries(NodeHandle * node)
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
            NodeHandle node{deviceName,
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
            controller->devices.list.insert({deviceName, node});
            reply.set_name(controller->ctrl_systemName);
            reply.set_experiment(controller->ctrl_experimentName);
            status = FINISH;
            responder.Finish(reply, Status::OK, this);
            spdlog::get("container_agent")->info("Device {} is connected to the system", request.device_name());
            controller->queryInDeviceNetworkEntries(&(controller->devices.list[deviceName]));
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

    void Controller::StartContainer(std::pair<std::string, ContainerHandle *> & container, int slo, std::string source,
                                    int replica, bool easy_allocation)
    {
        std::cout << "Starting container: " << container.first << std::endl;
        ContainerConfig request;
        ClientContext context;
        EmptyMessage reply;
        Status status;
        request.set_pipeline_name(container.second->task->tk_name);
        request.set_model(container.second->model);
        request.set_model_file(container.second->model_file[replica - 1]);
        request.set_batch_size(container.second->batch_size[replica - 1]);
        request.set_replica_id(replica);
        request.set_allocation_mode(easy_allocation);
        request.set_device(container.second->cuda_device[replica - 1]);
        request.set_slo(slo);
        for (auto dim : container.second->dimensions)
        {
            request.add_input_dimensions(dim);
        }
        for (auto dwnstr : container.second->downstreams)
        {
            Neighbor *dwn = request.add_downstream();
            dwn->set_name(dwnstr->name);
            dwn->set_ip(absl::StrFormat("%s:%d", dwnstr->device_agent->ip, dwnstr->recv_port[replica - 1]));
            dwn->set_class_of_interest(dwnstr->class_of_interest);
            if (dwnstr->model == Sink)
            {
                dwn->set_gpu_connection(false);
            }
            else
            {
                dwn->set_gpu_connection((container.second->device_agent == dwnstr->device_agent) &&
                                        (container.second->cuda_device == dwnstr->cuda_device));
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
        if (container.second->model == DataSource || container.second->model == Yolov5nDsrc || container.second->model == RetinafaceDsrc)
        {
            Neighbor *up = request.add_upstream();
            up->set_name("video_source");
            up->set_ip(source);
            up->set_class_of_interest(-1);
            up->set_gpu_connection(false);
        }
        else
        {
            for (auto upstr : container.second->upstreams)
            {
                Neighbor *up = request.add_upstream();
                up->set_name(upstr->name);
                up->set_ip(absl::StrFormat("0.0.0.0:%d", container.second->recv_port[replica - 1]));
                up->set_class_of_interest(-2);
                up->set_gpu_connection((container.second->device_agent == upstr->device_agent) &&
                                       (container.second->cuda_device == upstr->cuda_device));
            }
        }
        std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            container.second->device_agent->stub->AsyncStartContainer(&context, request,
                                                                      container.second->device_agent->cq));
        finishGrpc(rpc, reply, status, container.second->device_agent->cq);
        if (!status.ok())
        {
            std::cout << status.error_code() << ": An error occured while sending the request" << std::endl;
        }
    }

    void Controller::MoveContainer(ContainerHandle * msvc, bool to_edge, int cuda_device, int replica)
    {
        NodeHandle *old_device = msvc->device_agent;
        NodeHandle *device;
        bool start_dsrc = false, merge_dsrc = false;
        if (to_edge)
        {
            device = msvc->upstreams[0]->device_agent;
            if (msvc->mergable)
            {
                merge_dsrc = true;
                if (msvc->model == Yolov5n)
                {
                    msvc->model = Yolov5nDsrc;
                }
                else if (msvc->model == Retinaface)
                {
                    msvc->model = RetinafaceDsrc;
                }
            }
        }
        else
        {
            device = &devices.list["server"];
            if (msvc->mergable)
            {
                start_dsrc = true;
                if (msvc->model == Yolov5nDsrc)
                {
                    msvc->model = Yolov5n;
                }
                else if (msvc->model == RetinafaceDsrc)
                {
                    msvc->model = Retinaface;
                }
            }
        }
        msvc->device_agent = device;
        msvc->recv_port[replica - 1] = device->next_free_port++;
        device->containers.insert({msvc->name, msvc});
        msvc->cuda_device[replica - 1] = cuda_device;
        std::pair<std::string, ContainerHandle *> pair = {msvc->name, msvc};
        StartContainer(pair, msvc->task->tk_slo, msvc->task->tk_source, replica, !(start_dsrc || merge_dsrc));
        for (auto upstr : msvc->upstreams)
        {
            if (start_dsrc)
            {
                std::pair<std::string, ContainerHandle *> dsrc_pair = {upstr->name, upstr};
                StartContainer(dsrc_pair, upstr->task->tk_slo, msvc->task->tk_source, replica, false);
                SyncDatasource(msvc, upstr);
            }
            else if (merge_dsrc)
            {
                SyncDatasource(upstr, msvc);
                StopContainer(upstr->name, old_device);
            }
            else
            {
                AdjustUpstream(msvc->recv_port[replica - 1], upstr, device, msvc->name);
            }
        }
        StopContainer(msvc->name, old_device);
        old_device->containers.erase(msvc->name);
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

    void Controller::SyncDatasource(ContainerHandle * prev, ContainerHandle * curr)
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

    void Controller::AdjustBatchSize(ContainerHandle * msvc, int new_bs, int replica)
    {
        msvc->batch_size[replica - 1] = new_bs;
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

    void AdjustResolution(ContainerHandle * msvc, std::vector<int> new_resolution, int replica = 1)
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

    void Controller::StopContainer(std::string name, NodeHandle * device, bool forced)
    {
        ContainerSignal request;
        ClientContext context;
        EmptyMessage reply;
        Status status;
        request.set_name(name);
        request.set_forced(forced);
        std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            device->stub->AsyncStopContainer(&context, request, containers.list[name].device_agent->cq));
        finishGrpc(rpc, reply, status, device->cq);
    }

    /**
     * @brief
     *
     * @param container calculating queue sizes for the container before its official deployment.
     * @param modelType
     */
    void Controller::calculateQueueSizes(ContainerHandle & container, const ModelType modelType)
    {
        float preprocessRate = 1000000.f / container.expectedPreprocessLatency;                   // queries per second
        float postprocessRate = 1000000.f / container.expectedPostprocessLatency;                 // qps
        float inferRate = 1000000.f / (container.expectedInferLatency * container.batch_size[0]); // batch per second

        QueueLengthType minimumQueueSize = 30;

        // Receiver to Preprocessor
        // Utilization of preprocessor
        float preprocess_rho = container.arrival_rate / preprocessRate;
        QueueLengthType preprocess_inQueueSize = std::max((QueueLengthType)std::ceil(preprocess_rho * preprocess_rho / (2 * (1 - preprocess_rho))), minimumQueueSize);
        float preprocess_thrpt = std::min(preprocessRate, container.arrival_rate);

        // Preprocessor to Inferencer
        // Utilization of inferencer
        float infer_rho = preprocess_thrpt / container.batch_size[0] / inferRate;
        QueueLengthType infer_inQueueSize = std::max((QueueLengthType)std::ceil(infer_rho * infer_rho / (2 * (1 - infer_rho))), minimumQueueSize);
        float infer_thrpt = std::min(inferRate, preprocess_thrpt / container.batch_size[0]); // batch per second

        float postprocess_rho = (infer_thrpt * container.batch_size[0]) / postprocessRate;
        QueueLengthType postprocess_inQueueSize = std::max((QueueLengthType)std::ceil(postprocess_rho * postprocess_rho / (2 * (1 - postprocess_rho))), minimumQueueSize);
        float postprocess_thrpt = std::min(postprocessRate, infer_thrpt * container.batch_size[0]);

        QueueLengthType sender_inQueueSize = postprocess_inQueueSize * container.batch_size[0];

        container.queueSizes = {preprocess_inQueueSize, infer_inQueueSize, postprocess_inQueueSize, sender_inQueueSize};

        container.expectedThroughput = postprocess_thrpt;
    }

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
    NetworkEntryType Controller::initNetworkCheck(const NodeHandle &node, uint32_t minPacketSize, uint32_t maxPacketSize, uint32_t numLoops)
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
        network_check_buffer[node.name].clear();
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

            for (auto &[deviceName, nodeHandle] : devices.list)
            {
                if (deviceName == "server")
                {
                    continue;
                }
                networkEntries[deviceName] = {};
            }
            std::string tableName = ctrl_metricsServerConfigs.schema + "." + abbreviate(ctrl_experimentName) + "_serv_netw";
            std::string query = absl::StrFormat("SELECT sender_host, p95_transfer_duration_us, p95_total_package_size_b "
                                                "FROM %s ",
                                                tableName);

            pqxx::result res = pullSQL(*ctrl_metricsServerConn, query);
            // Getting the latest network entries into the networkEntries map
            for (pqxx::result::const_iterator row = res.begin(); row != res.end(); ++row)
            {
                std::string sender_host = row["sender_host"].as<std::string>();
                if (sender_host == "server" || sender_host == "serv")
                {
                    continue;
                }
                std::pair<uint32_t, uint64_t> entry = {row["p95_total_package_size_b"].as<uint32_t>(), row["p95_transfer_duration_us"].as<uint64_t>()};
                networkEntries[sender_host].emplace_back(entry);
            }

            // Updating NodeHandle object with the latest network entries
            for (auto &[deviceName, entries] : networkEntries)
            {
                // If entry belongs to a device that is not in the list of devices, ignore it
                if (devices.list.find(deviceName) == devices.list.end() || deviceName != "server")
                {
                    continue;
                }
                std::unique_lock<std::mutex> lock(devices.list[deviceName].nodeHandleMutex);
                devices.list[deviceName].latestNetworkEntries["server"] = aggregateNetworkEntries(entries);
            }

            // If no network entries exist for a device, send a request to the device to perform network testing
            for (auto &[deviceName, nodeHandle] : devices.list)
            {
                if (nodeHandle.latestNetworkEntries.size() == 0)
                {
                    // TODO: Send a request to the device to perform network testing
                }
            }

            stopwatch.stop();
            std::this_thread::sleep_for(TimePrecisionType(60 * 1000000 - stopwatch.elapsed_microseconds()));
        }
    }

    // ----------------------------------------------------------------------------------------------------------------
    //                                         copy from scheduling-ppp.cpp
    // ----------------------------------------------------------------------------------------------------------------

    bool Controller::AddTask(const TaskDescription::TaskStruct &t)
    {
        std::cout << "Adding task: " << t.name << std::endl;
        TaskHandle *task = new TaskHandle{t.name, t.fullName, t.type, t.source, t.slo, {}, 0};

        std::unique_lock lock(devices.devicesMutex);
        if (devices.list.find(t.device) == devices.list.end())
        {
            spdlog::error("Device {0:s} is not connected", t.device);
            return false;
        }

        task->tk_pipelineModels = getModelsByPipelineType(t.type, t.device);
        std::unique_lock lock(ctrl_unscheduledPipelines.tasksMutex);

        ctrl_unscheduledPipelines.list.insert({task->tk_name, *task});
        lock.unlock();

        std::vector<std::pair<std::string, std::string>> possibleDevicePairList = {{"server", "server"}};
        std::map<std::pair<std::string, std::string>, NetworkEntryType> possibleNetworkEntryPairs;

        for (const auto &pair : possibleDevicePairList)
        {
            std::unique_lock lock(devices.list[pair.first].nodeHandleMutex);
            possibleNetworkEntryPairs[pair] = devices.list[pair.first].latestNetworkEntries[pair.second];
            lock.unlock();
        }

        std::vector<std::string> possibleDeviceList = {"server"};

        // CHECKME: need to modify here
        std::unique_lock lock(ctrl_unscheduledPipelines.tasksMutex);
        for (auto &model : task->tk_pipelineModels)
        {
            std::string containerName = model->name + "-" + possibleDevicePairList[0].second;
            if (containerName.find("datasource") != std::string::npos || containerName.find("sink") != std::string::npos)
            {
                // MODIFICATION
                if (containerName.find("datasource") != std::string::npos)
                {
                    model->arrivalProfiles.arrivalRates = 30;
                    this->client_profiles_jf.add(containerName, task->tk_slo, model->arrivalProfiles.arrivalRates, model,
                                                 task->tk_name, task->tk_source, possibleNetworkEntryPairs);
                }
                // END
                continue;
            }
            model->arrivalProfiles.arrivalRates = queryArrivalRate(
                *ctrl_metricsServerConn,
                ctrl_experimentName,
                ctrl_systemName,
                t.name,
                t.source,
                ctrl_containerLib[containerName].taskName,
                ctrl_containerLib[containerName].modelName);
            for (const auto &pair : possibleDevicePairList)
            {
                NetworkProfile test = queryNetworkProfile(
                    *ctrl_metricsServerConn,
                    ctrl_experimentName,
                    ctrl_systemName,
                    t.name,
                    t.source,
                    ctrl_containerLib[containerName].taskName,
                    ctrl_containerLib[containerName].modelName,
                    pair.first,
                    pair.second,
                    possibleNetworkEntryPairs[pair]);
                model->arrivalProfiles.d2dNetworkProfile[std::make_pair(pair.first, pair.second)] = test;
            }

            for (const auto deviceName : possibleDeviceList)
            {
                std::string deviceTypeName = getDeviceTypeName(devices[deviceName].type);
                ModelProfile profile = queryModelProfile(
                    *ctrl_metricsServerConn,
                    ctrl_experimentName,
                    ctrl_systemName,
                    t.name,
                    t.source,
                    deviceName,
                    deviceTypeName,
                    ctrl_containerLib[containerName].modelName);
                model->processProfiles[deviceTypeName] = profile;

                // MODIFICATION
                // collect the very first model of the pipeline, just use the yolo which is always the very first
                if (containerName.find("yolo"))
                {
                    // add all available batch_size profiling into consideration
                    for (auto it = profile.batchInfer.begin(); it != profile.batchInfer.end(); ++it)
                    {
                        BatchSizeType batch_size = it->first;
                        BatchInferProfile &batch_profile = it->second;

                        // note: the last three chars of the model name is the resolution it takes
                        int width = std::stoi(model->name.substr(model->name.length() - 3));

                        // check the accuracy indicator, use dummy value just to reflect the capacity of the model(evaluate their performance in general)
                        this->model_profiles_jf.add(model->name, ACC_LEVEL_MAP.at(model->name), static_cast<int>(batch_size),
                                                    static_cast<float>(batch_profile.p95inferLat), width, width, model); // height and width are the same
                    }
                }
            }

            // ModelArrivalProfile profile = queryModelArrivalProfile(
            //     *ctrl_metricsServerConn,
            //     ctrl_experimentName,
            //     ctrl_systemName,
            //     t.name,
            //     t.source,
            //     ctrl_containerLib[containerName].taskName,
            //     ctrl_containerLib[containerName].modelName,
            //     possibleDeviceList,
            //     possibleNetworkEntryPairs
            // );
            // std::cout << "sdfsdfasdf" << std::endl;
            // }
            std::cout << "Task added: " << t.name << std::endl;
            return true;
        }

        PipelineModelListType Controller::getModelsByPipelineType(PipelineType type, const std::string &startDevice)
        {
            switch (type)
            {
            case PipelineType::Traffic:
            {
                PipelineModel *datasource = new PipelineModel{startDevice, "datasource", true, {}, {}};
                PipelineModel *yolov5n = new PipelineModel{
                    "server",
                    "yolov5n",
                    true,
                    {},
                    {},
                    {},
                    {{datasource, -1}}};
                datasource->downstreams.push_back({yolov5n, -1});

                PipelineModel *retina1face = new PipelineModel{
                    "server",
                    "retina1face",
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 0}}};
                yolov5n->downstreams.push_back({retina1face, 0});

                PipelineModel *carbrand = new PipelineModel{
                    "server",
                    "carbrand",
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 2}}};
                yolov5n->downstreams.push_back({carbrand, 2});

                PipelineModel *platedet = new PipelineModel{
                    "server",
                    "platedet",
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 2}}};
                yolov5n->downstreams.push_back({platedet, 2});

                PipelineModel *sink = new PipelineModel{
                    "server",
                    "sink",
                    false,
                    {},
                    {},
                    {},
                    {{retina1face, -1}, {carbrand, -1}, {platedet, -1}}};
                retina1face->downstreams.push_back({sink, -1});
                carbrand->downstreams.push_back({sink, -1});
                platedet->downstreams.push_back({sink, -1});

                return {datasource, yolov5n, retina1face, carbrand, platedet, sink};
            }
            case PipelineType::Building_Security:
            {
                PipelineModel *datasource = new PipelineModel{startDevice, "datasource", true, {}, {}};
                PipelineModel *yolov5n = new PipelineModel{
                    "server",
                    "yolov5n",
                    true,
                    {},
                    {},
                    {},
                    {{datasource, -1}}};
                datasource->downstreams.push_back({yolov5n, -1});

                PipelineModel *retina1face = new PipelineModel{
                    "server",
                    "retina1face",
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 0}}};
                yolov5n->downstreams.push_back({retina1face, 0});

                PipelineModel *movenet = new PipelineModel{
                    "server",
                    "movenet",
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 0}}};
                yolov5n->downstreams.push_back({movenet, 0});

                PipelineModel *gender = new PipelineModel{
                    "server",
                    "gender",
                    false,
                    {},
                    {},
                    {},
                    {{retina1face, -1}}};
                retina1face->downstreams.push_back({gender, -1});

                PipelineModel *age = new PipelineModel{
                    "server",
                    "age",
                    false,
                    {},
                    {},
                    {},
                    {{retina1face, -1}}};
                retina1face->downstreams.push_back({age, -1});

                PipelineModel *sink = new PipelineModel{
                    "server",
                    "sink",
                    false,
                    {},
                    {},
                    {},
                    {{gender, -1}, {age, -1}, {movenet, -1}}};
                gender->downstreams.push_back({sink, -1});
                age->downstreams.push_back({sink, -1});
                movenet->downstreams.push_back({sink, -1});

                return {datasource, yolov5n, retina1face, movenet, gender, age, sink};
            }
            case PipelineType::Video_Call:
            {
                PipelineModel *datasource = new PipelineModel{startDevice, "datasource", true, {}, {}};
                PipelineModel *retina1face = new PipelineModel{
                    "server",
                    "retina1face",
                    true,
                    {},
                    {},
                    {},
                    {{datasource, -1}}};
                datasource->downstreams.push_back({retina1face, -1});

                PipelineModel *emotionnet = new PipelineModel{
                    "server",
                    "emotionnet",
                    false,
                    {},
                    {},
                    {},
                    {{retina1face, -1}}};
                retina1face->downstreams.push_back({emotionnet, -1});

                PipelineModel *age = new PipelineModel{
                    "server",
                    "age",
                    false,
                    {},
                    {},
                    {},
                    {{retina1face, -1}}};
                retina1face->downstreams.push_back({age, -1});

                PipelineModel *gender = new PipelineModel{
                    "server",
                    "gender",
                    false,
                    {},
                    {},
                    {},
                    {{retina1face, -1}}};
                retina1face->downstreams.push_back({gender, -1});

                PipelineModel *arcface = new PipelineModel{
                    "server",
                    "arcface",
                    false,
                    {},
                    {},
                    {},
                    {{retina1face, -1}}};
                retina1face->downstreams.push_back({arcface, -1});

                PipelineModel *sink = new PipelineModel{
                    "server",
                    "sink",
                    false,
                    {},
                    {},
                    {},
                    {{emotionnet, -1}, {age, -1}, {gender, -1}, {arcface, -1}}};
                emotionnet->downstreams.push_back({sink, -1});
                age->downstreams.push_back({sink, -1});
                gender->downstreams.push_back({sink, -1});
                arcface->downstreams.push_back({sink, -1});

                return {datasource, retina1face, emotionnet, age, gender, arcface, sink};
            }
            default:
                return {};
            }
        }

        /**
         * @brief Recursively traverse the model tree and try shifting models to edge devices
         *
         * @param models
         * @param slo
         */
        void Controller::shiftModelToEdge(TaskHandle & models, const ModelType &currModel, uint64_t slo)
        {
        }

        /**
         * @brief
         *
         * @param models
         * @param slo
         * @param nObjects
         * @return std::map<ModelType, int>
         */
        void Controller::getInitialBatchSizes(
            TaskHandle & models, uint64_t slo,
            int nObjects)
        {

            // for (auto &m: models) {
            //     ModelType modelType  = std::get<0>(m);
            //     m.second.batchSize = 1;
            //     m.second.numReplicas = 1;
            // }

            // // DFS-style recursively estimate the latency of a pipeline from source to sin
            // estimatePipelineLatency(models, models.begin()->first, 0);

            // uint64_t expectedE2ELatency = models.at(ModelType::Sink).expectedStart2HereLatency;

            // if (slo < expectedE2ELatency) {
            //     spdlog::info("SLO is too low for the pipeline to meet. Expected E2E latency: {0:d}, SLO: {1:d}", expectedE2ELatency, slo);
            // }

            // // Increase number of replicas to avoid bottlenecks
            // for (auto &m: models) {
            //     incNumReplicas(m.second, m.second.device);
            // }

            // // Find near-optimal batch sizes
            // auto foundBest = true;
            // while (foundBest) {
            //     foundBest = false;
            //     uint64_t bestCost = models.at(ModelType::Sink).estimatedStart2HereCost;
            //     PipelineModelListType tmp_models = models;
            //     for (auto &m: tmp_models) {
            //         m.second.batchSize *= 2;
            //         estimatePipelineLatency(tmp_models, tmp_models.begin()->first, 0);
            //         expectedE2ELatency = tmp_models.at(ModelType::Sink).expectedStart2HereLatency;
            //         if (expectedE2ELatency < slo) {
            //             // If increasing the batch size of model `m` creates a pipeline that meets the SLO, we should keep it
            //             uint64_t estimatedE2Ecost = tmp_models.at(ModelType::Sink).estimatedStart2HereCost;
            //             // Unless the estimated E2E cost is better than the best cost, we should not consider it as a candidate
            //             if (estimatedE2Ecost < bestCost) {
            //                 bestCost = estimatedE2Ecost;
            //                 models = tmp_models;
            //                 foundBest = true;
            //             }
            //             if (!foundBest) {
            //                 continue;
            //             }
            //             // If increasing the batch size meets the SLO, we can try decreasing the number of replicas
            //             decNumReplicas(m.second, m.second.device);
            //             estimatedE2Ecost = tmp_models.at(ModelType::Sink).estimatedStart2HereCost;
            //             if (estimatedE2Ecost < bestCost) {
            //                 models = tmp_models;
            //                 foundBest = true;
            //             }
            //         } else {
            //             m.second.batchSize /= 2;
            //         }
            //     }
            // }
        }

        /**
         * @brief estimate the different types of latency, in microseconds
         * Due to batch inference's nature, the queries that come later has to wait for more time both in preprocessor and postprocessor.
         *
         * @param model infomation about the model
         * @param modelType
         */
        void Controller::estimateModelLatency(PipelineModel * currModel, const std::string &deviceName)
        {
            ModelProfile profile = currModel->processProfiles[deviceName];
            uint64_t preprocessLatency = profile.p95prepLat;
            BatchSizeType batchSize = currModel->batchSize;
            uint64_t inferLatency = profile.batchInfer[batchSize].p95inferLat;
            uint64_t postprocessLatency = profile.p95postLat;
            float preprocessRate = 1000000.f / preprocessLatency;

            currModel->expectedQueueingLatency = calculateQueuingLatency(currModel->arrivalProfiles.arrivalRates, preprocessRate);
            currModel->expectedAvgPerQueryLatency = preprocessLatency + inferLatency * batchSize + postprocessLatency;
            currModel->expectedMaxProcessLatency = preprocessLatency * batchSize + inferLatency * batchSize + postprocessLatency * batchSize;
            currModel->estimatedPerQueryCost = currModel->expectedAvgPerQueryLatency + currModel->expectedQueueingLatency + currModel->expectedTransferLatency;
        }

        /**
         * @brief DFS-style recursively estimate the latency of a pipeline from source to sink
         *
         * @param pipeline provides all information about the pipeline needed for scheduling
         * @param currModel
         */
        void Controller::estimatePipelineLatency(PipelineModel * currModel, const uint64_t start2HereLatency)
        {
            // estimateModelLatency(currModel, currModel->device);

            // Update the expected latency to reach the current model
            // In case a model has multiple upstreams, the expected latency to reach the model is the maximum of the expected latency
            // to reach from each upstream.
            currModel->expectedStart2HereLatency = std::max(
                currModel->expectedStart2HereLatency,
                start2HereLatency + currModel->expectedMaxProcessLatency + currModel->expectedTransferLatency + currModel->expectedQueueingLatency);

            // Cost of the pipeline until the current model
            currModel->estimatedStart2HereCost += currModel->estimatedPerQueryCost;

            std::vector<std::pair<PipelineModel *, int>> downstreams = currModel->downstreams;
            for (const auto &d : downstreams)
            {
                estimatePipelineLatency(d.first, currModel->expectedStart2HereLatency);
            }

            if (currModel->downstreams.size() == 0)
            {
                return;
            }
        }

        /**
         * @brief Attempts to increase the number of replicas to meet the arrival rate
         *
         * @param model the model to be scaled
         * @param deviceName
         * @return uint8_t The number of replicas to be added
         */
        uint8_t Controller::incNumReplicas(const PipelineModel *model)
        {
            uint8_t numReplicas = model->numReplicas;
            std::string deviceTypeName = model->deviceTypeName;
            ModelProfile profile = model->processProfiles.at(deviceTypeName);
            uint64_t inferenceLatency = profile.batchInfer.at(model->batchSize).p95inferLat;
            float indiProcessRate = 1 / (inferenceLatency + profile.p95prepLat + profile.p95postLat);
            float processRate = indiProcessRate * numReplicas;
            while (processRate < model->arrivalProfiles.arrivalRates)
            {
                numReplicas++;
                processRate = indiProcessRate * numReplicas;
            }
            return numReplicas - model->numReplicas;
        }

        /**
         * @brief Decrease the number of replicas as long as it is possible to meet the arrival rate
         *
         * @param model
         * @return uint8_t The number of replicas to be removed
         */
        uint8_t Controller::decNumReplicas(const PipelineModel *model)
        {
            uint8_t numReplicas = model->numReplicas;
            std::string deviceTypeName = model->deviceTypeName;
            ModelProfile profile = model->processProfiles.at(deviceTypeName);
            uint64_t inferenceLatency = profile.batchInfer.at(model->batchSize).p95inferLat;
            float indiProcessRate = 1 / (inferenceLatency + profile.p95prepLat + profile.p95postLat);
            float processRate = indiProcessRate * numReplicas;
            while (numReplicas > 1)
            {
                numReplicas--;
                processRate = indiProcessRate * numReplicas;
                // If the number of replicas is no longer enough to meet the arrival rate, we should not decrease the number of replicas anymore.
                if (processRate < model->arrivalProfiles.arrivalRates)
                {
                    numReplicas++;
                    break;
                }
            }
            return model->numReplicas - numReplicas;
        }

        /**
         * @brief Calculate queueing latency for each query coming to the preprocessor's queue, in microseconds
         * Queue type is expected to be M/D/1
         *
         * @param arrival_rate
         * @param preprocess_rate
         * @return uint64_t
         */
        uint64_t Controller::calculateQueuingLatency(const float &arrival_rate, const float &preprocess_rate)
        {
            float rho = arrival_rate / preprocess_rate;
            float numQueriesInSystem = rho / (1 - rho);
            float averageQueueLength = rho * rho / (1 - rho);
            return (uint64_t)(averageQueueLength / arrival_rate * 1000000);
        }

        // ----------------------------------------------------------------------------------------------------------------
        //                                         end of copy from scheduling-ppp.cpp
        // ----------------------------------------------------------------------------------------------------------------

        // ----------------------------------------------------------------------------------------------------------------
        //                                             implementations
        // ----------------------------------------------------------------------------------------------------------------
        ModelInfoJF::ModelInfoJF() {}

        ModelInfoJF::ModelInfoJF(int bs, float il, int w, int h, std::string n, float acc, PipelineModel *m)
        {
            batch_size = bs;

            // the inference_latency is us
            inference_latency = il;

            // throughput is req/s
            // CHECKME: validate the unit of the time stamp and the gcd of all throughputs,
            // now the time stamp is us, and the gcd of all throughputs is 10, maybe need change to ease the dp table
            throughput = (int(bs / (il * 1e-6)) / 10) * 10; // round it to be devidisble by 10 for better dp computing
            width = w;
            height = h;
            name = n;
            accuracy = acc;
            model = m;
        }

        ClientInfoJF::ClientInfoJF(std::string _ip, float _budget, int _req_rate,
                                   PipelineModel *_model, std::string _task_name, std::string _task_source,
                                   std::map<std::pair<std::string, std::string>, NetworkEntryType> _network_pairs)
        {
            ip = _ip;
            budget = _budget;
            req_rate = _req_rate;
            model = _model;
            task_name = _task_name;
            task_source = _task_source;
            transmission_latency = -1;
            network_pairs = _network_pairs;
        }

        void ClientInfoJF::set_transmission_latency(int lat)
        {
            this->transmission_latency = lat;
        }

        bool ModelSetCompare::operator()(
            const std::tuple<std::string, float> &lhs,
            const std::tuple<std::string, float> &rhs) const
        {
            return std::get<1>(lhs) < std::get<1>(rhs);
        }

        // -------------------------------------------------------------------------------------------
        //                               implementation of ModelProfilesJF
        // -------------------------------------------------------------------------------------------

        /**
         * @brief add profiled information of model
         *
         * @param model_type
         * @param accuracy
         * @param batch_size
         * @param inference_latency
         * @param throughput
         */
        void ModelProfilesJF::add(std::string name, float accuracy, int batch_size, float inference_latency, int width, int height, PipelineModel *m)
        {
            auto key = std::tuple<std::string, float>{name, accuracy};
            ModelInfoJF value(batch_size, inference_latency, width, height, name, accuracy, m);
            infos[key].push_back(value);
        }

        void ModelProfilesJF::add(const ModelInfoJF &model_info)
        {
            auto key =
                std::tuple<std::string, float>{model_info.name, model_info.accuracy};
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
                    std::cout << "batch size: " << model_info.batch_size << ", latency: " << model_info.inference_latency
                              << ", width: " << model_info.width << ", height: " << model_info.height << std::endl;
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
        void ClientProfilesJF::sortBudgetDescending(std::vector<ClientInfoJF> & clients)
        {
            std::sort(clients.begin(), clients.end(),
                      [](const ClientInfoJF &a, const ClientInfoJF &b)
                      {
                          return a.budget - a.transmission_latency > b.budget - b.transmission_latency;
                      });
        }

        void ClientProfilesJF::add(const std::string &ip, float budget, int req_rate,
                                   PipelineModel *model, std::string task_name, std::string task_source,
                                   std::map<std::pair<std::string, std::string>, NetworkEntryType> network_pairs)
        {
            infos.push_back(ClientInfoJF(ip, budget, req_rate, model, task_name, task_source, network_pairs));
        }

        void ClientProfilesJF::debugging()
        {
            std::cout << "===================================ClientProfiles Debugging==========================" << std::endl;
            for (const auto &client_info : infos)
            {
                std::cout << "Unique id: " << client_info.ip << ", buget: " << client_info.budget << ", req_rate: " << client_info.req_rate << std::endl;
            }
        }

        // -------------------------------------------------------------------------------------------
        //                               implementation of scheduling algorithms
        // -------------------------------------------------------------------------------------------

        std::vector<ClientInfoJF> findOptimalClients(const std::vector<ModelInfoJF> &models,
                                                     std::vector<ClientInfoJF> &clients)
        {
            // sort clients
            ClientProfilesJF::sortBudgetDescending(clients);
            std::cout << "findOptimal start" << std::endl;
            std::cout << "available sorted clients: " << std::endl;
            for (auto &client : clients)
            {
                std::cout << client.ip << " " << client.budget - client.transmission_latency << " " << client.req_rate
                          << std::endl;
            }
            std::cout << "available models: " << std::endl;
            for (auto &model : models)
            {
                std::cout << model.name << " " << model.accuracy << " " << model.batch_size << " " << model.throughput << " " << model.inference_latency << std::endl;
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
                if (model.throughput > max_throughput)
                {
                    max_throughput = model.throughput;
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
                std::cout << "client ip: " << client.ip << ", max_batch_size: " << max_batch_size << ", max_index: "
                          << max_index << std::endl;
                if (max_batch_size <= 0)
                {
                    break;
                }
                int cols_upperbound = int(models[max_index].throughput / h);
                int lambda_i = client.req_rate;
                int v_i = client.req_rate;
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

            std::vector<ClientInfoJF> selected_clients;

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
                    int w_i = c.req_rate;
                    row = row - 1;
                    col = int((w - w_i) / h);
                    w = col * h;
                    assert(w == dp_mat[row][col]);
                    selected_clients.push_back(c);
                }
            }

            std::cout << "findOptimal end" << std::endl;
            std::cout << "selected clients" << std::endl;
            for (auto &sc : selected_clients)
            {
                std::cout << sc.ip << " " << sc.budget << " " << sc.req_rate << std::endl;
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
            std::tuple<std::tuple<std::string, float>, std::vector<ClientInfoJF>, int>>
        mapClient(ClientProfilesJF client_profile, ModelProfilesJF model_profiles)
        {
            std::cout << " ======================= mapClient ==========================" << std::endl;

            std::vector<
                std::tuple<std::tuple<std::string, float>, std::vector<ClientInfoJF>, int>>
                mappings;
            std::vector<ClientInfoJF> clients = client_profile.infos;

            int map_size = model_profiles.infos.size();
            int key_index = 0;
            for (auto it = model_profiles.infos.begin(); it != model_profiles.infos.end();
                 ++it)
            {
                key_index++;
                std::cout << "before filtering" << std::endl;
                for (auto &c : clients)
                {
                    std::cout << c.ip << " " << c.budget << " " << c.req_rate << std::endl;
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
                        std::cout << c.ip << " " << c.budget << " " << c.req_rate << std::endl;
                    }
                    assert(clients.size() == 0);
                }

                int batch_size = check_and_assign(it->second, selected_clients);

                std::cout << "model throughput: " << it->second[0].throughput << std::endl;
                std::cout << "batch size: " << batch_size << std::endl;

                mappings.push_back(
                    std::make_tuple(it->first, selected_clients, batch_size));
                std::cout << "start removing collected clients" << std::endl;
                differenceClients(clients, selected_clients);
                std::cout << "after filtering" << std::endl;
                for (auto &c : clients)
                {
                    std::cout << c.ip << " " << c.budget << " " << c.req_rate << std::endl;
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
                    std::cout << "client name: " << client.ip << ", req rate: " << client.req_rate << ", budget-lat: " << client.budget << std::endl;
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
        int check_and_assign(std::vector<ModelInfoJF> & model,
                             std::vector<ClientInfoJF> & selected_clients)
        {
            int total_req_rate = 0;
            // sum all selected req rate
            for (auto &client : selected_clients)
            {
                total_req_rate += client.req_rate;
            }
            int max_batch_size = 1;

            for (auto &model_info : model)
            {
                if (model_info.throughput > total_req_rate &&
                    max_batch_size < model_info.batch_size)
                {
                    max_batch_size = model_info.batch_size;
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
        std::tuple<int, int> findMaxBatchSize(const std::vector<ModelInfoJF> &models,
                                              const ClientInfoJF &client, int max_available_batch_size)
        {
            int max_batch_size = 0;
            float budget = client.budget;
            int index = 0;
            int max_index = 0;
            for (const auto &model : models)
            {
                // CHECKME: the inference time should be limited by (budget - transmission time)
                if (model.inference_latency * 2.0 < client.budget - client.transmission_latency &&
                    model.batch_size > max_batch_size && model.batch_size <= max_available_batch_size)
                {
                    max_batch_size = model.batch_size;
                    max_index = index;
                }
                index++;
            }
            return std::make_tuple(max_batch_size, max_index);
        }

        /**
         * @brief remove the selected clients
         *
         * @param src
         * @param diff
         */
        void differenceClients(std::vector<ClientInfoJF> & src,
                               const std::vector<ClientInfoJF> &diff)
        {
            auto is_in_diff = [&diff](const ClientInfoJF &client)
            {
                return std::find(diff.begin(), diff.end(), client) != diff.end();
            };
            src.erase(std::remove_if(src.begin(), src.end(), is_in_diff), src.end());
        }

        // -------------------------------------------------------------------------------------------
        //                                  end of implementations
        // -------------------------------------------------------------------------------------------