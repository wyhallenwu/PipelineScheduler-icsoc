#include "controller.h"

ABSL_FLAG(std::string, ctrl_configPath, "../jsons/experiments/base-experiment.json",
          "Path to the configuration file for this experiment.");
ABSL_FLAG(uint16_t, ctrl_verbose, 0, "Verbosity level of the controller.");
ABSL_FLAG(uint16_t, ctrl_loggingMode, 0, "Logging mode of the controller. 0:stdout, 1:file, 2:both");
ABSL_FLAG(std::string, ctrl_logPath, "../logs", "Path to the log dir for the controller.");

void Controller::readConfigFile(const std::string &path) {
    std::ifstream file(path);
    json j = json::parse(file);

    ctrl_experimentName = j["expName"];
    ctrl_systemName = j["systemName"];
    ctrl_runtime = j["runtime"];
    initialTasks = j["initial_pipelines"];

}

void TaskDescription::from_json(const nlohmann::json &j, TaskDescription::TaskStruct &val) {
    j.at("pipeline_name").get_to(val.name);
    j.at("pipeline_target_slo").get_to(val.slo);
    j.at("pipeline_type").get_to(val.type);
    j.at("video_source").get_to(val.source);
    j.at("pipeline_source_device").get_to(val.device);
}

Controller::Controller(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);
    readConfigFile(absl::GetFlag(FLAGS_ctrl_configPath));

    ctrl_logPath = absl::GetFlag(FLAGS_ctrl_logPath);
    ctrl_logPath += "/" + ctrl_experimentName;
    std::filesystem::create_directories(
            std::filesystem::path(ctrl_logPath)
    );
    ctrl_logPath += "/" + ctrl_systemName;
    std::filesystem::create_directories(
            std::filesystem::path(ctrl_logPath)
    );
    ctrl_verbose = absl::GetFlag(FLAGS_ctrl_verbose);
    ctrl_loggingMode = absl::GetFlag(FLAGS_ctrl_loggingMode);

    setupLogger(
            ctrl_logPath,
            "controller",
            ctrl_loggingMode,
            ctrl_verbose,
            ctrl_loggerSinks,
            ctrl_logger
    );

    ctrl_containerLib = getContainerLib();

    json metricsCfgs = json::parse(std::ifstream("../jsons/metricsserver.json"));
    ctrl_metricsServerConfigs.from_json(metricsCfgs);
    ctrl_metricsServerConfigs.schema = ctrl_experimentName + "_" + ctrl_systemName;
    ctrl_metricsServerConfigs.user = "controller";
    ctrl_metricsServerConfigs.password = "agent";
    ctrl_metricsServerConn = connectToMetricsServer(ctrl_metricsServerConfigs, "controller");



    running = true;
    devices = std::map<std::string, NodeHandle>();
    tasks = std::map<std::string, TaskHandle>();
    containers = std::map<std::string, ContainerHandle>();

    std::string server_address = absl::StrFormat("%s:%d", "0.0.0.0", 60001);
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    cq = builder.AddCompletionQueue();
    server = builder.BuildAndStart();
}

Controller::~Controller() {
    for (auto &msvc: containers) {
        StopContainer(msvc.first, msvc.second.device_agent, true);
    }
    for (auto &device: devices) {
        device.second.cq->Shutdown();
        void *got_tag;
        bool ok = false;
        while (device.second.cq->Next(&got_tag, &ok));
    }
    server->Shutdown();
    cq->Shutdown();
}

void Controller::HandleRecvRpcs() {
    new DeviseAdvertisementHandler(&service, cq.get(), this);
    while (running) {
        void *tag;
        bool ok;
        if (!cq->Next(&tag, &ok)) {
            break;
        }
        static_cast<RequestHandler *>(tag)->Proceed();
    }
}

void Controller::Scheduling() {
    // TODO: please out your scheduling loop inside of here
    while (running) {
        // use list of devices, tasks and containers to schedule depending on your algorithm
        // put helper functions as a private member function of the controller and write them at the bottom of this file.
        std::this_thread::sleep_for(std::chrono::milliseconds(
                5000)); // sleep time can be adjusted to your algorithm or just left at 5 seconds for now
    }
}

void Controller::AddTask(const TaskDescription::TaskStruct &t) {
    std::cout << "Adding task: " << t.name << std::endl;
    // tasks.insert({t.name, {t.name, t.type, t.source, t.slo, {}, 0, {}}});
    // TaskHandle *task = &tasks[t.name];
    // NodeHandle *device = &devices[t.device];
    // auto models = getModelsByPipelineType(t.type);

    // std::string tmp = t.name;
    // containers.insert({tmp.append(":datasource"), {tmp, DataSource, device, task, 33, 1, {0}}});
    // task->subtasks.insert({tmp, &containers[tmp]});
    // task->subtasks[tmp]->recv_port = device->next_free_port++;
    // device->containers.insert({tmp, task->subtasks[tmp]});
    // device = &devices["server"];

    // // TODO: get correct initial batch size, cuda devices, and number of replicas
    // auto batch_sizes = getInitialBatchSizes(models, t.slo, 10);
    // int cuda_device = 1;
    // int replicas = 1;
    // for (const auto &m: models) {
    //     tmp = t.name;

    //     containers.insert(
    //             {tmp.append(MODEL_INFO[m.first][0]), {tmp, m.first, device, task, batch_sizes[m.first], 1, {cuda_device},
    //                                                   -1, device->next_free_port++, {}, {}, {}, {}}});
    //     task->subtasks.insert({tmp, &containers[tmp]});
    //     device->containers.insert({tmp, task->subtasks[tmp]});
    // }

    // task->subtasks[t.name + ":datasource"]->downstreams.push_back(task->subtasks[t.name + MODEL_INFO[models[0].first][0]]);
    // task->subtasks[t.name + MODEL_INFO[models[0].first][0]]->upstreams.push_back(task->subtasks[t.name + ":datasource"]);
    // for (const auto &m: models) {
    //     for (const auto &d: m.second) {
    //         tmp = t.name;
    //         task->subtasks[tmp.append(MODEL_INFO[d.first][0])]->class_of_interest = d.second;
    //         task->subtasks[tmp]->upstreams.push_back(task->subtasks[t.name + MODEL_INFO[m.first][0]]);
    //         task->subtasks[t.name + MODEL_INFO[m.first][0]]->downstreams.push_back(task->subtasks[tmp]);
    //     }
    // }

    // for (std::pair<std::string, ContainerHandle *> msvc: task->subtasks) {
    //     StartContainer(msvc, task->slo, t.source, replicas);
    // }
}

void Controller::DeviseAdvertisementHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestAdvertiseToController(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new DeviseAdvertisementHandler(service, cq, controller);
        std::string target_str = absl::StrFormat("%s:%d", request.ip_address(), 60002);
        controller->devices.insert({request.device_name(),
                                    {request.device_name(),
                                     request.ip_address(),
                                     ControlCommunication::NewStub(
                                             grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials())),
                                     new CompletionQueue(),
                                     static_cast<SystemDeviceType>(request.device_type()),
                                     request.processors(), std::vector<double>(request.processors(), 0.0),
                                     std::vector<unsigned long>(request.memory().begin(), request.memory().end()),
                                     std::vector<double>(request.processors(), 0.0), 55001, {}}});
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void Controller::StartContainer(std::pair<std::string, ContainerHandle *> &container, int slo, std::string source,
                                int replica, bool easy_allocation) {
    std::cout << "Starting container: " << container.first << std::endl;
    ContainerConfig request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_pipeline_name(container.second->task->name);
    request.set_model(container.second->model);
    request.set_batch_size(container.second->batch_size[replica -1]);
    request.set_replica_id(replica);
    request.set_allocation_mode(easy_allocation);
    request.set_device(container.second->cuda_device[replica - 1]);
    request.set_slo(slo);
    for (auto dim: container.second->dimensions) {
        request.add_input_dimensions(dim);
    }
    for (auto dwnstr: container.second->downstreams) {
        Neighbor *dwn = request.add_downstream();
        dwn->set_name(dwnstr->name);
        dwn->set_ip(absl::StrFormat("%s:%d", dwnstr->device_agent->ip, dwnstr->recv_port[replica - 1]));
        dwn->set_class_of_interest(dwnstr->class_of_interest);
        if (dwnstr->model == Sink) {
            dwn->set_gpu_connection(false);
        } else {
            dwn->set_gpu_connection((container.second->device_agent == dwnstr->device_agent) &&
                                    (container.second->cuda_device == dwnstr->cuda_device));
        }
    }
    if (request.downstream_size() == 0) {
        Neighbor *dwn = request.add_downstream();
        dwn->set_name("video_sink");
        dwn->set_ip("./out.log"); //output log file
        dwn->set_class_of_interest(-1);
        dwn->set_gpu_connection(false);
    }
    if (container.second->model == DataSource || container.second->model == Yolov5nDsrc || container.second->model == RetinafaceDsrc) {
        Neighbor *up = request.add_upstream();
        up->set_name("video_source");
        up->set_ip(source);
        up->set_class_of_interest(-1);
        up->set_gpu_connection(false);
    } else {
        for (auto upstr: container.second->upstreams) {
            Neighbor *up = request.add_upstream();
            up->set_name(upstr->name);
            up->set_ip(absl::StrFormat("0.0.0.0:%d", container.second->recv_port[replica -1]));
            up->set_class_of_interest(-2);
            up->set_gpu_connection((container.second->device_agent == upstr->device_agent) &&
                                   (container.second->cuda_device == upstr->cuda_device));
        }
    }
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            container.second->device_agent->stub->AsyncStartContainer(&context, request,
                                                                      container.second->device_agent->cq));
    rpc->Finish(&reply, &status, (void *) 1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(container.second->device_agent->cq->Next(&got_tag, &ok));
    GPR_ASSERT(ok);
    if (!status.ok()) {
        std::cout << status.error_code() << ": An error occured while sending the request" << std::endl;
    }
}

void Controller::MoveContainer(ContainerHandle *msvc, bool to_edge, int cuda_device, int replica) {
    NodeHandle *old_device = msvc->device_agent;
    NodeHandle *device;
    bool start_dsrc = false, merge_dsrc = false;
    if (to_edge) {
        device = msvc->upstreams[0]->device_agent;
        if (msvc->mergable) {
            merge_dsrc = true;
            if (msvc->model == Yolov5n) {
                msvc->model = Yolov5nDsrc;
            } else if (msvc->model == Retinaface) {
                msvc->model = RetinafaceDsrc;
            }
        }
    } else {
        device = &devices["server"];
        if (msvc->mergable) {
            start_dsrc = true;
            if (msvc->model == Yolov5nDsrc) {
                msvc->model = Yolov5n;
            } else if (msvc->model == RetinafaceDsrc) {
                msvc->model = Retinaface;
            }
        }
    }
    msvc->device_agent = device;
    msvc->recv_port[replica - 1] = device->next_free_port++;
    device->containers.insert({msvc->name, msvc});
    msvc->cuda_device[replica - 1] = cuda_device;
    std::pair<std::string, ContainerHandle *> pair = {msvc->name, msvc};
    StartContainer(pair, msvc->task->slo, msvc->task->source, replica, !(start_dsrc || merge_dsrc));
    for (auto upstr: msvc->upstreams) {
        if (start_dsrc) {
            std::pair<std::string, ContainerHandle *> dsrc_pair = {upstr->name, upstr};
            StartContainer(dsrc_pair, upstr->task->slo, msvc->task->source, replica, false);
            SyncDatasource(msvc, upstr);
        } else if (merge_dsrc) {
            SyncDatasource(upstr, msvc);
            StopContainer(upstr->name, old_device);
        } else {
            AdjustUpstream(msvc->recv_port[replica - 1], upstr, device, msvc->name);
        }
    }
    StopContainer(msvc->name, old_device);
    old_device->containers.erase(msvc->name);
}

void Controller::AdjustUpstream(int port, ContainerHandle *upstr, NodeHandle *new_device,
                                const std::string &dwnstr) {
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
    rpc->Finish(&reply, &status, (void *) 1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(upstr->device_agent->cq->Next(&got_tag, &ok));
    GPR_ASSERT(ok);
}

void Controller::SyncDatasource(ContainerHandle *prev, ContainerHandle *curr) {
    ContainerLink request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_name(prev->name);
    request.set_downstream_name(curr->name);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            curr->device_agent->stub->AsyncSyncDatasource(&context, request, curr->device_agent->cq));
    rpc->Finish(&reply, &status, (void *) 1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(curr->device_agent->cq->Next(&got_tag, &ok));
    GPR_ASSERT(ok);
}

void Controller::AdjustBatchSize(ContainerHandle *msvc, int new_bs, int replica) {
    msvc->batch_size[replica - 1] = new_bs;
    ContainerInt request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_name(msvc->name);
    request.set_value(new_bs);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            msvc->device_agent->stub->AsyncUpdateBatchSize(&context, request, msvc->device_agent->cq));
    rpc->Finish(&reply, &status, (void *) 1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(msvc->device_agent->cq->Next(&got_tag, &ok));
    GPR_ASSERT(ok);
}

void Controller::StopContainer(std::string name, NodeHandle *device, bool forced) {
    ContainerSignal request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_name(name);
    request.set_forced(forced);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            device->stub->AsyncStopContainer(&context, request, containers[name].device_agent->cq));
    rpc->Finish(&reply, &status, (void *) 1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(device->cq->Next(&got_tag, &ok));
    GPR_ASSERT(ok);
}

void Controller::optimizeBatchSizeStep(
        const Pipeline &models,
        std::map<ModelType, int> &batch_sizes, std::map<ModelType, int> &estimated_infer_times, int nObjects) {
    ModelType candidate;
    int max_saving = 0;
    std::vector<ModelType> blacklist;
    for (const auto &m: models) {
        int saving;
        if (max_saving == 0) {
            saving =
                    estimated_infer_times[m.first] - InferTimeEstimator(m.first, batch_sizes[m.first] * 2);
        } else {
            if (batch_sizes[m.first] == 64 ||
                std::find(blacklist.begin(), blacklist.end(), m.first) != blacklist.end()) {
                continue;
            }
            for (const auto &d: m.second) {
                if (batch_sizes[d.first] > batch_sizes[m.first]) {
                    blacklist.push_back(d.first);
                }
            }
            saving = estimated_infer_times[m.first] -
                     (InferTimeEstimator(m.first, batch_sizes[m.first] * 2) * (nObjects / batch_sizes[m.first] * 2));
        }
        if (saving > max_saving) {
            max_saving = saving;
            candidate = m.first;
        }
    }
    batch_sizes[candidate] *= 2;
    estimated_infer_times[candidate] -= max_saving;
}

std::map<ModelType, int> Controller::getInitialBatchSizes(
        const Pipeline &models, int slo,
        int nObjects) {
    std::map<ModelType, int> batch_sizes = {};
    std::map<ModelType, int> estimated_infer_times = {};

    for (const auto &m: models) {
        batch_sizes[m.first] = 1;
        if (estimated_infer_times.size() == 0) {
            estimated_infer_times[m.first] = (InferTimeEstimator(m.first, 1));
        } else {
            estimated_infer_times[m.first] = (InferTimeEstimator(m.first, 1) * nObjects);
        }
    }

    int sum = std::accumulate(estimated_infer_times.begin(), estimated_infer_times.end(), 0,
                              [](int acc, const std::pair<ModelType, int> &p) {
                                  return acc + p.second;
                              });

    while (slo < sum) {
        optimizeBatchSizeStep(models, batch_sizes, estimated_infer_times, nObjects);
        sum = std::accumulate(estimated_infer_times.begin(), estimated_infer_times.end(), 0,
                              [](int acc, const std::pair<ModelType, int> &p) {
                                  return acc + p.second;
                              });
    }
    optimizeBatchSizeStep(models, batch_sizes, estimated_infer_times, nObjects);
    return batch_sizes;
}

PipelineModelListType Controller::getModelsByPipelineType(PipelineType type, const std::string &startDevice) {
    switch (type) {
        case PipelineType::Traffic:
            return {
                {
                    ModelType::DataSource, 
                    {startDevice, true, {}, {}, {{ModelType::Yolov5n, 0}}}
                },
                {
                    ModelType::Yolov5n,
                    {
                        "server", true, {}, {},       
                        {{ModelType::Retinaface, 0}, {ModelType::CarBrand, 2}, {ModelType::PlateDet, 2}},
                        {{ModelType::DataSource, -1}}
                    },
                },
                {
                    ModelType::Retinaface, 
                    {
                        "server", false, {}, {},
                        {{ModelType::Arcface,    -1}},
                        {{ModelType::Yolov5n, -1}}
                    }
                },
                {
                    ModelType::Arcface,
                    {
                        "server", false, {}, {},
                        {{ModelType::Sink,   -1}},
                        {{ModelType::Retinaface, -1}}
                    }
                },
                {
                    ModelType::CarBrand,
                    {
                        "server", false, {}, {},
                        {{ModelType::Sink,   -1}},
                        {{ModelType::Yolov5n, -1}}
                    }
                },
                {
                    ModelType::PlateDet,
                    {
                        "server", false, {}, {}, {{ModelType::Sink,   -1}},
                        {{ModelType::Yolov5n, -1}}
                    }
                },
                {
                    ModelType::Sink,
                    {
                        "server", false, {}, {},
                        {},
                        {{ModelType::Arcface, -1}, {ModelType::CarBrand, -1}, {ModelType::PlateDet, -1}}
                    }
                }
            };
        case PipelineType::Video_Call:
            return {
                {
                    ModelType::DataSource,
                    {startDevice, true, {}, {}, {{ModelType::Retinaface, 0}}}
                },
                {
                    ModelType::Retinaface,
                    {
                        "server", true, {}, {},
                        {{ModelType::Emotionnet, -1}, {ModelType::Age, -1}, {ModelType::Gender, -1}, {ModelType::Arcface, -1}},
                        {{ModelType::DataSource, -1}}
                    }
                },
                {
                    ModelType::Gender,
                    {
                        "server", false, {}, {},
                        {{ModelType::Sink,   -1}},
                        {{ModelType::Retinaface, -1}}
                    }
                },
                {
                    ModelType::Age,
                    {
                        "server", false, {}, {},
                        {{ModelType::Sink,   -1}},
                        {{ModelType::Retinaface, -1}}
                    }
                },
                {
                    ModelType::Emotionnet,
                    {
                        "server", false, {}, {},
                        {{ModelType::Sink,   -1}},
                        {{ModelType::Retinaface, -1}}
                    }
                },
                {
                    ModelType::Arcface,
                    {
                        "server", false, {}, {},
                        {{ModelType::Sink,   -1}},
                        {{ModelType::Retinaface, -1}}
                    }
                },
                {
                    ModelType::Sink,
                    {
                        "server", false, {}, {},
                        {},
                        {{ModelType::Emotionnet, -1}, {ModelType::Age, -1}, {ModelType::Gender, -1}, {ModelType::Arcface, -1}}
                    }
                }
            };
        case PipelineType::Building_Security:
            return {
                {
                    ModelType::DataSource,
                    {startDevice, true, {}, {}, {{ModelType::Yolov5n, 0}}}
                },
                {
                    ModelType::Yolov5n,
                    {
                        "server", true, {}, {},
                        {{ModelType::Retinaface, 0}},
                        {{ModelType::DataSource, -1}}
                    }
                },
                {
                    ModelType::Retinaface,
                    {
                        "server", false, {}, {},
                        {{ModelType::Gender,     -1}, {ModelType::Age, -1}},
                        {{ModelType::Yolov5n, -1}}
                    }
                },
                {
                    ModelType::Movenet,
                    {
                        "server", false, {}, {},
                        {{ModelType::Sink,   -1}},
                        {{ModelType::Yolov5n, -1}}
                    }
                },
                {
                    ModelType::Gender,
                    {
                        "server", false, {}, {},
                        {{ModelType::Sink,   -1}},
                        {{ModelType::Retinaface, -1}}
                    }
                },
                {
                    ModelType::Age,
                    {
                        "server", false, {}, {},
                        {{ModelType::Sink,   -1}},
                        {{ModelType::Retinaface, -1}}
                    }
                },
                {
                    ModelType::Sink,
                    {
                        "server", false, {}, {},
                        {},
                        {{ModelType::Age, -1}, {ModelType::Gender, -1}, {ModelType::Movenet, -1}}
                    }
                }
            };
        default:
            return {};
    }
}

double Controller::LoadTimeEstimator(const char *model_path, double input_mem_size) {
    // Load the pre-trained model
    BoosterHandle booster;
    int num_iterations = 1;
    int ret = LGBM_BoosterCreateFromModelfile(model_path, &num_iterations, &booster);

    // Prepare the input data
    std::vector<double> input_data = {input_mem_size};

    // Perform inference
    int64_t out_len;
    std::vector<double> out_result(1);
    ret = LGBM_BoosterPredictForMat(booster,
                                    input_data.data(),
                                    C_API_DTYPE_FLOAT64,
                                    1,  // Number of rows
                                    1,  // Number of columns
                                    1,  // Is row major
                                    C_API_PREDICT_NORMAL,  // Predict type
                                    0,  // Start iteration
                                    -1,  // Number of iterations, -1 means use all
                                    "",  // Parameter
                                    &out_len,
                                    out_result.data());
    if (ret != 0) {
        std::cout << "Failed to perform inference!" << std::endl;
        exit(ret);
    }

    // Print the predicted value
    std::cout << "Predicted value: " << out_result[0] << std::endl;

    // Free the booster handle
    LGBM_BoosterFree(booster);

    return out_result[0];
}


/**
 * @brief
 *
 * @param model to specify model
 * @param batch_size for targeted batch size (binary)
 * @return int for inference time per full batch in nanoseconds
 */
int Controller::InferTimeEstimator(ModelType model, int batch_size) {
    // std::map<int, int> time_per_frame;
    // switch (model) {
    //     case ModelType::Yolov5n:
    //         time_per_frame = {{1,  3602348},
    //                           {2,  2726377},
    //                           {4,  2467065},
    //                           {8,  2575456},
    //                           {16, 3220761},
    //                           {32, 4680154},
    //                           {64, 7773959}};
    //         break;
    //     case ModelType::Yolov5n416:
    //         time_per_frame = {{1,  2649396},
    //                           {2,  2157968},
    //                           {4,  1897505},
    //                           {8,  2076971},
    //                           {16, 2716276},
    //                           {32, 4172530},
    //                           {64, 7252059}};
    //         break;
    //     case ModelType::Yolov5s:
    //         time_per_frame = {{1,  4515118},
    //                           {2,  3399807},
    //                           {4,  3044100},
    //                           {8,  3008503},
    //                           {16, 3672566},
    //                           {32, 5116321},
    //                           {64, 8237824}};
    //         break;
    //     case ModelType::Yolov5m:
    //         time_per_frame = {{1,  7263238},
    //                           {2,  5905167},
    //                           {4,  4446144},
    //                           {8,  4449675},
    //                           {16, 4991818},
    //                           {32, 6543270},
    //                           {64, 9579015}};
    //         break;
    //     case ModelType::Yolov5nDsrc:
    //         time_per_frame = {{1,  3602348},
    //                           {2,  2726377},
    //                           {4,  2467065},
    //                           {8,  2575456},
    //                           {16, 3220761},
    //                           {32, 4680154},
    //                           {64, 7773959}};
    //         break;
    //     case ModelType::Retinaface:
    //         time_per_frame = {{1,  1780280},
    //                           {2,  1527410},
    //                           {4,  1357906},
    //                           {8,  1164929},
    //                           {16, 2177011},
    //                           {32, 3399701},
    //                           {64, 8146690}};
    //         break;
    //     case ModelType::CarBrand:
    //         time_per_frame = {{1,  4998407},
    //                           {2,  3335101},
    //                           {4,  2344440},
    //                           {8,  2176385},
    //                           {16, 2483317},
    //                           {32, 2357686},
    //                           {64, 1155050}};
    //         break;
    //     case ModelType::PlateDet:
    //         time_per_frame = {{1,  7304176},
    //                           {2,  4909581},
    //                           {4,  3225549},
    //                           {8,  2883803},
    //                           {16, 2871236},
    //                           {32, 2004165},
    //                           {64, 3094331}};
    //         break;
    //     case ModelType::Movenet:
    //         time_per_frame = {{1,  1644526},
    //                           {2,  3459537},
    //                           {4,  2703916},
    //                           {8,  2377614},
    //                           {16, 2647643},
    //                           {32, 2900894},
    //                           {64, 2197719}};
    //         break;
    //     case ModelType::Arcface:
    //         time_per_frame = {{1,  18120029},
    //                           {2,  11226197},
    //                           {4,  7883673},
    //                           {8,  6364369},
    //                           {16, 5620677},
    //                           {32, 3370018},
    //                           {64, 3206726}};
    //         break;
    //     case ModelType::Emotionnet:
    //         time_per_frame = {{1,  3394144},
    //                           {2,  1365037},
    //                           {4,  1615653},
    //                           {8,  1967143},
    //                           {16, 1500867},
    //                           {32, 1665680},
    //                           {64, 1957914}};
    //         break;
    //     case ModelType::Age:
    //         time_per_frame = {{1,  14729041},
    //                           {2,  9050828},
    //                           {4,  6112501},
    //                           {8,  5015442},
    //                           {16, 3927934},
    //                           {32, 3523500},
    //                           {64, 2899034}};
    //         break;
    //     case ModelType::Gender:
    //         time_per_frame = {{1,  1357500},
    //                           {2,  831649},
    //                           {4,  687484},
    //                           {8,  749792},
    //                           {16, 1021500},
    //                           {32, 1800263},
    //                           {64, 4002824}};
    //         break;
    //     default:
    //         return 0;
    // }
    // int i = 1;
    // while (i < batch_size) {
    //     i *= 2;
    // }
    // return time_per_frame[batch_size] * batch_size;
    return 0;
}

std::map<ModelType, std::vector<int>> Controller::InitialRequestCount(const std::string &input, const Pipeline &models,
                                                                      int fps) {
    std::map<ModelType, std::vector<int>> request_counts = {};
    std::vector<int> fps_values = {fps, fps * 3, fps * 7, fps * 15, fps * 30, fps * 60};

    request_counts[models[0].first] = fps_values;
    json objectCount = json::parse(std::ifstream("../jsons/object_count.json"))[input];

    for (const auto &m: models) {
        if (m.first == ModelType::Sink) {
            request_counts[m.first] = std::vector<int>(6, 0);
            continue;
        }

        for (const auto &d: m.second) {
            if (d.second == -1) {
                request_counts[d.first] = request_counts[m.first];
            } else {
                std::vector<int> objects = (d.second == 0 ? objectCount["person"]
                                                          : objectCount["car"]).get<std::vector<int>>();

                for (int j: fps_values) {
                    int count = std::accumulate(objects.begin(), objects.begin() + j, 0);
                    request_counts[d.first].push_back(request_counts[m.first][0] * count);
                }
            }
        }
    }
    return request_counts;
}