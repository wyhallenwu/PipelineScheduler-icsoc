#include "controller.h"

void TaskDescription::to_json(nlohmann::json &j, const TaskDescription::TaskStruct &val) {
    j = json{{"name",   val.name},
             {"slo",    val.slo},
             {"type",   val.type},
             {"source", val.source},
             {"device", val.device}};
}

void TaskDescription::from_json(const nlohmann::json &j, TaskDescription::TaskStruct &val) {
    j.at("name").get_to(val.name);
    j.at("slo").get_to(val.slo);
    j.at("type").get_to(val.type);
    j.at("source").get_to(val.source);
    j.at("device").get_to(val.device);
}

Controller::Controller() {
    running = true;
    devices = std::map<std::string, NodeHandle>();
    tasks = std::map<std::string, TaskHandle>();
    containers = std::map<std::string, ContainerHandle>();

    std::string server_address = absl::StrFormat("%s:%d", "localhost", 60001);
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
    new LightMetricsRequestHandler(&service, cq.get(), this);
    new FullMetricsRequestHandler(&service, cq.get(), this);
    while (running) {
        void *tag;
        bool ok;
        if (!cq->Next(&tag, &ok)) {
            break;
        }
        static_cast<RequestHandler *>(tag)->Proceed();
    }
}

void Controller::AddTask(const TaskDescription::TaskStruct &t) {
    std::cout << "Adding task: " << t.name << std::endl;
    tasks.insert({t.name, {t.slo, t.type, {}}});
    TaskHandle *task = &tasks[t.name];
    NodeHandle *device = &devices[t.device];
    auto models = getModelsByPipelineType(t.type);

    std::string tmp = t.name;
    containers.insert({tmp.append(":datasource"), {tmp, DataSource, device, task, 33, 0, -1}});
    task->subtasks.insert({tmp, &containers[tmp]});
    task->subtasks[tmp]->recv_port = device->next_free_port++;
    device->containers.insert({tmp, task->subtasks[tmp]});
    device = &devices["server"];

    std::map<std::string, int> batch_sizes = getInitialBatchSizes(models, t.slo, 10);
    for (const auto &m: models) {
        tmp = t.name;
        // TODO: get correct initial batch sizes and cuda devices based on TaskDescription and System State
        int cuda_device = 1;
        containers.insert(
                {tmp.append(m.first), {tmp, MODEL_TYPES[m.first], device, task, batch_sizes[m.first], cuda_device,
                                       -1, device->next_free_port++, {}, {}, {}, {}}});
        task->subtasks.insert({tmp, &containers[tmp]});
        device->containers.insert({tmp, task->subtasks[tmp]});
    }

    task->subtasks[t.name + ":datasource"]->downstreams.push_back(task->subtasks[t.name + models[0].first]);
    task->subtasks[t.name + models[0].first]->upstreams.push_back(task->subtasks[t.name + ":datasource"]);
    for (const auto &m: models) {
        for (const auto &d: m.second) {
            tmp = t.name;
            task->subtasks[tmp.append(d.first)]->class_of_interest = d.second;
            task->subtasks[tmp]->upstreams.push_back(task->subtasks[t.name + m.first]);
            task->subtasks[t.name + m.first]->downstreams.push_back(task->subtasks[tmp]);
        }
    }

    for (std::pair<std::string, ContainerHandle *> msvc: task->subtasks) {
        StartContainer(msvc, task->slo, t.source);
    }
}

void Controller::UpdateLightMetrics(google::protobuf::RepeatedPtrField<LightMetrics> metrics) {
    for (auto metric: metrics) {
        containers[metric.name()].queue_lengths = metric.queue_size();
        containers[metric.name()].metrics.requestRate = metric.request_rate();
    }
}

void Controller::UpdateFullMetrics(google::protobuf::RepeatedPtrField<FullMetrics> metrics) {
    for (auto metric: metrics) {
        containers[metric.name()].queue_lengths = metric.queue_size();
        Metrics *m = &containers[metric.name()].metrics;
        m->requestRate = metric.request_rate();
        m->cpuUsage = metric.cpu_usage();
        m->memUsage = metric.mem_usage();
        m->gpuUsage = metric.gpu_usage();
        m->gpuMemUsage = metric.gpu_mem_usage();
    }
}

void Controller::LightMetricsRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestSendLightMetrics(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new LightMetricsRequestHandler(service, cq, controller);
        controller->UpdateLightMetrics(request.metrics());
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void Controller::FullMetricsRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestSendFullMetrics(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new FullMetricsRequestHandler(service, cq, controller);
        controller->UpdateFullMetrics(request.metrics());
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void Controller::DeviseAdvertisementHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestAdvertiseToController(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new DeviseAdvertisementHandler(service, cq, controller);
        std::string target_str = absl::StrFormat("%s:%d", request.ip_address(), 60002);
        controller->devices.insert({request.device_name(),
                                    {request.ip_address(),
                                     ControlCommunication::NewStub(
                                             grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials())),
                                     new CompletionQueue(),
                                     static_cast<DeviceType>(request.device_type()),
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

void Controller::StartContainer(std::pair<std::string, ContainerHandle *> &container, int slo, std::string source) {
    std::cout << "Starting container: " << container.first << std::endl;
    ContainerConfig request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_name(container.first);
    request.set_model(container.second->model);
    request.set_batch_size(container.second->batch_size);
    request.set_recv_port(container.second->recv_port);
    request.set_slo(slo);
    request.set_device(container.second->cuda_device);
    for (auto dwnstr: container.second->downstreams) {
        Neighbor *dwn = request.add_downstream();
        dwn->set_name(dwnstr->name);
        dwn->set_ip(absl::StrFormat("%s:%d", dwnstr->device_agent->ip, dwnstr->recv_port));
        dwn->set_class_of_interest(dwnstr->class_of_interest);
        dwn->set_gpu_connection((container.second->device_agent == dwnstr->device_agent) &&
                                (container.second->cuda_device == dwnstr->cuda_device));
    }
    if (request.downstream_size() == 0) {
        Neighbor *dwn = request.add_downstream();
        dwn->set_name("video_sink");
        dwn->set_ip("./out.log"); //output log file
        dwn->set_class_of_interest(-1);
        dwn->set_gpu_connection(false);
    }
    if (container.second->model == DataSource) {
        Neighbor *up = request.add_upstream();
        up->set_name("video_source");
        up->set_ip(source);
        up->set_class_of_interest(-1);
        up->set_gpu_connection(false);
    } else {
        for (auto upstr: container.second->upstreams) {
            Neighbor *up = request.add_upstream();
            up->set_name(upstr->name);
            up->set_ip(absl::StrFormat("%s:%d", upstr->device_agent->ip, upstr->recv_port));
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

void Controller::MoveContainer(ContainerHandle *msvc, int cuda_device, bool to_edge) {
    NodeHandle *old_device = msvc->device_agent;
    NodeHandle *device;
    if (to_edge) {
        device = msvc->upstreams[0]->device_agent;
    } else {
        device = &devices["server"];
    }
    msvc->device_agent = device;
    msvc->recv_port = device->next_free_port++;
    device->containers.insert({msvc->name, msvc});
    msvc->cuda_device = cuda_device;
    std::pair<std::string, ContainerHandle *> pair = {msvc->name, msvc};
    StartContainer(pair, msvc->task->slo, "");
    for (auto upstr: msvc->upstreams) {
        AdjustUpstream(msvc->recv_port, upstr, device, msvc->name);
    }
    StopContainer(msvc->name, old_device);
    old_device->containers.erase(msvc->name);
}

void Controller::AdjustUpstream(int port, Controller::ContainerHandle *upstr, Controller::NodeHandle *new_device,
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

void Controller::AdjustBatchSize(Controller::ContainerHandle *msvc, int new_bs) {
    msvc->batch_size = new_bs;
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
        const std::vector<std::pair<std::string, std::vector<std::pair<std::string, int>>>> &models,
        std::map<std::string, int> &batch_sizes, std::map<std::string, int> &estimated_infer_times, int nObjects) {
    std::string candidate;
    int max_saving = 0;
    std::vector<std::string> blacklist;
    for (const auto &m: models) {
        std::cout << "Model: " << m.first << " : " << batch_sizes[m.first] << std::endl;
        int saving;
        if (max_saving == 0) {
            saving = estimated_infer_times[m.first] - InferTimeEstimator(MODEL_TYPES[m.first], batch_sizes[m.first] * 2);
        } else {
            if (std::find(blacklist.begin(), blacklist.end(), m.first) != blacklist.end()) {
                continue;
            }
            for (const auto &d: m.second) {
                std::cout << "Downstream: " << batch_sizes[d.first] << std::endl;
                if (batch_sizes[d.first] > (batch_sizes[m.first])) {
                    blacklist.push_back(d.first);
                }
            }
            saving = estimated_infer_times[m.first] -
                     (InferTimeEstimator(MODEL_TYPES[m.first], batch_sizes[m.first] * 2) * nObjects);
        }
        if (saving > max_saving) {
            max_saving = saving;
            candidate = m.first;
        }
    }
    std::cout << "Optimizing batch size for: " << candidate << std::endl;
    batch_sizes[candidate] *= 2;
    estimated_infer_times[candidate] -= max_saving;
}

std::map<std::string, int> Controller::getInitialBatchSizes(
        const std::vector<std::pair<std::string, std::vector<std::pair<std::string, int>>>> &models, int slo,
        int nObjects) {
    std::map<std::string, int> batch_sizes = {};
    std::map<std::string, int> estimated_infer_times = {};

    for (const auto &m: models) {
        batch_sizes[m.first] = 1;
        if (estimated_infer_times.size() == 0) {
            estimated_infer_times[m.first] = (InferTimeEstimator(MODEL_TYPES[m.first], 1));
        } else {
            estimated_infer_times[m.first] = (InferTimeEstimator(MODEL_TYPES[m.first], 1) * nObjects);
        }
    }

    int sum = std::accumulate(estimated_infer_times.begin(), estimated_infer_times.end(), 0,
                              [](int acc, const std::pair<std::string, int>& p) {
                                  return acc + p.second;
                              });

    while (slo < sum) {
        std::cout << "Optimizing batch sizes: " << sum << std::endl;
        optimizeBatchSizeStep(models, batch_sizes, estimated_infer_times, nObjects);
        sum = std::accumulate(estimated_infer_times.begin(), estimated_infer_times.end(), 0,
                              [](int acc, const std::pair<std::string, int>& p) {
                                  return acc + p.second;
                              });
        std::cout << "Batch sizes: " << sum << std::endl;
    }
    optimizeBatchSizeStep(models, batch_sizes, estimated_infer_times, nObjects);
    for (const auto &m: models) {
        std::cout << m.first << ": " << batch_sizes[m.first] << std::endl;
    }
    return batch_sizes;
}

std::vector<std::pair<std::string, std::vector<std::pair<std::string, int>>>>
Controller::getModelsByPipelineType(PipelineType type) {
    switch (type) {
        case PipelineType::Traffic:
            return {{":yolov5",     {{":retinaface", 0}, {":carbrand", 2}, {":plate", 2}}},
                    {":retinaface", {{":arcface",    -1}}},
                    {":arcface",    {{":basesink",   -1}}},
                    {":carbrand",   {{":basesink",   -1}}},
                    {":plate",      {{":basesink",   -1}}},
                    {":basesink",   {}}};
        case PipelineType::Video_Call:
            return {{":retinaface", {{":emotion",  -1}, {":age", -1}, {":gender", -1}, {":arcface", -1}}},
                    {":gender",     {{":basesink", -1}}},
                    {":age",        {{":basesink", -1}}},
                    {":emotion",    {{":basesink", -1}}},
                    {":arcface",    {{":basesink", -1}}},
                    {":basesink",   {}}};
        case PipelineType::Building_Security:
            return {{":yolov5",     {{":retinaface", 0}}},
                    {":retinaface", {{":gender",     -1}, {":age", -1}}},
                    {":movenet",    {{":basesink",   -1}}},
                    {"gender",      {{":basesink",   -1}}},
                    {":age",        {{":basesink",   -1}}},
                    {":basesink",   {}}};
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

// returns the profiling results for inference time per frame in a full batch in nanoseconds
int Controller::InferTimeEstimator(ModelType model, int batch_size) {
    std::map<int, int> time_per_frame;
    switch (model) {
        case ModelType::Yolov5:
            time_per_frame = {{1,  3602348},
                              {2,  2726377},
                              {4,  2467065},
                              {8,  2575456},
                              {16, 3220761},
                              {32, 4680154},
                              {64, 7773959}};
            break;
        case ModelType::Yolov5Datasource:
            time_per_frame = {{1,  3602348},
                              {2,  2726377},
                              {4,  2467065},
                              {8,  2575456},
                              {16, 3220761},
                              {32, 4680154},
                              {64, 7773959}};
            break;
        case ModelType::Retinaface:
            time_per_frame = {{1,  1780280},
                              {2,  1527410},
                              {4,  1357906},
                              {8,  1164929},
                              {16, 2177011},
                              {32, 3399701},
                              {64, 8146690}};
            break;
        case ModelType::CarBrand:
            time_per_frame = {{1,  4998407},
                              {2,  3335101},
                              {4,  2344440},
                              {8,  2176385},
                              {16, 2483317},
                              {32, 2357686},
                              {64, 1155050}};
            break;
        case ModelType::Yolov5_Plate:
            time_per_frame = {{1,  7304176},
                              {2,  4909581},
                              {4,  3225549},
                              {8,  2883803},
                              {16, 2871236},
                              {32, 2004165},
                              {64, 3094331}};
            break;
        case ModelType::Movenet:
            time_per_frame = {{1,  1644526},
                              {2,  3459537},
                              {4,  2703916},
                              {8,  2377614},
                              {16, 2647643},
                              {32, 2900894},
                              {64, 2197719}};
            break;
        case ModelType::Arcface:
            time_per_frame = {{1,  18120029},
                              {2,  11226197},
                              {4,  7883673},
                              {8,  6364369},
                              {16, 5620677},
                              {32, 3370018},
                              {64, 3206726}};
            break;
        case ModelType::Emotionnet:
            time_per_frame = {{1,  3394144},
                              {2,  1365037},
                              {4,  1615653},
                              {8,  1967143},
                              {16, 1500867},
                              {32, 1665680},
                              {64, 1957914}};
            break;
        case ModelType::Age:
            time_per_frame = {{1,  14729041},
                              {2,  9050828},
                              {4,  6112501},
                              {8,  5015442},
                              {16, 3927934},
                              {32, 3523500},
                              {64, 2899034}};
            break;
        case ModelType::Gender:
            time_per_frame = {{1,  1357500},
                              {2,  831649},
                              {4,  687484},
                              {8,  749792},
                              {16, 1021500},
                              {32, 1800263},
                              {64, 4002824}};
            break;
        default:
            return 0;
    }
    int i = 1;
    while (i < batch_size) {
        i *= 2;
    }
    return time_per_frame[batch_size];
}

int main() {
    auto controller = new Controller();
    std::thread receiver_thread(&Controller::HandleRecvRpcs, controller);
    receiver_thread.detach();
    std::ifstream file("../jsons/experiment.json");
    std::vector<TaskDescription::TaskStruct> tasks = json::parse(file);
    std::string command;

    while (controller->isRunning()) {
        TaskDescription::TaskStruct task;
        std::cout << "Enter command {init, traffic, video_call, people, exit): ";
        std::cin >> command;
        if (command == "exit") {
            controller->Stop();
            break;
        } else if (command == "init") {
            for (auto &t: tasks) {
                controller->AddTask(t);
            }
            continue;
        } else if (command == "traffic") {
            task.type = PipelineType::Traffic;
        } else if (command == "video_call") {
            task.type = PipelineType::Video_Call;
        } else if (command == "people") {
            task.type = PipelineType::Building_Security;
        } else {
            std::cout << "Invalid command" << std::endl;
            continue;
        }
        std::cout << "Enter name of task: ";
        std::cin >> task.name;
        std::cout << "Enter SLO in ns: ";
        std::cin >> task.slo;
        std::cout << "Enter total path to source file: ";
        std::cin >> task.source;
        std::cout << "Enter name of source device: ";
        std::cin >> task.device;
        controller->AddTask(task);
    }
    delete controller;
    return 0;
}
