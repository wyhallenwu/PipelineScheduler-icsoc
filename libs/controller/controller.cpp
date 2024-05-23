#include "controller.h"

std::map<ModelType, std::vector<std::string>> MODEL_INFO = {
    {DataSource, {":datasource", "./Container_DataSource"}},
    {Sink, {":basesink", "./runSink"}},
    {Yolov5, {":yolov5", "./Container_Yolov5"}},
    {Yolov5Datasource, {":yolov5datasource", "./Container_Yolov5"}},
    {Retinaface, {":retinaface", "./Container_RetinaFace"}},
    {Yolov5_Plate, {":platedetection", "./Container_Yolov5-plate"}},
    {Movenet, {":movenet", "./Container_MoveNet"}},
    {Emotionnet, {":emotionnet", "./Container_EmotionNet"}},
    {Arcface, {":arcface", "./Container_ArcFace"}},
    {Age, {":age", "./Container_Age"}},
    {Gender, {":gender", "./Container_Gender"}},
    {CarBrand, {":carbrand", "./Container_CarBrand"}},
};

void TaskDescription::to_json(nlohmann::json &j, const TaskDescription::TaskStruct &val)
{
    j = json{{"name", val.name},
             {"slo", val.slo},
             {"type", val.type},
             {"source", val.source},
             {"device", val.device}};
}

void TaskDescription::from_json(const nlohmann::json &j, TaskDescription::TaskStruct &val)
{
    j.at("name").get_to(val.name);
    j.at("slo").get_to(val.slo);
    j.at("type").get_to(val.type);
    j.at("source").get_to(val.source);
    j.at("device").get_to(val.device);
}

Controller::Controller()
{
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

Controller::~Controller()
{
    for (auto &msvc : containers)
    {
        StopContainer(msvc.first, msvc.second.device_agent, true);
    }
    for (auto &device : devices)
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
    while (running)
    {
        void *tag;
        bool ok;
        if (!cq->Next(&tag, &ok))
        {
            break;
        }
        static_cast<RequestHandler *>(tag)->Proceed();
    }
}

void Controller::AddTask(const TaskDescription::TaskStruct &t)
{
    std::cout << "Adding task: " << t.name << std::endl;
    tasks.insert({t.name, {t.slo, t.type, {}}});
    TaskHandle *task = &tasks[t.name];
    NodeHandle *device = &devices[t.device];
    auto models = getModelsByPipelineType(t.type);

    std::string tmp = t.name;
    containers.insert({tmp.append(":datasource"), {tmp, DataSource, device, task, 33, 1, {-1}}});
    task->subtasks.insert({tmp, &containers[tmp]});
    task->subtasks[tmp]->recv_port = device->next_free_port++;
    device->containers.insert({tmp, task->subtasks[tmp]});
    device = &devices["server"];

    auto batch_sizes = getInitialBatchSizes(models, t.slo, 10);
    for (const auto &m : models)
    {
        tmp = t.name;
        // TODO: get correct initial cuda devices based on TaskDescription and System State
        int cuda_device = 1;
        containers.insert(
            {tmp.append(MODEL_INFO[m.first][0]), {tmp, m.first, device, task, batch_sizes[m.first], 1, {cuda_device}, -1, device->next_free_port++, {}, {}, {}, {}}});
        task->subtasks.insert({tmp, &containers[tmp]});
        device->containers.insert({tmp, task->subtasks[tmp]});
    }

    task->subtasks[t.name + ":datasource"]->downstreams.push_back(task->subtasks[t.name + MODEL_INFO[models[0].first][0]]);
    task->subtasks[t.name + MODEL_INFO[models[0].first][0]]->upstreams.push_back(task->subtasks[t.name + ":datasource"]);
    for (const auto &m : models)
    {
        for (const auto &d : m.second)
        {
            tmp = t.name;
            task->subtasks[tmp.append(MODEL_INFO[d.first][0])]->class_of_interest = d.second;
            task->subtasks[tmp]->upstreams.push_back(task->subtasks[t.name + MODEL_INFO[m.first][0]]);
            task->subtasks[t.name + MODEL_INFO[m.first][0]]->downstreams.push_back(task->subtasks[tmp]);
        }
    }

    for (std::pair<std::string, ContainerHandle *> msvc : task->subtasks)
    {
        StartContainer(msvc, task->slo, t.source);
    }
}

void Controller::UpdateLightMetrics()
{
    // TODO: Replace with Database Scraping
    //    for (auto metric: metrics) {
    //        containers[metric.name()].queue_lengths = metric.queue_size();
    //        containers[metric.name()].metrics.requestRate = metric.request_rate();
    //    }
}

void Controller::UpdateFullMetrics()
{
    // TODO: Replace with Database Scraping
    //    for (auto metric: metrics) {
    //        containers[metric.name()].queue_lengths = metric.queue_size();
    //        Metrics *m = &containers[metric.name()].metrics;
    //        m->requestRate = metric.request_rate();
    //        m->cpuUsage = metric.cpu_usage();
    //        m->memUsage = metric.mem_usage();
    //        m->gpuUsage = metric.gpu_usage();
    //        m->gpuMemUsage = metric.gpu_mem_usage();
    //    }
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
        std::string target_str = absl::StrFormat("%s:%d", request.ip_address(), 60002);
        controller->devices.insert({request.device_name(),
                                    {request.ip_address(),
                                     ControlCommunication::NewStub(
                                         grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials())),
                                     new CompletionQueue(),
                                     static_cast<SystemDeviceType>(request.device_type()),
                                     request.processors(),
                                     std::vector<double>(request.processors(), 0.0),
                                     std::vector<unsigned long>(request.memory().begin(), request.memory().end()),
                                     std::vector<double>(request.processors(), 0.0),
                                     55001,
                                     {}}});
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    }
    else
    {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void Controller::StartContainer(std::pair<std::string, ContainerHandle *> &container, int slo, std::string source,
                                int replica)
{
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
    request.set_device(container.second->cuda_device[replica - 1]);
    for (auto dwnstr : container.second->downstreams)
    {
        Neighbor *dwn = request.add_downstream();
        dwn->set_name(dwnstr->name);
        dwn->set_ip(absl::StrFormat("%s:%d", dwnstr->device_agent->ip, dwnstr->recv_port));
        dwn->set_class_of_interest(dwnstr->class_of_interest);
        dwn->set_gpu_connection((container.second->device_agent == dwnstr->device_agent) &&
                                (container.second->cuda_device == dwnstr->cuda_device));
    }
    if (request.downstream_size() == 0)
    {
        Neighbor *dwn = request.add_downstream();
        dwn->set_name("video_sink");
        dwn->set_ip("./out.log"); // output log file
        dwn->set_class_of_interest(-1);
        dwn->set_gpu_connection(false);
    }
    if (container.second->model == DataSource)
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
            up->set_ip(absl::StrFormat("%s:%d", upstr->device_agent->ip, upstr->recv_port));
            up->set_class_of_interest(-2);
            up->set_gpu_connection((container.second->device_agent == upstr->device_agent) &&
                                   (container.second->cuda_device == upstr->cuda_device));
        }
    }
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
        container.second->device_agent->stub->AsyncStartContainer(&context, request,
                                                                  container.second->device_agent->cq));
    rpc->Finish(&reply, &status, (void *)1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(container.second->device_agent->cq->Next(&got_tag, &ok));
    GPR_ASSERT(ok);
    if (!status.ok())
    {
        std::cout << status.error_code() << ": An error occured while sending the request" << std::endl;
    }
}

void Controller::MoveContainer(ContainerHandle *msvc, int cuda_device, bool to_edge, int replica)
{
    NodeHandle *old_device = msvc->device_agent;
    NodeHandle *device;
    if (to_edge)
    {
        device = msvc->upstreams[0]->device_agent;
    }
    else
    {
        device = &devices["server"];
    }
    msvc->device_agent = device;
    msvc->recv_port = device->next_free_port++;
    device->containers.insert({msvc->name, msvc});
    msvc->cuda_device[replica - 1] = cuda_device;
    std::pair<std::string, ContainerHandle *> pair = {msvc->name, msvc};
    StartContainer(pair, msvc->task->slo, "");
    for (auto upstr : msvc->upstreams)
    {
        AdjustUpstream(msvc->recv_port, upstr, device, msvc->name);
    }
    StopContainer(msvc->name, old_device);
    old_device->containers.erase(msvc->name);
}

void Controller::AdjustUpstream(int port, Controller::ContainerHandle *upstr, Controller::NodeHandle *new_device,
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
    rpc->Finish(&reply, &status, (void *)1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(upstr->device_agent->cq->Next(&got_tag, &ok));
    GPR_ASSERT(ok);
}

void Controller::AdjustBatchSize(Controller::ContainerHandle *msvc, int new_bs)
{
    msvc->batch_size = new_bs;
    ContainerInt request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_name(msvc->name);
    request.set_value(new_bs);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
        msvc->device_agent->stub->AsyncUpdateBatchSize(&context, request, msvc->device_agent->cq));
    rpc->Finish(&reply, &status, (void *)1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(msvc->device_agent->cq->Next(&got_tag, &ok));
    GPR_ASSERT(ok);
}

void Controller::StopContainer(std::string name, NodeHandle *device, bool forced)
{
    ContainerSignal request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_name(name);
    request.set_forced(forced);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
        device->stub->AsyncStopContainer(&context, request, containers[name].device_agent->cq));
    rpc->Finish(&reply, &status, (void *)1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(device->cq->Next(&got_tag, &ok));
    GPR_ASSERT(ok);
}

void Controller::optimizeBatchSizeStep(
    const std::vector<std::pair<ModelType, std::vector<std::pair<ModelType, int>>>> &models,
    std::map<ModelType, int> &batch_sizes, std::map<ModelType, int> &estimated_infer_times, int nObjects)
{
    ModelType candidate;
    int max_saving = 0;
    std::vector<ModelType> blacklist;
    for (const auto &m : models)
    {
        int saving;
        if (max_saving == 0)
        {
            saving =
                estimated_infer_times[m.first] - InferTimeEstimator(m.first, batch_sizes[m.first] * 2);
        }
        else
        {
            if (batch_sizes[m.first] == 64 ||
                std::find(blacklist.begin(), blacklist.end(), m.first) != blacklist.end())
            {
                continue;
            }
            for (const auto &d : m.second)
            {
                if (batch_sizes[d.first] > batch_sizes[m.first])
                {
                    blacklist.push_back(d.first);
                }
            }
            saving = estimated_infer_times[m.first] -
                     (InferTimeEstimator(m.first, batch_sizes[m.first] * 2) * (nObjects / batch_sizes[m.first] * 2));
        }
        if (saving > max_saving)
        {
            max_saving = saving;
            candidate = m.first;
        }
    }
    batch_sizes[candidate] *= 2;
    estimated_infer_times[candidate] -= max_saving;
}

std::map<ModelType, int> Controller::getInitialBatchSizes(
    const std::vector<std::pair<ModelType, std::vector<std::pair<ModelType, int>>>> &models, int slo,
    int nObjects)
{
    std::map<ModelType, int> batch_sizes = {};
    std::map<ModelType, int> estimated_infer_times = {};

    for (const auto &m : models)
    {
        batch_sizes[m.first] = 1;
        if (estimated_infer_times.size() == 0)
        {
            estimated_infer_times[m.first] = (InferTimeEstimator(m.first, 1));
        }
        else
        {
            estimated_infer_times[m.first] = (InferTimeEstimator(m.first, 1) * nObjects);
        }
    }

    int sum = std::accumulate(estimated_infer_times.begin(), estimated_infer_times.end(), 0,
                              [](int acc, const std::pair<ModelType, int> &p)
                              {
                                  return acc + p.second;
                              });

    while (slo < sum)
    {
        optimizeBatchSizeStep(models, batch_sizes, estimated_infer_times, nObjects);
        sum = std::accumulate(estimated_infer_times.begin(), estimated_infer_times.end(), 0,
                              [](int acc, const std::pair<ModelType, int> &p)
                              {
                                  return acc + p.second;
                              });
    }
    optimizeBatchSizeStep(models, batch_sizes, estimated_infer_times, nObjects);
    return batch_sizes;
}

std::vector<std::pair<ModelType, std::vector<std::pair<ModelType, int>>>>
Controller::getModelsByPipelineType(PipelineType type)
{
    switch (type)
    {
    case PipelineType::Traffic:
        return {{ModelType::Yolov5, {{ModelType::Retinaface, 0}, {ModelType::CarBrand, 2}, {ModelType::Yolov5_Plate, 2}}},
                {ModelType::Retinaface, {{ModelType::Arcface, -1}}},
                {ModelType::Arcface, {{ModelType::Sink, -1}}},
                {ModelType::CarBrand, {{ModelType::Sink, -1}}},
                {ModelType::Yolov5_Plate, {{ModelType::Sink, -1}}},
                {ModelType::Sink, {}}};
    case PipelineType::Video_Call:
        return {{ModelType::Retinaface, {{ModelType::Emotionnet, -1}, {ModelType::Age, -1}, {ModelType::Gender, -1}, {ModelType::Arcface, -1}}},
                {ModelType::Gender, {{ModelType::Sink, -1}}},
                {ModelType::Age, {{ModelType::Sink, -1}}},
                {ModelType::Emotionnet, {{ModelType::Sink, -1}}},
                {ModelType::Arcface, {{ModelType::Sink, -1}}},
                {ModelType::Sink, {}}};
    case PipelineType::Building_Security:
        return {{ModelType::Yolov5, {{ModelType::Retinaface, 0}}},
                {ModelType::Retinaface, {{ModelType::Gender, -1}, {ModelType::Age, -1}}},
                {ModelType::Movenet, {{ModelType::Sink, -1}}},
                {ModelType::Gender, {{ModelType::Sink, -1}}},
                {ModelType::Age, {{ModelType::Sink, -1}}},
                {ModelType::Sink, {}}};
    default:
        return {};
    }
}

double Controller::LoadTimeEstimator(const char *model_path, double input_mem_size)
{
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
                                    1,                    // Number of rows
                                    1,                    // Number of columns
                                    1,                    // Is row major
                                    C_API_PREDICT_NORMAL, // Predict type
                                    0,                    // Start iteration
                                    -1,                   // Number of iterations, -1 means use all
                                    "",                   // Parameter
                                    &out_len,
                                    out_result.data());
    if (ret != 0)
    {
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
int Controller::InferTimeEstimator(ModelType model, int batch_size)
{
    std::map<int, int> time_per_frame;
    switch (model)
    {
    case ModelType::Yolov5:
        time_per_frame = {{1, 3602348},
                          {2, 2726377},
                          {4, 2467065},
                          {8, 2575456},
                          {16, 3220761},
                          {32, 4680154},
                          {64, 7773959}};
        break;
    case ModelType::Yolov5Datasource:
        time_per_frame = {{1, 3602348},
                          {2, 2726377},
                          {4, 2467065},
                          {8, 2575456},
                          {16, 3220761},
                          {32, 4680154},
                          {64, 7773959}};
        break;
    case ModelType::Retinaface:
        time_per_frame = {{1, 1780280},
                          {2, 1527410},
                          {4, 1357906},
                          {8, 1164929},
                          {16, 2177011},
                          {32, 3399701},
                          {64, 8146690}};
        break;
    case ModelType::CarBrand:
        time_per_frame = {{1, 4998407},
                          {2, 3335101},
                          {4, 2344440},
                          {8, 2176385},
                          {16, 2483317},
                          {32, 2357686},
                          {64, 1155050}};
        break;
    case ModelType::Yolov5_Plate:
        time_per_frame = {{1, 7304176},
                          {2, 4909581},
                          {4, 3225549},
                          {8, 2883803},
                          {16, 2871236},
                          {32, 2004165},
                          {64, 3094331}};
        break;
    case ModelType::Movenet:
        time_per_frame = {{1, 1644526},
                          {2, 3459537},
                          {4, 2703916},
                          {8, 2377614},
                          {16, 2647643},
                          {32, 2900894},
                          {64, 2197719}};
        break;
    case ModelType::Arcface:
        time_per_frame = {{1, 18120029},
                          {2, 11226197},
                          {4, 7883673},
                          {8, 6364369},
                          {16, 5620677},
                          {32, 3370018},
                          {64, 3206726}};
        break;
    case ModelType::Emotionnet:
        time_per_frame = {{1, 3394144},
                          {2, 1365037},
                          {4, 1615653},
                          {8, 1967143},
                          {16, 1500867},
                          {32, 1665680},
                          {64, 1957914}};
        break;
    case ModelType::Age:
        time_per_frame = {{1, 14729041},
                          {2, 9050828},
                          {4, 6112501},
                          {8, 5015442},
                          {16, 3927934},
                          {32, 3523500},
                          {64, 2899034}};
        break;
    case ModelType::Gender:
        time_per_frame = {{1, 1357500},
                          {2, 831649},
                          {4, 687484},
                          {8, 749792},
                          {16, 1021500},
                          {32, 1800263},
                          {64, 4002824}};
        break;
    default:
        return 0;
    }
    int i = 1;
    while (i < batch_size)
    {
        i *= 2;
    }
    return time_per_frame[batch_size] * batch_size;
}

std::pair<std::vector<NodeHandle>, std::vector<NodeHandle>> categorizeNodes(const std::vector<NodeHandle> &nodes)
{
    std::vector<NodeHandle> edges;
    std::vector<NodeHandle> servers;

    for (const auto &node : nodes)
    {
        if (node.type == Edge)
        {
            edges.push_back(node);
        }
        else if (node.type == Server)
        {
            servers.push_back(node);
        }
    }

    return {edges, servers};
}

int calculateTotalprocessedRate(const std::vector<NodeHandle> &nodes, bool is_edge)
{
    auto [edges, servers] = categorizeNodes(nodes);
    double totalEdgeRequestRate = 0;
    double totalServerRequestRate = 0;
    double totalLastRequestRate = 0;
    if (is_edge)
    {
        for (const NodeHandle &edge : edges)
        {
            for (const auto &microservicePair : edge.microservices)
            {
                const MicroserviceHandle *microservice = microservicePair.second;
                if (microservice)
                {
                    //  read the model type from the microservice, then find the corresponding type in the InferTimeEstimator function and read the value of batch = 8.
                    double timePerFrame = InferTimeEstimator(microservice->model, 8) / 1e9;
                    double requestRate = 1000000.0 / timePerFrame;
                    totalEdgeRequestRate += requestRate;
                }
            }
        }

        return totalEdgeRequestRate;
    }
    else
    {
        for (const NodeHandle &server : servers)
        {
            totalLastRequestRate += server.lastRequestRate;
            for (const auto &microservicePair : server.microservices)
            {
                const MicroserviceHandle *microservice = microservicePair.second;
                if (microservice)
                {
                    //  read the model type from the microservice, then find the corresponding type in the InferTimeEstimator function and read the value of batch = 32.
                    double timePerFrame = InferTimeEstimator(microservice->model, 32) / 1e9;
                    // Similar calculation as for edge.
                    double requestRate = 1000000.0 / timePerFrame;
                    totalServerRequestRate += requestRate;
                }
            }
        }

        return totalServerRequestRate;
    }
}

int calculateTotalQueue(const std::vector<NodeHandle> &nodes, bool is_edge)
{
    auto [edges, servers] = categorizeNodes(nodes);
    double totalEdgeQueue = 0;
    double totalServerQueue = 0;
    if (is_edge)
    {

        for (const NodeHandle &edge : edges)
        {
            for (const auto &microservicePair : edge.microservices)
            {
                const MicroserviceHandle *microservice = microservicePair.second;
                if (microservice)
                {
                    totalEdgeQueue += std::accumulate(microservice->queue_lengths.begin(), microservice->queue_lengths.end(), 0);
                }
            }
        }

        return totalEdgeQueue;
    }
    else
    {
        for (const NodeHandle &server : servers)
        {
            for (const auto &microservicePair : server.microservices)
            {
                const MicroserviceHandle *microservice = microservicePair.second;
                if (microservice)
                {
                    totalServerQueue += std::accumulate(microservice->queue_lengths.begin(), microservice->queue_lengths.end(), 0);
                }
            }
        }

        return totalServerQueue;
    }
}

double getMaxTP(std::vector<NodeHandle> nodes, bool is_edge)
{
    int processedRate = calculateTotalprocessedRate(nodes, is_edge);
    if (calculateTotalQueue(nodes, is_edge) == 0.0)
    {
        return 0;
    }
    else
    {
        return processedRate;
    }
}

void scheduleBaseParPointLoop(Partitioner* partitioner,std::vector<NodeHandle> nodes, std::vector<ContainerHandle> Microservices)
{
    float TPedgesAvg = 0.0f;
    float TPserverAvg = 0.0f;
    const float smooth = 0.4f;

    while (true)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        float TPEdges = 0.0f;

        auto [edges, servers] = categorizeNodes(nodes);
        for (NodeHandle &edge : edges)
        {
            int totalRequestRate = calculateTotalprocessedRate(nodes, true);
            if (totalRequestRate != 0)
            {
                TPEdges += getMaxTP(nodes, true);
            }
            edge.lastRequestRate = totalRequestRate;
        }

        // init the TPedgesAvg and TPserverAvg based on the current runtime
        TPedgesAvg = smooth * TPedgesAvg + (1 - smooth) * TPEdges;
        TPserverAvg = smooth * TPserverAvg + (1 - smooth) * getMaxTP(nodes, false); // this is server throughput

        // partition the parpoint
        if (TPedgesAvg > TPserverAvg + 10 * partitioner->server->num_processors)
        {
            if (TPedgesAvg > 1.5 * TPserverAvg)
            {
                partitioner->BaseParPoint += 0.006f;
            }
            else if (TPedgesAvg > 1.3 * TPserverAvg)
            {
                partitioner->BaseParPoint += 0.003f;
            }
            else
            {
                partitioner->BaseParPoint += 0.001f;
            }
        }
        else if (TPedgesAvg < TPserverAvg - 10 * partitioner->server->num_processors)
        {
            if (1.5 * TPedgesAvg < TPserverAvg)
            {
                partitioner->BaseParPoint -= 0.006f;
            }
            else if (1.3 * TPedgesAvg < TPserverAvg)
            {
                partitioner->BaseParPoint -= 0.003f;
            }
            else
            {
                partitioner->BaseParPoint -= 0.001f;
            }
        }

        if (partitioner->BaseParPoint > 1)
        {
            partitioner->BaseParPoint = 1;
        }
        else if (partitioner->BaseParPoint < 0)
        {
            partitioner->BaseParPoint = 0;
        }
        break;
    }
}

float ComputeAveragedNormalizedWorkload(const std::vector<NodeHandle> &nodes, bool is_edge)
{
    float sum = 0.0;
    int N = nodes.size();
    float edgeQueueCapacity = 200.0; // need to know the  real Capacity

    if (N == 0)
        return 0; // incase N=0

    for (const auto &node : nodes)
    {
        float tmp = calculateTotalQueue(nodes, is_edge) / edgeQueueCapacity;
        sum += tmp;
    }
    float norm = sum / static_cast<float>(N);
    return norm;
}

void scheduleFineGrainedParPointLoop(Partitioner *partitioner, const std::vector<NodeHandle> &nodes)
{
    float w;
    int totalServerQueue;
    float edgeQueueCapacity = 400.0;
    while (true)
    {
        // std::this_thread::sleep_for(std::chrono::milliseconds(250));  // every 250 weakup
        auto [edges, servers] = categorizeNodes(nodes);

        float wbar = ComputeAveragedNormalizedWorkload(edges, true);
        for (NodeHandle &edge : edges)
        {
            for (const auto &microservicePair : edge.microservices)
            {
                const MicroserviceHandle *microservice = microservicePair.second;
                if (microservice)
                {
                    totalServerQueue += std::accumulate(microservice->queue_lengths.begin(), microservice->queue_lengths.end(), 0);
                    w = static_cast<float>(totalServerQueue) / edgeQueueCapacity;
                    float tmp = 0.0f;
                    if (w == 0)
                    {
                        tmp = 1.0f;
                    }
                    else
                    {
                        tmp = (wbar - w) / std::max(wbar, w);
                    }
                    partitioner->FineGrainedOffset = tmp * partitioner->BaseParPoint;
                    break;
                }
            }
        }
        break;
    }
}

// didn't debug for DecideAndMoveContainer but the calculate of the parpoint is correct

void DecideAndMoveContainer(const std::vector<NodeHandle> &nodes, Partitioner &partitioner)
{
    float decisionPoint = partitioner.BaseParPoint + partitioner.FineGrainedOffset;
    float GPUratio = calculateTotalGPU(nodes);
    float tolerance = 0.1;
    auto [edges, servers] = categorizeNodes(nodes);
    float minGPUUsage = std::numeric_limits<float>::max();
    MicroserviceHandle *leastUsedContainer = nullptr;
    NodeHandle *leastUsedCudaNode = nullptr;
    do
    {
        GPUratio = calculateTotalGPU(nodes);
        // decided based on parpoint
        if (decisionPoint > GPUratio + tolerance)
        {
            // Iterate through all the edge nodes to select the container with the smallest currently used GPU to the emptiest cuda_device, and give the smallest nodes to msvc
            for (NodeHandle &edge : edges)
            {
                for (const auto &microservicePair : edge.microservices)
                {
                    // Select the smallest edge node in microservice->metrics.gpuUsage as the msvc for the move
                    if (microservice && microservice->metrics.gpuUsage < minGPUUsage)
                    {
                        minGPUUsage = microservice->metrics.gpuUsage;
                        leastUsedContainer = microservice;
                        leastUsedCudaNode = microservice->device_agent;
                    }

                    MoveContainer(leastUsedCudaNode, cuda_device, false);
                }
            }
        }
        // same logic but to move the container from server to edge
        else if (decisionPoint < GPUratio - tolerance)
        {
            for (NodeHandle &server : servers)
            {
                for (const auto &microservicePair : server.microservices)
                {
                    if (microservice && microservice->metrics.gpuUsage < minGPUUsage)
                    {
                        minGPUUsage = microservice->metrics.gpuUsage;
                        leastUsedContainer = microservice;
                        leastUsedCudaNode = microservice->device_agent;
                    }

                    MoveContainer(leastUsedCudaNode, cuda_device, false);
                }
            }
        }
    } while (decisionPoint < GPUratio - tolerance || decisionPoint > GPUratio + tolerance);
}

void periodicFunction(Partitioner *partitioner, std::vector<NodeHandle> nodes, std::vector<MicroserviceHandle> microservices) {
    while (true) {
        scheduleFineGrainedParPointLoop(partitioner, nodes);
        scheduleBaseParPointLoop(partitioner, nodes, microservices);
        DecideAndMoveContainer(nodes, *partitioner);

        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
}