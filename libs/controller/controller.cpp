#include "controller.h"

std::map<ModelType, std::vector<std::string>> MODEL_INFO = {
    {DataSource, {":datasource", "./Container_DataSource"}},
    {Sink, {":basesink", "./runSink"}},
    {Yolov5, {":yolov5", "./Container_Yolov5"}},
    {Yolov5n320, {":yolov5", "./Container_Yolov5"}},
    {Yolov5s, {":yolov5", "./Container_Yolov5"}},
    {Yolov5m, {":yolov5", "./Container_Yolov5"}},
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

/**
 * @brief Query the request rate in a given time period (1 minute, 2 minutes...)
 *
 * @param name
 * @param time_period
 * @return uint64_t
 */
float Controller::queryRequestRateInPeriod(const std::string &name, const uint32_t &period)
{
    std::string query = absl::StrFormat("SELECT COUNT (*) FROM %s WHERE to_timestamp(arrival_timestamps / 1000000.0) >= NOW() - INTERVAL '", name);
    query += std::to_string(period) + " seconds';";

    pqxx::nontransaction session(*ctl_metricsServerConn);
    pqxx::result res = session.exec(query);

    int count = 0;
    for (const auto &row : res)
    {
        count = row[0].as<int>();
    }

    return (float)count / period;
}

Controller::Controller()
{
    json metricsCfgs = json::parse(std::ifstream("../jsons/metricsserver.json"));
    ctl_metricsServerConfigs.from_json(metricsCfgs);
    ctl_metricsServerConfigs.user = "controller";
    ctl_metricsServerConfigs.password = "agent";

    ctl_metricsServerConn = connectToMetricsServer(ctl_metricsServerConfigs, "controller");

    running = true;
    devices = std::map<std::string, NodeHandle>();
    // TODO: Remove Test Code
    devices.insert({"server",
                    {"server",
                     {},
                     new CompletionQueue(),
                     SystemDeviceType::Server,
                     4,
                     std::vector<double>(4, 0.0),
                     {8000, 8000, 8000, 8000},
                     std::vector<double>(4, 0.0),
                     55001,
                     {}}});
    devices.insert({"edge1",
                    {"edge1",
                     {},
                     new CompletionQueue(),
                     SystemDeviceType::Edge,
                     1,
                     std::vector<double>(1, 0.0),
                     {4000},
                     std::vector<double>(1, 0.0),
                     55001,
                     {}}});
    devices.insert({"edge2",
                    {"edge2",
                     {},
                     new CompletionQueue(),
                     SystemDeviceType::Edge,
                     1,
                     std::vector<double>(1, 0.0),
                     {4000},
                     std::vector<double>(1, 0.0),
                     55001,
                     {}}});
    devices.insert({"edge3",
                    {"edge3",
                     {},
                     new CompletionQueue(),
                     SystemDeviceType::Edge,
                     1,
                     std::vector<double>(1, 0.0),
                     {4000},
                     std::vector<double>(1, 0.0),
                     55001,
                     {}}});
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

void Controller::Scheduling()
{
    // add mutex to control the data
    //  TODO: @Jinghang, @Quang, @Yuheng please out your scheduling loop inside of here
    while (running)
    {
        // use list of devices, tasks and containers to schedule depending on your algorithm
        // put helper functions as a private member function of the controller and write them at the bottom of this file.
        NodeHandle *edgePointer = nullptr;
        NodeHandle *serverPointer = nullptr;
        unsigned long totalEdgeMemory = 0, totalServerMemory = 0;
        // std::vector<std::unique_ptr<NodeHandle>> nodes;
        // int cuda_device = 2; // need to be add
        nodes.clear();
        {
            std::unique_lock<std::mutex> lock(mtx);
            for (const auto &devicePair : devices)
            {
                nodes.push_back(devicePair.second);
            }
            // init Partitioner
            Partitioner partitioner;
            float ratio = calculateRatio(nodes);

            partitioner.BaseParPoint = ratio;

            scheduleBaseParPointLoop(&partitioner, nodes);
            scheduleFineGrainedParPointLoop(&partitioner, nodes);
            DecideAndMoveContainer(nodes, &partitioner, 2);
            std::cout << "end_scheduleBaseParPoint " << partitioner.BaseParPoint << std::endl;
            std::cout << "end_FineGrainedParPoint " << partitioner.FineGrainedOffset << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2500)); // sleep time can be adjusted to your algorithm or just left at 5 seconds for now
    }
}

// void Controller::AddTask(const TaskDescription::TaskStruct &t)
// {
//     std::cout << "Adding task: " << t.name << std::endl;
//     tasks.insert({t.name, {t.slo, t.type, {}}});
//     TaskHandle *task = &tasks[t.name];
//     NodeHandle *device = &devices[t.device];
//     auto models = getModelsByPipelineType(t.type);

//     std::string tmp = t.name;
//     containers.insert({tmp.append(":datasource"), {tmp, DataSource, device, task, 33, 1, {0}}});
//     task->subtasks.insert({tmp, &containers[tmp]});
//     task->subtasks[tmp]->recv_port = device->next_free_port++;
//     device->containers.insert({tmp, task->subtasks[tmp]});
//     device = &devices["server"];

//     // TODO: @Jinghang, @Quang, @Yuheng get correct initial batch size, cuda devices, and number of replicas
//     // based on TaskDescription and System State if one of them does not apply to your algorithm just leave it at 1
//     // all of you should use different cuda devices at the server!
//     auto batch_sizes = std::map<ModelType, int>();
//     int cuda_device = 2;
//     int replicas = 2;
//     for (const auto &m : models)
//     {
//         tmp = t.name;

//         containers.insert(
//             {tmp.append(MODEL_INFO[m.first][0]), {tmp, m.first, device, task, batch_sizes[m.first], 1, {cuda_device}, -1, device->next_free_port++, {}, {}, {}, {}}});
//         task->subtasks.insert({tmp, &containers[tmp]});
//         device->containers.insert({tmp, task->subtasks[tmp]});
//     }

//     task->subtasks[t.name + ":datasource"]->downstreams.push_back(task->subtasks[t.name + MODEL_INFO[models[0].first][0]]);
//     task->subtasks[t.name + MODEL_INFO[models[0].first][0]]->upstreams.push_back(task->subtasks[t.name + ":datasource"]);
//     for (const auto &m : models)
//     {
//         for (const auto &d : m.second)
//         {
//             tmp = t.name;
//             task->subtasks[tmp.append(MODEL_INFO[d.first][0])]->class_of_interest = d.second;
//             task->subtasks[tmp]->upstreams.push_back(task->subtasks[t.name + MODEL_INFO[m.first][0]]);
//             task->subtasks[t.name + MODEL_INFO[m.first][0]]->downstreams.push_back(task->subtasks[tmp]);
//         }
//     }

//     for (std::pair<std::string, ContainerHandle *> msvc : task->subtasks)
//     {
//         // StartContainer(msvc, task->slo, t.source, replicas);

//         FakeStartContainer(msvc, task->slo, replicas);
//     }
// }
void Controller::AddTask(const TaskDescription::TaskStruct &t)
{
    std::cout << "Adding task: " << t.name << std::endl;
    tasks.insert({t.name, {t.slo, t.type, {}}});
    TaskHandle *task = &tasks[t.name];
    NodeHandle *device = &devices[t.device];
    auto models = getModelsByPipelineType(t.type);

    std::string tmp = t.name;
    containers.insert({tmp.append(":datasource"), {tmp, DataSource, device, task, 33, 1, {0}}});
    task->subtasks.insert({tmp, &containers[tmp]});
    task->subtasks[tmp]->recv_port = device->next_free_port++;
    device->containers.insert({tmp, task->subtasks[tmp]});
    std::cout << "Initialized container: " << tmp << ", Model: DataSource" << std::endl; // 打印数据源容器信息

    NodeHandle *serverDevice = &devices["server"];
    NodeHandle *edgeDevice = &devices["edge"];

    std::cout << "Initial containers and models:" << std::endl;
    auto batch_sizes = std::map<ModelType, int>();
    int cuda_device = 2;
    int replicas = 2;
    for (const auto &m : models)
    {
        tmp = t.name;
        NodeHandle *targetDevice = serverDevice;
        if (MODEL_INFO[m.first][0] == ":yolov5")
        {
            targetDevice = edgeDevice;
            std::cout << "targetDevice " << MODEL_INFO[m.first][0] << " - " << (targetDevice == edgeDevice ? "edge" : "server") << std::endl;
        }

        containers.insert(
            {tmp.append(MODEL_INFO[m.first][0]), {tmp, m.first, targetDevice, task, batch_sizes[m.first], 1, {cuda_device}, -1, targetDevice->next_free_port++, {}, {}, {}, {}}});
        task->subtasks.insert({tmp, &containers[tmp]});
        targetDevice->containers.insert({tmp, task->subtasks[tmp]});
        std::cout << "Model used in container: " << MODEL_INFO[m.first][0] << ", Container Name: " << tmp << ", Device: " << (targetDevice == edgeDevice ? "edge" : "server") << std::endl;
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
            std::cout << "Subtask for model " << MODEL_INFO[d.first][0] << ", Container Name: " << tmp << std::endl;
        }
    }

    for (std::pair<std::string, ContainerHandle *> msvc : task->subtasks)
    {
        FakeStartContainer(msvc, task->slo, replicas);
    }
}

void Controller::FakeContainer(ContainerHandle *cont, int slo)
{
    // @Jinghang, @Quang, @Yuheng this is a fake container that updates metrics every 1.2 seconds you can adjust the values etc. to have different scheduling results
    while (cont->running)
    {

        {
            std::unique_lock<std::mutex> lock(mtx);
            cont->metrics.cpuUsage = (rand() % 100) / 100.0;
            cont->metrics.memUsage = (rand() % 1500 + 500) / 1000.0;
            cont->metrics.gpuUsage = (rand() % 100) / 100.0;
            cont->metrics.gpuMemUsage = (rand() % 100) / 100.0;
            cont->metrics.requestRate = (rand() % 70 + 30) / 100.0;
            cont->queue_lengths = {};
            for (int i = 0; i < 5; i++)
            {
                cont->queue_lengths.Add((rand() % 10));
            }
            // change
            //  cont->task->last_latency = 100;//(cont->task->last_latency + slo * 0.8 + (rand() % (int)(slo * 0.4))) / 2;
            //  cont->device_agent->processors_utilization[cont->cuda_device[0]] += (rand() % 100) / 100.0;
            //  cont->device_agent->processors_utilization[cont->cuda_device[0]] /= 2;

            // @Jinghang, @Quang, @Yuheng change this logging to help you verify your algorithm probably add the batch size or something else
            spdlog::info("Container {} is running on device {} and metrics are updated", cont->name, cont->device_agent->ip);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1200));
        // lock.unlock();
    }
}

void Controller::FakeStartContainer(std::pair<std::string, ContainerHandle *> &cont, int slo, int replica)
{
    cont.second->running = true;
    for (int i = 0; i < replica; i++)
    {
        std::cout << "Starting container: " << cont.first << std::endl;
        std::thread t(&Controller::FakeContainer, this, cont.second, slo);
        t.detach();
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
            up->set_ip(absl::StrFormat("0.0.0.0:%d", upstr->recv_port));
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
    std::cout << "MoveContainer" << std::endl;
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
    // removed for test environ
    /*    StartContainer(pair, msvc->task->slo, "");
        for (auto upstr: msvc->upstreams) {
            AdjustUpstream(msvc->recv_port, upstr, device, msvc->name);
        }
        StopContainer(msvc->name, old_device);*/
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
    // GPR_ASSERT(upstr->device_agent->cq->Next(&got_tag, &ok));
    // GPR_ASSERT(ok);
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
    /*std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            msvc->device_agent->stub->AsyncUpdateBatchSize(&context, request, msvc->device_agent->cq));
    rpc->Finish(&reply, &status, (void *) 1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(msvc->device_agent->cq->Next(&got_tag, &ok));
    GPR_ASSERT(ok);*/
}

void Controller::StopContainer(std::string name, NodeHandle *device, bool forced)
{
    ContainerSignal request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_name(name);
    request.set_forced(forced);
    containers[name].running = false;
    /*std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            device->stub->AsyncStopContainer(&context, request, containers[name].device_agent->cq));
    rpc->Finish(&reply, &status, (void *) 1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(device->cq->Next(&got_tag, &ok));
    GPR_ASSERT(ok);*/
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

/**
 * @brief
 *
 * @param model to specify model
 * @param batch_size for targeted batch size (binary)
 * @return int for inference time per full batch in nanoseconds
 */
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
    case ModelType::Yolov5n320:
        time_per_frame = {{1, 2649396},
                          {2, 2157968},
                          {4, 1897505},
                          {8, 2076971},
                          {16, 2716276},
                          {32, 4172530},
                          {64, 7252059}};
        break;
    case ModelType::Yolov5s:
        time_per_frame = {{1, 4515118},
                          {2, 3399807},
                          {4, 3044100},
                          {8, 3008503},
                          {16, 3672566},
                          {32, 5116321},
                          {64, 8237824}};
        break;
    case ModelType::Yolov5m:
        time_per_frame = {{1, 7263238},
                          {2, 5905167},
                          {4, 4446144},
                          {8, 4449675},
                          {16, 4991818},
                          {32, 6543270},
                          {64, 9579015}};
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

std::pair<std::vector<Controller::NodeHandle>, std::vector<Controller::NodeHandle>> Controller::categorizeNodes(const std::vector<Controller::NodeHandle> &nodes)
{
    std::vector<Controller::NodeHandle> edges;
    std::vector<Controller::NodeHandle> servers;

    for (const auto &node : nodes)
    {
        if (node.type == Edge)
        {
            edges.push_back(node);
            //  std::cout << "edge_push " << node.ip << std::endl;
        }
        else if (node.type == Server)
        {
            servers.push_back(node);
            // std::cout << "server_push " << node.ip << std::endl;
        }
    }

    return {edges, servers};
}

int Controller::calculateTotalprocessedRate(const std::vector<NodeHandle> &nodes, bool is_edge)
{
    auto [edges, servers] = categorizeNodes(nodes);
    double totalRequestRate = 0;
    if (is_edge)
    {
        for (const NodeHandle &edge : edges)
        {
            for (const auto &containerPair : edge.containers)
            {
                const ContainerHandle *container = containerPair.second;
                std::cout << "edge container->model" << container->model << std::endl;
                if (container && container->model != 0)
                {
                    //  read the model type from the microservice, then find the corresponding type in the InferTimeEstimator function and read the value of batch = 8.
                    int timePerFrame = InferTimeEstimator(container->model, 8);
                    std::cout << "edge container->model" << container->model << std::endl;
                    float requestRate;
                    if (timePerFrame == 0)
                    {
                        requestRate = 0.0;
                    }
                    else
                    {
                        requestRate = 1000000000.0 / timePerFrame;
                    }
                    totalRequestRate += requestRate;
                }
            }
        }
    }
    else
    {
        for (const NodeHandle &server : servers)
        {
            for (const auto &containerPair : server.containers)
            {
                const ContainerHandle *container = containerPair.second;
                std::cout << "server container->model" << container->model << std::endl;
                if (container && container->model != 0) // && container->model != DataSource && container->model != Sink)
                {
                    //  read the model type from the microservice, then find the corresponding type in the InferTimeEstimator function and read the value of batch = 32.
                    int timePerFrame = InferTimeEstimator(container->model, 32);
                    float requestRate;
                    if (timePerFrame == 0)
                    {
                        requestRate = 0.0;
                    }
                    else
                    {
                        requestRate = 1000000000.0 / timePerFrame;
                    }
                    totalRequestRate += requestRate;
                }
            }
        }
    }
    return totalRequestRate;
}

int Controller::calculateTotalQueue(const std::vector<NodeHandle> &nodes, bool is_edge)
{
    auto [edges, servers] = categorizeNodes(nodes);
    double totalEdgeQueue = 0;
    double totalServerQueue = 0;
    if (is_edge)
    {

        for (const NodeHandle &edge : edges)
        {
            for (const auto &containerPair : edge.containers)
            {
                const ContainerHandle *container = containerPair.second;
                if (container)
                {
                    totalEdgeQueue += std::accumulate(container->queue_lengths.begin(), container->queue_lengths.end(), 0);
                }
            }
        }

        return totalEdgeQueue;
    }
    else
    {
        for (const NodeHandle &server : servers)
        {
            for (const auto &containerPair : server.containers)
            {
                const ContainerHandle *container = containerPair.second;
                if (container)
                {
                    totalServerQueue += std::accumulate(container->queue_lengths.begin(), container->queue_lengths.end(), 0);
                }
            }
        }

        return totalServerQueue;
    }
}

double Controller::getMaxTP(std::vector<NodeHandle> nodes, bool is_edge)
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

void Controller::scheduleBaseParPointLoop(Partitioner *partitioner, std::vector<NodeHandle> nodes)
{
    float TPedgesAvg = 0.0f;
    float TPserverAvg = 0.0f;
    const float smooth = 0.4f;

    while (true)
    {
        // std::this_thread::sleep_for(std::chrono::milliseconds(250));
        // float TPEdges = 0.0f;

        // auto [edges, servers] = categorizeNodes(nodes);
        float TPEdges = getMaxTP(nodes, true);
        std::cout << "TPEdges: " << TPEdges << std::endl;
        float TPServer = getMaxTP(nodes, false);
        std::cout << "TPServer: " << TPServer << std::endl;

        // init the TPedgesAvg and TPserverAvg based on the current runtime
        TPedgesAvg = smooth * TPedgesAvg + (1 - smooth) * TPEdges;
        TPserverAvg = smooth * TPserverAvg + (1 - smooth) * TPServer; // this is server throughput
        std::cout << " TPserverAvg:" << TPserverAvg << std::endl;

        // partition the parpoint
        if (TPedgesAvg > TPserverAvg + 10) //* 4)
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
        else if (TPedgesAvg < TPserverAvg - 10) //* 4)
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

float Controller::ComputeAveragedNormalizedWorkload(const std::vector<NodeHandle> &nodes, bool is_edge)
{
    float sum = 0.0;
    int N = nodes.size();
    float edgeQueueCapacity = 200.0; // need to know the  real Capacity

    if (N == 0)
        return 0; // incase N=0

    float tmp = calculateTotalQueue(nodes, is_edge) / edgeQueueCapacity;
    sum += tmp;

    // for (const auto &node : nodes)
    // {
    //     float tmp = calculateTotalQueue(nodes, is_edge) / edgeQueueCapacity;
    //     sum += tmp;
    // }
    float norm = sum / static_cast<float>(N);
    return norm;
}

void Controller::scheduleFineGrainedParPointLoop(Partitioner *partitioner, const std::vector<NodeHandle> &nodes)
{
    float w;
    int totalServerQueue;
    float ServerCapacity = 5000.0;
    float tmp;
    while (true)
    {
        // std::this_thread::sleep_for(std::chrono::milliseconds(250));  // every 250 weakup
        auto [edges, servers] = categorizeNodes(nodes);

        float wbar = ComputeAveragedNormalizedWorkload(edges, true);
        // std::cout << "wbar " << wbar << std::endl;
        float totalServerQueue = calculateTotalQueue(nodes, false);
        // std::cout << "totalServerQueue " << totalServerQueue << std::endl;
        float w = totalServerQueue / ServerCapacity;
        // std::cout << "w " << w << std::endl;
        if (w == 0)
        {
            float tmp = 1.0f;
            partitioner->FineGrainedOffset = tmp * partitioner->BaseParPoint;
        }
        else
        {
            float tmp = (wbar - w) / std::max(wbar, w);
            // std::cout << "tmp " << tmp << std::endl;
            // std::cout << "(wbar - w) " << (wbar - w) << std::endl;
            // std::cout << "std::max(wbar, w) " << std::max(wbar, w) << std::endl;
            partitioner->FineGrainedOffset = tmp * partitioner->BaseParPoint;
        }
        // std::cout << "tmp " << tmp << std::endl;
        break;
    }
}

float Controller::calculateRatio(const std::vector<NodeHandle> &nodes)
{
    auto [edges, servers] = categorizeNodes(nodes);
    float edgeMem = 0.0f;
    float serverMem = 0.0f;
    float ratio = 0.0f;
    NodeHandle *edgePointer = nullptr;
    NodeHandle *serverPointer = nullptr;

    for (const NodeHandle &node : nodes)
    {
        if (node.type == SystemDeviceType::Edge)
        {
            edgePointer = const_cast<NodeHandle *>(&node);
            edgeMem += std::accumulate(node.mem_size.begin(), node.mem_size.end(), 0UL);
        }
        else
        {
            serverPointer = const_cast<NodeHandle *>(&node);
            serverMem += std::accumulate(node.mem_size.begin(), node.mem_size.end(), 0UL);
        }
    }

    if (edgePointer == nullptr)
    {
        std::cout << "No edge device found.\n";
    }

    std::cout << "Total serverMem: " << serverMem << std::endl;
    std::cout << "Total edgeMem: " << edgeMem << std::endl;

    if (serverMem != 0)
    {
        ratio = edgeMem / serverMem;
    }
    else
    {
        ratio = 0.0f;
    }

    std::cout << "Calculated Ratio: " << ratio << std::endl;
    return ratio;
}

void Controller::DecideAndMoveContainer(std::vector<NodeHandle> &nodes, Partitioner *partitioner, int cuda_device)
{
    float decisionPoint = partitioner->BaseParPoint + partitioner->FineGrainedOffset;
    float ratio = 0.7;
    float tolerance = 0.1;
    auto [edges, servers] = categorizeNodes(nodes);
    float minModel = 100;
    float maxModel = 1;
    ContainerHandle *selectedContainer = nullptr;

    // while (decisionPoint < ratio - tolerance || decisionPoint > ratio + tolerance)
    // {
    if (decisionPoint > ratio + tolerance)
    {
        for (NodeHandle &edge : edges)
        {
            for (auto &containerPair : edge.containers)
            {
                ContainerHandle *container = containerPair.second;
                if (container && container->model != 0 && container->model > maxModel)
                {
                    maxModel = container->model;   
                    selectedContainer = container; 
                }
                // ContainerHandle *container = containerPair.second;
                // std::vector<ModelType> modelPriority = {CarBrand, Gender, Age, Arcface, Emotionnet, Movenet, Yolov5_Plate, Retinaface, Yolov5Datasource, Yolov5m, Yolov5s, Yolov5n320, Yolov5};
                // for (ModelType priorityModel : modelPriority)
                // {
                //     if (container && container->model == priorityModel)
                //     {
                //     selectedContainer = container;
                //     std::cout << "Move Container from edge to server based on model priority" << container-> name<< std::endl;
                //     MoveContainer(selectedContainer, 1, false, 1);
                //     return;
                //     }
                // }
            }
        }
        if (selectedContainer)
        {
            std::cout << "Move Container from edge to server based on model priority: " << selectedContainer->name << std::endl;
            MoveContainer(selectedContainer, 1, false, 1);
        }
    }
    // Similar logic for the server side
    if (decisionPoint < ratio - tolerance)
    {
        for (NodeHandle &server : servers)
        {
            for (auto &containerPair : server.containers)
            {
                ContainerHandle *container = containerPair.second;
                // std::vector<ModelType> modelPriority = {Yolov5, Yolov5n320, Yolov5s, Yolov5m, Yolov5Datasource, Retinaface, Yolov5_Plate, Movenet, Emotionnet, Arcface, Age, Gender, CarBrand};
                if (container && container->model != 0 && container->model  < minModel)
                {
                    minModel = container->model;   
                    selectedContainer = container; 
                
                    // if (container && container->model == priorityModel)
                    // {
                    // selectedContainer = container;
                    // std::cout << "Move Container from edge to server based on model priority"<< container-> name<< std::endl;
                    // MoveContainer(selectedContainer, 1, false, 1);
                    // return;
                    // }
                }
            }
        }
        if (selectedContainer)
        {
            std::cout << "Move Container from server to edge based on model priority: " << selectedContainer->name << std::endl;
            MoveContainer(selectedContainer, 1, true, 1);
        }
    }
}
