#include "controller.h"

std::map<ModelType, std::vector<std::string>> MODEL_INFO = {
        {DataSource,        {":datasource",         "./Container_DataSource"}},
        {Sink,              {":basesink",           "./runSink"}},
        {Yolov5,            {":yolov5",             "./Container_Yolov5"}},
        {Yolov5n320,        {":yolov5",             "./Container_Yolov5"}},
        {Yolov5s,           {":yolov5",             "./Container_Yolov5"}},
        {Yolov5m,           {":yolov5",             "./Container_Yolov5"}},
        {Yolov5Datasource,  {":yolov5datasource",   "./Container_Yolov5"}},
        {Retinaface,        {":retinaface",         "./Container_RetinaFace"}},
        {Yolov5_Plate,      {":platedetection",     "./Container_Yolov5-plate"}},
        {Movenet,           {":movenet",            "./Container_MoveNet"}},
        {Emotionnet,        {":emotionnet",         "./Container_EmotionNet"}},
        {Arcface,           {":arcface",            "./Container_ArcFace"}},
        {Age,               {":age",                "./Container_Age"}},
        {Gender,            {":gender",             "./Container_Gender"}},
        {CarBrand,          {":carbrand",           "./Container_CarBrand"}},
};

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

/**
 * @brief Query the request rate in a given time period (1 minute, 2 minutes...)
 * 
 * @param name 
 * @param time_period 
 * @return uint64_t 
 */
float Controller::queryRequestRateInPeriod(const std::string &name, const uint32_t &period) {
    std::string query = absl::StrFormat("SELECT COUNT (*) FROM %s WHERE to_timestamp(arrival_timestamps / 1000000.0) >= NOW() - INTERVAL '", name);
    query += std::to_string(period) + " seconds';";

    pqxx::nontransaction session(*ctl_metricsServerConn);
    pqxx::result res = session.exec(query);

    int count = 0;
    for (const auto& row : res) {
        count = row[0].as<int>();
    }

    return (float) count / period;
}

Controller::Controller() {
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
                     4, std::vector<double>(4, 0.2),
                     {8000, 8000, 8000, 8000},
                     std::vector<double>(4, 0.2), 55001, {}}});
    devices.insert({"edge1",
                    {"AGX",
                     {},
                     new CompletionQueue(),
                     SystemDeviceType::Edge,
                     1, std::vector<double>(1, 0.3),
                     {4000},
                     std::vector<double>(1, 0.3), 55001, {}}});
    devices.insert({"edge2",
                    {"Nano",
                     {},
                     new CompletionQueue(),
                     SystemDeviceType::Edge,
                     1, std::vector<double>(1, 0.4),
                     {4000},
                     std::vector<double>(1, 0.4), 55001, {}}});
    devices.insert({"edge3",
                    {"Nano",
                     {},
                     new CompletionQueue(),
                     SystemDeviceType::Edge,
                     1, std::vector<double>(1, 0.4),
                     {4000},
                     std::vector<double>(1, 0.4), 55001, {}}});
    devices.insert({"edge4",
                    {"Nano",
                     {},
                     new CompletionQueue(),
                     SystemDeviceType::Server,
                     1, std::vector<double>(1, 0.4),
                     {4000},
                     std::vector<double>(1, 0.4), 55001, {}}});
    devices.insert({"edge5",
                    {"NX",
                     {},
                     new CompletionQueue(),
                     SystemDeviceType::Edge,
                     1, std::vector<double>(1, 0.5),
                     {4000},
                     std::vector<double>(1, 0.5), 55001, {}}});
    devices.insert({"edge6",
                    {"NX",
                     {},
                     new CompletionQueue(),
                     SystemDeviceType::Edge,
                     1, std::vector<double>(1, 0.5),
                     {4000},
                     std::vector<double>(1, 0.5), 55001, {}}});
    devices.insert({"edge7",
                    {"NX",
                     {},
                     new CompletionQueue(),
                     SystemDeviceType::Edge,
                     1, std::vector<double>(1, 0.5),
                     {4000},
                     std::vector<double>(1, 0.5), 55001, {}}});
    devices.insert({"edge8",
                    {"NX",
                     {},
                     new CompletionQueue(),
                     SystemDeviceType::Edge,
                     1, std::vector<double>(1, 0.5),
                     {4000},
                     std::vector<double>(1, 0.5), 55001, {}}});
    devices.insert({"edge9",
                    {"NX",
                     {},
                     new CompletionQueue(),
                     SystemDeviceType::Edge,
                     1, std::vector<double>(1, 0.5),
                     {4000},
                     std::vector<double>(1, 0.5), 55001, {}}});
                     
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
    // TODO: @Jinghang, @Quang, @Yuheng please out your scheduling loop inside of here
    while (running) {
        // use list of devices, tasks and containers to schedule depending on your algorithm
        // put helper functions as a private member function of the controller and write them at the bottom of this file.
        // Perform placement for the task
        // Iterate over all tasks
        {
            std::unique_lock<std::mutex> lock(mutex_);

            if (tasks.size() != 0) {
                for (const auto& task : tasks) {
                    performPlacement(task.second);
                }
                tasks.clear();
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // sleep time can be adjusted to your algorithm or just left at 5 seconds for now
    }
}

void Controller::AddTask(const TaskDescription::TaskStruct &t) {
    std::cout << "Adding task: " << t.name << std::endl;
    std::cout << t.slo << "----";
    tasks.insert({t.name, {0,t.slo, t.type, {}}});
    TaskHandle *task = &tasks[t.name];
    NodeHandle *device = &devices[t.device];
    auto models = getModelsByPipelineType(t.type);

    std::string tmp = t.name;
    containers.insert({tmp.append(":datasource"), {tmp, DataSource, device, task, 33, 1, {0}}});
    task->subtasks.insert({tmp, &containers[tmp]});
    task->subtasks[tmp]->recv_port = device->next_free_port++;

    // TODO: @Jinghang, @Quang, @Yuheng get correct initial batch size, cuda devices, and number of replicas
    // based on TaskDescription and System State if one of them does not apply to your algorithm just leave it at 1
    // all of you should use different cuda devices at the server!
    auto batch_sizes = std::map<ModelType, int>{
        {ModelType::DataSource, 32},
        {ModelType::Sink, 32},
        {ModelType::Yolov5, 8},
        {ModelType::Yolov5n320, 8},
        {ModelType::Yolov5s, 8},
        {ModelType::Yolov5m, 4},
        {ModelType::Yolov5Datasource, 8},
        {ModelType::Retinaface, 4},
        {ModelType::Yolov5_Plate, 4},
        {ModelType::Movenet, 16},
        {ModelType::Emotionnet, 16},
        {ModelType::Arcface, 8},
        {ModelType::Age, 16},
        {ModelType::Gender, 32},
        {ModelType::CarBrand, 16}
    };
    
    for (const auto &m: models) {
        tmp = t.name;

        containers.insert(
                {tmp.append(MODEL_INFO[m.first][0]), {tmp, m.first, device, task, batch_sizes[m.first], 1, {},
                                                      -1, device->next_free_port++, {}, {}, {}, {}}});
        task->subtasks.insert({tmp, &containers[tmp]});
    }

    task->subtasks[t.name + ":datasource"]->downstreams.push_back(task->subtasks[t.name + MODEL_INFO[models[0].first][0]]);
    task->subtasks[t.name + MODEL_INFO[models[0].first][0]]->upstreams.push_back(task->subtasks[t.name + ":datasource"]);

    for (const auto &m: models) {
        for (const auto &d: m.second) {
            tmp = t.name;
            task->subtasks[tmp.append(MODEL_INFO[d.first][0])]->class_of_interest = d.second;
            task->subtasks[tmp]->upstreams.push_back(task->subtasks[t.name + MODEL_INFO[m.first][0]]);
            task->subtasks[t.name + MODEL_INFO[m.first][0]]->downstreams.push_back(task->subtasks[tmp]);
        }
    }
}


void Controller::FakeContainer(ContainerHandle *cont, int slo) {
    // @Jinghang, @Quang, @Yuheng this is a fake container that updates metrics every 1.2 seconds you can adjust the values etc. to have different scheduling results
    while (cont->running) {
        // @Jinghang, @Quang, @Yuheng change this logging to help you verify your algorithm probably add the batch size or something else
        {
            // std::unique_lock<std::mutex> lock(mutex_);
            spdlog::info("------------------------------------------------------");

            spdlog::info("Container {} is running on device {} with the following metrics:", cont->name, cont->device_agent->ip);
            spdlog::info("  Processor Utilization: {}", cont->device_agent->processors_utilization[cont->cuda_device[0]]);
            spdlog::info("  Memory Utilization: {}", cont->device_agent->mem_utilization[cont->cuda_device[0]]);
        }
        // cont->device_agent->processors_utilization[cont->cuda_device[0]] = (rand() % 100) / 100.0;
        // cont->device_agent->mem_utilization[cont->cuda_device[0]] = (rand() % 100) / 100.0;
        // cont->device_agent->processors_utilization[cont->cuda_device[0]] /= 2;
        // cont->device_agent->mem_utilization[cont->cuda_device[0]] /= 2;
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

void Controller::FakeStartContainer(std::pair<std::string, ContainerHandle *> &cont, int slo, int replica) {
    cont.second->running = true;
    for (int i=0; i<replica; i++) {
        std::cout << "Starting container: " << cont.first << std::endl;
        std::thread t(&Controller::FakeContainer, this, cont.second, slo);
        t.detach();
    }

}

void Controller::UpdateLightMetrics() {
    // TODO: Replace with Database Scraping
//    for (auto metric: metrics) {
//        containers[metric.name()].queue_lengths = metric.queue_size();
//        containers[metric.name()].metrics.requestRate = metric.request_rate();
//    }
}

void Controller::UpdateFullMetrics() {
    //TODO: Replace with Database Scraping
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
                                int replica) {
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
    for (auto dwnstr: container.second->downstreams) {
        Neighbor *dwn = request.add_downstream();
        dwn->set_name(dwnstr->name);
        dwn->set_ip(absl::StrFormat("%s:%d", dwnstr->device_agent->ip, dwnstr->recv_port));
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
            up->set_ip(absl::StrFormat("0.0.0.0:%d", upstr->recv_port));
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

void Controller::MoveContainer(ContainerHandle *msvc, int cuda_device, bool to_edge, int replica) {
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
    msvc->cuda_device[replica -1] = cuda_device;
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
    /*std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            msvc->device_agent->stub->AsyncUpdateBatchSize(&context, request, msvc->device_agent->cq));
    rpc->Finish(&reply, &status, (void *) 1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(msvc->device_agent->cq->Next(&got_tag, &ok));
    GPR_ASSERT(ok);*/
}

void Controller::StopContainer(std::string name, NodeHandle *device, bool forced) {
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


std::vector<std::pair<ModelType, std::vector<std::pair<ModelType, int>>>>
Controller::getModelsByPipelineType(PipelineType type) {
    switch (type) {
        case PipelineType::Traffic:
            return {{ModelType::Yolov5,       {{ModelType::Retinaface, 0}, {ModelType::CarBrand, 2}, {ModelType::Yolov5_Plate, 2}}},
                    {ModelType::Retinaface,   {{ModelType::Arcface,    -1}}},
                    {ModelType::Arcface,      {{ModelType::Sink,   -1}}},
                    {ModelType::CarBrand,     {{ModelType::Sink,   -1}}},
                    {ModelType::Yolov5_Plate, {{ModelType::Sink,   -1}}},
                    {ModelType::Sink,     {}}};
        case PipelineType::Video_Call:
            return {{ModelType::Retinaface, {{ModelType::Emotionnet, -1}, {ModelType::Age, -1}, {ModelType::Gender, -1}, {ModelType::Arcface, -1}}},
                    {ModelType::Gender,     {{ModelType::Sink,   -1}}},
                    {ModelType::Age,        {{ModelType::Sink,   -1}}},
                    {ModelType::Emotionnet, {{ModelType::Sink,   -1}}},
                    {ModelType::Arcface,    {{ModelType::Sink,   -1}}},
                    {ModelType::Sink,   {}}};
        case PipelineType::Building_Security:
            return {{ModelType::Yolov5,     {{ModelType::Retinaface, 0}}},
                    {ModelType::Retinaface, {{ModelType::Gender,     -1}, {ModelType::Age, -1}}},
                    {ModelType::Movenet,    {{ModelType::Sink,   -1}}},
                    {ModelType::Gender,     {{ModelType::Sink,   -1}}},
                    {ModelType::Age,        {{ModelType::Sink,   -1}}},
                    {ModelType::Sink,   {}}};
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
int Controller::InferTimeEstimator(ModelType model, int batch_size,std::string device_type) {
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
        case ModelType::Yolov5n320:
            time_per_frame = {{1,  2649396},
                              {2,  2157968},
                              {4,  1897505},
                              {8,  2076971},
                              {16, 2716276},
                              {32, 4172530},
                              {64, 7252059}};
            break;
        case ModelType::Yolov5s:
            time_per_frame = {{1,  4515118},
                              {2,  3399807},
                              {4,  3044100},
                              {8,  3008503},
                              {16, 3672566},
                              {32, 5116321},
                              {64, 8237824}};
            break;
        case ModelType::Yolov5m:
            time_per_frame = {{1,  7263238},
                              {2,  5905167},
                              {4,  4446144},
                              {8,  4449675},
                              {16, 4991818},
                              {32, 6543270},
                              {64, 9579015}};
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
    // Apply device-specific performance factors
    double performance_factor = 1.0;
    if (device_type == "AGX") {
        performance_factor = 2.0;
    } else if (device_type == "Nano") {
        performance_factor = 2.0 * 1.3;
    } else if (device_type == "NX") {
        performance_factor = 2.0 * 1.3 * 1.2;
    }
    return static_cast<int>(time_per_frame[batch_size] * batch_size * performance_factor);
}


// ========================================================== added ================================================================

bool Controller::placeMDAGOnSingleWorker(const TaskHandle& task) {
    {
        // std::unique_lock<std::mutex> lock(mutex_);
        NodeHandle* bestFitWorker = nullptr;
        float minRemainingProcCapacity = 1.0;
        float minRemainingMemCapacity = 1.0;
        int cuda_device = 0;
        for (auto& worker : devices) {
            float totalRequiredCapacity = 0;
            for (const auto& container : task.subtasks) {
                float targetFps = 1 / (float(task.slo) * float(1e-9));
                float maxFps = container.second->batch_size / (float(InferTimeEstimator(container.second->model, container.second->batch_size,worker.second.ip))* float(1e-9));
                float requiredCapacity = targetFps / maxFps;
                totalRequiredCapacity += requiredCapacity;
            }
            for (int i = 0; i < worker.second.num_processors; i++) {
                if (worker.second.processors_utilization[i] + totalRequiredCapacity < 1.0 &&
                    worker.second.mem_utilization[i] + totalRequiredCapacity < 1.0) {
                    float remainingProcCapacity = 1.0 - worker.second.processors_utilization[i] - totalRequiredCapacity;
                    float remainingMemCapacity = 1.0 - worker.second.mem_utilization[i] - totalRequiredCapacity;

                    if (remainingProcCapacity < minRemainingProcCapacity 
                        && remainingMemCapacity < minRemainingMemCapacity) 
                    {
                        minRemainingProcCapacity = remainingProcCapacity;
                        minRemainingMemCapacity = remainingMemCapacity;
                        bestFitWorker = &worker.second;
                        bestFitWorker->processors_utilization[i] = 1.0 - minRemainingProcCapacity;
                        bestFitWorker->mem_utilization[i] = 1.0 - minRemainingMemCapacity;
                        cuda_device = i;
                    }
                }
            }
        }
        if (bestFitWorker) {
            for (const auto& container : task.subtasks) {
                bestFitWorker->containers.insert({container.second->name, container.second});
                container.second->device_agent = bestFitWorker;
                container.second->cuda_device = {cuda_device};
                container.second->recv_port = bestFitWorker->next_free_port++;
            }
            std::cout << "mDAG placed on a single worker: " << bestFitWorker->ip << std::endl;
            return true;
        }
        std::cout << "No suitable worker found for placing the mDAG." << std::endl;
        return false;
    }
}

void Controller::placeModulesOnWorkers(const TaskHandle& task) {

    {    
        // std::unique_lock<std::mutex> lock(mutex_);
        for (const auto& container : task.subtasks) {
            NodeHandle* bestFitWorker = nullptr;
            float minRemainingProcCapacity = 1.0;
            float minRemainingMemCapacity = 1.0;
            for (auto& worker : devices) {
                float targetFps = 1 / (float(task.slo) * float(1e-9));
                float maxFps = container.second->batch_size / (float(InferTimeEstimator(container.second->model, container.second->batch_size,worker.second.ip))* float(1e-9));
                float totalRequiredCapacity = targetFps / maxFps;
                for (int i = 0; i < worker.second.num_processors; i++) {
                    if (worker.second.processors_utilization[i] + totalRequiredCapacity < 1.0 &&
                    worker.second.mem_utilization[i] + totalRequiredCapacity < 1.0) {
                        float remainingProcCapacity = 1.0 - worker.second.processors_utilization[i] - totalRequiredCapacity;
                        float remainingMemCapacity = 1.0 - worker.second.mem_utilization[i] - totalRequiredCapacity;

                        if (remainingProcCapacity < minRemainingProcCapacity 
                            && remainingMemCapacity < minRemainingMemCapacity) 
                        {
                            minRemainingProcCapacity = remainingProcCapacity;
                            minRemainingMemCapacity = remainingMemCapacity;
                            bestFitWorker = &worker.second;
                            bestFitWorker->processors_utilization[i] = 1.0 - minRemainingProcCapacity;
                            bestFitWorker->mem_utilization[i] = 1.0 - minRemainingMemCapacity;
                            bestFitWorker->containers.insert({container.second->name, container.second});
                            container.second->device_agent = bestFitWorker;
                            container.second->cuda_device = {i};
                            container.second->recv_port = bestFitWorker->next_free_port++;
                            spdlog::info("Placed container {} on worker {}", container.second->name, bestFitWorker->ip);
                            return;
                        }
                    }
                }
            }
            std::cout << "No available worker found for container: " << container.second->name << std::endl;
        }
    }
}

void Controller::performPlacement(const TaskHandle& task) {
    if (!placeMDAGOnSingleWorker(task)) {
        placeModulesOnWorkers(task);
    }
    int replicas = 1;
    for (std::pair<std::string, ContainerHandle *> msvc: task.subtasks) {
        FakeStartContainer(msvc, task.slo, replicas);
    }
}