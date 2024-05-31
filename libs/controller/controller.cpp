#include "controller.h"

std::map<ModelType, std::vector<std::string>> MODEL_INFO = {
        {DataSource,        {":datasource",         "./Container_DataSource"}},
        {Sink,              {":basesink",           "./runSink"}},
        {Yolov5,            {":yolov5",             "./Container_Yolov5"}},
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

std::map<SystemDeviceType, std::string> DEVICE_INFO = {
    {Server, "server"},
    {XavierNX, "xaviernx"},
    {OrinNano, "orinano"}
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
void Controller::queryRequestRateInPeriod(const std::string &name, ArrivalRateType &rates) {
    for (auto &rate : rates) {
        uint32_t period = rate.first;
        std::string query = absl::StrFormat("SELECT COUNT (*) FROM %s WHERE to_timestamp(arrival_timestamps / 1000000.0) >= NOW() - INTERVAL '", name + "_arrival_table");
        query += std::to_string(period) + " seconds';";

        pqxx::nontransaction session(*ctl_metricsServerConn);
        pqxx::result res = session.exec(query);

        int count = 0;
        for (const auto& row : res) {
            count = row[0].as<int>();
        }

        rate.second = (float) count / period;
        if (rate.second < 4) {
            rate.second = 4;
        }
    }
}

/**
 * @brief Query the number of outputs per input for each model in a given time period (1 minute, 2 minutes...)
 * 
 * @param name 
 * @param time_period 
 * @return uint64_t 
 */
void Controller::queryScaleFactorInPeriod(const std::string &name, ScaleFactorType &scale_factors) {
    for (auto &factor : scale_factors) {
        uint32_t period = factor.first;
        std::string query = absl::StrFormat("SELECT MAX(scale_factors) FROM %s WHERE to_timestamp(post_process_timestamps / 1000000.0) >= NOW() - INTERVAL '", name + "_process_table");
        query += std::to_string(period) + " seconds';";

        pqxx::nontransaction session(*ctl_metricsServerConn);
        pqxx::result res = session.exec(query);

        if (!res.empty() && !res[0][0].is_null()) {
            factor.second = res[0][0].as<float>();
        } else {
            factor.second = 0;
        }
    }
}

/**
 * @brief 
 * 
 * @param name 
 * @return ModelProfileType 
 */
ModelProfile Controller::queryModelProfile(const std::string &name, const std::string &deviceName) {
    // Query the median preprocess and postprocess duration of the last 1000 records as these depend on real-time data's content
    std::string query = absl::StrFormat(R"(
        WITH last_entries AS (
            SELECT prep_duration, post_duration, input_size, output_size
            FROM %s
            ORDER BY postprocess_timestamps DESC
            LIMIT 1000
        )
        SELECT
            percentile(0.5) WITHIN GROUP (ORDER BY prep_duration) AS median_prep_duration,
            percentile(0.5) WITHIN GROUP (ORDER BY post_duration) AS median_post_duration,
            percentile(0.5) WITHIN GROUP (ORDER BY input_size) AS median_input_size,
            percentile(0.5) WITHIN GROUP (ORDER BY output_size) AS median_output_size
        FROM last_entries;
    )", name + "_process_table");

    pqxx::nontransaction session(*ctl_metricsServerConn);
    pqxx::result res = session.exec(query);

    uint64_t prep_duration, post_duration;
    int input_size, output_size;
    if (!res.empty()) {
        prep_duration = res[0]["median_prep_duration"].as<uint64_t>();
        post_duration = res[0]["median_post_duration"].as<uint64_t>();
        input_size = res[0]["median_input_size"].as<int>();
        output_size = res[0]["median_output_size"].as<int>();
    }

    // Query batch profile from the profile table
    // Profile could be a one-time-only measurement or a periodically updated one
    std::string modelName = splitString(name, '_').back();

    query = absl::StrFormat(R"(
        SELECT batch_size, per_query_latency FROM %s
    )", modelName + "_" + deviceName + "_profile_table");

    res = session.exec(query);

    BatchLatencyProfileType batch_profile;

    for (const auto& row : res) {
        batch_profile[row["batch_size"].as<BatchSizeType>()] = row["per_query_latency"].as<uint64_t>();
    }


    return {prep_duration, batch_profile, post_duration, input_size, output_size};

}

/**
 * @brief query and calculate expected latency in microseconds to transfer the output of the model to the next model
 * 
 * @param name 
 * @return uint64_t expected latency in microseconds
 */
uint64_t Controller::queryTransmitLatency(const int packageSize, const std::string &sourceName, const std::string &destName) {
    // TODO: Should consider two cases (1) Same device and (2) different devices
    if (sourceName == destName) {
        return 1000;
    }
    return 5000;
}

Controller::Controller() {
    json metricsCfgs = json::parse(std::ifstream("../jsons/metricsserver.json"));
    ctl_metricsServerConfigs.from_json(metricsCfgs);
    ctl_metricsServerConfigs.user = "controller";
    ctl_metricsServerConfigs.password = "agent";

    ctl_metricsServerConn = connectToMetricsServer(ctl_metricsServerConfigs, "controller");


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
                                    {   
                                        request.device_name(),
                                        request.ip_address(),
                                     ControlCommunication::NewStub(
                                             grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials())),
                                     new CompletionQueue(),
                                     static_cast<SystemDeviceType>(request.device_type()),
                                     100, // Mbps // TODO: dynamically query and set the bandwidth
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
    StartContainer(pair, msvc->task->slo, "");
    for (auto upstr: msvc->upstreams) {
        AdjustUpstream(msvc->recv_port, upstr, device, msvc->name);
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

void Controller::AdjustBatchSize(ContainerHandle *msvc, int new_bs) {
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

/**
 * @brief Calculate queueing latency for each query coming to the preprocessor's queue, in microseconds
 * Queue type is expected to be M/D/1
 * 
 * @param arrival_rate 
 * @param preprocess_rate 
 * @return uint64_t 
 */
inline uint64_t calculateQueuingLatency(const float &arrival_rate, const float &preprocess_rate) {
    float rho = arrival_rate / preprocess_rate;
    float numQueriesInSystem = rho / (1 - rho);
    float averageQueueLength = rho * rho / (1 - rho);
    return (uint64_t) (averageQueueLength / arrival_rate * 1000000);
}

/**
 * @brief estimate the different types of latency, in microseconds
 * Due to batch inference's nature, the queries that come later has to wait for more time both in preprocessor and postprocessor.
 * 
 * @param model infomation about the model
 * @param modelType 
 */
void Controller::estimateModelLatency(PipelineModel &model, const ModelType modelType) {
    uint64_t preprocessLatency = model.modelProfile.preprocessorLatency;
    BatchSizeType batchSize = model.batchSize;
    uint64_t inferLatency = InferTimeEstimator(modelType, batchSize);
    uint64_t postprocessLatency =  model.modelProfile.postprocessorLatency;
    float preprocessRate = 1000000.f / preprocessLatency;

    model.expectedQueueingLatency = calculateQueuingLatency(model.arrivalRate, preprocessRate);
    model.expectedAvgPerQueryLatency = preprocessLatency + inferLatency * batchSize + postprocessLatency;
    model.expectedMaxProcessLatency = preprocessLatency * batchSize + inferLatency * batchSize + postprocessLatency * batchSize;
    model.estimatedPerQueryCost = model.expectedAvgPerQueryLatency + model.expectedQueueingLatency + model.expectedTransmitLatency;
}

/**
 * @brief DFS-style recursively estimate the latency of a pipeline from source to sink
 * 
 * @param pipeline provides all information about the pipeline needed for scheduling
 * @param currModel 
 */
void Controller::estimatePipelineLatency(PipelineModelListType &pipeline, const ModelType &currModel, const uint64_t start2HereLatency) {
    estimateModelLatency(pipeline.at(currModel), currModel);

    // Update the expected latency to reach the current model
    // In case a model has multiple upstreams, the expected latency to reach the model is the maximum of the expected latency 
    // to reach from each upstream.
    pipeline.at(currModel).expectedStart2HereLatency = std::max(
        pipeline.at(currModel).expectedStart2HereLatency,
        start2HereLatency + pipeline.at(currModel).expectedMaxProcessLatency + pipeline.at(currModel).expectedTransmitLatency + pipeline.at(currModel).expectedQueueingLatency
    );

    // Cost of the pipeline until the current model
    pipeline.at(currModel).estimatedStart2HereCost += pipeline.at(currModel).estimatedPerQueryCost;

    std::vector<std::pair<ModelType, int>> downstreams = pipeline.at(currModel).downstreams;
    for (const auto &d: downstreams) {
        estimatePipelineLatency(pipeline, currModel, pipeline.at(currModel).expectedStart2HereLatency);
    }

    if (currModel == ModelType::Sink) {
        return;
    }
}

/**
 * @brief Increase the number of replicas until the arrival rate is met
 * 
 * @param model 
 */
void Controller::incNumReplicas(PipelineModel &model) {
    uint8_t numReplicas = model.numReplicas;
    uint64_t inferenceLatency = model.modelProfile.inferenceLatency[model.batchSize];
    float indiProcessRate = 1 / (inferenceLatency + model.modelProfile.preprocessorLatency + model.modelProfile.postprocessorLatency);
    float processRate = indiProcessRate * numReplicas;
    while (processRate < model.arrivalRate) {
        numReplicas++;
        processRate = indiProcessRate * numReplicas;
    }
    model.numReplicas = numReplicas;
}

/**
 * @brief Decrease the number of replicas as long as it is possible to meet the arrival rate
 * 
 * @param model 
 */
void Controller::decNumReplicas(PipelineModel &model) {
    uint8_t numReplicas = model.numReplicas;
    uint64_t inferenceLatency = model.modelProfile.inferenceLatency[model.batchSize];
    float indiProcessRate = 1 / (inferenceLatency + model.modelProfile.preprocessorLatency + model.modelProfile.postprocessorLatency);
    float processRate = indiProcessRate * numReplicas;
    while (numReplicas > 1) {
        numReplicas--;
        processRate = indiProcessRate * numReplicas;
        // If the number of replicas is no longer enough to meet the arrival rate, we should not decrease the number of replicas anymore.
        if (processRate < model.arrivalRate) {
            numReplicas++;
            break;
        }
    }
    model.numReplicas = numReplicas;
}

void Controller::optimizeBatchSizeStep(
        const std::vector<std::pair<ModelType, std::vector<std::pair<ModelType, int>>>> &models,
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

/**
 * @brief 
 * 
 * @param models 
 * @param slo 
 * @param nObjects 
 * @return std::map<ModelType, int> 
 */
void Controller::getInitialBatchSizes(
        PipelineModelListType &models, uint64_t slo,
        int nObjects) {

    for (auto &m: models) {
        ModelType modelType  = std::get<0>(m);
        m.second.batchSize = 1;
        m.second.numReplicas = 1;
    }

    // DFS-style recursively estimate the latency of a pipeline from source to sin
    estimatePipelineLatency(models, models.begin()->first, 0);

    uint64_t expectedE2ELatency = models.at(ModelType::Sink).expectedStart2HereLatency;

    if (slo < expectedE2ELatency) {
        spdlog::info("SLO is too low for the pipeline to meet. Expected E2E latency: {0:d}, SLO: {1:d}", expectedE2ELatency, slo);
    }

    // Increase number of replicas to avoid bottlenecks
    for (auto &m: models) {
        incNumReplicas(m.second);
    }

    // Find near-optimal batch sizes
    auto foundBest = true;
    while (foundBest) {
        foundBest = false;
        uint64_t bestCost = models.at(ModelType::Sink).estimatedStart2HereCost;
        PipelineModelListType tmp_models = models;
        for (auto &m: tmp_models) {
            m.second.batchSize *= 2;
            estimatePipelineLatency(tmp_models, tmp_models.begin()->first, 0);
            expectedE2ELatency = tmp_models.at(ModelType::Sink).expectedStart2HereLatency;
            if (expectedE2ELatency < slo) { 
                // If increasing the batch size of model `m` creates a pipeline that meets the SLO, we should keep it
                uint64_t estimatedE2Ecost = tmp_models.at(ModelType::Sink).estimatedStart2HereCost;
                // Unless the estimated E2E cost is better than the best cost, we should not consider it as a candidate
                if (estimatedE2Ecost < bestCost) {
                    bestCost = estimatedE2Ecost;
                    models = tmp_models;
                    foundBest = true;
                }
                if (!foundBest) {
                    continue;
                }
                // If increasing the batch size meets the SLO, we can try decreasing the number of replicas
                decNumReplicas(m.second);
                estimatedE2Ecost = tmp_models.at(ModelType::Sink).estimatedStart2HereCost;
                if (estimatedE2Ecost < bestCost) {
                    models = tmp_models;
                    foundBest = true;
                }
            } else {
                m.second.batchSize /= 2;
            }
        }   
    }
}

/**
 * @brief Recursively traverse the model tree and try shifting models to edge devices
 * 
 * @param models 
 * @param slo
 */
void Controller::shiftModelToEdge(PipelineModelListType &models, const ModelType &currModel, uint64_t slo) {
    if (currModel == ModelType::Sink) {
        return;
    }
    PipelineModelListType tmp_models = models;
    std::string startDevice = tmp_models.begin()->second.device;
    std::string currDevice = tmp_models.at(currModel).device;
    std::string currModelName = MODEL_INFO[currModel][0].substr(1);

    if (currDevice != startDevice) {
        int inputSize = tmp_models.at(currModel).modelProfile.avgInputSize;
        int outputSize = tmp_models.at(currModel).modelProfile.avgOutputSize;
        if (inputSize * 0.8 < outputSize) {
            tmp_models.at(currModel).device = startDevice;
            for (auto &d: tmp_models.at(currModel).downstreams) {
                tmp_models.at(currModel).expectedTransmitLatency = queryTransmitLatency(tmp_models.at(d.first).modelProfile.avgInputSize, startDevice, currDevice);
            }
            estimatePipelineLatency(tmp_models, tmp_models.begin()->first, 0);
            uint64_t expectedE2ELatency = tmp_models.at(ModelType::Sink).expectedStart2HereLatency;
            // if after shifting the model to the edge device, the pipeline still meets the SLO, we should keep it
            if (expectedE2ELatency < slo) {
                models = tmp_models;
            }
        }
    }
    for (auto &d: tmp_models.at(currModel).downstreams) {
        shiftModelToEdge(tmp_models, d.first, slo);
    }
}

void Controller::AddTask(const TaskDescription::TaskStruct &t) {
    std::cout << "Adding task: " << t.name << std::endl;
    tasks.insert({t.name, {t.slo, t.type, {}}});
    TaskHandle *task = &tasks[t.name];
    NodeHandle *device = &devices[t.device];
    auto models = getModelsByPipelineType(t.type, t.device);
    ArrivalRateType arrival_rates;

    ScaleFactorType scale_factors;
    // Query arrival rates of individual models
    for (auto &m: models) {
        arrival_rates = {
            {1, -1}, //1 second
            {3, -1},
            {7, -1},
            {15, -1},
            {30, -1},
            {60, -1}
        };

        scale_factors = {
            {1, 1},
            {3, 1},
            {7, 1},
            {15, 1},
            {30, 1},
            {60, 1}
        };

        // Get the name of the model
        // substr(1) is used to remove the colon at the beginning of the model name
        std::string model_name = t.name + "_" + MODEL_INFO[std::get<0>(m)][0].substr(1);

        // Query the request rate for each time period
        queryRequestRateInPeriod(model_name + "_arrival_table", arrival_rates);
        // Query the scale factor (ratio of number of outputs / each input) for each time period
        queryScaleFactorInPeriod(model_name + "_process_table", scale_factors);

        m.second.arrivalRate = std::max_element(arrival_rates.begin(), arrival_rates.end(),
                                              [](const std::pair<int, float> &p1, const std::pair<int, float> &p2) {
                                                  return p1.second < p2.second;
                                              })->second;
        m.second.scaleFactors = scale_factors;
        m.second.modelProfile = queryModelProfile(model_name, DEVICE_INFO[device->type]);
        m.second.expectedTransmitLatency = queryTransmitLatency(m.second.modelProfile.avgInputSize, t.source, m.second.device);
    }

    std::string tmp = t.name;
    containers.insert({tmp.append(":datasource"), {tmp, DataSource, device, task, 33, 9999, 0, 1, {-1}}});
    task->subtasks.insert({tmp, &containers[tmp]});
    task->subtasks[tmp]->recv_port = device->next_free_port++;
    device->containers.insert({tmp, task->subtasks[tmp]});
    device = &devices["server"];

    // Find an initial batch size and replica configuration that meets the SLO at the server
    getInitialBatchSizes(models, t.slo, 10);

    // Try to shift model to edge devices
    shiftModelToEdge(models, ModelType::DataSource, t.slo);

    // for (const auto &m: models) {
    //     tmp = t.name;
    //     // TODO: get correct initial cuda devices based on TaskDescription and System State
    //     int cuda_device = 1;
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
    //     StartContainer(msvc, task->slo, t.source);
    // }
}

PipelineModelListType Controller::getModelsByPipelineType(PipelineType type, const std::string &startDevice) {
    switch (type) {
        case PipelineType::Traffic:
            return {
                {
                    ModelType::DataSource, 
                    {startDevice, true, 0, {}, {}, {{ModelType::Yolov5, 0}}}
                },
                {
                    ModelType::Yolov5,
                    {
                        "server", true, 0, {}, {},       
                        {{ModelType::Retinaface, 0}, {ModelType::CarBrand, 2}, {ModelType::Yolov5_Plate, 2}},
                        {{ModelType::DataSource, -1}}
                    },
                },
                {
                    ModelType::Retinaface, 
                    {
                        "server", false, 0, {}, {},
                        {{ModelType::Arcface,    -1}},
                        {{ModelType::Yolov5, -1}}
                    }
                },
                {
                    ModelType::Arcface,
                    {
                        "server", false, 0, {}, {},
                        {{ModelType::Sink,   -1}},
                        {{ModelType::Retinaface, -1}}
                    }
                },
                {
                    ModelType::CarBrand,
                    {
                        "server", false, 0, {}, {},
                        {{ModelType::Sink,   -1}},
                        {{ModelType::Yolov5, -1}}
                    }
                },
                {
                    ModelType::Yolov5_Plate,
                    {
                        "server", false, 0, {}, {}, {{ModelType::Sink,   -1}},
                        {{ModelType::Yolov5, -1}}
                    }
                },
                {
                    ModelType::Sink,
                    {
                        "server", false, 0, {}, {},
                        {},
                        {{ModelType::Arcface, -1}, {ModelType::CarBrand, -1}, {ModelType::Yolov5_Plate, -1}}
                    }
                }
            };
        case PipelineType::Video_Call:
            return {
                {
                    ModelType::DataSource,
                    {startDevice, true, 0, {}, {}, {{ModelType::Retinaface, 0}}}
                },
                {
                    ModelType::Retinaface,
                    {
                        "server", true, 0, {}, {},
                        {{ModelType::Emotionnet, -1}, {ModelType::Age, -1}, {ModelType::Gender, -1}, {ModelType::Arcface, -1}},
                        {{ModelType::DataSource, -1}}
                    }
                },
                {
                    ModelType::Gender,
                    {
                        "server", false, 0, {}, {},
                        {{ModelType::Sink,   -1}},
                        {{ModelType::Retinaface, -1}}
                    }
                },
                {
                    ModelType::Age,
                    {
                        "server", false, 0, {}, {},
                        {{ModelType::Sink,   -1}},
                        {{ModelType::Retinaface, -1}}
                    }
                },
                {
                    ModelType::Emotionnet,
                    {
                        "server", false, 0, {}, {},
                        {{ModelType::Sink,   -1}},
                        {{ModelType::Retinaface, -1}}
                    }
                },
                {
                    ModelType::Arcface,
                    {
                        "server", false, 0, {}, {},
                        {{ModelType::Sink,   -1}},
                        {{ModelType::Retinaface, -1}}
                    }
                },
                {
                    ModelType::Sink,
                    {
                        "server", false, 0, {}, {},
                        {},
                        {{ModelType::Emotionnet, -1}, {ModelType::Age, -1}, {ModelType::Gender, -1}, {ModelType::Arcface, -1}}
                    }
                }
            };
        case PipelineType::Building_Security:
            return {
                {
                    ModelType::DataSource,
                    {startDevice, true, 0, {}, {}, {{ModelType::Yolov5, 0}}}
                },
                {
                    ModelType::Yolov5,
                    {
                        "server", true, 0, {}, {},
                        {{ModelType::Retinaface, 0}},
                        {{ModelType::DataSource, -1}}
                    }
                },
                {
                    ModelType::Retinaface,
                    {
                        "server", false, 0, {}, {},
                        {{ModelType::Gender,     -1}, {ModelType::Age, -1}},
                        {{ModelType::Yolov5, -1}}
                    }
                },
                {
                    ModelType::Movenet,
                    {
                        "server", false, 0, {}, {},
                        {{ModelType::Sink,   -1}},
                        {{ModelType::Yolov5, -1}}
                    }
                },
                {
                    ModelType::Gender,
                    {
                        "server", false, 0, {}, {},
                        {{ModelType::Sink,   -1}},
                        {{ModelType::Retinaface, -1}}
                    }
                },
                {
                    ModelType::Age,
                    {
                        "server", false, 0, {}, {},
                        {{ModelType::Sink,   -1}},
                        {{ModelType::Retinaface, -1}}
                    }
                },
                {
                    ModelType::Sink,
                    {
                        "server", false, 0, {}, {},
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
    return time_per_frame[batch_size] * batch_size;
}