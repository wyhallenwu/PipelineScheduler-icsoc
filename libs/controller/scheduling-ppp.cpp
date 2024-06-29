#include "scheduling-ppp.h"

bool Controller::AddTask(const TaskDescription::TaskStruct &t) {
    std::cout << "Adding task: " << t.name << std::endl;
    TaskHandle *task = new TaskHandle{t.name, t.fullName, t.type, t.source, t.slo, {}, 0};

    std::unique_lock lock(devices.devicesMutex);
    if (devices.list.find(t.device) == devices.list.end()) {
        spdlog::error("Device {0:s} is not connected", t.device);
        return false;
    }

    task->tk_pipelineModels = getModelsByPipelineType(t.type, t.device);
    std::unique_lock lock(ctrl_unscheduledPipelines.tasksMutex);

    ctrl_unscheduledPipelines.list.insert({task->tk_name, *task});
    lock.unlock();
    

    std::vector<std::pair<std::string, std::string>> possibleDevicePairList = {{"server", "server"}};
    std::map<std::pair<std::string, std::string>, NetworkEntryType> possibleNetworkEntryPairs;

    for (const auto &pair : possibleDevicePairList) {
        std::unique_lock lock(devices.list[pair.first].nodeHandleMutex);
        possibleNetworkEntryPairs[pair] = devices.list[pair.first].latestNetworkEntries[pair.second];
        lock.unlock();
    }

    std::vector<std::string> possibleDeviceList = {"server"};

    // for (auto& model: ctrl_unscheduledPipelines.tk_pipelineModels) {
    //     std::string containerName = model->name + "-" + possibleDevicePairList[0].second;
    //     if (containerName.find("datasource") != std::string::npos || containerName.find("sink") != std::string::npos) {
    //         continue;
    //     }
    //     model->arrivalProfiles.arrivalRates = queryArrivalRate(
    //         *ctrl_metricsServerConn,
    //         ctrl_experimentName,
    //         ctrl_systemName,
    //         t.name,
    //         t.source,
    //         ctrl_containerLib[containerName].taskName,
    //         ctrl_containerLib[containerName].modelName
    //     );
    //     for (const auto &pair : possibleDevicePairList) {
    //         NetworkProfile test = queryNetworkProfile(
    //             *ctrl_metricsServerConn,
    //             ctrl_experimentName,
    //             ctrl_systemName,
    //             t.name,
    //             t.source,
    //             ctrl_containerLib[containerName].taskName,
    //             ctrl_containerLib[containerName].modelName,
    //             pair.first,
    //             pair.second,
    //             possibleNetworkEntryPairs[pair]
    //         );   
    //         model->arrivalProfiles.d2dNetworkProfile[std::make_pair(pair.first, pair.second)] = test;
    //     }

    //     for (const auto deviceName : possibleDeviceList) {
    //         std::string deviceTypeName = getDeviceTypeName(devices[deviceName].type);
    //         ModelProfile profile = queryModelProfile(
    //             *ctrl_metricsServerConn,
    //             ctrl_experimentName,
    //             ctrl_systemName,
    //             t.name,
    //             t.source,
    //             deviceName,
    //             deviceTypeName,
    //             ctrl_containerLib[containerName].modelName
    //         );
    //         model->processProfiles[deviceTypeName] = profile;
    //     }
        
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

PipelineModelListType Controller::getModelsByPipelineType(PipelineType type, const std::string &startDevice) {
    switch (type) {
        case PipelineType::Traffic: {
            PipelineModel *datasource = new PipelineModel{startDevice, "datasource", true, {}, {}};
            datasource->possibleDevices = {startDevice};

            PipelineModel *yolov5n = new PipelineModel{
                "server",
                "yolov5n",
                true,
                {},
                {},
                {},
                {{datasource, -1}}
            };
            yolov5n->possibleDevices = {startDevice, "server"};
            datasource->downstreams.push_back({yolov5n, -1});

            PipelineModel *retina1face = new PipelineModel{
                "server",
                "retina1face",
                false,
                {},
                {},
                {},
                {{yolov5n, 0}}
            };
            retina1face->possibleDevices = {startDevice, "server"};
            yolov5n->downstreams.push_back({retina1face, 0});

            PipelineModel *carbrand = new PipelineModel{
                "server",
                "carbrand",
                false,
                {},
                {},
                {},
                {{yolov5n, 2}}
            };
            yolov5n->downstreams.push_back({carbrand, 2});

            PipelineModel *platedet = new PipelineModel{
                "server",
                "platedet",
                false,
                {},
                {},
                {},
                {{yolov5n, 2}}
            };
            yolov5n->downstreams.push_back({platedet, 2});

            PipelineModel *sink = new PipelineModel{
                "server",
                "sink",
                false,
                {},
                {},
                {},
                {{retina1face, -1}, {carbrand, -1}, {platedet, -1}}
            };
            sink->possibleDevices = {"server"};
            retina1face->downstreams.push_back({sink, -1});
            carbrand->downstreams.push_back({sink, -1});
            platedet->downstreams.push_back({sink, -1});

            return {datasource, yolov5n, retina1face, carbrand, platedet, sink};
        }
        case PipelineType::Building_Security: {
            PipelineModel *datasource = new PipelineModel{startDevice, "datasource", true, {}, {}};
            datasource->possibleDevices = {startDevice};
            PipelineModel *yolov5n = new PipelineModel{
                "server",
                "yolov5n",
                true,
                {},
                {},
                {},
                {{datasource, -1}}
            };
            yolov5n->possibleDevices = {startDevice, "server"};
            datasource->downstreams.push_back({yolov5n, -1});

            PipelineModel *retina1face = new PipelineModel{
                "server",
                "retina1face",
                false,
                {},
                {},
                {},
                {{yolov5n, 0}}
            };
            retina1face->possibleDevices = {startDevice, "server"};
            yolov5n->downstreams.push_back({retina1face, 0});

            PipelineModel *movenet = new PipelineModel{
                "server",
                "movenet",
                false,
                {},
                {},
                {},
                {{yolov5n, 0}}
            };
            movenet->possibleDevices = {"server"};
            yolov5n->downstreams.push_back({movenet, 0});

            PipelineModel *gender = new PipelineModel{
                "server",
                "gender",
                false,
                {},
                {},
                {},
                {{retina1face, -1}}
            };
            gender->possibleDevices = {"server"};
            retina1face->downstreams.push_back({gender, -1});

            PipelineModel *age = new PipelineModel{
                "server",
                "age",
                false,
                {},
                {},
                {},
                {{retina1face, -1}}
            };
            age->possibleDevices = {"server"};
            retina1face->downstreams.push_back({age, -1});

            PipelineModel *sink = new PipelineModel{
                "server",
                "sink",
                false,
                {},
                {},
                {},
                {{gender, -1}, {age, -1}, {movenet, -1}}
            };
            sink->possibleDevices = {"server"};
            gender->downstreams.push_back({sink, -1});
            age->downstreams.push_back({sink, -1});
            movenet->downstreams.push_back({sink, -1});

            return {datasource, yolov5n, retina1face, movenet, gender, age, sink};
        }
        case PipelineType::Video_Call: {
            PipelineModel *datasource = new PipelineModel{startDevice, "datasource", true, {}, {}};
            datasource->possibleDevices = {startDevice};
            PipelineModel *retina1face = new PipelineModel{
                "server",
                "retina1face",
                true,
                {},
                {},
                {},
                {{datasource, -1}}
            };
            retina1face->possibleDevices = {startDevice, "server"};
            datasource->downstreams.push_back({retina1face, -1});

            PipelineModel *emotionnet = new PipelineModel{
                "server",
                "emotionnet",
                false,
                {},
                {},
                {},
                {{retina1face, -1}}
            };
            emotionnet->possibleDevices = {"server"};
            retina1face->downstreams.push_back({emotionnet, -1});

            PipelineModel *age = new PipelineModel{
                "server",
                "age",
                false,
                {},
                {},
                {},
                {{retina1face, -1}}
            };
            age->possibleDevices = {startDevice, "server"};
            retina1face->downstreams.push_back({age, -1});

            PipelineModel *gender = new PipelineModel{
                "server",
                "gender",
                false,
                {},
                {},
                {},
                {{retina1face, -1}}
            };
            gender->possibleDevices = {startDevice, "server"};
            retina1face->downstreams.push_back({gender, -1});

            PipelineModel *arcface = new PipelineModel{
                "server",
                "arcface",
                false,
                {},
                {},
                {},
                {{retina1face, -1}}
            };
            arcface->possibleDevices = {"server"};
            retina1face->downstreams.push_back({arcface, -1});

            PipelineModel *sink = new PipelineModel{
                "server",
                "sink",
                false,
                {},
                {},
                {},
                {{emotionnet, -1}, {age, -1}, {gender, -1}, {arcface, -1}}
            };
            sink->possibleDevices = {"server"};
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
void Controller::shiftModelToEdge(TaskHandle &models, const ModelType &currModel, uint64_t slo) {
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
        TaskHandle &models, uint64_t slo,
        int nObjects) {

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
void Controller::estimateModelLatency(PipelineModel *currModel, const std::string& deviceName) {
    ModelProfile profile = currModel->processProfiles[deviceName];
    uint64_t preprocessLatency = profile.p95prepLat;
    BatchSizeType batchSize = currModel->batchSize;
    uint64_t inferLatency = profile.batchInfer[batchSize].p95inferLat;
    uint64_t postprocessLatency =  profile.p95postLat;
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
void Controller::estimatePipelineLatency(PipelineModel* currModel, const uint64_t start2HereLatency) {
    // estimateModelLatency(currModel, currModel->device);

    // Update the expected latency to reach the current model
    // In case a model has multiple upstreams, the expected latency to reach the model is the maximum of the expected latency 
    // to reach from each upstream.
    currModel->expectedStart2HereLatency = std::max(
        currModel->expectedStart2HereLatency,
        start2HereLatency + currModel->expectedMaxProcessLatency + currModel->expectedTransferLatency + currModel->expectedQueueingLatency
    );

    // Cost of the pipeline until the current model
    currModel->estimatedStart2HereCost += currModel->estimatedPerQueryCost;

    std::vector<std::pair<PipelineModel *, int>> downstreams = currModel->downstreams;
    for (const auto &d: downstreams) {
        estimatePipelineLatency(d.first, currModel->expectedStart2HereLatency);
    }

    if (currModel->downstreams.size() == 0) {
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
uint8_t Controller::incNumReplicas(const PipelineModel *model) {
    uint8_t numReplicas = model->numReplicas;
    std::string deviceTypeName = model->deviceTypeName;
    ModelProfile profile = model->processProfiles.at(deviceTypeName);
    uint64_t inferenceLatency = profile.batchInfer.at(model->batchSize).p95inferLat;
    float indiProcessRate = 1 / (inferenceLatency + profile.p95prepLat 
                            + profile.p95postLat);
    float processRate = indiProcessRate * numReplicas;
    while (processRate < model->arrivalProfiles.arrivalRates) {
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
uint8_t Controller::decNumReplicas(const PipelineModel *model) {
    uint8_t numReplicas = model->numReplicas;
    std::string deviceTypeName = model->deviceTypeName;
    ModelProfile profile = model->processProfiles.at(deviceTypeName);
    uint64_t inferenceLatency = profile.batchInfer.at(model->batchSize).p95inferLat;
    float indiProcessRate = 1 / (inferenceLatency + profile.p95prepLat + profile.p95postLat);
    float processRate = indiProcessRate * numReplicas;
    while (numReplicas > 1) {
        numReplicas--;
        processRate = indiProcessRate * numReplicas;
        // If the number of replicas is no longer enough to meet the arrival rate, we should not decrease the number of replicas anymore.
        if (processRate < model->arrivalProfiles.arrivalRates) {
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
uint64_t Controller::calculateQueuingLatency(const float &arrival_rate, const float &preprocess_rate) {
    float rho = arrival_rate / preprocess_rate;
    float numQueriesInSystem = rho / (1 - rho);
    float averageQueueLength = rho * rho / (1 - rho);
    return (uint64_t) (averageQueueLength / arrival_rate * 1000000);
}