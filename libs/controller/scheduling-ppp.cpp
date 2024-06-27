#include "scheduling-ppp.h"

void Controller::AddTask(const TaskDescription::TaskStruct &t) {
    std::cout << "Adding task: " << t.name << std::endl;
    tasks.insert({t.name, {t.name, t.type, t.source, t.slo, {}, 0, {}}});
    TaskHandle *task = &tasks[t.name];
    NodeHandle *device = &devices[t.device];
    PipelineModelListType models = getModelsByPipelineType(t.type, t.device);

    std::vector<std::pair<std::string, std::string>> possibleDeviceList = {{"serv", "serv"}};
    std::map<std::pair<std::string, std::string>, NetworkEntryType> possibleNetworkEntryPairs;

    for (const auto &pair : possibleDeviceList) {
        std::unique_lock lock(devices[pair.first].nodeHandleMutex);
        possibleNetworkEntryPairs[pair] = devices[pair.first].latestNetworkEntries[pair.second];
        lock.unlock();
    }

    for (auto &[modelType, modelObj]: models) {
        std::string modelName = getContainerName(devices[t.device].type, modelType);
        if (modelName.find("datasource") != std::string::npos || modelName.find("sink") != std::string::npos) {
            continue;
        }
        modelObj.arrivalProfiles = queryModelArrivalProfile(
            *ctrl_metricsServerConn,
            ctrl_experimentName,
            ctrl_systemName,
            t.name,
            t.source,
            ctrl_containerLib[modelName].taskName,
            ctrl_containerLib[modelName].modelName,
            possibleDeviceList,
            possibleNetworkEntryPairs
        );
    }
    // ScaleFactorType scale_factors;
    // // Query arrival rates of individual models
    // for (auto &m: models) {
    //     arrival_rates = {
    //         {1, -1}, //1 second
    //         {3, -1},
    //         {7, -1},
    //         {15, -1},
    //         {30, -1},
    //         {60, -1}
    //     };

    //     scale_factors = {
    //         {1, 1},
    //         {3, 1},
    //         {7, 1},
    //         {15, 1},
    //         {30, 1},
    //         {60, 1}
    //     };

    //     // Get the name of the model
    //     // substr(1) is used to remove the colon at the beginning of the model name
    //     std::string model_name = t.name + "_" + MODEL_INFO[std::get<0>(m)][0].substr(1);

    //     // Query the request rate for each time period
    //     queryRequestRateInPeriod(model_name + "_arrival_table", arrival_rates);
    //     // Query the scale factor (ratio of number of outputs / each input) for each time period
    //     queryScaleFactorInPeriod(model_name + "_process_table", scale_factors);

    //     m.second.arrivalRate = std::max_element(arrival_rates.begin(), arrival_rates.end(),
    //                                           [](const std::pair<int, float> &p1, const std::pair<int, float> &p2) {
    //                                               return p1.second < p2.second;
    //                                           })->second;
    //     m.second.scaleFactors = scale_factors;
    //     m.second.modelProfile = queryModelProfile(model_name, DEVICE_INFO[device->type]);
    //     m.second.expectedTransmitLatency = queryTransmitLatency(m.second.modelProfile.avgInputSize, t.source, m.second.device);
    // }

    // std::string tmp = t.name;
    // containers.insert({tmp.append(":datasource"), {tmp, DataSource, device, task, 33, 9999, 0, 1, {-1}}});
    // task->subtasks.insert({tmp, &containers[tmp]});
    // task->subtasks[tmp]->recv_port = device->next_free_port++;
    // device->containers.insert({tmp, task->subtasks[tmp]});
    // device = &devices["server"];

    // // Find an initial batch size and replica configuration that meets the SLO at the server
    // getInitialBatchSizes(models, t.slo, 10);

    // // Try to shift model to edge devices
    // shiftModelToEdge(models, ModelType::DataSource, t.slo);

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


/**
 * @brief Recursively traverse the model tree and try shifting models to edge devices
 * 
 * @param models 
 * @param slo
 */
void Controller::shiftModelToEdge(PipelineModelListType &models, const ModelType &currModel, uint64_t slo) {
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
        incNumReplicas(m.second, m.second.device);
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
                decNumReplicas(m.second, m.second.device);
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
 * @brief estimate the different types of latency, in microseconds
 * Due to batch inference's nature, the queries that come later has to wait for more time both in preprocessor and postprocessor.
 * 
 * @param model infomation about the model
 * @param modelType 
 */
void Controller::estimateModelLatency(PipelineModel &model, const ModelType modelType, const std::string& deviceName) {
    ModelProfile profile = model.processProfiles[deviceName];
    uint64_t preprocessLatency = profile.p95prepLat;
    BatchSizeType batchSize = model.batchSize;
    uint64_t inferLatency = profile.batchInfer[batchSize].p95inferLat;
    uint64_t postprocessLatency =  profile.p95postLat;
    float preprocessRate = 1000000.f / preprocessLatency;

    model.expectedQueueingLatency = calculateQueuingLatency(model.arrivalProfiles.arrivalRates, preprocessRate);
    model.expectedAvgPerQueryLatency = preprocessLatency + inferLatency * batchSize + postprocessLatency;
    model.expectedMaxProcessLatency = preprocessLatency * batchSize + inferLatency * batchSize + postprocessLatency * batchSize;
    model.estimatedPerQueryCost = model.expectedAvgPerQueryLatency + model.expectedQueueingLatency + model.expectedTransferLatency;
}

/**
 * @brief DFS-style recursively estimate the latency of a pipeline from source to sink
 * 
 * @param pipeline provides all information about the pipeline needed for scheduling
 * @param currModel 
 */
void Controller::estimatePipelineLatency(PipelineModelListType &pipeline, const ModelType &currModel, const uint64_t start2HereLatency) {
    estimateModelLatency(pipeline.at(currModel), currModel, pipeline.at(currModel).device);

    // Update the expected latency to reach the current model
    // In case a model has multiple upstreams, the expected latency to reach the model is the maximum of the expected latency 
    // to reach from each upstream.
    pipeline.at(currModel).expectedStart2HereLatency = std::max(
        pipeline.at(currModel).expectedStart2HereLatency,
        start2HereLatency + pipeline.at(currModel).expectedMaxProcessLatency + pipeline.at(currModel).expectedTransferLatency + pipeline.at(currModel).expectedQueueingLatency
    );

    // Cost of the pipeline until the current model
    pipeline.at(currModel).estimatedStart2HereCost += pipeline.at(currModel).estimatedPerQueryCost;

    std::vector<std::pair<ModelType, int>> downstreams = pipeline.at(currModel).downstreams;
    for (const auto &d: downstreams) {
        estimatePipelineLatency(pipeline, d.first, pipeline.at(currModel).expectedStart2HereLatency);
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
void Controller::incNumReplicas(PipelineModel &model, const std::string& deviceName) {
    uint8_t numReplicas = model.numReplicas;
    uint64_t inferenceLatency = model.processProfiles[deviceName].batchInfer[model.batchSize].p95inferLat;
    float indiProcessRate = 1 / (inferenceLatency + model.processProfiles[deviceName].p95prepLat + model.processProfiles[deviceName].p95postLat);
    float processRate = indiProcessRate * numReplicas;
    while (processRate < model.arrivalProfiles.arrivalRates) {
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
void Controller::decNumReplicas(PipelineModel &model, const std::string& deviceName) {
    uint8_t numReplicas = model.numReplicas;
    uint64_t inferenceLatency = model.processProfiles[deviceName].batchInfer[model.batchSize].p95inferLat;
    float indiProcessRate = 1 / (inferenceLatency + model.processProfiles[deviceName].p95prepLat + model.processProfiles[deviceName].p95postLat);
    float processRate = indiProcessRate * numReplicas;
    while (numReplicas > 1) {
        numReplicas--;
        processRate = indiProcessRate * numReplicas;
        // If the number of replicas is no longer enough to meet the arrival rate, we should not decrease the number of replicas anymore.
        if (processRate < model.arrivalProfiles.arrivalRates) {
            numReplicas++;
            break;
        }
    }
    model.numReplicas = numReplicas;
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