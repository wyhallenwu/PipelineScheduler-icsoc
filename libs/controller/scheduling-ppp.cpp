#include "scheduling-ppp.h"

std::string DeviceNameToType(std::string name) {
    if (name == "server") {
        return "server";
    } else {
        return name.substr(0, name.size() - 1);
    }
}

bool Controller::AddTask(const TaskDescription::TaskStruct &t) {
    std::cout << "Adding task: " << t.name << std::endl;
    TaskHandle *task = new TaskHandle{t.name, t.fullName, t.type, t.source, t.device, t.slo, {}, 0};

    std::map<std::string, NodeHandle*> deviceList = devices.getMap();

    if (deviceList.find(t.device) == deviceList.end()) {
        spdlog::error("Device {0:s} is not connected", t.device);
        return false;
    }

    while (!deviceList.at(t.device)->initialNetworkCheck) {
        spdlog::get("container_agent")->info("Waiting for device {0:s} to finish network check", t.device);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    task->tk_src_device = t.device;

    task->tk_pipelineModels = getModelsByPipelineType(t.type, t.device);

    for (auto &model: task->tk_pipelineModels) {
        if (model->name.find("sink") != std::string::npos) {
            continue;
        } else if (model->name.find("datasource") != std::string::npos) {
            model->arrivalProfiles.arrivalRates = 30;
            this->client_profiles_jf.add(model);
            continue;
        }
        model->deviceTypeName = getDeviceTypeName(deviceList.at(model->device)->type);
        std::vector<std::string> upstreamPossibleDeviceList = model->upstreams.front().first->possibleDevices;
        std::vector<std::string> thisPossibleDeviceList = model->possibleDevices;
        std::vector<std::pair<std::string, std::string>> possibleDevicePairList;
        for (const auto &deviceName : upstreamPossibleDeviceList) {
            for (const auto &deviceName2 : thisPossibleDeviceList) {
                if (deviceName == "server" && deviceName2 != deviceName) {
                    continue;
                }
                possibleDevicePairList.push_back({deviceName, deviceName2});
            }
        }
        std::string containerName = model->name + "-" + model->deviceTypeName;
        model->arrivalProfiles.arrivalRates = queryArrivalRate(
            *ctrl_metricsServerConn,
            ctrl_experimentName,
            ctrl_systemName,
            t.name,
            t.source,
            ctrl_containerLib[containerName].taskName,
            ctrl_containerLib[containerName].modelName
        );

        for (const auto &pair : possibleDevicePairList) {
            std::string senderDeviceType = getDeviceTypeName(deviceList.at(pair.first)->type);
            std::string receiverDeviceType = getDeviceTypeName(deviceList.at(pair.second)->type);
            containerName = model->name + "-" + receiverDeviceType;
            std::unique_lock lock(devices.list[pair.first]->nodeHandleMutex);
            NetworkEntryType entry = devices.list[pair.first]->latestNetworkEntries[receiverDeviceType];
            lock.unlock();
            NetworkProfile test = queryNetworkProfile(
                *ctrl_metricsServerConn,
                ctrl_experimentName,
                ctrl_systemName,
                t.name,
                t.source,
                ctrl_containerLib[containerName].taskName,
                ctrl_containerLib[containerName].modelName,
                pair.first,
                senderDeviceType,
                pair.second,
                receiverDeviceType,
                entry
            );
            model->arrivalProfiles.d2dNetworkProfile[std::make_pair(pair.first, pair.second)] = test;
            model->possibleNetworkEntryPairs[std::make_pair(pair.first, pair.second)] = entry;
        }

        for (const auto deviceName : model->possibleDevices) {
            std::string deviceTypeName = getDeviceTypeName(deviceList.at(deviceName)->type);
            containerName = model->name + "-" + deviceTypeName;
            ModelProfile profile = queryModelProfile(
                *ctrl_metricsServerConn,
                ctrl_experimentName,
                ctrl_systemName,
                t.name,
                t.source,
                deviceName,
                deviceTypeName,
                ctrl_containerLib[containerName].modelName
            );
            model->processProfiles[deviceTypeName] = profile;

            // MODIFICATION
            // collect the very first model of the pipeline, just use the yolo which is always the very first
            if (containerName.find("yolov") != std::string::npos) {
                // add all available batch_size profiling into consideration
                for (auto it = profile.batchInfer.begin(); it != profile.batchInfer.end(); ++it) {
                    BatchSizeType batch_size = it->first;
                    BatchInferProfile &batch_profile = it->second;

                    // note: the last three chars of the model name is the resolution it takes
                    int width = model->name == "yolov5n" ? 640 : std::stoi(model->name.substr(model->name.length() - 3));

                    // check the accuracy indicator, use dummy value just to reflect the capacity of the model(evaluate their performance in general)
                    this->model_profiles_jf.add(model->name, ACC_LEVEL_MAP.at(model->name),
                                                static_cast<int>(batch_size),
                                                static_cast<float>(batch_profile.p95inferLat), width, width,
                                                model); // height and width are the same
                }
            }
        }

    }
    std::lock_guard lock2(ctrl_unscheduledPipelines.tasksMutex);
    ctrl_unscheduledPipelines.list.insert({task->tk_name, task});

    std::cout << "Task added: " << t.name << std::endl;
    return true;
}

bool CheckMergable(const std::string &m) {
    if (m == "datasource") {
        return true;
    }
    if (m.find("yolov5") != std::string::npos) {
        return true;
    }
    if (m.find("retina1face") != std::string::npos) {
        return true;
    }
    return false;
}

ContainerHandle *Controller::TranslateToContainer(PipelineModel *model, NodeHandle *device, unsigned int i) {
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
    if (model->name == "datasource" || model->name == "yolov5ndsrc" || model->name == "retina1facedsrc") {
        container->dimensions = ctrl_containerLib[model->name].templateConfig["container"]["cont_pipeline"][0]["msvc_dataShape"][0].get<std::vector<int>>();
    } else if (model->name != "sink") {
        container->dimensions = ctrl_containerLib[model->name].templateConfig["container"]["cont_pipeline"][1]["msvc_dnstreamMicroservices"][0]["nb_expectedShape"][0].get<std::vector<int>>();
    }
    model->task->tk_subTasks[model->name].push_back(container);

    for (auto &downstream: model->downstreams) {
        for (auto &downstreamContainer: downstream.first->task->tk_subTasks[downstream.first->name]) {
            if (downstreamContainer->device_agent == device) {
                container->downstreams.push_back(downstreamContainer);
                downstreamContainer->upstreams.push_back(container);
            }
        }
    }
    for (auto &upstream: model->upstreams) {
        for (auto &upstreamContainer: upstream.first->task->tk_subTasks[upstream.first->name]) {
            if (upstreamContainer->device_agent == device) {
                container->upstreams.push_back(upstreamContainer);
                upstreamContainer->downstreams.push_back(container);
            }
        }
    }
    return container;
}

/**
 * @brief call this method after the pipeline models have been added to scheduled
 *
 */
void Controller::ApplyScheduling() {
    // collect all running containers by device and model name
    std::vector<ContainerHandle *> new_containers;
    std::unique_lock lock_devices(devices.devicesMutex);
    std::unique_lock lock_pipelines(ctrl_scheduledPipelines.tasksMutex);
    std::unique_lock lock_containers(containers.containersMutex);

    for (auto &pipe: ctrl_scheduledPipelines.list) {
        for (auto &model: pipe.second->tk_pipelineModels) {
            std::unique_lock lock_model(model->pipelineModelMutex);
            std::vector<ContainerHandle *> candidates = model->task->tk_subTasks[model->name];
            // make sure enough containers are running with the right configurations
            if (candidates.size() < model->numReplicas) {
                // start additional containers
                for (unsigned int i = candidates.size(); i < model->numReplicas; i++) {
                    ContainerHandle *container = TranslateToContainer(model, devices.list[model->device], i);
                    new_containers.push_back(container);
                }
            } else if (candidates.size() > model->numReplicas) {
                // remove the extra containers
                for (unsigned int i = model->numReplicas; i < candidates.size(); i++) {
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
            for (auto *candidate: candidates) {
                if (candidate->device_agent->name != model->device) {
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

    for (auto container: new_containers) {
        StartContainer(container);
        containers.list.insert({container->name, container});
    }
}

bool Controller::mergeArrivalProfiles(ModelArrivalProfile &mergedProfile, const ModelArrivalProfile &toBeMergedProfile) {
    mergedProfile.arrivalRates += toBeMergedProfile.arrivalRates;
    auto mergedD2DProfile = &mergedProfile.d2dNetworkProfile;
    auto toBeMergedD2DProfile = &toBeMergedProfile.d2dNetworkProfile;
    for (const auto &[pair, profile] : toBeMergedProfile.d2dNetworkProfile) {
        mergedD2DProfile->at(pair).p95TransferDuration = std::max(mergedD2DProfile->at(pair).p95TransferDuration,
                                                                  toBeMergedD2DProfile->at(pair).p95TransferDuration);
        mergedD2DProfile->at(pair).p95PackageSize = std::max(mergedD2DProfile->at(pair).p95PackageSize,
                                                             toBeMergedD2DProfile->at(pair).p95PackageSize);

    }
    return true;
}

bool Controller::mergeProcessProfiles(PerDeviceModelProfileType &mergedProfile, const PerDeviceModelProfileType &toBeMergedProfile) {
    for (const auto &[deviceName, profile] : toBeMergedProfile) {
        auto mergedProfileDevice = &mergedProfile[deviceName];
        auto toBeMergedProfileDevice = &toBeMergedProfile.at(deviceName);

        mergedProfileDevice->p95InputSize = std::max(mergedProfileDevice->p95InputSize, toBeMergedProfileDevice->p95InputSize);
        mergedProfileDevice->p95OutputSize = std::max(mergedProfileDevice->p95OutputSize, toBeMergedProfileDevice->p95OutputSize);
        mergedProfileDevice->p95prepLat = std::max(mergedProfileDevice->p95prepLat, toBeMergedProfileDevice->p95prepLat);
        mergedProfileDevice->p95postLat = std::max(mergedProfileDevice->p95postLat, toBeMergedProfileDevice->p95postLat);

        auto mergedBatchInfer = &mergedProfileDevice->batchInfer;
        // auto toBeMergedBatchInfer = &toBeMergedProfileDevice->batchInfer;

        for (const auto &[batchSize, p] : toBeMergedProfileDevice->batchInfer) {
            mergedBatchInfer->at(batchSize).p95inferLat = std::max(mergedBatchInfer->at(batchSize).p95inferLat, p.p95inferLat);
            mergedBatchInfer->at(batchSize).cpuUtil = std::max(mergedBatchInfer->at(batchSize).cpuUtil, p.cpuUtil);
            mergedBatchInfer->at(batchSize).gpuUtil = std::max(mergedBatchInfer->at(batchSize).gpuUtil, p.gpuUtil);
            mergedBatchInfer->at(batchSize).memUsage = std::max(mergedBatchInfer->at(batchSize).memUsage, p.memUsage);
            mergedBatchInfer->at(batchSize).rssMemUsage = std::max(mergedBatchInfer->at(batchSize).rssMemUsage, p.rssMemUsage);
            mergedBatchInfer->at(batchSize).gpuMemUsage = std::max(mergedBatchInfer->at(batchSize).gpuMemUsage, p.gpuMemUsage);
        }

    }
    return true;
}

bool Controller::mergeModels(PipelineModel *mergedModel, PipelineModel* toBeMergedModel) {
    // If the merged model is empty, we should just copy the model to be merged
    if (mergedModel->numReplicas == 0) {
        *mergedModel = *toBeMergedModel;
        return true;
    }
    // If the devices are different, we should not merge the models
    if (mergedModel->device != toBeMergedModel->device || toBeMergedModel->merged) {
        return false;
    }

    mergeArrivalProfiles(mergedModel->arrivalProfiles, toBeMergedModel->arrivalProfiles);
    mergeProcessProfiles(mergedModel->processProfiles, toBeMergedModel->processProfiles);


    bool merged = false;
    toBeMergedModel->merged = true;

}

TaskHandle Controller::mergePipelines(const std::string& taskName) {
    TaskHandle mergedPipeline;
    auto mergedPipelineModels = &(mergedPipeline.tk_pipelineModels);

    auto unscheduledTasks = ctrl_unscheduledPipelines.getMap();

    *mergedPipelineModels = getModelsByPipelineType(unscheduledTasks.at(taskName)->tk_type, "server");
    auto numModels = mergedPipeline.tk_pipelineModels.size();

    for (auto i = 0; i < numModels; i++) {
        if (mergedPipelineModels->at(i)->name == "datasource") {
            continue;
        }
        for (const auto& task : unscheduledTasks) {
            if (task.first == taskName) {
                continue;
            }
            mergeModels(mergedPipelineModels->at(i), task.second->tk_pipelineModels.at(i));
        }
    }
}

void Controller::mergePipelines() {
    std::vector<std::string> toMerge = {"traffic", "people"};
    TaskHandle mergedPipeline;

    for (const auto &taskName : toMerge) {
        mergedPipeline = mergePipelines(taskName);
    }
}

/**
 * @brief Recursively traverse the model tree and try shifting models to edge devices
 * 
 * @param models 
 * @param slo
 */
void Controller::shiftModelToEdge(PipelineModelListType &pipeline, PipelineModel *currModel, uint64_t slo,
                                  const std::string &edgeDevice) {
}

/**
 * @brief 
 * 
 * @param models 
 * @param slo 
 * @param nObjects 
 * @return std::map<ModelType, int> 
 */
void Controller::getInitialBatchSizes(TaskHandle &task, uint64_t slo) {
//
//    PipelineModelListType &models = task.tk_pipelineModels;
//
//    for (auto &m: models) {
//        m->batchSize = 1;
//        m->numReplicas = 1;
//
//        estimateModelLatency(m);
//    }
//
//
//    // DFS-style recursively estimate the latency of a pipeline from source to sink
//    // The first model should be the datasource
//    estimatePipelineLatency(models.front(), 0);
//
//    uint64_t expectedE2ELatency = models.back()->expectedStart2HereLatency;
//
//    if (slo < expectedE2ELatency) {
//        spdlog::info("SLO is too low for the pipeline to meet. Expected E2E latency: {0:d}, SLO: {1:d}", expectedE2ELatency, slo);
//    }
//
//    // Increase number of replicas to avoid bottlenecks
//    for (auto &m: models) {
//        auto numIncReplicas = incNumReplicas(m);
//        m->numReplicas += numIncReplicas;
//    }
//
//    // Find near-optimal batch sizes
//    auto foundBest = true;
//    while (foundBest) {
//        foundBest = false;
//        uint64_t bestCost = models.back()->estimatedStart2HereCost;
//        for (auto &m: models) {
//            BatchSizeType oldBatchsize = m->batchSize;
//            m->batchSize *= 2;
//            estimateModelLatency(m);
//            estimatePipelineLatency(models.front(), 0);
//            expectedE2ELatency = models.back()->expectedStart2HereLatency;
//            if (expectedE2ELatency < slo) {
//                // If increasing the batch size of model `m` creates a pipeline that meets the SLO, we should keep it
//                uint64_t estimatedE2Ecost = models.back()->estimatedStart2HereCost;
//                // Unless the estimated E2E cost is better than the best cost, we should not consider it as a candidate
//                if (estimatedE2Ecost < bestCost) {
//                    bestCost = estimatedE2Ecost;
//                    foundBest = true;
//                }
//                if (!foundBest) {
//                    m->batchSize = oldBatchsize;
//                    estimateModelLatency(m);
//                    continue;
//                }
//                // If increasing the batch size meets the SLO, we can try decreasing the number of replicas
//                auto numDecReplicas = decNumReplicas(m);
//                m->numReplicas -= numDecReplicas;
//            } else {
//                m->batchSize = oldBatchsize;
//                estimateModelLatency(m);
//            }
//        }
//    }
}

/**
 * @brief estimate the different types of latency, in microseconds
 * Due to batch inference's nature, the queries that come later has to wait for more time both in preprocessor and postprocessor.
 * 
 * @param model infomation about the model
 * @param modelType 
 */
void Controller::estimateModelLatency(PipelineModel *currModel) { std::string deviceName= currModel->device;
    // We assume datasource and sink models have no latency
    if (currModel->name == "datasource" || currModel->name == "sink") {
        currModel->expectedQueueingLatency = 0;
        currModel->expectedAvgPerQueryLatency = 0;
        currModel->expectedMaxProcessLatency = 0;
        currModel->estimatedPerQueryCost = 0;
        return;
    }
    ModelProfile profile = currModel->processProfiles[deviceName];
    uint64_t preprocessLatency = profile.p95prepLat;
    BatchSizeType batchSize = currModel->batchSize;
    uint64_t inferLatency = profile.batchInfer[batchSize].p95inferLat;
    uint64_t postprocessLatency = profile.p95postLat;
    float preprocessRate = 1000000.f / preprocessLatency;

    currModel->expectedQueueingLatency = calculateQueuingLatency(currModel->arrivalProfiles.arrivalRates,
                                                                 preprocessRate);
    currModel->expectedAvgPerQueryLatency = preprocessLatency + inferLatency * batchSize + postprocessLatency;
    currModel->expectedMaxProcessLatency =
            preprocessLatency * batchSize + inferLatency * batchSize + postprocessLatency * batchSize;
    currModel->estimatedPerQueryCost = currModel->expectedAvgPerQueryLatency + currModel->expectedQueueingLatency +
                                       currModel->expectedTransferLatency;
}

void Controller::estimateModelNetworkLatency(PipelineModel *currModel) {
    if (currModel->name == "datasource" || currModel->name == "sink") {
        currModel->expectedTransferLatency = 0;
        return;
    }

    currModel->expectedTransferLatency = currModel->arrivalProfiles.d2dNetworkProfile[std::make_pair(currModel->device, currModel->upstreams[0].first->device)].p95TransferDuration;
}

/**
 * @brief DFS-style recursively estimate the latency of a pipeline from source to sink
 * 
 * @param pipeline provides all information about the pipeline needed for scheduling
 * @param currModel 
 */
void Controller::estimatePipelineLatency(PipelineModel *currModel, const uint64_t start2HereLatency) {
    // estimateModelLatency(currModel, currModel->device);

    // Update the expected latency to reach the current model
    // In case a model has multiple upstreams, the expected latency to reach the model is the maximum of the expected latency 
    // to reach from each upstream.
    currModel->expectedStart2HereLatency = std::max(
            currModel->expectedStart2HereLatency,
            start2HereLatency + currModel->expectedMaxProcessLatency + currModel->expectedTransferLatency +
            currModel->expectedQueueingLatency
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

// ============================================================================================================================================
// ============================================================================================================================================
// ============================================================================================================================================
// ============================================================================================================================================

PipelineModelListType Controller::getModelsByPipelineType(PipelineType type, const std::string &startDevice) {
    switch (type) {
        case PipelineType::Traffic: {
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
                    {{datasource, -1}}
            };
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
                    {{yolov5n, 0}}
            };
            retina1face->possibleDevices = {startDevice, "server"};
            yolov5n->downstreams.push_back({retina1face, 0});

            auto *carbrand = new PipelineModel{
                    "server",
                    "carbrand",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 2}}
            };
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
                    {{yolov5n, 2}}
            };
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
                    {{retina1face, -1}, {carbrand, -1}, {platedet, -1}}
            };
            sink->possibleDevices = {"server"};
            retina1face->downstreams.push_back({sink, -1});
            carbrand->downstreams.push_back({sink, -1});
            platedet->downstreams.push_back({sink, -1});

            return {datasource, yolov5n, retina1face, carbrand, platedet, sink};
        }
        case PipelineType::Building_Security: {
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
                    {{datasource, -1}}
            };
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
                    {{yolov5n, 0}}
            };
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
                    {{yolov5n, 0}}
            };
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
                    {{retina1face, -1}}
            };
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
                    {{retina1face, -1}}
            };
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
                    {{gender, -1}, {age, -1}, {movenet, -1}}
            };
            sink->possibleDevices = {"server"};
            gender->downstreams.push_back({sink, -1});
            age->downstreams.push_back({sink, -1});
            movenet->downstreams.push_back({sink, -1});

            return {datasource, yolov5n, retina1face, movenet, gender, age, sink};
        }
        case PipelineType::Video_Call: {
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
                    {{datasource, -1}}
            };
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
                    {{retina1face, -1}}
            };
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
                    {{retina1face, -1}}
            };
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
                    {{retina1face, -1}}
            };
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
                    {{retina1face, -1}}
            };
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

// ----------------------------------------------------------------------------------------------------------------
//                                             implementations
// ----------------------------------------------------------------------------------------------------------------



bool ModelSetCompare::operator()(
        const std::tuple<std::string, float> &lhs,
        const std::tuple<std::string, float> &rhs) const {
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
void
ModelProfilesJF::add(std::string name, float accuracy, int batch_size, float inference_latency, int width, int height,
                     PipelineModel *m) {
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

void ModelProfilesJF::add(PipelineModel *model_info) {
    auto key =
            std::tuple<std::string, float>{model_info->name, model_info->accuracy};
    infos[key].push_back(model_info);
}

void ModelProfilesJF::debugging() {
    std::cout << "======================ModelProfiles Debugging=======================" << std::endl;
    for (auto it = infos.begin(); it != infos.end(); ++it) {
        auto key = it->first;
        auto profilings = it->second;
        std::cout << "*********************************************" << std::endl;
        std::cout << "Model: " << std::get<0>(key) << ", Accuracy: " << std::get<1>(key) << std::endl;
        for (const auto &model_info: profilings) {
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
void ClientProfilesJF::sortBudgetDescending(std::vector<PipelineModel *> &clients) {
    std::sort(clients.begin(), clients.end(),
              [](const PipelineModel *a, const PipelineModel *b) {
                  return a->task->tk_slo - a->expectedTransferLatency > b->task->tk_slo - b->expectedTransferLatency;
              });
}

void ClientProfilesJF::add(PipelineModel *m) {
    models.push_back(m);
}

void ClientProfilesJF::debugging() {
    std::cout << "===================================ClientProfiles Debugging==========================" << std::endl;
    for (const auto &client_model: models) {
        std::cout << "Unique id: " << client_model->device << ", buget: " << client_model->task->tk_slo << ", req_rate: "
                  << client_model->arrivalProfiles.arrivalRates << std::endl;
    }
}

// -------------------------------------------------------------------------------------------
//                               implementation of scheduling algorithms
// -------------------------------------------------------------------------------------------

void Controller::Scheduling() {
    while (running) {
        // update the networking time for each client-server pair
        // there is only one server in JellyFish, so get the server device from any recorded yolo model.
        // note: it's not allowed to be empty here, or it will cause the UB
        if (this->model_profiles_jf.infos.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5000));
            continue;
        }

        PipelineModel *model = this->model_profiles_jf.infos.begin()->second[0];
        std::unique_lock<std::mutex> lock(model->pipelineModelMutex);
        std::string server_device = model->device;
        lock.unlock();

        for (auto &client_model: client_profiles_jf.models) {
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

        for (auto &mapping: mappings) {
            // retrieve the mapping for one model and its paired clients
            auto model_info = std::get<0>(mapping);
            auto selected_clients = std::get<1>(mapping);
            int batch_size = std::get<2>(mapping);

            // find the PipelineModel* of that model
            PipelineModel *m = this->model_profiles_jf.infos[model_info][0];
            for (auto &model: this->model_profiles_jf.infos[model_info]) {
                if (model->batchSize == batch_size) {
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
            for (auto &client: selected_clients) {
                m->upstreams.push_back(std::make_pair(client, -1));
                std::unique_lock<std::mutex> client_lock(client->pipelineModelMutex);
                client->downstreams.clear();
                client->downstreams.push_back(std::make_pair(m, -1));

                // retrieve new resolution
                int width = m->width;
                int height = m->height;

                client_lock.unlock();

                std::unique_lock<std::mutex> container_lock(this->containers.containersMutex);
                for (auto it = this->containers.list.begin(); it != this->containers.list.end(); ++it) {
                    if (it->first == client->device) {
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
                                             std::vector<PipelineModel *> &clients) {
    // sort clients
    ClientProfilesJF::sortBudgetDescending(clients);
    std::cout << "findOptimal start" << std::endl;
    std::cout << "available sorted clients: " << std::endl;
    for (auto &client: clients) {
        std::cout << client->device << " " << client->task->tk_slo - client->expectedTransferLatency << " " << client->arrivalProfiles.arrivalRates
                  << std::endl;
    }
    std::cout << "available models: " << std::endl;
    for (auto &model: models) {
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
    for (auto &model: models) {
        if (model->throughput > max_throughput) {
            max_throughput = model->throughput;
        }
    }
    // init matrix
    int cols = max_throughput / h + 1;
    std::cout << "max_throughput: " << max_throughput << std::endl;
    std::cout << "row: " << rows << " cols: " << cols << std::endl;
    std::vector<std::vector<int>> dp_mat(rows, std::vector<int>(cols, 0));
    // iterating
    for (int client_index = 1; client_index < clients.size(); client_index++) {
        auto &client = clients[client_index];
        auto result = findMaxBatchSize(models, client, max_batch_size);
        max_batch_size = std::get<0>(result);
        max_index = std::get<1>(result);
        std::cout << "client ip: " << client->device << ", max_batch_size: " << max_batch_size << ", max_index: "
                  << max_index << std::endl;
        if (max_batch_size <= 0) {
            break;
        }
        int cols_upperbound = int(models[max_index]->throughput / h);
        int lambda_i = client->arrivalProfiles.arrivalRates;
        int v_i = client->arrivalProfiles.arrivalRates;
        std::cout << "cols_up " << cols_upperbound << ", req " << lambda_i
                  << std::endl;
        for (int k = 1; k <= cols_upperbound; k++) {

            int w_k = k * h;
            if (lambda_i <= w_k) {
                int k_prime = (w_k - lambda_i) / h;
                int v = v_i + dp_mat[client_index - 1][k_prime];
                if (v > dp_mat[client_index - 1][k]) {
                    dp_mat[client_index][k] = v;
                }
                if (v > best_value) {
                    best_cell = std::make_tuple(client_index, k);
                    best_value = v;
                }
            } else {
                dp_mat[client_index][k] = dp_mat[client_index - 1][k];
            }
        }
    }

    std::cout << "updated dp_mat" << std::endl;
    for (auto &row: dp_mat) {
        for (auto &v: row) {
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
    while (row > 0 && col > 0) {
        std::cout << row << " " << col << std::endl;
        if (dp_mat[row][col] == dp_mat[row - 1][col]) {
            row--;
        } else {
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
    for (auto sc: selected_clients) {
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
        std::tuple<std::tuple<std::string, float>, std::vector<PipelineModel* >, int>>
mapClient(ClientProfilesJF client_profile, ModelProfilesJF model_profiles) {
    std::cout << " ======================= mapClient ==========================" << std::endl;

    std::vector<
            std::tuple<std::tuple<std::string, float>, std::vector<PipelineModel *>, int>>
            mappings;
    std::vector<PipelineModel *> clients = client_profile.models;

    int map_size = model_profiles.infos.size();
    int key_index = 0;
    for (auto it = model_profiles.infos.begin(); it != model_profiles.infos.end();
         ++it) {
        key_index++;
        std::cout << "before filtering" << std::endl;
        for (auto &c: clients) {
            std::cout << c->device << " " << c->task->tk_slo << " " << c->arrivalProfiles.arrivalRates << std::endl;
        }

        auto selected_clients = findOptimalClients(it->second, clients);

        // tradeoff:
        // assign all left clients to the last available model
        if (key_index == map_size) {
            std::cout << "assign all rest clients" << std::endl;
            selected_clients = clients;
            clients.clear();
            std::cout << "selected clients assgined" << std::endl;
            for (auto &c: selected_clients) {
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
        for (auto &sc: selected_clients) {
            clients.erase(std::remove_if(clients.begin(), clients.end(),
                                         [&sc](const PipelineModel *c) {
                                             return c->device == sc->device;
                                         }), clients.end());
        }
        std::cout << "after filtering" << std::endl;
        for (auto &c: clients) {
            std::cout << c->device << " " << c->task->tk_slo << " " << c->arrivalProfiles.arrivalRates << std::endl;
        }
        if (clients.size() == 0) {
            break;
        }
    }

    std::cout << "mapping relation" << std::endl;
    for (auto &t: mappings) {
        std::cout << "======================" << std::endl;
        auto [model_info, clients, batch_size] = t;
        std::cout << std::get<0>(model_info) << " " << std::get<1>(model_info)
                  << " " << batch_size << std::endl;
        for (auto &client: clients) {
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
                     std::vector<PipelineModel> &selected_clients) {
    int total_req_rate = 0;
    // sum all selected req rate
    for (auto &client: selected_clients) {
        total_req_rate += client.arrivalProfiles.arrivalRates;
    }
    int max_batch_size = 1;

    for (auto &model_info: model) {
        if (model_info.throughput > total_req_rate &&
            max_batch_size < model_info.batchSize) {
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
int check_and_assign(std::vector<PipelineModel*> & model,
                     std::vector<PipelineModel*> & selected_clients)
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
                                      const PipelineModel *client, int max_available_batch_size) {
    int max_batch_size = 0;
    int index = 0;
    int max_index = 0;
    for (const auto &model: models) {
        // CHECKME: the inference time should be limited by (budget - transmission time)
        if (model->expectedMaxProcessLatency * 2.0 < client->task->tk_slo - client->expectedTransferLatency &&
            model->batchSize > max_batch_size && model->batchSize <= max_available_batch_size) {
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