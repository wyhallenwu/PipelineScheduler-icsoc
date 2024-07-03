#include "scheduling-ppp.h"

void Controller::initiateGPULanes(NodeHandle &node) {
    if (node.name == "server") {
        node.numGPULanes = NUM_GPUS * NUM_LANES_PER_GPU;
    } else {
        node.numGPULanes = 1;
    }

    for (auto &lane : node.gpuLanes) {
        delete lane;
    }

    node.gpuLanes.clear();
    for (auto &portion : node.freeGPUPortions.list) {
        delete portion;
    }
    node.freeGPUPortions.list.clear();
    node.freeGPUPortions = {};

    // Initialize the GPU execution portions
    for (auto i = 0; i < node.numGPULanes; i++) {
        GPULane *gpuLane = new GPULane{i / node.numGPULanes, i};
        node.gpuLanes.push_back(gpuLane);
    }

    // Initialize the GPU portions
    GPUPortion *prevPortion = nullptr;
    for (auto i = 0; i < node.numGPULanes; i++) {
        GPUPortion *gpuPortion = new GPUPortion{};
        gpuPortion->lane = node.gpuLanes[i];
        node.freeGPUPortions.list.push_back(gpuPortion);
        if (i == 0) {
            node.freeGPUPortions.head = gpuPortion;
        }
        prevPortion = gpuPortion;
        if (i > 0) {
            gpuPortion->prev = prevPortion;
            prevPortion->next = gpuPortion;
        }
    }
}

bool Controller::AddTask(const TaskDescription::TaskStruct &t) {
    std::cout << "Adding task: " << t.name << std::endl;
    TaskHandle *task = new TaskHandle{t.name, t.fullName, t.type, t.source, t.device, t.slo, {}, 0};

    std::map<std::string, NodeHandle> *deviceList = devices.getMap();

    if (devices.list.find(t.device) == deviceList->end()) {
        spdlog::error("Device {0:s} is not connected", t.device);
        return false;
    }

    task->tk_pipelineModels = getModelsByPipelineType(t.type, t.device);
    std::unique_lock lock(ctrl_unscheduledPipelines.tasksMutex);

    ctrl_unscheduledPipelines.list.insert({task->tk_name, *task});
    lock.unlock();


    std::vector<std::pair<std::string, std::string>> possibleDevicePairList = {{"server", "server"}};
    std::map<std::pair<std::string, std::string>, NetworkEntryType> possibleNetworkEntryPairs;

    for (const auto &pair: possibleDevicePairList) {
        std::unique_lock lock(devices.list[pair.first].nodeHandleMutex);
        possibleNetworkEntryPairs[pair] = devices.list[pair.first].latestNetworkEntries[pair.second];
        lock.unlock();
    }

    std::vector<std::string> possibleDeviceList = {"server"};

    for (auto& model: task->tk_pipelineModels) {
        std::string containerName = model->name + "-" + possibleDevicePairList[0].second;
        if (containerName.find("datasource") != std::string::npos || containerName.find("sink") != std::string::npos) {
            continue;
        }
        model->arrivalProfiles.arrivalRates = queryArrivalRate(
            *ctrl_metricsServerConn,
            ctrl_experimentName,
            ctrl_systemName,
            t.name,
            t.source,
            ctrl_containerLib[containerName].taskName,
            ctrl_containerLib[containerName].modelName
        );
        std::vector<std::string> upstreamPossibleDeviceList = model->upstreams.front().first->possibleDevices;
        std::vector<std::string> thisPossibleDeviceList = model->possibleDevices;
        for (const auto &deviceName : upstreamPossibleDeviceList) {
            for (const auto &deviceName2 : thisPossibleDeviceList) {
                if (deviceName == "server" && deviceName2 != deviceName) {
                    continue;
                }
                possibleDevicePairList.push_back({deviceName, deviceName2});
            }
        }

        for (const auto &pair : possibleDevicePairList) {
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
                possibleNetworkEntryPairs[pair]
            );
            model->arrivalProfiles.d2dNetworkProfile[std::make_pair(pair.first, pair.second)] = test;
        }

        for (const auto deviceName : possibleDeviceList) {
            std::string deviceTypeName = getDeviceTypeName(deviceList->at(deviceName).type);
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
    }
    std::lock_guard lock2(ctrl_unscheduledPipelines.tasksMutex);
    ctrl_unscheduledPipelines.list.insert({task->tk_name, *task});

    std::cout << "Task added: " << t.name << std::endl;
    return true;
}

bool CheckMergable(const std::string &m) {
    return m == "datasource" || m == "yolov5n" || m == "retina1face" || m == "yolov5ndsrc" || m == "retina1facedsrc";
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
        for (auto &model: pipe.second.tk_pipelineModels) {
            std::unique_lock lock_model(model->pipelineModelMutex);
            std::vector<ContainerHandle *> candidates = model->task->tk_subTasks[model->name];
            // make sure enough containers are running with the right configurations
            if (candidates.size() < model->numReplicas) {
                // start additional containers
                for (unsigned int i = candidates.size(); i < model->numReplicas; i++) {
                    ContainerHandle *container = TranslateToContainer(model, &devices.list[model->device], i);
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
                    MoveContainer(candidate, &devices.list[model->device]);
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
        auto toBeMergedBatchInfer = &toBeMergedProfileDevice->batchInfer;

        for (const auto &[batchSize, profile] : toBeMergedProfileDevice->batchInfer) {
            mergedBatchInfer->at(batchSize).p95inferLat = std::max(mergedBatchInfer->at(batchSize).p95inferLat, profile.p95inferLat);
            mergedBatchInfer->at(batchSize).cpuUtil = std::max(mergedBatchInfer->at(batchSize).cpuUtil, profile.cpuUtil);
            mergedBatchInfer->at(batchSize).gpuUtil = std::max(mergedBatchInfer->at(batchSize).gpuUtil, profile.gpuUtil);
            mergedBatchInfer->at(batchSize).memUsage = std::max(mergedBatchInfer->at(batchSize).memUsage, profile.memUsage);
            mergedBatchInfer->at(batchSize).rssMemUsage = std::max(mergedBatchInfer->at(batchSize).rssMemUsage, profile.rssMemUsage);
            mergedBatchInfer->at(batchSize).gpuMemUsage = std::max(mergedBatchInfer->at(batchSize).gpuMemUsage, profile.gpuMemUsage);
        }

    }
    return true;
}

bool Controller::mergeModels(PipelineModel *mergedModel, PipelineModel* toBeMergedModel) {
    // If the merged model is empty, we should just copy the model to be merged
    if (mergedModel->numReplicas == -1) {
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

    *mergedPipelineModels = getModelsByPipelineType(unscheduledTasks->at(taskName).tk_type, "server");
    auto numModels = mergedPipeline.tk_pipelineModels.size();

    for (auto i = 0; i < numModels; i++) {
        if (mergedPipelineModels->at(i)->name == "datasource") {
            continue;
        }
        for (const auto& task : *unscheduledTasks) {
            if (task.first == taskName) {
                continue;
            }
            mergeModels(mergedPipelineModels->at(i), task.second.tk_pipelineModels.at(i));
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
void Controller::shiftModelToEdge(PipelineModelListType &pipeline, PipelineModel *currModel, uint64_t slo, const std::string& edgeDevice) {
    if (currModel->name == "sink") {
        return;
    }
    if (currModel->name == "datasource") {
        if (currModel->device != edgeDevice) {
            spdlog::get("container_agent")->warn("Edge device {0:s} is not identical to the datasource device {1:s}", edgeDevice, currModel->device);
        }
        return;
    }

    if (currModel->device == edgeDevice) {
        for (auto &d: currModel->downstreams) {
            shiftModelToEdge(pipeline, d.first, slo, edgeDevice);
        }
    }

    // If the edge device is not in the list of possible devices, we should not consider it
    if (std::find(currModel->possibleDevices.begin(), currModel->possibleDevices.end(), edgeDevice) == currModel->possibleDevices.end()) {
        return;
    }

    std::string deviceTypeName = getDeviceTypeName(devices.list[edgeDevice].type);

    uint32_t inputSize = currModel->processProfiles.at(deviceTypeName).p95InputSize;
    uint32_t outputSize = currModel->processProfiles.at(deviceTypeName).p95OutputSize;

    if (inputSize * 0.3 < outputSize) {
        currModel->device = edgeDevice;
        estimateModelLatency(currModel);
        for (auto &downstream : currModel->downstreams) {
            estimateModelLatency(downstream.first);
        }
        estimatePipelineLatency(currModel, currModel->expectedStart2HereLatency);
        uint64_t expectedE2ELatency = pipeline.back()->expectedStart2HereLatency;
        // if after shifting the model to the edge device, the pipeline still meets the SLO, we should keep it

        // However, if the pipeline does not meet the SLO, we should shift reverse the model back to the server
        if (expectedE2ELatency > slo) {
            currModel->device = "server";
            estimateModelLatency(currModel);
            for (auto &downstream : currModel->downstreams) {
                estimateModelLatency(downstream.first);
            }
            estimatePipelineLatency(currModel, currModel->expectedStart2HereLatency);
        }
    }
    // Shift downstream models to the edge device
    for (auto &d: currModel->downstreams) {
        shiftModelToEdge(pipeline, d.first, slo, edgeDevice);
    }
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

    PipelineModelListType &models = task.tk_pipelineModels;

    for (auto &m: models) {
        m->batchSize = 1;
        m->numReplicas = 1;

        estimateModelLatency(m);
    }


    // DFS-style recursively estimate the latency of a pipeline from source to sink
    // The first model should be the datasource
    estimatePipelineLatency(models.front(), 0);

    uint64_t expectedE2ELatency = models.back()->expectedStart2HereLatency;

    if (slo < expectedE2ELatency) {
        spdlog::info("SLO is too low for the pipeline to meet. Expected E2E latency: {0:d}, SLO: {1:d}", expectedE2ELatency, slo);
    }

    // Increase number of replicas to avoid bottlenecks
    for (auto &m: models) {
        auto numIncReplicas = incNumReplicas(m);
        m->numReplicas += numIncReplicas;
    }

    // Find near-optimal batch sizes
    auto foundBest = true;
    while (foundBest) {
        foundBest = false;
        uint64_t bestCost = models.back()->estimatedStart2HereCost;
        for (auto &m: models) {
            BatchSizeType oldBatchsize = m->batchSize;
            m->batchSize *= 2;
            estimateModelLatency(m);
            estimatePipelineLatency(models.front(), 0);
            expectedE2ELatency = models.back()->expectedStart2HereLatency;
            if (expectedE2ELatency < slo) {
                // If increasing the batch size of model `m` creates a pipeline that meets the SLO, we should keep it
                uint64_t estimatedE2Ecost = models.back()->estimatedStart2HereCost;
                // Unless the estimated E2E cost is better than the best cost, we should not consider it as a candidate
                if (estimatedE2Ecost < bestCost) {
                    bestCost = estimatedE2Ecost;
                    foundBest = true;
                }
                if (!foundBest) {
                    m->batchSize = oldBatchsize;
                    estimateModelLatency(m);
                    continue;
                }
                // If increasing the batch size meets the SLO, we can try decreasing the number of replicas
                auto numDecReplicas = decNumReplicas(m);
                m->numReplicas -= numDecReplicas;
            } else {
                m->batchSize = oldBatchsize;
                estimateModelLatency(m);
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