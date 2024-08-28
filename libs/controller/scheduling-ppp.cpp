#include "scheduling-ppp.h"

// =========================================================GPU Lanes/Portions Control===========================================================
// ==============================================================================================================================================
// ==============================================================================================================================================
// ==============================================================================================================================================

void Controller::initiateGPULanes(NodeHandle &node) {
    // Currently only support powerful GPUs capable of running multiple models in parallel
    if (node.name == "sink") {
        return;
    }
    auto deviceList = devices.list;
    
    if (deviceList.find(node.name) == deviceList.end()) {
        spdlog::get("container_agent")->error("Device {0:s} is not found in the device list", node.name);
        return;
    }

    if (node.type == SystemDeviceType::Server) {
        node.numGPULanes = NUM_LANES_PER_GPU * NUM_GPUS;
    } else {
        node.numGPULanes = 1;
    }
    node.gpuHandles.clear();
    node.freeGPUPortions.list.clear();
    

    for (auto i = 0; i < node.numGPULanes; i++) {
        node.gpuLanes.push_back(new GPULane{});
        node.gpuLanes.back()->laneNum = i;
        node.gpuLanes.back()->dutyCycle = 0;
        node.gpuLanes.back()->gpuHandle = node.gpuHandles[i / NUM_LANES_PER_GPU];
        // Initially the number of portions is the number of lanes
        node.freeGPUPortions.list.push_back(new GPUPortion{});
        node.freeGPUPortions.list.back()->lane = node.gpuLanes.back();
        node.gpuLanes.back()->gpuHandle->freeGPUPortions.push_back(node.freeGPUPortions.list.back());

        if (i == 0) {
            node.freeGPUPortions.head = node.freeGPUPortions.list.back();
        } else {
            node.freeGPUPortions.list[i - 1]->next = node.freeGPUPortions.list.back();
            node.freeGPUPortions.list.back()->prev = node.freeGPUPortions.list[i - 1];
        }
    }
}


// ==============================================================================================================================================
// ==============================================================================================================================================
// ==============================================================================================================================================

// ==================================================================Scheduling==================================================================
// ==============================================================================================================================================
// ==============================================================================================================================================
// ==============================================================================================================================================

void Controller::queryingProfiles(TaskHandle *task) {

    std::map<std::string, NodeHandle*> deviceList = devices.getMap();

    auto pipelineModels = &task->tk_pipelineModels;

    for (auto model: *pipelineModels) {
        if (model->name.find("datasource") != std::string::npos || model->name.find("sink") != std::string::npos) {
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
        std::string containerName = model->name + "_" + model->deviceTypeName;
        if (!task->tk_newlyAdded) {
            model->arrivalProfiles.arrivalRates = queryArrivalRate(
                *ctrl_metricsServerConn,
                ctrl_experimentName,
                ctrl_systemName,
                task->tk_name,
                task->tk_source,
                ctrl_containerLib[containerName].taskName,
                ctrl_containerLib[containerName].modelName,
                ctrl_systemFPS
            );
        }

        for (const auto &pair : possibleDevicePairList) {
            std::string senderDeviceType = getDeviceTypeName(deviceList.at(pair.first)->type);
            std::string receiverDeviceType = getDeviceTypeName(deviceList.at(pair.second)->type);
            containerName = model->name + "_" + receiverDeviceType;
            std::unique_lock lock(devices.list[pair.first]->nodeHandleMutex);
            NetworkEntryType entry = devices.list[pair.first]->latestNetworkEntries[receiverDeviceType];
            lock.unlock();
            NetworkProfile test = queryNetworkProfile(
                *ctrl_metricsServerConn,
                ctrl_experimentName,
                ctrl_systemName,
                task->tk_name,
                task->tk_source,
                ctrl_containerLib[containerName].taskName,
                ctrl_containerLib[containerName].modelName,
                pair.first,
                senderDeviceType,
                pair.second,
                receiverDeviceType,
                entry,
                ctrl_systemFPS
            );
            model->arrivalProfiles.d2dNetworkProfile[std::make_pair(pair.first, pair.second)] = test;
        }

        for (const auto &deviceName : model->possibleDevices) {
            std::string deviceTypeName = getDeviceTypeName(deviceList.at(deviceName)->type);
            containerName = model->name + "_" + deviceTypeName;
            ModelProfile profile = queryModelProfile(
                *ctrl_metricsServerConn,
                ctrl_experimentName,
                ctrl_systemName,
                task->tk_name,
                task->tk_source,
                deviceName,
                deviceTypeName,
                ctrl_containerLib[containerName].modelName,
                ctrl_systemFPS
            );
            model->processProfiles[deviceTypeName] = profile;
            model->processProfiles[deviceTypeName].maxBatchSize = std::max_element(
                profile.batchInfer.begin(),
                profile.batchInfer.end(),
                [](const auto &p1, const auto &p2) {
                    return p1.first < p2.first;
                }
            )->first;
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
}

void Controller::Scheduling() {
    while (running) {
        // Check if it is the next scheduling period
        Stopwatch schedulingSW;
        schedulingSW.start();
        if (std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now() - ctrl_nextSchedulingTime).count() < 10) {
            continue;
        }
        ctrl_unscheduledPipelines = ctrl_savedUnscheduledPipelines;
        auto taskList = ctrl_unscheduledPipelines.getMap();
        if (!isPipelineInitialised) {
            continue;
        }

        for (auto &[taskName, taskHandle]: taskList) {
            queryingProfiles(taskHandle);
            getInitialBatchSizes(taskHandle, taskHandle->tk_slo / 2);
            shiftModelToEdge(taskHandle->tk_pipelineModels, taskHandle->tk_pipelineModels.front(), taskHandle->tk_slo / 2, taskHandle->tk_pipelineModels.front()->device);
            for (auto &model: taskHandle->tk_pipelineModels) {
                model->name = taskName + "_" + model->name;
            }
            estimateModelTiming(taskHandle->tk_pipelineModels.front(), 0);
            taskHandle->tk_newlyAdded = false;
        }

        mergePipelines();

        auto mergedTasks = ctrl_mergedPipelines.getMap();
        for (auto &[taskName, taskHandle]: mergedTasks) {
            for (auto &model: taskHandle->tk_pipelineModels) {
                for (auto i = 0; i < model->numReplicas; i++) {
                    model->cudaDevices.push_back(0); // Add dummy cuda device value to create a container manifestation
                    // model->manifestations.push_back(TranslateToContainer(model, devices.list[model->device], 0));
                    // model->manifestations.back()->task = taskHandle;
                }
                if (model->name.find("sink") != std::string::npos) {
                    model->device = "sink";
                }
            }
        }

        estimatePipelineTiming();
        ctrl_scheduledPipelines = ctrl_mergedPipelines;
        ApplyScheduling();
        schedulingSW.stop();
        ctrl_nextSchedulingTime = std::chrono::system_clock::now() + std::chrono::seconds(ctrl_schedulingIntervalSec);
        std::this_thread::sleep_for(TimePrecisionType((ctrl_schedulingIntervalSec + 1) * 1000000 - schedulingSW.elapsed_microseconds()));
    }
}

/**
 * @brief insert the newly created free portion into the sorted list of free portions
 * Since the list is sorted, we can insert the new portion by traversing the list from the head
 * Complexity: O(n)
 * 
 * @param head 
 * @param freePortion 
 */
void Controller::insertFreeGPUPortion(GPUPortionList &portionList, GPUPortion *freePortion) {
    auto &head = portionList.head;
    if (head == nullptr) {
        head = freePortion;
        return;
    }
    GPUPortion *curr = head;
    auto it = portionList.list.begin();
    while (true) {
        if ((curr->end - curr->start) >= (freePortion->end - freePortion->start)) {
            if (curr == head) {
                freePortion->next = curr;
                curr->prev = freePortion;
                head = freePortion;
                portionList.list.insert(it, freePortion);
                return;
            } else if ((curr->prev->end - curr->prev->start) < (freePortion->end - freePortion->start)) {
                freePortion->next = curr;
                freePortion->prev = curr->prev;
                curr->prev = freePortion;
                freePortion->prev->next = freePortion;
                head = freePortion;
                portionList.list.insert(it, freePortion);
                return;
            }
        } else {
            if (curr->next == nullptr) {
                curr->next = freePortion;
                freePortion->prev = curr;
                portionList.list.push_back(freePortion);
                return;
            } else if ((curr->next->end - curr->next->start) > (freePortion->end - freePortion->start)) {
                freePortion->next = curr->next;
                freePortion->prev = curr;
                curr->next = freePortion;
                freePortion->next->prev = freePortion;
                portionList.list.insert(it, freePortion);
                return;
            } else {
                curr = curr->next;
            }
        }
        it++;
    }
}

GPUPortion* Controller::findFreePortionForInsertion(GPUPortionList &portionList, ContainerHandle *container) {
    auto &head = portionList.head;
    GPUPortion *curr = head;
    while (true) {
        auto laneDutyCycle = curr->lane->dutyCycle;
        if (curr->start <= container->startTime && 
            curr->end >= container->endTime &&
            container->pipelineModel->localDutyCycle >= laneDutyCycle) {
            return curr;
        }
        if (curr->next == nullptr) {
            return nullptr;
        }
        curr = curr->next;
    }
}

/**
 * @brief 
 * 
 * @param node 
 * @param scheduledPortion 
 * @param toBeDividedFreePortion 
 */
std::pair<GPUPortion *, GPUPortion *> Controller::insertUsedGPUPortion(GPUPortionList &portionList, ContainerHandle *container, GPUPortion *toBeDividedFreePortion) {
    auto &head = portionList.head;
    // new portion on the left
    uint64_t newStart = toBeDividedFreePortion->start;
    uint64_t newEnd = container->startTime;

    auto gpuLane = toBeDividedFreePortion->lane;
    auto gpu = gpuLane->gpuHandle;

    GPUPortion* leftPortion = nullptr;
    GPUPortion* rightPortion = nullptr;
    // Create a new portion on the left only if it is large enough
    if (newEnd - newStart >= MINIMUM_PORTION_SIZE) {
        leftPortion = new GPUPortion{};
        leftPortion->start = newStart;
        leftPortion->end = newEnd;
        leftPortion->lane = gpuLane;
        gpu->freeGPUPortions.push_back(leftPortion);
    }

    // new portion on the right
    newStart = container->endTime;
    auto laneDutyCycle = gpuLane->dutyCycle;
    if (laneDutyCycle == 0) {
        if (container->pipelineModel->localDutyCycle == 0) {
            throw std::runtime_error("Duty cycle of the container 0");
        }
        int64_t slack = container->pipelineModel->task->tk_slo - container->pipelineModel->localDutyCycle * 2;
        if (slack < 0) {
            throw std::runtime_error("Slack is negative. Duty cycle is larger than the SLO");
        }
        laneDutyCycle = container->pipelineModel->localDutyCycle;
        newEnd = container->pipelineModel->localDutyCycle;
    } else {
        newEnd = toBeDividedFreePortion->end;
    }
    // Create a new portion on the right only if it is large enough
    if (newEnd - newStart >= MINIMUM_PORTION_SIZE) {
        rightPortion = new GPUPortion{};
        rightPortion->start = newStart;
        rightPortion->end = newEnd;
        rightPortion->lane = gpuLane;
        gpu->freeGPUPortions.push_back(rightPortion);
    }

    gpuLane->dutyCycle = laneDutyCycle;

    auto it = std::find(portionList.list.begin(), portionList.list.end(), toBeDividedFreePortion);
    portionList.list.erase(it);
    it = std::find(gpu->freeGPUPortions.begin(), gpu->freeGPUPortions.end(), toBeDividedFreePortion);
    gpu->freeGPUPortions.erase(it);



    // Delete the old portion as it has been divided into two new free portions and an occupied portion
    if (toBeDividedFreePortion->prev != nullptr) {
        toBeDividedFreePortion->prev->next = toBeDividedFreePortion->next;
    } else {
        head = toBeDividedFreePortion->next;
    }
    if (toBeDividedFreePortion->next != nullptr) {
        toBeDividedFreePortion->next->prev = toBeDividedFreePortion->prev;
    }
    // delete toBeDividedFreePortion;

    if (leftPortion != nullptr) {
        insertFreeGPUPortion(portionList, leftPortion);
    }

    if (rightPortion != nullptr) {
        insertFreeGPUPortion(portionList, rightPortion);
    }

    return {leftPortion, rightPortion};
}

bool Controller::containerTemporalScheduling(ContainerHandle *container) {
    std::string deviceName = container->device_agent->name;
    auto portion = findFreePortionForInsertion(devices.list[deviceName]->freeGPUPortions, container);

    if (portion == nullptr) {
        spdlog::get("container_agent")->error("No free portion found for container {0:s}", container->name);
        return false;
    }
    container->executionPortion = portion;
    container->gpuHandle = portion->lane->gpuHandle;
    auto newPortions = insertUsedGPUPortion(devices.list[deviceName]->freeGPUPortions, container, portion);

    return true;
}

bool Controller::modelTemporalScheduling(PipelineModel *pipelineModel, unsigned int replica_id) {
    if (pipelineModel->gpuScheduled) { return true; }
    if (pipelineModel->name.find("datasource") == std::string::npos &&
        (pipelineModel->name.find("dsrc") == std::string::npos ||
         pipelineModel->name.find("yolov5ndsrc") != std::string::npos) &&
        pipelineModel->name.find("sink") == std::string::npos) {
        for (auto &container : pipelineModel->task->tk_subTasks[pipelineModel->name]) {
            if (container->replica_id == replica_id) {
                container->startTime = pipelineModel->startTime;
                container->endTime = pipelineModel->endTime;
                container->batchingDeadline = pipelineModel->batchingDeadline;
                containerTemporalScheduling(container);
            }
        }
    }
    bool allScheduled = true;
    for (auto downstream : pipelineModel->downstreams) {
        if (!modelTemporalScheduling(downstream.first, replica_id)) allScheduled = false;
    }
    if (!allScheduled) return false;
    if (replica_id == pipelineModel->numReplicas - 1) {
        pipelineModel->gpuScheduled = true;
        return true;
    }
    return false;
}

void Controller::temporalScheduling() {
    for (auto &[deviceName, deviceHandle]: devices.list) {
        initiateGPULanes(*deviceHandle);
    }
    bool process_flag = true;
    unsigned int replica_id = 0;
    while (process_flag) {
        process_flag = false;
        for (auto &[taskName, taskHandle]: ctrl_scheduledPipelines.list) {
            auto front_model = taskHandle->tk_pipelineModels.front();
            if (!front_model->gpuScheduled) {
                process_flag = process_flag || !modelTemporalScheduling(front_model, replica_id);
            }
        }
        replica_id++;
    }
}

bool Controller::mergeArrivalProfiles(ModelArrivalProfile &mergedProfile, const ModelArrivalProfile &toBeMergedProfile, const std::string& device, const std::string& upstreamDevice) {
    mergedProfile.arrivalRates += toBeMergedProfile.arrivalRates;
    auto mergedD2DProfile = &mergedProfile.d2dNetworkProfile;
    auto toBeMergedD2DProfile = &toBeMergedProfile.d2dNetworkProfile;
    float coefficient1 = mergedProfile.arrivalRates / (mergedProfile.arrivalRates + toBeMergedProfile.arrivalRates);
    float coefficient2 = toBeMergedProfile.arrivalRates / (mergedProfile.arrivalRates + toBeMergedProfile.arrivalRates);

    // There should be only 1 pair in the d2dNetworkProfile with key {"merged-...", device}
    D2DNetworkProfile newProfile = {};
    auto mergedPair = mergedProfile.d2dNetworkProfile.key_comp();
    for (const auto &[pair1, profile2] : mergedProfile.d2dNetworkProfile) {
        for (const auto &[pair2, profile1] : toBeMergedProfile.d2dNetworkProfile) {
            if (pair2.first != upstreamDevice || pair2.second != device || pair2.second != pair1.second) {
                continue;
            }
            std::pair<std::string, std::string> newPair = {pair1.first + "_" + upstreamDevice, pair1.second};
            newProfile.insert({newPair, {}});
            newProfile[newPair].p95TransferDuration = 
                mergedD2DProfile->at(pair1).p95TransferDuration * coefficient1 +
                toBeMergedD2DProfile->at(pair2).p95TransferDuration * coefficient2;
            newProfile[newPair].p95PackageSize = 
                mergedD2DProfile->at(pair1).p95PackageSize * coefficient1 +
                toBeMergedD2DProfile->at(pair2).p95PackageSize * coefficient2;
        }
    }
    mergedProfile.d2dNetworkProfile = newProfile;
    return true;
}

bool Controller::mergeProcessProfiles(
    PerDeviceModelProfileType &mergedProfile,
    float arrivalRate1,
    const PerDeviceModelProfileType &toBeMergedProfile,
    float arrivalRate2,
    const std::string& device
) {
    float coefficient1 = arrivalRate1 / (arrivalRate1 + arrivalRate2);
    float coefficient2 = arrivalRate2 / (arrivalRate1 + arrivalRate2);
    for (const auto &[deviceName, profile] : toBeMergedProfile) {
        if (deviceName != device) {
            continue;
        }
        auto mergedProfileDevice = &mergedProfile[deviceName];
        auto toBeMergedProfileDevice = &toBeMergedProfile.at(deviceName);

        mergedProfileDevice->p95InputSize = std::max(mergedProfileDevice->p95InputSize, toBeMergedProfileDevice->p95InputSize);
        mergedProfileDevice->p95OutputSize = std::max(mergedProfileDevice->p95OutputSize, toBeMergedProfileDevice->p95OutputSize);

        auto mergedBatchInfer = &mergedProfileDevice->batchInfer;
        // auto toBeMergedBatchInfer = &toBeMergedProfileDevice->batchInfer;

        for (const auto &[batchSize, p] : toBeMergedProfileDevice->batchInfer) {
            mergedBatchInfer->at(batchSize).p95inferLat =
                mergedBatchInfer->at(batchSize).p95inferLat * coefficient1 + p.p95inferLat * coefficient2;
            mergedBatchInfer->at(batchSize).p95prepLat =
                mergedBatchInfer->at(batchSize).p95prepLat * coefficient1 + p.p95prepLat * coefficient2;
            mergedBatchInfer->at(batchSize).p95postLat =
                mergedBatchInfer->at(batchSize).p95postLat * coefficient1 + p.p95postLat * coefficient2;
            mergedBatchInfer->at(batchSize).cpuUtil = std::max(mergedBatchInfer->at(batchSize).cpuUtil, p.cpuUtil);
            mergedBatchInfer->at(batchSize).gpuUtil = std::max(mergedBatchInfer->at(batchSize).gpuUtil, p.gpuUtil);
            mergedBatchInfer->at(batchSize).memUsage = std::max(mergedBatchInfer->at(batchSize).memUsage, p.memUsage);
            mergedBatchInfer->at(batchSize).rssMemUsage = std::max(mergedBatchInfer->at(batchSize).rssMemUsage, p.rssMemUsage);
            mergedBatchInfer->at(batchSize).gpuMemUsage = std::max(mergedBatchInfer->at(batchSize).gpuMemUsage, p.gpuMemUsage);
        }

    }
    return true;
}

bool Controller::mergeModels(PipelineModel *mergedModel, PipelineModel* toBeMergedModel, const std::string& device) {
    // If the merged model is empty, we should just copy the model to be merged
    if (mergedModel->numReplicas == 255) {
        *mergedModel = *toBeMergedModel;
        std::string upStreamDevice = toBeMergedModel->upstreams.front().first->device;
        for (auto &[pair, profile] : mergedModel->arrivalProfiles.d2dNetworkProfile) {
            if (pair.second != device || pair.first != upStreamDevice) {
                continue;
            }
            mergedModel->arrivalProfiles.d2dNetworkProfile[{"merged-" + pair.first, device}] = profile;
        }
        std::vector<decltype(mergedModel->arrivalProfiles.d2dNetworkProfile.begin())> keysToErase;
        for (auto &[pair, profile] : mergedModel->arrivalProfiles.d2dNetworkProfile) {
            if (pair.first.find("merged") != std::string::npos) {
                continue;
            }
            keysToErase.push_back(mergedModel->arrivalProfiles.d2dNetworkProfile.find(pair));
        }
        for (auto key : keysToErase) {
            mergedModel->arrivalProfiles.d2dNetworkProfile.erase(key);
        }
        return true;
    }
    // If the devices are different, we should not merge the models
    if (mergedModel->device != toBeMergedModel->device || 
        toBeMergedModel->merged || mergedModel->device != device || toBeMergedModel->device != device) {
        return false;
    }

    
    float rate1 = mergedModel->arrivalProfiles.arrivalRates;
    float rate2 = toBeMergedModel->arrivalProfiles.arrivalRates;

    mergeArrivalProfiles(mergedModel->arrivalProfiles, toBeMergedModel->arrivalProfiles, device, toBeMergedModel->upstreams.front().first->device);
    mergeProcessProfiles(mergedModel->processProfiles, rate1, toBeMergedModel->processProfiles, rate2, device);

    return true;
}

TaskHandle* Controller::mergePipelines(const std::string& taskName) {

    auto unscheduledTasks = ctrl_unscheduledPipelines.getMap();

    PipelineType tk_type;
    for (const auto& task : unscheduledTasks) {
        if (task.first.find(taskName) == std::string::npos) {
            continue;
        }
        tk_type = task.second->tk_type;
        break;
    }


    TaskHandle* mergedPipeline = new TaskHandle{};
    bool found;
    for (const auto& task : unscheduledTasks) {
        if (task.first.find(taskName) == std::string::npos) {
            continue;
        }
        found = true;
        // Initialize the merged pipeline with one of the added tasks in the task type
        *mergedPipeline = *task.second;
    }
    if (!found) {
        spdlog::info("No task with type {0:s} has been added", taskName);
        return nullptr;
    }
    uint16_t numModels = mergedPipeline->tk_pipelineModels.size();

    for (auto &model : mergedPipeline->tk_pipelineModels) {
        model->toBeRun = true;
    }

    // Loop through all the models in the pipeline and merge models of the same type
    for (uint16_t i = 0; i < numModels; i++) {
        // Find this model in all the scheduled tasks
        for (const auto& task : unscheduledTasks) {
            // This task is the one we used to initialize the merged pipeline, we should not merge it
            if (task.second->tk_name == mergedPipeline->tk_name) {
                continue;
            }
            // Cannot merge if model is they do not belong to the same task group
            if (task.first.find(taskName) == std::string::npos) {
                continue;
            }
            // If model is not scheduled to be run on the server, we should not merge it.
            // However, the model is still 
            if (task.second->tk_pipelineModels[i]->device != "server") {
                mergedPipeline->tk_pipelineModels.emplace_back(new PipelineModel(*task.second->tk_pipelineModels[i]));
                task.second->tk_pipelineModels[i]->merged = true;
                task.second->tk_pipelineModels[i]->toBeRun = false;
                mergedPipeline->tk_pipelineModels.back()->toBeRun = false;
                continue;
            }
            // We attempt to merge to model i of this unscheduled task into the model i of the merged pipeline
            bool merged = mergeModels(mergedPipeline->tk_pipelineModels[i], task.second->tk_pipelineModels.at(i), "server");
            task.second->tk_pipelineModels.at(i)->merged = true;
            task.second->tk_pipelineModels.at(i)->toBeRun = false;
        }
        // auto numIncReps = incNumReplicas(mergedPipeline.tk_pipelineModels[i]);
        // mergedPipeline.tk_pipelineModels[i]->numReplicas += numIncReps;
        // auto deviceList = devices.getMap();
        // for (auto j = 0; j < mergedPipeline.tk_pipelineModels[i]->numReplicas; j++) {
        //     mergedPipeline.tk_pipelineModels[i]->manifestations.emplace_back(new ContainerHandle{});
        //     mergedPipeline.tk_pipelineModels[i]->manifestations.back()->task = &mergedPipeline;
        //     mergedPipeline.tk_pipelineModels[i]->manifestations.back()->device_agent = deviceList.at(mergedPipeline.tk_pipelineModels[i]->device);
        // }
    }
    for (auto &model : mergedPipeline->tk_pipelineModels) {
        // If toBeRun is true means the model is not a newly added one, there's no need to modify its up and downstreams
        if (model->toBeRun) {
            continue;
        }
        for (auto &oldDownstream : model->downstreams) {
            std::string oldDnstreamModelName = splitString(oldDownstream.first->name, "_").back();
            for (auto &newDownstream : mergedPipeline->tk_pipelineModels) {
                std::string newDownstreamModelName = splitString(newDownstream->name, "_").back();
                if (oldDnstreamModelName == newDownstreamModelName && 
                    oldDownstream.first->device == newDownstream->device &&
                    oldDownstream.first != newDownstream) {
                    model->downstreams.emplace_back(std::make_pair(newDownstream, oldDownstream.second));
                    newDownstream->upstreams.emplace_back(std::make_pair(model, oldDownstream.second));
                    break;
                }
            }
        }
        model->toBeRun = true;
    }
    for (auto &model : mergedPipeline->tk_pipelineModels) {
        for (auto it = model->downstreams.begin(); it != model->downstreams.end();) {
            if (!(it->first->toBeRun)) {
                it = model->downstreams.erase(it);
            } else {
                it++;
            }
        }
        for (auto it = model->upstreams.begin(); it != model->upstreams.end();) {
            if (!(it->first->toBeRun)) {
                it = model->upstreams.erase(it);
            } else {
                it++;
            }
        }
    }
    for (auto &model : mergedPipeline->tk_pipelineModels) {
        if (model->device != "server") {
            continue;
        }
        auto names = splitString(model->name, "_");
        model->name = taskName + "_" + names[1];
    }
    mergedPipeline->tk_src_device = "merged";
    mergedPipeline->tk_name = taskName.substr(0, taskName.length());
    mergedPipeline->tk_source  = "merged";
    return mergedPipeline;
}

void Controller::mergePipelines() {
    std::vector<std::string> toMerge = {"traffic", "people"};
    TaskHandle* mergedPipeline;

    for (const auto &taskName : toMerge) {
        mergedPipeline = mergePipelines(taskName);
        if (mergedPipeline == nullptr) {
            continue;
        }
        // Increase the number of replicas to avoid bottlenecks
        for (auto &mergedModel : mergedPipeline->tk_pipelineModels) {
            // only models scheduled to run on the server are merged and to be considered
            if (mergedModel->device != "server") {
                continue;
            }
            auto numIncReps = incNumReplicas(mergedModel);
            mergedModel->numReplicas += numIncReps;
            estimateModelLatency(mergedModel);
        }
        for (auto &mergedModel : mergedPipeline->tk_pipelineModels) {
            if (mergedModel->name.find("datasource") == std::string::npos) {
                continue;
            }
            estimatePipelineLatency(mergedModel, mergedModel->expectedStart2HereLatency);
        }
        for (auto &mergedModel : mergedPipeline->tk_pipelineModels) {
            mergedModel->task = mergedPipeline;
        }
        std::lock_guard lock(ctrl_mergedPipelines.tasksMutex);
        ctrl_mergedPipelines.list.insert({mergedPipeline->tk_name, mergedPipeline});
    }
}

/**
 * @brief Recursively traverse the model tree and try shifting models to edge devices
 * 
 * @param models 
 * @param slo
 */
void Controller::shiftModelToEdge(PipelineModelListType &pipeline, PipelineModel *currModel, uint64_t slo, const std::string& edgeDevice) {
    if (currModel->name.find("sink") != std::string::npos) {
        return;
    }
    if (currModel->name.find("datasource") != std::string::npos) {
        if (currModel->device != edgeDevice) {
            spdlog::get("container_agent")->warn("Edge device {0:s} is not identical to the datasource device {1:s}", edgeDevice, currModel->device);
            return;
        }
    }

    if (currModel->device == edgeDevice) {
        for (auto d: currModel->downstreams) {
            shiftModelToEdge(pipeline, d.first, slo, edgeDevice);
        }
        return;
    }

    // If the edge device is not in the list of possible devices, we should not consider it
    if (std::find(currModel->possibleDevices.begin(), currModel->possibleDevices.end(), edgeDevice) == currModel->possibleDevices.end()) {
        return;
    }

    std::string deviceTypeName = getDeviceTypeName(devices.list[edgeDevice]->type);

    uint32_t inputSize = currModel->processProfiles.at(deviceTypeName).p95InputSize;
    uint32_t outputSize = currModel->processProfiles.at(deviceTypeName).p95OutputSize;
    bool shifted = false;

    if (inputSize * 0.6 > outputSize) {
        PipelineModel oldModel = *currModel;
        currModel->device = edgeDevice;
        currModel->deviceTypeName = deviceTypeName;
        for (auto downstream : currModel->downstreams) {
            estimateModelLatency(downstream.first);
        }
        currModel->batchSize = 1;
        if (currModel->batchSize <= currModel->processProfiles[deviceTypeName].maxBatchSize) {
            estimateModelLatency(currModel);
            estimatePipelineLatency(currModel, currModel->expectedStart2HereLatency);
            uint64_t expectedE2ELatency = pipeline.back()->expectedStart2HereLatency;
            if (expectedE2ELatency > slo) {
                *currModel = oldModel;
                // break;
            }
            oldModel = *currModel;
            shifted = true;
            // currModel->batchSize *= 2;
        }
        // if after shifting the model to the edge device, the pipeline still meets the SLO, we should keep it

        // However, if the pipeline does not meet the SLO, we should shift reverse the model back to the server
        if (!shifted) {
            *currModel = oldModel;
            estimateModelLatency(currModel);
            for (auto &downstream : currModel->downstreams) {
                estimateModelLatency(downstream.first);
            }
            estimatePipelineLatency(currModel, currModel->expectedStart2HereLatency);
            return;
        }
    }
    estimateModelLatency(currModel);
    estimatePipelineLatency(currModel, currModel->expectedStart2HereLatency);
    // Shift downstream models to the edge device
    for (auto d: currModel->downstreams) {
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
void Controller::getInitialBatchSizes(TaskHandle *task, uint64_t slo) {

    PipelineModelListType *models = &(task->tk_pipelineModels);

    for (auto m: *models) {
        m->batchSize = 1;
        m->numReplicas = 1;

        estimateModelLatency(m);
        if (m->name.find("datasource") == std::string::npos) {
            m->device = "server";
            m->deviceTypeName = "server";
        }
    }


    // DFS-style recursively estimate the latency of a pipeline from source to sink
    // The first model should be the datasource
    estimatePipelineLatency(models->front(), 0);

    uint64_t expectedE2ELatency = models->back()->expectedStart2HereLatency;

    if (slo < expectedE2ELatency) {
        spdlog::info("SLO is too low for the pipeline to meet. Expected E2E latency: {0:d}, SLO: {1:d}", expectedE2ELatency, slo);
    }

    // Increase number of replicas to avoid bottlenecks
    for (auto m: *models) {
        auto numIncReplicas = incNumReplicas(m);
        m->numReplicas += numIncReplicas;
    }
    estimatePipelineLatency(models->front(), 0);

    // Find near-optimal batch sizes
    auto foundBest = true;
    while (foundBest) {
        foundBest = false;
        uint64_t bestCost = models->back()->estimatedStart2HereCost;
        for (auto m: *models) {
            if (m->name.find("datasource") != std::string::npos || m->name.find("sink") != std::string::npos) {
                continue;
            }
            BatchSizeType oldBatchsize = m->batchSize;
            m->batchSize *= 2;
            if (m->batchSize > m->processProfiles[m->deviceTypeName].maxBatchSize) {
                m->batchSize = oldBatchsize;
                continue;
            }
            estimateModelLatency(m);
            estimatePipelineLatency(m, m->expectedStart2HereLatency);
            expectedE2ELatency = models->back()->expectedStart2HereLatency;
            if (expectedE2ELatency < slo) {
                // If increasing the batch size of model `m` creates a pipeline that meets the SLO, we should keep it
                uint64_t estimatedE2Ecost = models->back()->estimatedStart2HereCost;
                // Unless the estimated E2E cost is better than the best cost, we should not consider it as a candidate
                if (estimatedE2Ecost < bestCost) {
                    bestCost = estimatedE2Ecost;
                    foundBest = true;
                }
                if (!foundBest) {
                    m->batchSize = oldBatchsize;
                    estimateModelLatency(m);
                    estimatePipelineLatency(m, m->expectedStart2HereLatency);
                    continue;
                }
                // If increasing the batch size meets the SLO, we can try decreasing the number of replicas
                auto numDecReplicas = decNumReplicas(m);
                m->numReplicas -= numDecReplicas;
            } else {
                m->batchSize = oldBatchsize;
                estimateModelLatency(m);
                estimatePipelineLatency(m, m->expectedStart2HereLatency);
            }
        }
    }
    return;
}

/**
 * @brief estimate the different types of latency, in microseconds
 * Due to batch inference's nature, the queries that come later has to wait for more time both in preprocessor and postprocessor.
 * 
 * @param model infomation about the model
 * @param modelType 
 */
void Controller::estimateModelLatency(PipelineModel *currModel) {
    std::string deviceTypeName = currModel->deviceTypeName;
    // We assume datasource and sink models have no latency
    if (currModel->name.find("datasource") != std::string::npos || currModel->name.find("sink") != std::string::npos) {
        currModel->expectedQueueingLatency = 0;
        currModel->expectedAvgPerQueryLatency = 0;
        currModel->expectedMaxProcessLatency = 0;
        currModel->estimatedPerQueryCost = 0;
        currModel->expectedStart2HereLatency = 0;
        currModel->estimatedStart2HereCost = 0;
        return;
    }
    ModelProfile profile = currModel->processProfiles[deviceTypeName];
    BatchSizeType batchSize = currModel->batchSize;
    uint64_t preprocessLatency = profile.batchInfer[batchSize].p95prepLat;
    uint64_t inferLatency = profile.batchInfer[batchSize].p95inferLat;
    uint64_t postprocessLatency = profile.batchInfer[batchSize].p95postLat;
    float preprocessRate = 1000000.f / preprocessLatency;

    currModel->expectedQueueingLatency = calculateQueuingLatency(currModel->arrivalProfiles.arrivalRates,
                                                                 preprocessRate);
    currModel->expectedAvgPerQueryLatency = preprocessLatency + inferLatency * batchSize + postprocessLatency;
    currModel->expectedMaxProcessLatency =
            preprocessLatency * batchSize + inferLatency * batchSize + postprocessLatency * batchSize;
    currModel->estimatedPerQueryCost = preprocessLatency + inferLatency + postprocessLatency + currModel->expectedTransferLatency;
    currModel->expectedStart2HereLatency = 0;
    currModel->estimatedStart2HereCost = 0;
}

void Controller::estimateModelNetworkLatency(PipelineModel *currModel) {
    if (currModel->name.find("datasource") != std::string::npos || currModel->name.find("sink") != std::string::npos) {
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
    if (currModel->name.find("datasource") != std::string::npos) {
        currModel->expectedStart2HereLatency = start2HereLatency;
        currModel->estimatedStart2HereCost = 0;
    } else {
        currModel->estimatedStart2HereCost = currModel->estimatedPerQueryCost;
        currModel->expectedStart2HereLatency = 0;
        for (auto &upstream : currModel->upstreams) {
            currModel->estimatedStart2HereCost += upstream.first->estimatedStart2HereCost;
            currModel->expectedStart2HereLatency = std::max(
                currModel->expectedStart2HereLatency,
                upstream.first->expectedStart2HereLatency + currModel->expectedMaxProcessLatency + currModel->expectedTransferLatency +
                currModel->expectedQueueingLatency
            );
        }
            
    }

    std::vector<std::pair<PipelineModel *, int>> downstreams = currModel->downstreams;
    for (const auto &d: downstreams) {
        estimatePipelineLatency(d.first, currModel->expectedStart2HereLatency);
    }

    if (currModel->downstreams.size() == 0) {
        return;
    }
}

void Controller::estimateTimeBudgetLeft(PipelineModel *currModel)
{
    if (currModel->name.find("sink") != std::string::npos)
    {
        currModel->timeBudgetLeft = 0;
        return;
    } else if (currModel->name.find("datasource") != std::string::npos) {
        currModel->timeBudgetLeft = currModel->task->tk_slo;
    }
    
    uint64_t dnstreamBudget = 0;
    for (const auto &d : currModel->downstreams)
    {
        estimateTimeBudgetLeft(d.first);
        dnstreamBudget = std::max(dnstreamBudget, d.first->timeBudgetLeft);
    }
    currModel->timeBudgetLeft = dnstreamBudget * 1.2 +
                                (currModel->expectedQueueingLatency + currModel->expectedMaxProcessLatency) * 1.2;
}

void Controller::estimateModelTiming(PipelineModel *currModel, const uint64_t start2HereLatency) {

    if (currModel->name.find("datasource") != std::string::npos) {
        currModel->batchingDeadline = 0;
        currModel->startTime = 0;
        currModel->endTime = 0;
        // if (currModel->name.find("sink") != std::string::npos) {
        
    }
    else if (currModel->name.find("sink") != std::string::npos) {
        currModel->batchingDeadline = 0;
        currModel->startTime = 0;
        currModel->endTime = 0;
        for (auto &upstream : currModel->upstreams) {
            currModel->localDutyCycle = std::max(currModel->localDutyCycle, upstream.first->endTime);
        }
        return;
    } else {
        auto batchSize = currModel->batchSize;
        auto profile = currModel->processProfiles.at(currModel->deviceTypeName);

        uint64_t maxStartTime = std::max(currModel->startTime, start2HereLatency);
        for (auto &upstream : currModel->upstreams) {
            if (upstream.first->device != currModel->device) {
                continue;
            }
            // TODO: Add in-device transfer latency
            maxStartTime = std::max(maxStartTime, upstream.first->endTime);
        }
        currModel->startTime = maxStartTime;
        currModel->endTime = currModel->startTime + currModel->expectedMaxProcessLatency;
        currModel->batchingDeadline = currModel->endTime -
                                    profile.batchInfer.at(batchSize).p95inferLat * batchSize * 1.05 -
                                    profile.batchInfer.at(batchSize).p95postLat;
    }
    
    uint64_t maxDnstreamDutyCycle = currModel->localDutyCycle;
    for (auto &downstream : currModel->downstreams) {
        if (downstream.first->device != currModel->device &&
            downstream.first->name.find("sink") == std::string::npos) {
            estimateModelTiming(downstream.first, 0);
            maxDnstreamDutyCycle = std::max(maxDnstreamDutyCycle, start2HereLatency + currModel->expectedMaxProcessLatency);
            continue;
        }
        estimateModelTiming(downstream.first, start2HereLatency + currModel->expectedMaxProcessLatency);
        maxDnstreamDutyCycle = std::max(maxDnstreamDutyCycle, downstream.first->localDutyCycle);
    }
    currModel->localDutyCycle = maxDnstreamDutyCycle;
}

void Controller::estimatePipelineTiming() {
    auto tasks = ctrl_mergedPipelines.getMap();
    for (auto &[taskName, task]: tasks) {
        for (auto &model: task->tk_pipelineModels) {
            // If the model has already been estimated, we should not estimate it again
            if (model->endTime != 0 && model->startTime != 0) {
                continue;
            }
            // TODO
            estimateModelTiming(model, 0);
        }
        uint64_t localDutyCycle;
        // for (auto &model: task->tk_pipelineModels) {
        //     if (model->name.find("sink") != std::string::npos) {
        //         localDutyCycle = model->localDutyCycle;
        //     }
        // }
        // for (auto &model: task->tk_pipelineModels) {
        //     model->localDutyCycle = localDutyCycle;
        // }
        for (auto &model: task->tk_pipelineModels) {
            if (model->name.find("datasource") == std::string::npos &&
                model->name.find("dsrc") == std::string::npos) {
                
                continue;
            }
            estimateTimeBudgetLeft(model);
        }
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
    if (model->name.find("datasource") != std::string::npos || model->name.find("sink") != std::string::npos) {
        return 0;
    }
    uint8_t numReplicas = model->numReplicas;
    std::string deviceTypeName = model->deviceTypeName;
    ModelProfile profile = model->processProfiles.at(deviceTypeName);
    uint64_t inferenceLatency = profile.batchInfer.at(model->batchSize).p95inferLat;
    float indiProcessRate = 1000000.f / (inferenceLatency + profile.batchInfer.at(model->batchSize).p95prepLat
                                 + profile.batchInfer.at(model->batchSize).p95postLat);
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
    float indiProcessRate = 1 / (inferenceLatency + profile.batchInfer.at(model->batchSize).p95prepLat
                                 + profile.batchInfer.at(model->batchSize).p95postLat);
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
    if (rho > 1) {
        return 999999999;
    }
    float numQueriesInSystem = rho / (1 - rho);
    float averageQueueLength = rho * rho / (1 - rho);
    return (uint64_t) (averageQueueLength / arrival_rate * 1000000);
}

// ============================================================================================================================================
// ============================================================================================================================================
// ============================================================================================================================================
// ============================================================================================================================================