#include "scheduling-dis.h"
// #include "controller.h"

// ==================================================================Scheduling==================================================================
// ==============================================================================================================================================
// ==============================================================================================================================================
// ==============================================================================================================================================

void Controller::queryingProfiles(TaskHandle *task)
{

    std::map<std::string, NodeHandle *> deviceList = devices.getMap();

    auto pipelineModels = &task->tk_pipelineModels;

    for (auto model : *pipelineModels)
    {
        if (model->name.find("datasource") != std::string::npos || model->name.find("sink") != std::string::npos)
        {
            continue;
        }
        model->deviceTypeName = getDeviceTypeName(deviceList.at(model->device)->type);
        std::vector<std::string> upstreamPossibleDeviceList = model->upstreams.front().first->possibleDevices;
        std::vector<std::string> thisPossibleDeviceList = model->possibleDevices;
        std::vector<std::pair<std::string, std::string>> possibleDevicePairList;
        for (const auto &deviceName : upstreamPossibleDeviceList)
        {
            for (const auto &deviceName2 : thisPossibleDeviceList)
            {
                if (deviceName == "server" && deviceName2 != deviceName)
                {
                    continue;
                }
                possibleDevicePairList.push_back({deviceName, deviceName2});
            }
        }
        std::string containerName = model->name + "_" + model->deviceTypeName;
        if (!task->tk_newlyAdded)
        {
            model->arrivalProfiles.arrivalRates = queryArrivalRate(
                *ctrl_metricsServerConn,
                ctrl_experimentName,
                ctrl_systemName,
                task->tk_name,
                task->tk_source,
                ctrl_containerLib[containerName].taskName,
                ctrl_containerLib[containerName].modelName,
                ctrl_systemFPS);
        }

        for (const auto &pair : possibleDevicePairList)
        {
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
                ctrl_systemFPS);
            model->arrivalProfiles.d2dNetworkProfile[std::make_pair(pair.first, pair.second)] = test;
        }

        for (const auto deviceName : model->possibleDevices)
        {
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
                ctrl_systemFPS);
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
}

void Controller::Scheduling()
{
    while (running)
    {
        // use list of devices, tasks and containers to schedule depending on your algorithm
        // put helper functions as a private member function of the controller and write them at the bottom of this file.
        // std::vector<NodeHandle*> nodes;

        NodeHandle *edgePointer = nullptr;
        NodeHandle *serverPointer = nullptr;
        unsigned long totalEdgeMemory = 0, totalServerMemory = 0;
        // std::vector<std::unique_ptr<NodeHandle>> nodes;
        // int cuda_device = 2; // need to be add
        // std::unique_lock<std::mutex> lock(nodeHandleMutex);
        // std::unique_lock<std::mutex> lock(devices.devicesMutex);
        // for (const auto &devicePair : devices.list)
        // {
        //     nodes.push_back(devicePair.second);
        // }
        nodes.clear();

        auto pointers = devices.getList();

        {
            std::vector<NodeHandle> localNodes;
            for (const auto &ptr : pointers)
            {
                if (ptr != nullptr)
                {
                    localNodes.push_back(*ptr);
                }
            }

            nodes.swap(localNodes);
        }

        ctrl_unscheduledPipelines = ctrl_savedUnscheduledPipelines;
        auto taskList = ctrl_unscheduledPipelines.getMap();

        if (!isPipelineInitialised) {
            continue;
        }

        for (auto &taskPair : taskList)
        {
            auto task = taskPair.second;
            queryingProfiles(task);
            // Adding taskname to model name for clarity
            for (auto &model : task->tk_pipelineModels)
            {
                model->name = task->tk_name + "_" + model->name;
            }
        }



        // init Partitioner
        Partitioner partitioner;
        PipelineModel model;
        float ratio = 0.3;

        partitioner.BaseParPoint = ratio;

        scheduleBaseParPointLoop(&model, &partitioner, nodes);
        scheduleFineGrainedParPointLoop(&partitioner, nodes);
        DecideAndMoveContainer(&model, nodes, &partitioner, 2);

        for (auto &taskPair : taskList)
        {
            for (auto &model : taskPair.second->tk_pipelineModels)
            {
                if (model->name.find("datasource") != std::string::npos || model->name.find("sink") != std::string::npos)
                {
                    continue;
                }
                if (model->device == "server" && model->name.find("yolov5") != std::string::npos)
                {
                    model->batchSize = 16;
                }
                else if (model->device != "server")
                {
                    edgePointer = devices.list[model->device];
                    model->batchSize = 8;
                }
                else
                {
                    model->batchSize = 32;
                }
            }
        }

        ctrl_scheduledPipelines = ctrl_unscheduledPipelines;
//        int test = 0;
//        /**
//         * @brief Testing loop, removed in production
//         *
//         */
//        while (true) {
//            for (auto &taskPair : ctrl_scheduledPipelines.list)
//            {
//                std::vector<std::string> testDevices = {"server", taskPair.second->tk_pipelineModels.at(0)->device};
//                // Aritificialy change position of yolo model to test
//                taskPair.second->tk_pipelineModels.at(2)->device = testDevices.at(test % 2);
//            }
//            test++;
//            ApplyScheduling();
//            // Wait for 1 minute for the last scheduling to take effect
//            std::this_thread::sleep_for(std::chrono::milliseconds(60000));
//        }

        ApplyScheduling();
        std::cout << "end_scheduleBaseParPoint " << partitioner.BaseParPoint << std::endl;
        std::cout << "end_FineGrainedParPoint " << partitioner.FineGrainedOffset << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(
                ctrl_schedulingIntervalSec)); // sleep time can be adjusted to your algorithm or just left at 5 seconds for now
    }
}

bool Controller::containerTemporalScheduling(ContainerHandle *container)
{
}

bool Controller::modelTemporalScheduling(PipelineModel *pipelineModel)
{
    if (pipelineModel->name == "datasource" || pipelineModel->name == "sink")
    {
        return true;
    }
    for (auto container : pipelineModel->manifestations)
    {
        containerTemporalScheduling(container);
    }
    for (auto downstream : pipelineModel->downstreams)
    {
        modelTemporalScheduling(downstream.first);
    }
    return true;
}

void Controller::temporalScheduling()
{
    for (auto &[taskName, taskHandle] : ctrl_scheduledPipelines.list)
    {
    }
}

bool Controller::mergeArrivalProfiles(ModelArrivalProfile &mergedProfile, const ModelArrivalProfile &toBeMergedProfile)
{
    mergedProfile.arrivalRates += toBeMergedProfile.arrivalRates;
    auto mergedD2DProfile = &mergedProfile.d2dNetworkProfile;
    auto toBeMergedD2DProfile = &toBeMergedProfile.d2dNetworkProfile;
    for (const auto &[pair, profile] : toBeMergedProfile.d2dNetworkProfile)
    {
        mergedD2DProfile->at(pair).p95TransferDuration = std::max(mergedD2DProfile->at(pair).p95TransferDuration,
                                                                  toBeMergedD2DProfile->at(pair).p95TransferDuration);
        mergedD2DProfile->at(pair).p95PackageSize = std::max(mergedD2DProfile->at(pair).p95PackageSize,
                                                             toBeMergedD2DProfile->at(pair).p95PackageSize);
    }
    return true;
}

bool Controller::mergeProcessProfiles(PerDeviceModelProfileType &mergedProfile, const PerDeviceModelProfileType &toBeMergedProfile)
{
    for (const auto &[deviceName, profile] : toBeMergedProfile)
    {
        auto mergedProfileDevice = &mergedProfile[deviceName];
        auto toBeMergedProfileDevice = &toBeMergedProfile.at(deviceName);

        BatchSizeType batchSize =

            mergedProfileDevice->p95InputSize = std::max(mergedProfileDevice->p95InputSize, toBeMergedProfileDevice->p95InputSize);
        mergedProfileDevice->p95OutputSize = std::max(mergedProfileDevice->p95OutputSize, toBeMergedProfileDevice->p95OutputSize);
        // mergedProfileDevice->p95prepLat = std::max(mergedProfileDevice->p95prepLat, toBeMergedProfileDevice->p95prepLat);
        // mergedProfileDevice->p95postLat = std::max(mergedProfileDevice->p95postLat, toBeMergedProfileDevice->p95postLat);

        auto mergedBatchInfer = &mergedProfileDevice->batchInfer;
        // auto toBeMergedBatchInfer = &toBeMergedProfileDevice->batchInfer;

        for (const auto &[batchSize, p] : toBeMergedProfileDevice->batchInfer)
        {
            mergedBatchInfer->at(batchSize).p95inferLat = std::max(mergedBatchInfer->at(batchSize).p95inferLat, p.p95inferLat);
            mergedBatchInfer->at(batchSize).p95prepLat = std::max(mergedBatchInfer->at(batchSize).p95prepLat, p.p95prepLat);
            mergedBatchInfer->at(batchSize).p95postLat = std::max(mergedBatchInfer->at(batchSize).p95postLat, p.p95postLat);
            mergedBatchInfer->at(batchSize).cpuUtil = std::max(mergedBatchInfer->at(batchSize).cpuUtil, p.cpuUtil);
            mergedBatchInfer->at(batchSize).gpuUtil = std::max(mergedBatchInfer->at(batchSize).gpuUtil, p.gpuUtil);
            mergedBatchInfer->at(batchSize).memUsage = std::max(mergedBatchInfer->at(batchSize).memUsage, p.memUsage);
            mergedBatchInfer->at(batchSize).rssMemUsage = std::max(mergedBatchInfer->at(batchSize).rssMemUsage, p.rssMemUsage);
            mergedBatchInfer->at(batchSize).gpuMemUsage = std::max(mergedBatchInfer->at(batchSize).gpuMemUsage, p.gpuMemUsage);
        }
    }
    return true;
}

bool Controller::mergeModels(PipelineModel *mergedModel, PipelineModel *toBeMergedModel)
{
    // If the merged model is empty, we should just copy the model to be merged
    if (mergedModel->numReplicas == 0)
    {
        *mergedModel = *toBeMergedModel;
        return true;
    }
    // If the devices are different, we should not merge the models
    if (mergedModel->device != toBeMergedModel->device || toBeMergedModel->merged)
    {
        return false;
    }

    mergeArrivalProfiles(mergedModel->arrivalProfiles, toBeMergedModel->arrivalProfiles);
    mergeProcessProfiles(mergedModel->processProfiles, toBeMergedModel->processProfiles);

    bool merged = false;
    toBeMergedModel->merged = true;
}

TaskHandle Controller::mergePipelines(const std::string &taskName)
{
    TaskHandle mergedPipeline;
    auto mergedPipelineModels = &(mergedPipeline.tk_pipelineModels);

    auto unscheduledTasks = ctrl_unscheduledPipelines.getMap();

    *mergedPipelineModels = getModelsByPipelineType(unscheduledTasks.at(taskName)->tk_type, "server");
    uint16_t numModels = mergedPipeline.tk_pipelineModels.size();

    for (uint16_t i = 0; i < numModels; i++)
    {
        if (mergedPipelineModels->at(i)->name == "datasource")
        {
            continue;
        }
        for (const auto &task : unscheduledTasks)
        {
            if (task.first == taskName)
            {
                continue;
            }
            mergeModels(mergedPipelineModels->at(i), task.second->tk_pipelineModels.at(i));
        }
        auto numIncReps = incNumReplicas(mergedPipelineModels->at(i));
        mergedPipelineModels->at(i)->numReplicas += numIncReps;
        auto deviceList = devices.getMap();
        for (auto j = 0; j < mergedPipelineModels->at(i)->numReplicas; j++)
        {
            mergedPipelineModels->at(i)->manifestations.emplace_back(new ContainerHandle{});
            mergedPipelineModels->at(i)->manifestations.back()->task = &mergedPipeline;
            mergedPipelineModels->at(i)->manifestations.back()->device_agent = deviceList.at(mergedPipelineModels->at(i)->device);
        }
    }
}

void Controller::mergePipelines()
{
    std::vector<std::string> toMerge = {"traffic", "people"};
    TaskHandle mergedPipeline;

    for (const auto &taskName : toMerge)
    {
        mergedPipeline = mergePipelines(taskName);
        std::lock_guard lock(ctrl_scheduledPipelines.tasksMutex);
        ctrl_scheduledPipelines.list.insert({mergedPipeline.tk_name, &mergedPipeline});
    }
}

/**
 * @brief Recursively traverse the model tree and try shifting models to edge devices
 *
 * @param models
 * @param slo
 */
void Controller::shiftModelToEdge(PipelineModelListType &pipeline, PipelineModel *currModel, uint64_t slo, const std::string &edgeDevice)
{
    if (currModel->name == "sink")
    {
        return;
    }
    if (currModel->name == "datasource")
    {
        if (currModel->device != edgeDevice)
        {
            spdlog::get("container_agent")->warn("Edge device {0:s} is not identical to the datasource device {1:s}", edgeDevice, currModel->device);
        }
        return;
    }

    if (currModel->device == edgeDevice)
    {
        for (auto &d : currModel->downstreams)
        {
            shiftModelToEdge(pipeline, d.first, slo, edgeDevice);
        }
    }

    // If the edge device is not in the list of possible devices, we should not consider it
    if (std::find(currModel->possibleDevices.begin(), currModel->possibleDevices.end(), edgeDevice) == currModel->possibleDevices.end())
    {
        return;
    }

    std::string deviceTypeName = getDeviceTypeName(devices.list[edgeDevice]->type);

    uint32_t inputSize = currModel->processProfiles.at(deviceTypeName).p95InputSize;
    uint32_t outputSize = currModel->processProfiles.at(deviceTypeName).p95OutputSize;

    if (inputSize * 0.3 < outputSize)
    {
        currModel->device = edgeDevice;
        estimateModelLatency(currModel);
        for (auto &downstream : currModel->downstreams)
        {
            estimateModelLatency(downstream.first);
        }
        estimatePipelineLatency(currModel, currModel->expectedStart2HereLatency);
        uint64_t expectedE2ELatency = pipeline.back()->expectedStart2HereLatency;
        // if after shifting the model to the edge device, the pipeline still meets the SLO, we should keep it

        // However, if the pipeline does not meet the SLO, we should shift reverse the model back to the server
        if (expectedE2ELatency > slo)
        {
            currModel->device = "server";
            estimateModelLatency(currModel);
            for (auto &downstream : currModel->downstreams)
            {
                estimateModelLatency(downstream.first);
            }
            estimatePipelineLatency(currModel, currModel->expectedStart2HereLatency);
        }
    }
    // Shift downstream models to the edge device
    for (auto &d : currModel->downstreams)
    {
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
void Controller::getInitialBatchSizes(TaskHandle *task, uint64_t slo)
{

    PipelineModelListType *models = &(task->tk_pipelineModels);

    for (auto m : *models)
    {
        m->batchSize = 1;
        m->numReplicas = 1;

        estimateModelLatency(m);
    }

    // DFS-style recursively estimate the latency of a pipeline from source to sink
    // The first model should be the datasource
    estimatePipelineLatency(models->front(), 0);

    uint64_t expectedE2ELatency = models->back()->expectedStart2HereLatency;

    if (slo < expectedE2ELatency)
    {
        spdlog::info("SLO is too low for the pipeline to meet. Expected E2E latency: {0:d}, SLO: {1:d}", expectedE2ELatency, slo);
    }

    // Increase number of replicas to avoid bottlenecks
    for (auto m : *models)
    {
        auto numIncReplicas = incNumReplicas(m);
        m->numReplicas += numIncReplicas;
    }

    // Find near-optimal batch sizes
    auto foundBest = true;
    while (foundBest)
    {
        foundBest = false;
        uint64_t bestCost = models->back()->estimatedStart2HereCost;
        for (auto m : *models)
        {
            BatchSizeType oldBatchsize = m->batchSize;
            m->batchSize *= 2;
            estimateModelLatency(m);
            estimatePipelineLatency(models->front(), 0);
            expectedE2ELatency = models->back()->expectedStart2HereLatency;
            if (expectedE2ELatency < slo)
            {
                // If increasing the batch size of model `m` creates a pipeline that meets the SLO, we should keep it
                uint64_t estimatedE2Ecost = models->back()->estimatedStart2HereCost;
                // Unless the estimated E2E cost is better than the best cost, we should not consider it as a candidate
                if (estimatedE2Ecost < bestCost)
                {
                    bestCost = estimatedE2Ecost;
                    foundBest = true;
                }
                if (!foundBest)
                {
                    m->batchSize = oldBatchsize;
                    estimateModelLatency(m);
                    continue;
                }
                // If increasing the batch size meets the SLO, we can try decreasing the number of replicas
                auto numDecReplicas = decNumReplicas(m);
                m->numReplicas -= numDecReplicas;
            }
            else
            {
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
void Controller::estimateModelLatency(PipelineModel *currModel)
{
    std::string deviceName = currModel->device;
    // We assume datasource and sink models have no latency
    if (currModel->name == "datasource" || currModel->name == "sink")
    {
        currModel->expectedQueueingLatency = 0;
        currModel->expectedAvgPerQueryLatency = 0;
        currModel->expectedMaxProcessLatency = 0;
        currModel->estimatedPerQueryCost = 0;
        currModel->expectedStart2HereLatency = 0;
        currModel->estimatedStart2HereCost = 0;
        return;
    }
    ModelProfile profile = currModel->processProfiles[deviceName];
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
    currModel->estimatedPerQueryCost = currModel->expectedAvgPerQueryLatency + currModel->expectedQueueingLatency +
                                       currModel->expectedTransferLatency;
    currModel->expectedStart2HereLatency = 0;
    currModel->estimatedStart2HereCost = 0;
}

void Controller::estimateModelNetworkLatency(PipelineModel *currModel)
{
    if (currModel->name == "datasource" || currModel->name == "sink")
    {
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
void Controller::estimatePipelineLatency(PipelineModel *currModel, const uint64_t start2HereLatency)
{
    // estimateModelLatency(currModel, currModel->device);

    // Update the expected latency to reach the current model
    // In case a model has multiple upstreams, the expected latency to reach the model is the maximum of the expected latency
    // to reach from each upstream.
    if (currModel->name == "datasource")
    {
        currModel->expectedStart2HereLatency = start2HereLatency;
    }
    else
    {
        currModel->expectedStart2HereLatency = std::max(
            currModel->expectedStart2HereLatency,
            start2HereLatency + currModel->expectedMaxProcessLatency + currModel->expectedTransferLatency +
                currModel->expectedQueueingLatency);
    }

    // Cost of the pipeline until the current model
    currModel->estimatedStart2HereCost += currModel->estimatedPerQueryCost;

    std::vector<std::pair<PipelineModel *, int>> downstreams = currModel->downstreams;
    for (const auto &d : downstreams)
    {
        estimatePipelineLatency(d.first, currModel->expectedStart2HereLatency);
    }

    if (currModel->downstreams.size() == 0)
    {
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
uint8_t Controller::incNumReplicas(const PipelineModel *model)
{
    uint8_t numReplicas = model->numReplicas;
    std::string deviceTypeName = model->deviceTypeName;
    ModelProfile profile = model->processProfiles.at(deviceTypeName);
    uint64_t inferenceLatency = profile.batchInfer.at(model->batchSize).p95inferLat;
    float indiProcessRate = 1 / (inferenceLatency + profile.batchInfer.at(model->batchSize).p95prepLat + profile.batchInfer.at(model->batchSize).p95postLat);
    float processRate = indiProcessRate * numReplicas;
    while (processRate < model->arrivalProfiles.arrivalRates)
    {
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
uint8_t Controller::decNumReplicas(const PipelineModel *model)
{
    uint8_t numReplicas = model->numReplicas;
    std::string deviceTypeName = model->deviceTypeName;
    ModelProfile profile = model->processProfiles.at(deviceTypeName);
    uint64_t inferenceLatency = profile.batchInfer.at(model->batchSize).p95inferLat;
    float indiProcessRate = 1 / (inferenceLatency + profile.batchInfer.at(model->batchSize).p95prepLat + profile.batchInfer.at(model->batchSize).p95postLat);
    float processRate = indiProcessRate * numReplicas;
    while (numReplicas > 1)
    {
        numReplicas--;
        processRate = indiProcessRate * numReplicas;
        // If the number of replicas is no longer enough to meet the arrival rate, we should not decrease the number of replicas anymore.
        if (processRate < model->arrivalProfiles.arrivalRates)
        {
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
uint64_t Controller::calculateQueuingLatency(const float &arrival_rate, const float &preprocess_rate)
{
    float rho = arrival_rate / preprocess_rate;
    float numQueriesInSystem = rho / (1 - rho);
    float averageQueueLength = rho * rho / (1 - rho);
    return (uint64_t)(averageQueueLength / arrival_rate * 1000000);
}

///////////////////////////////////////////////////////////////////////distream add//////////////////////////////////////////////////////////////////////////////////////

double Controller::calculateTotalprocessedRate(const PipelineModel *model, const std::vector<NodeHandle> &nodes, bool is_edge)
{
    double totalRequestRate = 0.0;
    std::map<std::string, NodeHandle *> deviceList = devices.getMap();

    // Iterate over all unscheduled pipeline tasks
    for (const auto &taskPair : ctrl_unscheduledPipelines.list)
    {
        const auto &task = taskPair.second;
        // Iterate over all models in the task's pipeline
        for (auto &model : task->tk_pipelineModels)
        {
            if (deviceList.find(model->device) == deviceList.end())
            {
                continue;
            }

            // get devicename for the information for get the batchinfer for next step
            std::string deviceType = getDeviceTypeName(deviceList.at(model->device)->type);
            // std::cout << "calculateTotalprocessedRate deviceType " << deviceType << std::endl;
            // make sure the calculation is only for edge / server, because we need to is_edge to make sure which side information we need.
            if ((is_edge && deviceType != "server") || (!is_edge && deviceType == "server" && model->name.find("sink") == std::string::npos))
            {
                int batchInfer;
                if (is_edge)
                {
                    // calculate the info only on edge side
                    batchInfer = model->processProfiles[deviceType].batchInfer[8].p95inferLat;
                    // std::cout << "edge_batchInfer" << batchInfer << std::endl;
                }
                else
                {
                    // calculate info only the server side
                    batchInfer = model->processProfiles[deviceType].batchInfer[16].p95inferLat;
                    // std::cout << "server_batchInfer" << batchInfer << std::endl;
                }

                // calculate the tp because is ms so we need devided by 1000000
                double requestRate = (batchInfer == 0) ? 0.0 : 1000000.0 / batchInfer;
                totalRequestRate += requestRate;
                // std::cout << "totalRequestRate " << totalRequestRate << std::endl;
            }
        }
    }

    return totalRequestRate;
}

// calculate the queue based on arrival rate
int Controller::calculateTotalQueue(const std::vector<NodeHandle> &nodes, bool is_edge)
{
    // init the info
    double totalQueue = 0.0;
    std::map<std::string, NodeHandle *> deviceList = devices.getMap();

    // for loop every model in the system
    for (const auto &taskPair : ctrl_unscheduledPipelines.list)
    {
        const auto &task = taskPair.second;
        for (auto &model : task->tk_pipelineModels)
        {
            if (deviceList.find(model->device) == deviceList.end())
            {
                continue;
            }

            std::string deviceType = getDeviceTypeName(deviceList.at(model->device)->type);
            // std::cout << "calculateTotalprocessedRate deviceType " << deviceType << std::endl;
            // make sure the calculation is only for edge / server, because we need to is_edge to make sure which side information we need.
            if ((is_edge && deviceType != "server" && model->name.find("datasource") == std::string::npos) || 
                (!is_edge && deviceType == "server" && model->name.find("sink") == std::string::npos))
            {
                int queue;
                if (is_edge)
                {
                    // calculate the queue only on edge
                    queue = model->arrivalProfiles.arrivalRates;
                    // std::cout << "edge_queue" << queue << std::endl;
                }
                else
                {
                    // calculate the queue only on server
                    queue = model->arrivalProfiles.arrivalRates;
                    // std::cout << "server_queue" << queue << std::endl;
                }

                // add all the nodes queue
                double totalqueue = (queue == 0) ? 0.0 : queue;
                totalQueue += totalqueue;
                // std::cout << "totalRequestRate " << totalQueue << std::endl;
            }
        }
    }

    return totalQueue;
}

// calculate the BaseParPoint based on the TP
void Controller::scheduleBaseParPointLoop(const PipelineModel *model, Partitioner *partitioner, std::vector<NodeHandle> nodes)
{
    // init the data
    float TPedgesAvg = 0.0f;
    float TPserverAvg = 0.0f;
    const float smooth = 0.4f;

    while (true)
    {
        // get the TP on edge and server sides.
        float TPEdges = calculateTotalprocessedRate(model, nodes, true);
        std::cout << "TPEdges: " << TPEdges << std::endl;
        float TPServer = calculateTotalprocessedRate(model, nodes, false);
        std::cout << "TPServer: " << TPServer << std::endl;

        // init the TPedgesAvg and TPserverAvg based on the current runtime
        TPedgesAvg = smooth * TPedgesAvg + (1 - smooth) * TPEdges;
        TPserverAvg = smooth * TPserverAvg + (1 - smooth) * TPServer; // this is server throughput
        std::cout << " TPserverAvg:" << TPserverAvg << std::endl;

        // partition the parpoint, calculate based on the TP
        if (TPedgesAvg > TPserverAvg + 10 * 4)
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
        else if (TPedgesAvg < TPserverAvg - 10 * 4)
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

// fine grained the parpoint based on the queue
void Controller::scheduleFineGrainedParPointLoop(Partitioner *partitioner, const std::vector<NodeHandle> &nodes)
{
    float w;
    int totalServerQueue;
    float tmp;
    while (true)
    {

        // get edge and server sides queue data
        float wbar = calculateTotalQueue(nodes, true);
        std::cout << "wbar " << wbar << std::endl;
        float w = calculateTotalQueue(nodes, false);
        std::cout << "w " << w << std::endl;
        // based on the queue sides to claculate the fine grained point
        //  If there's no queue on the edge, set a default adjustment factor
        if (wbar == 0)
        {
            float tmp = 1.0f;
            partitioner->FineGrainedOffset = tmp * partitioner->BaseParPoint;
        }
        // Otherwise, calculate the fine grained offset based on the relative queue sizes
        else
        {
            float tmp = (wbar - w) / std::max(wbar, w);
            partitioner->FineGrainedOffset = tmp * partitioner->BaseParPoint;
        }
        break;
    }
}

void Controller::DecideAndMoveContainer(const PipelineModel *model, std::vector<NodeHandle> &nodes, Partitioner *partitioner, int cuda_device)
{
    // Calculate the decision point by adding the base and fine grained partition
    float decisionPoint = partitioner->BaseParPoint + partitioner->FineGrainedOffset*0.2;
    // tolerance threshold for decision making
    float tolerance = 0.1;
    // ratio for current worload 
    float ratio = 0.0f;//calculateTotalQueue(nodes, true) /calculateTotalQueue(nodes, false);
    // ContainerHandle *selectedContainer = nullptr;

    if (calculateTotalQueue(nodes, false) != 0)
    {                                                  
        ratio = calculateTotalQueue(nodes, true) /calculateTotalQueue(nodes, false);
        ratio = std::max(0.0f, std::min(1.0f, ratio)); 
    }
    else
    {
        ratio = 0.0f; 
    }

    // the decisionpoint is much larger than the current workload that means we need give the edge more work
    if (decisionPoint > ratio + tolerance)
    {
        std::cout << "Move Container from server to edge based on model priority: " << std::endl;
        // for loop every model to find out the current splitpoint.
        for (const auto &taskPair : ctrl_unscheduledPipelines.list)
        {
            const auto &task = taskPair.second;

            for (auto &model : task->tk_pipelineModels)
            {
                // we don't move the datasource and sink because it has to be on edge or server
                if (model->isSplitPoint && model->name.find("datasource") == std::string::npos && model->name.find("sink") == std::string::npos)
                {
                    std::lock_guard<std::mutex> lock(model->pipelineModelMutex);

                    // change the device from server to the source of edge device
                    if (model->device == "server")
                    {
                        model->device = task->tk_src_device;
                    }
                }
            }
        }
    }
    // Similar logic for the server side
    if (decisionPoint < ratio - tolerance)
    {
        std::cout << "Move Container from edge to server based on model priority: " << std::endl;
        for (const auto &taskPair : ctrl_unscheduledPipelines.list)
        {
            const auto &task = taskPair.second;

            for (auto &model : task->tk_pipelineModels)
            {
                if (model->isSplitPoint && 
                    model->name.find("datasource") == std::string::npos && 
                    model->name.find("sink") == std::string::npos)
                {
                    std::lock_guard<std::mutex> lock(model->pipelineModelMutex);

                    if (model->device != "server")
                    {
                        model->device = "server";
                    }
                }
                {
                    // because we need tp move container from edge to server so we have to move the upstream.
                    for (auto &upstreamPair : model->upstreams)
                    {
                        auto *upstreamModel = upstreamPair.first; // upstream pointer

                        // lock for change information
                        std::lock_guard<std::mutex> lock(upstreamModel->pipelineModelMutex);

                        // move the container from edge to server
                        if (upstreamModel->device != "server")
                        {
                            upstreamModel->device = "server";
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================================================================================
// ============================================================================================================================================
// ============================================================================================================================================
// ============================================================================================================================================