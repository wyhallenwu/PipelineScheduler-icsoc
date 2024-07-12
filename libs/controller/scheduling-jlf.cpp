#include "scheduling-jlf.h"

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
        std::string containerName = model->name + "-" + model->deviceTypeName;
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
            containerName = model->name + "-" + receiverDeviceType;
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
            containerName = model->name + "-" + deviceTypeName;
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
        // Check if it is the next scheduling period
        Stopwatch schedulingSW;
        schedulingSW.start();
        if (std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now() - ctrl_nextSchedulingTime)
                .count() < 10)
        {
            continue;
        }
        auto taskList = ctrl_unscheduledPipelines.getMap();
        if (taskList.empty())
        {
            continue;
        }

        // // FIXME: already empty here
        // for (auto &[task_name, task] : ctrl_unscheduledPipelines.list)
        // {
        //     std::cout << "s task name: " << task_name << std::endl;
        //     std::cout << "s full name" << task->tk_fullName << ", last_latency: " << task->tk_lastLatency << ", slo: " << task->tk_slo << std::endl;
        //     for (auto &model : task->tk_pipelineModels)
        //     {
        //         std::unique_lock<std::mutex> model_lock(model->pipelineModelMutex);
        //         std::cout << "s model device name: " << model->deviceTypeName
        //                   << ", s model batch size: " << model->batchSize << std::endl;
        //         auto downstream = model->downstreams.front().first;
        //         std::unique_lock<std::mutex> d_lock(downstream->pipelineModelMutex);
        //         std::cout << "s dstream name: " << downstream->deviceTypeName << std::endl;
        //     }
        // }

        // collect all information
        model_profiles_jf.infos.clear();
        client_profiles_jf.infos.clear();
        for (auto &[task_name, task] : taskList)
        {
            std::cout << "task name: " << task_name << std::endl;
            // std::unique_lock<std::mutex> lock_task(task->tk_mutex);
            for (auto model : task->tk_pipelineModels)
            {

                std::unique_lock<std::mutex> lock_pipeline_model(model->pipelineModelMutex);
                if (model->name.find("yolo") != std::string::npos)
                {
                    // collect model information
                    std::string name = model->name;
                    // CHECKME: what is the system FPS
                    BatchInferProfileListType batch_proilfes = queryBatchInferLatency(
                        *ctrl_metricsServerConn.get(),
                        ctrl_experimentName,
                        ctrl_systemName,
                        task->tk_name,
                        task->tk_source,
                        model->device,
                        model->deviceTypeName,
                        name,
                        ctrl_systemFPS);

                    // CHECKME: get width, height
                    // parse the resolution of the model
                    std::size_t pos = name.find("-");
                    std::string yolo = name.substr(0, pos);
                    int rs;
                    try
                    {
                        size_t pos;
                        rs = std::stoi(yolo.substr(name.length() - 3, 3), &pos);
                        if (pos != 3)
                        {
                            throw std::invalid_argument("yolov5n, set the default resolution 640");
                        }
                    }
                    catch (const std::invalid_argument &e)
                    {
                        rs = 640;
                    }
                    int width = rs;
                    int height = rs;

                    for (auto &[batch_size, profile] : batch_proilfes)
                    {
                        std::cout << "name is: " << name << std::endl;
                        model_profiles_jf.add(name, ACC_LEVEL_MAP.at(name), batch_size, profile.p95inferLat, width, height, model);
                    }
                }
                else if (model->name.find("datasource") != std::string::npos)
                {
                    // collect information of data source

                    // CHECKME: find the very first downstream device as the network entry pair
                    auto downstream = model->downstreams.front();
                    std::cout << "name of model: " << model->name << std::endl;
                    // FIXME: empty downstream here

                    std::cout << "before access" << std::endl;
                    std::cout << "name of the upstream: " << model->upstreams.front().first->name << std::endl;
                    // auto downstream_device = downstream.first->deviceTypeName;
                    // std::cout << "downstream device: " << downstream_device << std::endl;
                    // auto entry = model->deviceAgent->latestNetworkEntries.at(downstream_device);
                    // std::cout << "empty here" << std::endl;
                    // CHECKME: req rate correctness
                    // client_profiles_jf.add(model->name, task->tk_slo, ctrl_systemFPS, model, task->tk_name, task->tk_source, entry);
                }

                lock_pipeline_model.unlock();
                // lock_task.unlock();
            }
        }

        // just extract the PipelineModel* from the first profile of each model
        auto model = this->model_profiles_jf.infos.begin()->second[0].model;
        std::unique_lock<std::mutex> model_lock(model->pipelineModelMutex);
        auto server_device = model->device;
        auto server_device_type = model->deviceTypeName;
        model_lock.unlock();

        for (auto &client_info : client_profiles_jf.infos)
        {
            auto client_model = client_info.model;
            std::unique_lock<std::mutex> client_lock(client_model->pipelineModelMutex);
            auto client_device = client_model->device;
            auto client_device_type = client_model->deviceTypeName;
            client_lock.unlock();

            // CHECKME: NetworkEntry retrieval correctness
            NetworkProfile network_proflie = queryNetworkProfile(
                *ctrl_metricsServerConn,
                ctrl_experimentName,
                ctrl_systemName,
                client_info.task_name,
                client_info.task_source,
                ctrl_containerLib[client_info.name].taskName,
                ctrl_containerLib[client_info.name].modelName,
                client_device,
                client_device_type,
                server_device,
                server_device_type,
                client_info.network_entry);
            auto lat = network_proflie.p95TransferDuration;
            client_info.set_transmission_latency(lat);
        }

        // start scheduling

        auto mappings = mapClient(this->client_profiles_jf, this->model_profiles_jf);

        for (auto &mapping : mappings)
        {
            // retrieve the mapping for one model and its paired clients
            auto model_info = std::get<0>(mapping);
            auto selected_clients = std::get<1>(mapping);
            int batch_size = std::get<2>(mapping);

            // find the PipelineModel* of that model
            ModelInfoJF m = this->model_profiles_jf.infos[model_info][0];
            for (auto &model : this->model_profiles_jf.infos[model_info])
            {
                if (model.batch_size == batch_size)
                {
                    // note: if occurs core dump, it's possible that there is no matchable pointer
                    // and the p is null
                    m = model;
                    break;
                }
            }
            // clear the upstream of that model
            // CHECKME: lock correctness here
            std::unique_lock<std::mutex> model_lock(m.model->pipelineModelMutex);
            m.model->upstreams.clear();

            // adjust downstream, upstream and resolution
            // CHECKME: vaildate the class of interest here, default to 1 for simplicity
            for (auto &client : selected_clients)
            {
                m.model->upstreams.push_back(std::make_pair(client.model, 1));
                std::unique_lock<std::mutex> client_lock(client.model->pipelineModelMutex);
                client.model->downstreams.clear();
                client.model->downstreams.push_back(std::make_pair(m.model, 1));

                // retrieve new resolution
                int width = m.width;
                int height = m.height;

                client_lock.unlock();

                std::unique_lock<std::mutex> container_lock(this->containers.containersMutex);
                for (auto it = this->containers.list.begin(); it != this->containers.list.end(); ++it)
                {
                    if (it->first == client.name)
                    {
                        // CHECKME: excute resolution adjustment
                        std::vector<int> rs = {width, height, 3};
                        AdjustResolution(it->second, rs);
                    }
                }
                container_lock.unlock();
            }
            model_lock.unlock();
        }

        // TODO: apply the scheduling
        // ctrl_scheduledPipelines = ctrl_unscheduledPipelines;
        // ApplyScheduling();

        // for (auto [taskName, taskHandle]: taskList) {
        //     queryingProfiles(taskHandle);
        //     getInitialBatchSizes(taskHandle, taskHandle->tk_slo);
        //     shiftModelToEdge(taskHandle->tk_pipelineModels, taskHandle->tk_pipelineModels.front(), taskHandle->tk_slo, "edge");
        //     taskHandle->tk_newlyAdded = false;
        // }

        // mergePipelines();
        // // temporalScheduling();
        // schedulingSW.stop();
        // ctrl_nextSchedulingTime = std::chrono::system_clock::now() + std::chrono::seconds(ctrl_schedulingIntervalSec);
        // std::this_thread::sleep_for(TimePrecisionType((ctrl_schedulingIntervalSec + 1) * 1000000 - schedulingSW.elapsed_microseconds()));
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

// ============================================================================================================================================
// ============================================================================================================================================
// ============================================================================================================================================
// ============================================================================================================================================

// ----------------------------------------------------------------------------------------------------------------
//                                             implementations
// ----------------------------------------------------------------------------------------------------------------
ModelInfoJF::ModelInfoJF() {}

ModelInfoJF::ModelInfoJF(int bs, float il, int w, int h, std::string n, float acc, PipelineModel *m)
{
    batch_size = bs;

    // the inference_latency is us
    inference_latency = il;

    // throughput is req/s
    // CHECKME: validate the unit of the time stamp and the gcd of all throughputs,
    // now the time stamp is us, and the gcd of all throughputs is 10, maybe need change to ease the dp table
    throughput = (int(bs / (il * 1e-6)) / 10) * 10; // round it to be devidisble by 10 for better dp computing
    width = w;
    height = h;
    name = n;
    accuracy = acc;
    model = m;
}

ClientInfoJF::ClientInfoJF(std::string _name, float _budget, int _req_rate,
                           PipelineModel *_model, std::string _task_name, std::string _task_source,
                           NetworkEntryType _network_entry)
{
    name = _name;
    budget = _budget;
    req_rate = _req_rate;
    model = _model;
    task_name = _task_name;
    task_source = _task_source;
    transmission_latency = -1;
    network_entry = _network_entry;
}

void ClientInfoJF::set_transmission_latency(int lat)
{
    this->transmission_latency = lat;
}

bool ModelSetCompare::operator()(
    const std::tuple<std::string, float> &lhs,
    const std::tuple<std::string, float> &rhs) const
{
    return std::get<1>(lhs) < std::get<1>(rhs);
}

// -------------------------------------------------------------------------------------------
//                               implementation of ModelProfilesJF
// -------------------------------------------------------------------------------------------

/**
 * @brief add profiled information of model
 *
 * @param model_type
 * @param accuracy
 * @param batch_size
 * @param inference_latency
 * @param throughput
 */
void ModelProfilesJF::add(std::string name, float accuracy, int batch_size, float inference_latency, int width, int height, PipelineModel *m)
{
    auto key = std::tuple<std::string, float>{name, accuracy};
    ModelInfoJF value(batch_size, inference_latency, width, height, name, accuracy, m);
    infos[key].push_back(value);
}

void ModelProfilesJF::add(const ModelInfoJF &model_info)
{
    auto key =
        std::tuple<std::string, float>{model_info.name, model_info.accuracy};
    infos[key].push_back(model_info);
}

void ModelProfilesJF::debugging()
{
    std::cout << "======================ModelProfiles Debugging=======================" << std::endl;
    for (auto it = infos.begin(); it != infos.end(); ++it)
    {
        auto key = it->first;
        auto profilings = it->second;
        std::cout << "*********************************************" << std::endl;
        std::cout << "Model: " << std::get<0>(key) << ", Accuracy: " << std::get<1>(key) << std::endl;
        for (const auto &model_info : profilings)
        {
            std::cout << "batch size: " << model_info.batch_size << ", latency: " << model_info.inference_latency
                      << ", width: " << model_info.width << ", height: " << model_info.height << std::endl;
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
void ClientProfilesJF::sortBudgetDescending(std::vector<ClientInfoJF> &clients)
{
    std::sort(clients.begin(), clients.end(),
              [](const ClientInfoJF &a, const ClientInfoJF &b)
              {
                  return a.budget - a.transmission_latency > b.budget - b.transmission_latency;
              });
}

void ClientProfilesJF::add(const std::string &name, float budget, int req_rate,
                           PipelineModel *model, std::string task_name, std::string task_source,
                           NetworkEntryType network_entry)
{
    infos.push_back(ClientInfoJF(name, budget, req_rate, model, task_name, task_source, network_entry));
}

void ClientProfilesJF::debugging()
{
    std::cout << "===================================ClientProfiles Debugging==========================" << std::endl;
    for (const auto &client_info : infos)
    {
        std::cout << "Unique id: " << client_info.name << ", buget: " << client_info.budget << ", req_rate: " << client_info.req_rate << std::endl;
    }
}

// -------------------------------------------------------------------------------------------
//                               implementation of scheduling algorithms
// -------------------------------------------------------------------------------------------

std::vector<ClientInfoJF> findOptimalClients(const std::vector<ModelInfoJF> &models,
                                             std::vector<ClientInfoJF> &clients)
{
    // sort clients
    ClientProfilesJF::sortBudgetDescending(clients);
    std::cout << "findOptimal start" << std::endl;
    std::cout << "available sorted clients: " << std::endl;
    for (auto &client : clients)
    {
        std::cout << client.name << " " << client.budget - client.transmission_latency << " " << client.req_rate
                  << std::endl;
    }
    std::cout << "available models: " << std::endl;
    for (auto &model : models)
    {
        std::cout << model.name << " " << model.accuracy << " " << model.batch_size << " " << model.throughput << " " << model.inference_latency << std::endl;
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
    for (auto &model : models)
    {
        if (model.throughput > max_throughput)
        {
            max_throughput = model.throughput;
        }
    }
    // init matrix
    int cols = max_throughput / h + 1;
    std::cout << "max_throughput: " << max_throughput << std::endl;
    std::cout << "row: " << rows << " cols: " << cols << std::endl;
    std::vector<std::vector<int>> dp_mat(rows, std::vector<int>(cols, 0));
    // iterating
    for (int client_index = 1; client_index < clients.size(); client_index++)
    {
        auto &client = clients[client_index];
        auto result = findMaxBatchSize(models, client, max_batch_size);
        max_batch_size = std::get<0>(result);
        max_index = std::get<1>(result);
        std::cout << "client name: " << client.name << ", max_batch_size: " << max_batch_size << ", max_index: "
                  << max_index << std::endl;
        if (max_batch_size <= 0)
        {
            break;
        }
        int cols_upperbound = int(models[max_index].throughput / h);
        int lambda_i = client.req_rate;
        int v_i = client.req_rate;
        std::cout << "cols_up " << cols_upperbound << ", req " << lambda_i
                  << std::endl;
        for (int k = 1; k <= cols_upperbound; k++)
        {

            int w_k = k * h;
            if (lambda_i <= w_k)
            {
                int k_prime = (w_k - lambda_i) / h;
                int v = v_i + dp_mat[client_index - 1][k_prime];
                if (v > dp_mat[client_index - 1][k])
                {
                    dp_mat[client_index][k] = v;
                }
                if (v > best_value)
                {
                    best_cell = std::make_tuple(client_index, k);
                    best_value = v;
                }
            }
            else
            {
                dp_mat[client_index][k] = dp_mat[client_index - 1][k];
            }
        }
    }

    std::cout << "updated dp_mat" << std::endl;
    for (auto &row : dp_mat)
    {
        for (auto &v : row)
        {
            std::cout << v << " ";
        }
        std::cout << std::endl;
    }

    // perform backtracing from (row, col)
    // using dp_mat, best_cell, best_value

    std::vector<ClientInfoJF> selected_clients;

    auto [row, col] = best_cell;

    std::cout << "best cell: " << row << " " << col << std::endl;
    int w = dp_mat[row][col];
    while (row > 0 && col > 0)
    {
        std::cout << row << " " << col << std::endl;
        if (dp_mat[row][col] == dp_mat[row - 1][col])
        {
            row--;
        }
        else
        {
            auto c = clients[row - 1];
            int w_i = c.req_rate;
            row = row - 1;
            col = int((w - w_i) / h);
            w = col * h;
            assert(w == dp_mat[row][col]);
            selected_clients.push_back(c);
        }
    }

    std::cout << "findOptimal end" << std::endl;
    std::cout << "selected clients" << std::endl;
    for (auto &sc : selected_clients)
    {
        std::cout << sc.name << " " << sc.budget << " " << sc.req_rate << std::endl;
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
    std::tuple<std::tuple<std::string, float>, std::vector<ClientInfoJF>, int>>
mapClient(ClientProfilesJF client_profile, ModelProfilesJF model_profiles)
{
    std::cout << " ======================= mapClient ==========================" << std::endl;

    std::vector<
        std::tuple<std::tuple<std::string, float>, std::vector<ClientInfoJF>, int>>
        mappings;
    std::vector<ClientInfoJF> clients = client_profile.infos;

    int map_size = model_profiles.infos.size();
    int key_index = 0;
    for (auto it = model_profiles.infos.begin(); it != model_profiles.infos.end();
         ++it)
    {
        key_index++;
        std::cout << "before filtering" << std::endl;
        for (auto &c : clients)
        {
            std::cout << c.name << " " << c.budget << " " << c.req_rate << std::endl;
        }

        auto selected_clients = findOptimalClients(it->second, clients);

        // tradeoff:
        // assign all left clients to the last available model
        if (key_index == map_size)
        {
            std::cout << "assign all rest clients" << std::endl;
            selected_clients = clients;
            clients.clear();
            std::cout << "selected clients assgined" << std::endl;
            for (auto &c : selected_clients)
            {
                std::cout << c.name << " " << c.budget << " " << c.req_rate << std::endl;
            }
            assert(clients.size() == 0);
        }

        int batch_size = check_and_assign(it->second, selected_clients);

        std::cout << "model throughput: " << it->second[0].throughput << std::endl;
        std::cout << "batch size: " << batch_size << std::endl;

        mappings.push_back(
            std::make_tuple(it->first, selected_clients, batch_size));
        std::cout << "start removing collected clients" << std::endl;
        differenceClients(clients, selected_clients);
        std::cout << "after filtering" << std::endl;
        for (auto &c : clients)
        {
            std::cout << c.name << " " << c.budget << " " << c.req_rate << std::endl;
        }
        if (clients.size() == 0)
        {
            break;
        }
    }

    std::cout << "mapping relation" << std::endl;
    for (auto &t : mappings)
    {
        std::cout << "======================" << std::endl;
        auto [model_info, clients, batch_size] = t;
        std::cout << std::get<0>(model_info) << " " << std::get<1>(model_info)
                  << " " << batch_size << std::endl;
        for (auto &client : clients)
        {
            std::cout << "client name: " << client.name << ", req rate: " << client.req_rate << ", budget-lat: " << client.budget << std::endl;
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
int check_and_assign(std::vector<ModelInfoJF> &model,
                     std::vector<ClientInfoJF> &selected_clients)
{
    int total_req_rate = 0;
    // sum all selected req rate
    for (auto &client : selected_clients)
    {
        total_req_rate += client.req_rate;
    }
    int max_batch_size = 1;

    for (auto &model_info : model)
    {
        if (model_info.throughput > total_req_rate &&
            max_batch_size < model_info.batch_size)
        {
            max_batch_size = model_info.batch_size;
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
std::tuple<int, int> findMaxBatchSize(const std::vector<ModelInfoJF> &models,
                                      const ClientInfoJF &client, int max_available_batch_size)
{
    int max_batch_size = 0;
    float budget = client.budget;
    int index = 0;
    int max_index = 0;
    for (const auto &model : models)
    {
        // CHECKME: the inference time should be limited by (budget - transmission time)
        if (model.inference_latency * 2.0 < client.budget - client.transmission_latency &&
            model.batch_size > max_batch_size && model.batch_size <= max_available_batch_size)
        {
            max_batch_size = model.batch_size;
            max_index = index;
        }
        index++;
    }
    return std::make_tuple(max_batch_size, max_index);
}

/**
 * @brief remove the selected clients
 *
 * @param src
 * @param diff
 */
void differenceClients(std::vector<ClientInfoJF> &src,
                       const std::vector<ClientInfoJF> &diff)
{
    auto is_in_diff = [&diff](const ClientInfoJF &client)
    {
        return std::find(diff.begin(), diff.end(), client) != diff.end();
    };
    src.erase(std::remove_if(src.begin(), src.end(), is_in_diff), src.end());
}

// -------------------------------------------------------------------------------------------
//                                  end of implementations
// -------------------------------------------------------------------------------------------