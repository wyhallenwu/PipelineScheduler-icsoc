#include "controller.h"

// ====================================================== IMPLEMENTATION OF RIM  ===========================================================

void Controller::performPlacement(TaskHandle& task) {
    Pipeline pipeline;
    for (const auto& subtask : task.subtasks) {
        PipelineModel* model = new PipelineModel();
        model->name = subtask.first;
        model->arrivalProfiles.arrivalRates = subtask.second->arrival_rate;
        copyContainerDataToModel(subtask.second, model);
        pipeline.pipelineModels.push_back(model);
    }

    auto [fits, fitScore, bestNode, bestProcessor] = findBestFitForPipeline(pipeline, task);
    if (fits) {
        placeMDAGOnNode(pipeline, *bestNode, bestProcessor, task);
        ctrl_logger->info("Successfully placed mDAG on node: {} for task: {}", bestNode->name, task.name);
    } else {
        placeModulesOnWorkers(pipeline, task);
    }

    updateContainerHandles(task, pipeline);
}

std::tuple<bool, double, NodeHandle*, int> Controller::findBestFitForPipeline(const Pipeline& pipeline, const TaskHandle& task) {
    auto edgeFit = findBestFitOnEdge(pipeline, task);
    if (std::get<0>(edgeFit)) {
        return edgeFit;
    }
    return findBestFitOnServer(pipeline, task);
}

std::tuple<bool, double, NodeHandle*, int> Controller::findBestFitOnEdge(const Pipeline& pipeline, const TaskHandle& task) {
    bool fits = false;
    double bestFitScore = std::numeric_limits<double>::max();
    NodeHandle* bestNode = nullptr;
    int bestProcessor = -1;

    for (auto& device_pair : devices) {
        NodeHandle& node = device_pair.second;
        if (node.type == Server) continue;

        auto [nodeFits, nodeFitScore, nodePtr, nodeProcessor] = canFitPipeline(pipeline, node, task);
        if (nodeFits && nodeFitScore < bestFitScore) {
            fits = true;
            bestFitScore = nodeFitScore;
            bestNode = nodePtr;
            bestProcessor = nodeProcessor;
        }
    }

    return std::make_tuple(fits, bestFitScore, bestNode, bestProcessor);
}

std::tuple<bool, double, NodeHandle*, int> Controller::findBestFitOnServer(const Pipeline& pipeline, const TaskHandle& task) {
    for (auto& device_pair : devices) {
        NodeHandle& node = device_pair.second;
        if (node.type != Server) continue;

        auto [fits, fitScore, nodePtr, processor] = canFitPipeline(pipeline, node, task);
        return std::make_tuple(fits, fitScore, nodePtr, processor);
    }

    return std::make_tuple(false, std::numeric_limits<double>::max(), nullptr, -1);
}

std::tuple<bool, double, NodeHandle*, int> Controller::canFitPipeline(const Pipeline& pipeline, NodeHandle& node, const TaskHandle& task) {
    double totalRequiredCapacity = 0;
    uint64_t totalLatency = 0;

    for (const auto& model : pipeline.pipelineModels) {
        estimateModelLatency(model, node.name);
        float throughput = 1000000.0f / model->expectedAvgPerQueryLatency;
        float requiredCapacity = static_cast<float>(model->arrivalProfiles.arrivalRates) / throughput;
        totalRequiredCapacity += requiredCapacity;
        totalLatency += model->expectedTransferLatency + 
                        model->expectedQueueingLatency + 
                        model->expectedMaxProcessLatency;
    }

    bool fits = false;
    double bestFitScore = std::numeric_limits<double>::max();
    int bestProcessor = -1;

    for (int i = 0; i < node.num_processors; i++) {
        double processorUtilization = node.processors_utilization[i] + totalRequiredCapacity;
        double memoryUtilization = node.mem_utilization[i] + totalRequiredCapacity;

        if (processorUtilization <= 1.0 && memoryUtilization <= 1.0) {
            double fitScore = std::max(processorUtilization, memoryUtilization);
            if (fitScore < bestFitScore) {
                fits = true;
                bestFitScore = fitScore;
                bestProcessor = i;
            }
        }
    }

    bool meetsLatency = totalLatency <= static_cast<uint64_t>(task.slo);
    return std::make_tuple(fits && meetsLatency, bestFitScore, &node, bestProcessor);
}

void Controller::placeModulesOnWorkers(Pipeline& pipeline, const TaskHandle& task) {
    for (auto& model : pipeline.pipelineModels) {
        auto [fits, fitScore, bestNode, bestProcessor] = findBestFitForModule(*model, task);
        if (fits) {
            placeModelOnNode(model, bestNode, bestProcessor);
        } else {
            ctrl_logger->error("Failed to place module: {} for task: {}", model->name, task.name);
        }
    }
}

std::tuple<bool, double, NodeHandle*, int> Controller::findBestFitForModule(const PipelineModel& model, const TaskHandle& task) {
    auto edgeFit = findBestFitModuleOnEdge(model, task);
    if (std::get<0>(edgeFit)) {
        return edgeFit;
    }
    return findBestFitModuleOnServer(model, task);
}

std::tuple<bool, double, NodeHandle*, int> Controller::findBestFitOnEdge(const Pipeline& pipeline, const TaskHandle& task) {
    bool fits = false;
    double bestFitScore = std::numeric_limits<double>::max();
    NodeHandle* bestNode = nullptr;
    int bestProcessor = -1;

    for (auto& device_pair : devices) {
        NodeHandle& node = device_pair.second;
        if (node.type == Server) continue;

        auto [nodeFits, nodeFitScore, nodePtr, nodeProcessor] = canFitPipeline(pipeline, node, task);
        if (nodeFits && nodeFitScore < bestFitScore) {
            fits = true;
            bestFitScore = nodeFitScore;
            bestNode = nodePtr;
            bestProcessor = nodeProcessor;
        }
    }

    return std::make_tuple(fits, bestFitScore, bestNode, bestProcessor);
}

std::tuple<bool, double, NodeHandle*, int> Controller::findBestFitModuleOnServer(const PipelineModel& model, const TaskHandle& task) {
    for (auto& device_pair : devices) {
        NodeHandle& node = device_pair.second;
        if (node.type != Server) continue;

        auto [fits, fitScore, processor] = canFitModule(model, node, task);
        return std::make_tuple(fits, fitScore, &node, processor);
    }

    return std::make_tuple(false, std::numeric_limits<double>::max(), nullptr, -1);
}

std::tuple<bool, double, int> Controller::canFitModule(const PipelineModel& model, NodeHandle& node, const TaskHandle& task) {
    estimateModelLatency(&model, node.name);
    float throughput = 1000000.0f / model.expectedAvgPerQueryLatency;
    float requiredCapacity = static_cast<float>(model.arrivalProfiles.arrivalRates) / throughput;

    bool fits = false;
    double bestFitScore = std::numeric_limits<double>::max();
    int bestProcessor = -1;

    for (int i = 0; i < node.num_processors; i++) {
        double processorUtilization = node.processors_utilization[i] + requiredCapacity;
        double memoryUtilization = node.mem_utilization[i] + requiredCapacity;

        if (processorUtilization <= 1.0 && memoryUtilization <= 1.0) {
            double fitScore = std::max(processorUtilization, memoryUtilization);
            if (fitScore < bestFitScore) {
                fits = true;
                bestFitScore = fitScore;
                bestProcessor = i;
            }
        }
    }

    uint64_t totalLatency = model.expectedTransferLatency + 
                            model.expectedQueueingLatency + 
                            model.expectedMaxProcessLatency;
    bool meetsLatency = totalLatency <= static_cast<uint64_t>(task.slo);
    return std::make_tuple(fits && meetsLatency, bestFitScore, bestProcessor);
}

void Controller::placeMDAGOnNode(Pipeline& pipeline, NodeHandle& node, int processor, const TaskHandle& task) {
    for (auto& model : pipeline.pipelineModels) {
        placeModelOnNode(model, &node, processor);
    }
}

void Controller::placeModelOnNode(PipelineModel* model, NodeHandle* node, int processor) {
    model->device = node->name;
    model->deviceTypeName = getDeviceTypeName(node->type);
}

void Controller::updateContainerHandles(TaskHandle& task, const Pipeline& pipeline) {
    for (const auto& model : pipeline.pipelineModels) {
        if (task.subtasks.find(model->name) != task.subtasks.end()) {
            ContainerHandle* container = task.subtasks[model->name];
            
            container->device_agent = &devices[model->device];
            container->cuda_device = {findProcessorForModel(model, container->device_agent)};
            
            float throughput = 1000000.0f / model->expectedAvgPerQueryLatency;
            float requiredCapacity = static_cast<float>(model->arrivalProfiles.arrivalRates) / throughput;
            container->device_agent->processors_utilization[container->cuda_device[0]] += requiredCapacity;
            container->device_agent->mem_utilization[container->cuda_device[0]] += requiredCapacity;

            container->batch_size = {model->batchSize};
            container->num_replicas = model->numReplicas;
            container->arrival_rate = model->arrivalProfiles.arrivalRates;

            container->device_agent->containers[container->name] = container;
            containers[container->name] = *container;
        }
    }
}

void Controller::copyContainerDataToModel(const ContainerHandle* container, PipelineModel* model) {
    model->batchSize = container->batch_size[0];
    model->numReplicas = container->num_replicas;
    // Copy other relevant fields...
}

int Controller::findProcessorForModel(const PipelineModel* model, const NodeHandle* node) {
    float throughput = 1000000.0f / model->expectedAvgPerQueryLatency;
    float requiredCapacity = static_cast<float>(model->arrivalProfiles.arrivalRates) / throughput;
    
    for (int i = 0; i < node->num_processors; i++) {
        if (node->processors_utilization[i] + requiredCapacity <= 1.0) {
            return i;
        }
    }
    return -1;
}
