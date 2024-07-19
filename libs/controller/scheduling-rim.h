#include "controller.h"

// ====================================================== IMPLEMENTATION OF RIM  ===========================================================

// Helper functions

std::string device_type_str(NodeHandle *device)
{
    // std::lock_guard<std::mutex> lock(device->nodeHandleMutex);
    std::string device_type;
    switch (device->type)
    {
    case AGXXavier:
        device_type = "agxavier";
        break;
    case NXXavier:
        device_type = "nxavier";
        break;
    case OrinNano:
        device_type = "orinano";
        break;
    default:
        device_type = "server";
        break;
    }
    return device_type;
}

uint64_t calc_model_fps(PipelineModel *currModel, NodeHandle *device)
{
    std::lock_guard<std::mutex> model_lock(currModel->pipelineModelMutex);
    // std::lock_guard<std::mutex> device_lock(device->nodeHandleMutex);

    uint64_t batchSize = 8;

    std::string device_type = device_type_str(device);

    ModelProfile profile = currModel->processProfiles[device_type];
    uint64_t preprocessLatency = profile.batchInfer[batchSize].p95prepLat;
    uint64_t inferLatency = profile.batchInfer[batchSize].p95inferLat;
    uint64_t postprocessLatency = profile.batchInfer[batchSize].p95postLat;

    currModel->expectedAvgPerQueryLatency = preprocessLatency + inferLatency * batchSize + postprocessLatency;
    return static_cast<uint64_t>(std::round(1000000.0 / static_cast<double>(currModel->expectedAvgPerQueryLatency)));
}

double calc_pipe_utilization(const std::list<std::string> &subgraph,
                             const std::unordered_map<std::string, PipelineModel *> &model_map,
                             NodeHandle *device, uint64_t desiredFps)
{
    // std::lock_guard<std::mutex> device_lock(device->nodeHandleMutex);

    double device_utilization = device->mem_utilization[0];
    for (const auto &node_name : subgraph)
    {
        auto it = model_map.find(node_name);
        if (it != model_map.end())
        {
            PipelineModel *node = it->second;
            if (node_name == "datasource" || node_name == "sink") {
                continue;
            }
            uint64_t max_model_fps = calc_model_fps(node, device);
            device_utilization += static_cast<double>(desiredFps) / max_model_fps;
        }
    }
    return device_utilization;
}

void generate_subgraphs_helper(PipelineModel *node, std::vector<PipelineModel *> &current_path, std::vector<std::list<std::string>> &subgraphs)
{
    if (!node) return;

    std::unique_lock<std::mutex> lock(node->pipelineModelMutex);

    current_path.push_back(node);

    // Generate subgraphs for the current path
    for (size_t i = 0; i < current_path.size(); ++i)
    {
        std::list<std::string> subgraph;
        for (size_t j = i; j < current_path.size(); ++j)
        {
            subgraph.push_back(current_path[j]->name);
        }
        subgraphs.push_back(subgraph);
    }

    // Continue with downstream nodes
    auto downstreams = node->downstreams;
    lock.unlock();

    for (auto &downstream : downstreams)
    {
        generate_subgraphs_helper(downstream.first, current_path, subgraphs);
    }

    current_path.pop_back();
}

std::vector<std::list<std::string>> generate_subgraphs(PipelineModel *root)
{
    std::vector<std::list<std::string>> subgraphs;
    std::vector<PipelineModel *> current_path;
    generate_subgraphs_helper(root, current_path, subgraphs);

    // Sort the subgraphs by size in descending order
    std::sort(subgraphs.begin(), subgraphs.end(), [](const std::list<std::string> &a, const std::list<std::string> &b) {
        return a.size() > b.size();
    });

    return subgraphs;
}

std::unordered_map<std::string, PipelineModel *> build_model_map(PipelineModel *root)
{
    std::unordered_map<std::string, PipelineModel *> model_map;
    if (!root) return model_map;

    std::queue<PipelineModel *> q;
    q.push(root);

    while (!q.empty())
    {
        PipelineModel *current = q.front();
        q.pop();

        std::unique_lock<std::mutex> lock(current->pipelineModelMutex);

        if (model_map.find(current->name) == model_map.end())
        {
            model_map[current->name] = current;

            for (const auto &downstream : current->downstreams)
            {
                q.push(downstream.first);
            }
        }

        lock.unlock();

    }

    return model_map;
}

// End of helper functions

std::optional<std::list<std::string>> choosing_subgraph_for_edge(const std::vector<std::list<std::string>> &available_subgraphs,
                                                       PipelineModel *root, std::map<std::string, NodeHandle *> devices, uint64_t desiredFps)
{
    double best_score = 0.0;
    auto model_map = build_model_map(root);
    std::list<std::string> best_subgraph;
    NodeHandle *best_device = nullptr;

    std::unique_lock<std::mutex> root_lock(root->pipelineModelMutex);
    auto possible_devices = root->possibleDevices;
    root_lock.unlock();

    for (std::map<std::string, NodeHandle *>::iterator it = devices.begin(); it != devices.end(); ++it)
    {
        if (std::find(possible_devices.begin(), possible_devices.end(), it->first) != possible_devices.end()) {
            for (const auto &subgraph : available_subgraphs)
            {
                double score = calc_pipe_utilization(subgraph, model_map, it->second, desiredFps);
                if (score > best_score && score <= 1.0)
                {
                    best_score = score;
                    best_subgraph = subgraph;
                    std::unique_lock<std::mutex> device_lock(it->second->nodeHandleMutex);
                    best_device = it->second;
                    device_lock.unlock();
                }
            }
        }
    }

    if (best_score == 0.0)
    {
        return std::nullopt;
    }

    std::unique_lock<std::mutex> device_lock(best_device->nodeHandleMutex);

    best_device->mem_utilization[0] = best_score;

    // Update the device attribute for nodes in the best subgraph
    for (const auto &node_name : best_subgraph)
    {
        auto it = model_map.find(node_name);
        if (it != model_map.end())
        {
            PipelineModel *node = it->second;
            std::unique_lock<std::mutex> node_lock(node->pipelineModelMutex);
            if (node_name == "sink" || node_name == "datasource") {
                continue;
            }
            else {
                node->deviceAgent = best_device;
                node->device = best_device->name;
                node->deviceTypeName = device_type_str(best_device);
            }
            node_lock.unlock();
        }
    }
    device_lock.unlock();

    return best_subgraph;
}

void update_available_subgraphs(std::vector<std::list<std::string>> &available_subgraphs,
                                std::vector<std::list<std::string>> &selected_subgraphs,
                                const std::list<std::string> &chosen_subgraph)
{
    // Add the chosen subgraph to selected subgraphs
    selected_subgraphs.push_back(chosen_subgraph);

    // Create a set of nodes in the chosen subgraph for quick lookup
    std::set<std::string> chosen_nodes(chosen_subgraph.begin(), chosen_subgraph.end());

    // Remove subgraphs from available_subgraphs that have any node in chosen_nodes
    available_subgraphs.erase(
        std::remove_if(available_subgraphs.begin(), available_subgraphs.end(),
                       [&chosen_nodes](const std::list<std::string> &subgraph) {
                           for (const auto &node : subgraph)
                           {
                               if (chosen_nodes.find(node) != chosen_nodes.end())
                               {
                                   return true;
                               }
                           }
                           return false;
                       }),
        available_subgraphs.end());
}

void place_on_edge(PipelineModel *root, std::vector<std::list<std::string>> &available_subgraphs,
                   std::vector<std::list<std::string>> &selected_subgraphs,
                   std::map<std::string, NodeHandle *> &devices, uint64_t desiredFps)
{

    while (!available_subgraphs.empty())
    {
        // Choose a subgraph
        auto chosen_subgraph = choosing_subgraph_for_edge(available_subgraphs, root, devices, desiredFps);

        // If no subgraph could be chosen, return false
        if (!chosen_subgraph.has_value())
        {
            return;
        }

        // Update available and selected subgraphs
        update_available_subgraphs(available_subgraphs, selected_subgraphs, chosen_subgraph.value());
    }
}



void place_on_server(PipelineModel *root, std::vector<std::list<std::string>> &available_subgraphs,
                     std::vector<std::list<std::string>> &selected_subgraphs,
                     std::map<std::string, NodeHandle *> &devices, uint64_t desiredFps)
{
    auto model_map = build_model_map(root);
    std::list<std::string> best_subgraph;
    NodeHandle *server_node = devices.at("server");


    while (!available_subgraphs.empty())
    {
        // Update the device attribute for nodes in the best subgraph
        best_subgraph = available_subgraphs.front();
        for (const auto &node_name : best_subgraph)
        {
            auto it = model_map.find(node_name);
            if (it != model_map.end())
            {
                if (node_name == "sink" || node_name == "datasource") {
                    continue;
                }
                PipelineModel *node = it->second;
                node->deviceAgent = server_node;
                node->device = server_node->name;
            }
        }

        // Update available and selected subgraphs
        update_available_subgraphs(available_subgraphs, selected_subgraphs, best_subgraph);
    }
}

void print_subgraphs(const std::vector<std::list<std::string>> &subgraphs)
{
    for (size_t i = 0; i < subgraphs.size(); ++i)
    {
        std::cout << i + 1 << ". ";
        bool first = true;
        for (const auto &node : subgraphs[i])
        {
            if (!first)
                std::cout << "-";
            std::cout << node;
            first = false;
        }
        std::cout << std::endl;
    }
}

void print_pipeline_levels(PipelineModel* root) {
    std::cout << "Pipeline Graph Levels:" << std::endl;

    std::queue<PipelineModel*> current_level;
    current_level.push(root);

    int level = 0;
    while (!current_level.empty()) {
        std::cout << "Level " << level << ": ";
        
        std::queue<PipelineModel*> next_level;
        std::unordered_set<PipelineModel*> level_set;

        while (!current_level.empty()) {
            PipelineModel* node = current_level.front();
            current_level.pop();

            if (level_set.find(node) == level_set.end()) {
                std::cout << node->name << " ";
                level_set.insert(node);

                for (const auto& [downstream, _] : node->downstreams) {
                    if (level_set.find(downstream) == level_set.end()) {
                        next_level.push(downstream);
                    }
                }
            }
        }

        std::cout << std::endl;
        current_level = next_level;
        level++;
    }
}

void Controller::rim_action(TaskHandle *task)
{
    // The desired fps
    std::lock_guard<std::mutex> task_lock(task->tk_mutex);

    uint64_t desiredFps = static_cast<uint64_t>(std::round(1000000.0 / static_cast<double>(task->tk_slo)));
    PipelineModel *root = task->tk_pipelineModels[0];
    std::vector<std::list<std::string>> remaining_subgraphs = generate_subgraphs(root);
    std::vector<std::list<std::string>> selected_subgraphs;

    // print_pipeline_levels(root);

    // print_subgraphs(remaining_subgraphs);

    std::map<std::string, NodeHandle *> edges;
    std::map<std::string, NodeHandle *> servers;

    {
        std::lock_guard<std::mutex> devices_lock(devices.devicesMutex);
        for (const auto &pair : devices.list)
        {
            if (pair.second->name == "server")
            {
                servers[pair.first] = pair.second;
            }
            else
            {
                edges[pair.first] = pair.second;
            }
        }
    }
    

    //update device for datasource
    {
        std::lock_guard<std::mutex> source_lock(root->pipelineModelMutex);
        root->deviceAgent = edges.at(root->device);
        root->deviceTypeName = device_type_str(root->deviceAgent);
    }


    place_on_edge(root, remaining_subgraphs, selected_subgraphs, edges, desiredFps);
    if (!remaining_subgraphs.empty()) {
        place_on_server(root, remaining_subgraphs, selected_subgraphs, servers, desiredFps);
    }

    //update device for sink
    PipelineModel *sink = task->tk_pipelineModels[task->tk_pipelineModels.size() - 1];
    {
        std::lock_guard<std::mutex> sink_lock(sink->pipelineModelMutex);
        sink->deviceAgent = servers.at("server");
        sink->deviceTypeName = device_type_str(sink->deviceAgent);
    }
}

// ====================================================== END OF IMPLEMENTATION OF RIM  ===========================================================

// 2. Comment out all the important functions
// 3. The behaviour of unscheduled and scheduled pipeline is not correct