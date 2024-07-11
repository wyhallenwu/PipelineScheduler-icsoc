#include "controller.h"

// ====================================================== IMPLEMENTATION OF RIM  ===========================================================

// Helper functions
uint64_t calc_model_fps(PipelineModel *currModel, NodeHandle *device)
{
    uint64_t batchSize = 16;
    ModelProfile profile = currModel->processProfiles[device->name];
    uint64_t preprocessLatency = profile.batchInfer[batchSize].p95prepLat;
    uint64_t inferLatency = profile.batchInfer[batchSize].p95inferLat;
    uint64_t postprocessLatency = profile.batchInfer[batchSize].p95postLat;

    currModel->expectedAvgPerQueryLatency = preprocessLatency + inferLatency * batchSize + postprocessLatency;
    return 1 / (currModel->expectedAvgPerQueryLatency / 1000000);
}

double calc_pipe_utilization(const std::set<std::string> &subgraph,
                             const std::unordered_map<std::string, PipelineModel *> &model_map,
                             NodeHandle *device, uint64_t desiredFps)
{
    double device_utilization = device->mem_utilization[0];
    for (const auto &node_name : subgraph)
    {
        auto it = model_map.find(node_name);
        if (it != model_map.end())
        {
            PipelineModel *node = it->second;
            device_utilization += static_cast<double>(desiredFps) / calc_model_fps(node, device);
        }
    }
    return device_utilization;
}

void dfs(PipelineModel *node, std::set<std::string> &current_subgraph, std::set<std::set<std::string>> &subgraphs)
{
    if (!node)
        return;

    current_subgraph.insert(node->name);
    subgraphs.insert(current_subgraph);

    for (const auto &child_pair : node->downstreams)
    {
        PipelineModel *child = child_pair.first;
        std::set<std::string> new_subgraph = current_subgraph;
        dfs(child, new_subgraph, subgraphs);
    }
}

std::vector<std::set<std::string>> generate_subgraphs(PipelineModel *root)
{
    std::set<std::set<std::string>> subgraphs;
    std::set<std::string> current_subgraph;
    dfs(root, current_subgraph, subgraphs);

    std::vector<std::set<std::string>> result(subgraphs.begin(), subgraphs.end());
    std::sort(result.begin(), result.end(),
              [](const std::set<std::string> &a, const std::set<std::string> &b)
              {
                  return a.size() > b.size();
              });

    return result;
}

std::unordered_map<std::string, PipelineModel *> build_model_map(PipelineModel *root)
{
    std::unordered_map<std::string, PipelineModel *> model_map;
    std::function<void(PipelineModel *)> dfs = [&](PipelineModel *node)
    {
        if (!node || model_map.find(node->name) != model_map.end())
            return;
        model_map[node->name] = node;
        for (const auto &child_pair : node->downstreams)
        {
            dfs(child_pair.first);
        }
    };
    dfs(root);
    return model_map;
}

// End of helper functions

std::optional<std::set<std::string>> choosing_subgraph(const std::vector<std::set<std::string>> &available_subgraphs,
                                                       PipelineModel *root, std::map<std::string, NodeHandle *> devices, uint64_t desiredFps)
{
    auto model_map = build_model_map(root);
    std::set<std::string> best_subgraph;
    double best_score = 0.0;
    NodeHandle *best_device = nullptr;

    for (std::map<std::string, NodeHandle *>::iterator it = devices.begin(); it != devices.end(); ++it)
    {
        for (const auto &subgraph : available_subgraphs)
        {
            double score = calc_pipe_utilization(subgraph, model_map, it->second, desiredFps);
            if (score > best_score && score <= 1.0)
            {
                best_score = score;
                best_subgraph = subgraph;
                best_device = it->second;
            }
        }
    }

    if (best_score == 0.0)
    {
        return std::nullopt;
    }

    // Update the device attribute for nodes in the best subgraph
    for (const auto &node_name : best_subgraph)
    {
        auto it = model_map.find(node_name);
        if (it != model_map.end())
        {
            PipelineModel *node = it->second;
            node->deviceAgent = best_device;
        }
    }

    return best_subgraph;
}

void update_available_subgraphs(std::vector<std::set<std::string>> &available_subgraphs,
                                std::vector<std::set<std::string>> &selected_subgraphs,
                                const std::set<std::string> &chosen_subgraph)
{
    selected_subgraphs.push_back(chosen_subgraph);

    auto it = available_subgraphs.begin();
    while (it != available_subgraphs.end())
    {
        bool should_remove = false;
        for (const auto &node : chosen_subgraph)
        {
            if (it->find(node) != it->end())
            {
                should_remove = true;
                break;
            }
        }
        if (should_remove)
        {
            it = available_subgraphs.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

bool place_on_edge(PipelineModel *root, std::vector<std::set<std::string>> &available_subgraphs,
                   std::vector<std::set<std::string>> selected_subgraphs,
                   std::map<std::string, NodeHandle *> devices, uint64_t desiredFps)
{

    while (!available_subgraphs.empty())
    {
        // Choose a subgraph
        auto chosen_subgraph = choosing_subgraph(available_subgraphs, root, devices, desiredFps);

        // If no subgraph could be chosen, return false
        if (!chosen_subgraph.has_value())
        {
            return false;
        }
        // Update available and selected subgraphs
        update_available_subgraphs(available_subgraphs, selected_subgraphs, chosen_subgraph.value());
    }

    return true;
}

void place_on_server(PipelineModel *root, std::vector<std::set<std::string>> &available_subgraphs,
                     std::vector<std::set<std::string>> selected_subgraphs,
                     std::map<std::string, NodeHandle *> devices, uint64_t desiredFps)
{
    // Generate initial subgraphs

    while (!available_subgraphs.empty())
    {
        // Choose a subgraph
        auto chosen_subgraph = choosing_subgraph(available_subgraphs, root, devices, desiredFps);

        // Update available and selected subgraphs
        update_available_subgraphs(available_subgraphs, selected_subgraphs, chosen_subgraph.value());
    }
}

void Controller::rim_action(TaskHandle *task)
{
    // The desired fps
    uint64_t desiredFps = 1 / (task->tk_slo / 1000000);
    // Should the root be the data source or the first model ??
    PipelineModel *root = task->tk_pipelineModels[0];
    std::vector<std::set<std::string>> remaining_subgraphs = generate_subgraphs(root);
    std::vector<std::set<std::string>> selected_subgraphs;

    std::map<std::string, NodeHandle *> edges;
    std::map<std::string, NodeHandle *> servers;

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

    if (!place_on_edge(root, remaining_subgraphs, selected_subgraphs, edges, desiredFps))
    {
        place_on_server(root, remaining_subgraphs, selected_subgraphs, servers, desiredFps);
    }
}

void Controller::schedule_rim(std::map<std::string, TaskHandle*> &tasks){
    for (auto [taskName, taskHandle]: tasks) {
        rim_action(taskHandle);
    } 
}

// Task to do tomorrow
// 1. Implement the place_on_server function, ask carefully the question of how to place the model on the server
// 2. Make sure the devices in the place on edge function are the edge devices.
