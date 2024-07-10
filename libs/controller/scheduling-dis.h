#include "controller.h"


// std::mutex nodeHandleMutex;
//     // std::vector<NodeHandle*> nodes;
//     struct Partitioner
//     {
//         // NodeHandle& edge;
//         // NodeHandle& server;
//         // need server here
//         float BaseParPoint;
//         float FineGrainedOffset;
//     };

//     struct Partitioner;
//     std::vector<NodeHandle> nodes;
//     std::pair<std::vector<NodeHandle>, std::vector<NodeHandle>> categorizeNodes(const std::vector<NodeHandle> &nodes);
//     double calculateTotalprocessedRate(const PipelineModel *model, const std::vector<NodeHandle> &nodes, bool is_edge);
//     int calculateTotalQueue(const std::vector<NodeHandle> &nodes, bool is_edge);
//     double getMaxTP(const PipelineModel *model, std::vector<NodeHandle> nodes, bool is_edge);
//     void scheduleBaseParPointLoop(const PipelineModel *model, Partitioner *partitioner, std::vector<NodeHandle> nodes);
//     float ComputeAveragedNormalizedWorkload(const std::vector<NodeHandle> &nodes, bool is_edge);
//     void scheduleFineGrainedParPointLoop(Partitioner *partitioner, const std::vector<NodeHandle> &nodes);
//     void DecideAndMoveContainer(const PipelineModel *model, std::vector<NodeHandle> &nodes, Partitioner *partitioner, int cuda_device);
//     float calculateRatio(const std::vector<NodeHandle> &nodes);