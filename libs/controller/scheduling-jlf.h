#include "controller.h"

// the accuracy value here is dummy, just use for ranking models
const std::map<std::string, float> ACC_LEVEL_MAP = {
        {"yolov5n320", 0.30},
        {"yolov5n512", 0.40},
        {"yolov5n640", 0.50},
        {"yolov5s640", 0.55},
        {"yolov5m640", 0.60},
};

// --------------------------------------------------------------------------------------------------------
//                                     start of jellyfish scheduling implementation
// --------------------------------------------------------------------------------------------------------

std::vector<std::tuple<std::tuple<std::string, float>, std::vector<ClientInfoJF>, int>> mapClient(ClientProfilesJF &client_profile, ModelProfilesJF &model_profiles);
std::vector<ClientInfoJF> findOptimalClients(const std::vector<ModelInfoJF> &models, std::vector<ClientInfoJF> &clients);
int check_and_assign(std::vector<ModelInfoJF> &model, std::vector<ClientInfoJF> &selected_clients);

std::tuple<int, int> findMaxBatchSize(const std::vector<ModelInfoJF> &models, const ClientInfoJF &client, int max_available_batch_size = 16);
void differenceClients(std::vector<ClientInfoJF> &src, const std::vector<ClientInfoJF> &diff);

// --------------------------------------------------------------------------------------------------------
//                                      end of jellyfish scheduling implementation
// --------------------------------------------------------------------------------------------------------