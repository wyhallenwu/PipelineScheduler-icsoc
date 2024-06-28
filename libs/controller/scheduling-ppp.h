// /*
// Jellyfish controller implementation.

// Compared with the original jellyfish paper, we make some slight modification for our case:
// (1) the req rate is the very first model of the pipeline(usually yolo in our case)
// (2) we only map the datasource to different very first analysis model in the pipeline

// The following code implement the scheduling algorithm of jellyfish whicn contains 1.data adaptation, 2.client-DNN mapping 3. dynamic batching
// */

// // the accuracy value here is dummy, just use for ranking models
// const std::map<std::string, float> ACC_LEVEL_MAP = {
//     {"yolov5n320", 0.30},
//     {"yolov5n416", 0.35},
//     {"yolov5n512", 0.40},
//     {"yolov5n608", 0.45},
//     {"Yolov5n640", 0.50},
//     {"yolov5s640", 0.55},
//     {"yolov5m640", 0.60},
// };

// /**
//  * @brief ModelInfo is a collection of a single running model. A helper class for scheduling algorithm
//  *
//  */
// struct ModelInfoJF
// {
//     int batch_size;
//     float inference_latency;
//     int throughput;
//     int width;
//     int height;
//     std::string name;
//     float accuracy;
//     PipelineModel *model;

//     ModelInfoJF(int bs, float il, int w, int h, std::string n, float acc, PipelineModel *m);
// };

// /**
//  * @brief comparison of the key of ModelProfiles, for sorting in the ModelProfiles::infos
//  *
//  */
// struct ModelSetCompare
// {
//     bool operator()(const std::tuple<std::string, float> &lhs, const std::tuple<std::string, float> &rhs) const;
// };

// /**
//  * @brief ModelProfiles is a collection of all runing models' profile.
//  *
//  */
// class ModelProfilesJF
// {
// public:
//     // key: (model type, accuracy) value: (model_info)
//     std::map<std::tuple<std::string, float>, std::vector<ModelInfoJF>, ModelSetCompare> infos;

//     void add(std::string name, float accuracy, int batch_size, float inference_latency, int width, int height, PipelineModel *model);
//     void add(const ModelInfoJF &model_info);

//     void debugging();
// };

// /**
//  * @brief ClientInfo is a collection of single client's information. A helper class for scheduling algorithm
//  *
//  */
// struct ClientInfoJF
// {
//     std::string ip;           // can be anything, just a unique identification for differenct clients(datasource)
//     float budget;             // slo
//     int req_rate;             // request rate (how many frame are sent to remote per second)
//     PipelineModel *model;     // pointer to that component
//     int transmission_latency; // networking time, useful for scheduling

//     ClientInfoJF(std::string _ip, float _budget, int _req_rate, PipelineModel *model);

//     bool operator==(const ClientInfoJF &other) const
//     {
//         return ip == other.ip &&
//                budget == other.budget &&
//                req_rate == other.req_rate;
//     }

//     void set_transmission_latency(int lat);

//     // void set_bandwidth(float bw);
// };

// /**
//  * @brief ClientProfiles is a collection of all clients' profile.
//  *
//  */
// class ClientProfilesJF
// {
// public:
//     std::vector<ClientInfoJF> infos;

//     static void sortBudgetDescending(std::vector<ClientInfoJF> &clients);

//     void add(const std::string &ip, float budget, int req_rate, PipelineModel *model);

//     void debugging();
// };

// // --------------------------------------------------------------------------------------------------------
// //                                      jellyfish scheduling functions
// // --------------------------------------------------------------------------------------------------------

// std::vector<std::tuple<std::tuple<std::string, float>, std::vector<ClientInfoJF>, int>> mapClient(ClientProfilesJF client_profile, ModelProfilesJF model_profiles);
// std::vector<ClientInfoJF> findOptimalClients(const std::vector<ModelInfoJF> &models, std::vector<ClientInfoJF> &clients);
// int check_and_assign(std::vector<ModelInfoJF> &model, std::vector<ClientInfoJF> &selected_clients);

// // helper functions

// std::tuple<int, int> findMaxBatchSize(const std::vector<ModelInfoJF> &models, const ClientInfoJF &client, int max_available_batch_size = 64);
// void differenceClients(std::vector<ClientInfoJF> &src, const std::vector<ClientInfoJF> &diff);

// // --------------------------------------------------------------------------------------------------------
// //                                      end of jellyfish scheduling functions
// // --------------------------------------------------------------------------------------------------------
