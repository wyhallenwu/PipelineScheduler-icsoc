// #include "scheduling-ppp.h"

// void Controller::AddTask(const TaskDescription::TaskStruct &t)
// {
//     std::cout << "Adding task: " << t.name << std::endl;
//     tasks.insert({t.name, {t.name, t.type, t.source, t.slo, {}, 0, {}}});
//     TaskHandle *task = &tasks[t.name];
//     NodeHandle *device = &devices[t.device];
//     Pipeline pipe = {getModelsByPipelineType(t.type, t.device)};
//     ctrl_unscheduledPipelines.emplace_back(pipe);

//     std::vector<std::pair<std::string, std::string>> possibleDevicePairList = {{"server", "server"}};
//     std::map<std::pair<std::string, std::string>, NetworkEntryType> possibleNetworkEntryPairs;

//     ClientProfilesJF client_profiles_jf;
//     ModelProfilesJF model_profiles_jf;

//     for (const auto &pair : possibleDevicePairList)
//     {
//         std::unique_lock lock(devices[pair.first].nodeHandleMutex);
//         possibleNetworkEntryPairs[pair] = devices[pair.first].latestNetworkEntries[pair.second];
//         lock.unlock();
//     }

//     std::vector<std::string> possibleDeviceList = {"server"};

//     for (auto &model : ctrl_unscheduledPipelines.back().pipelineModels)
//     {
//         std::string containerName = model->name + "-" + possibleDevicePairList[0].second;
//         if (containerName.find("datasource") != std::string::npos || containerName.find("sink") != std::string::npos)
//         {
//             // MODIFICATION -----
//             if (containerName.find("datasource"))
//             {
//                 // set the req rate, in jellyfish, it's the fps of edge cameras
//                 model->arrivalProfiles.arrivalRates = 30;

//                 // check if the name of the datasource is unique
//                 client_profiles_jf.add(containerName, t.slo, model->arrivalProfiles.arrivalRates, model);
//             }
//             // -------------------
//             continue;
//         }

//         model->arrivalProfiles.arrivalRates = queryArrivalRate(
//             *ctrl_metricsServerConn,
//             ctrl_experimentName,
//             ctrl_systemName,
//             t.name,
//             t.source,
//             ctrl_containerLib[containerName].taskName,
//             ctrl_containerLib[containerName].modelName);
//         for (const auto &pair : possibleDevicePairList)
//         {
//             NetworkProfile test = queryNetworkProfile(
//                 *ctrl_metricsServerConn,
//                 ctrl_experimentName,
//                 ctrl_systemName,
//                 t.name,
//                 t.source,
//                 ctrl_containerLib[containerName].taskName,
//                 ctrl_containerLib[containerName].modelName,
//                 pair.first,
//                 pair.second,
//                 possibleNetworkEntryPairs[pair]);
//             model->arrivalProfiles.d2dNetworkProfile[std::make_pair(pair.first, pair.second)] = test;
//         }

//         for (const auto deviceName : possibleDeviceList)
//         {
//             std::string deviceTypeName = getDeviceTypeName(devices[deviceName].type);
//             ModelProfile profile = queryModelProfile(
//                 *ctrl_metricsServerConn,
//                 ctrl_experimentName,
//                 ctrl_systemName,
//                 t.name,
//                 t.source,
//                 deviceName,
//                 deviceTypeName,
//                 ctrl_containerLib[containerName].modelName);
//             model->processProfiles[deviceTypeName] = profile;
//             // MODIFICATION ----------------

//             // collect the very first model of the pipeline, just use the yolo which is always the very first
//             if (containerName.find("yolo"))
//             {
//                 // add all available batch_size profiling into consideration
//                 for (auto it = profile.batchInfer.begin(); it != profile.batchInfer.end(); ++it)
//                 {
//                     BatchSizeType batch_size = it->first;
//                     BatchInferProfile &batch_profile = it->second;

//                     // note: the last three chars of the model name is the resolution it takes
//                     int width = std::stoi(model->name.substr(model->name.length() - 3));

//                     // check the accuracy indicator, use dummy value just to reflect the capacity of the model(evaluate their performance in general)
//                     model_profiles_jf.add(model->name, ACC_LEVEL_MAP.at(model->name), static_cast<int>(batch_size),
//                                           static_cast<float>(batch_profile.p95inferLat), width, width, model); // height and width are the same
//                 }
//             }

//             // -----------------------------
//         }

//         // ModelArrivalProfile profile = queryModelArrivalProfile(
//         //     *ctrl_metricsServerConn,
//         //     ctrl_experimentName,
//         //     ctrl_systemName,
//         //     t.name,
//         //     t.source,
//         //     ctrl_containerLib[containerName].taskName,
//         //     ctrl_containerLib[containerName].modelName,
//         //     possibleDeviceList,
//         //     possibleNetworkEntryPairs
//         // );
//         // std::cout << "sdfsdfasdf" << std::endl;
//     }
//     std::cout << "Task added: " << t.name << std::endl;
// }

// PipelineModelListType Controller::getModelsByPipelineType(PipelineType type, const std::string &startDevice)
// {
//     switch (type)
//     {
//     case PipelineType::Traffic:
//     {
//         PipelineModel *datasource = new PipelineModel{startDevice, "datasource", true, {}, {}};
//         PipelineModel *yolov5n = new PipelineModel{
//             "server",
//             "yolov5n",
//             true,
//             {},
//             {},
//             {},
//             {{datasource, -1}}};
//         datasource->downstreams.push_back({yolov5n, -1});

//         PipelineModel *retina1face = new PipelineModel{
//             "server",
//             "retina1face",
//             false,
//             {},
//             {},
//             {},
//             {{yolov5n, 0}}};
//         yolov5n->downstreams.push_back({retina1face, 0});

//         PipelineModel *carbrand = new PipelineModel{
//             "server",
//             "carbrand",
//             false,
//             {},
//             {},
//             {},
//             {{yolov5n, 2}}};
//         yolov5n->downstreams.push_back({carbrand, 2});

//         PipelineModel *platedet = new PipelineModel{
//             "server",
//             "platedet",
//             false,
//             {},
//             {},
//             {},
//             {{yolov5n, 2}}};
//         yolov5n->downstreams.push_back({platedet, 2});

//         PipelineModel *sink = new PipelineModel{
//             "server",
//             "sink",
//             false,
//             {},
//             {},
//             {},
//             {{retina1face, -1}, {carbrand, -1}, {platedet, -1}}};
//         retina1face->downstreams.push_back({sink, -1});
//         carbrand->downstreams.push_back({sink, -1});
//         platedet->downstreams.push_back({sink, -1});

//         return {datasource, yolov5n, retina1face, carbrand, platedet, sink};
//     }
//     case PipelineType::Building_Security:
//     {
//         PipelineModel *datasource = new PipelineModel{startDevice, "datasource", true, {}, {}};
//         PipelineModel *yolov5n = new PipelineModel{
//             "server",
//             "yolov5n",
//             true,
//             {},
//             {},
//             {},
//             {{datasource, -1}}};
//         datasource->downstreams.push_back({yolov5n, -1});

//         PipelineModel *retina1face = new PipelineModel{
//             "server",
//             "retina1face",
//             false,
//             {},
//             {},
//             {},
//             {{yolov5n, 0}}};
//         yolov5n->downstreams.push_back({retina1face, 0});

//         PipelineModel *movenet = new PipelineModel{
//             "server",
//             "movenet",
//             false,
//             {},
//             {},
//             {},
//             {{yolov5n, 0}}};
//         yolov5n->downstreams.push_back({movenet, 0});

//         PipelineModel *gender = new PipelineModel{
//             "server",
//             "gender",
//             false,
//             {},
//             {},
//             {},
//             {{retina1face, -1}}};
//         retina1face->downstreams.push_back({gender, -1});

//         PipelineModel *age = new PipelineModel{
//             "server",
//             "age",
//             false,
//             {},
//             {},
//             {},
//             {{retina1face, -1}}};
//         retina1face->downstreams.push_back({age, -1});

//         PipelineModel *sink = new PipelineModel{
//             "server",
//             "sink",
//             false,
//             {},
//             {},
//             {},
//             {{gender, -1}, {age, -1}, {movenet, -1}}};
//         gender->downstreams.push_back({sink, -1});
//         age->downstreams.push_back({sink, -1});
//         movenet->downstreams.push_back({sink, -1});

//         return {datasource, yolov5n, retina1face, movenet, gender, age, sink};
//     }
//     case PipelineType::Video_Call:
//     {
//         PipelineModel *datasource = new PipelineModel{startDevice, "datasource", true, {}, {}};
//         PipelineModel *retina1face = new PipelineModel{
//             "server",
//             "retina1face",
//             true,
//             {},
//             {},
//             {},
//             {{datasource, -1}}};
//         datasource->downstreams.push_back({retina1face, -1});

//         PipelineModel *emotionnet = new PipelineModel{
//             "server",
//             "emotionnet",
//             false,
//             {},
//             {},
//             {},
//             {{retina1face, -1}}};
//         retina1face->downstreams.push_back({emotionnet, -1});

//         PipelineModel *age = new PipelineModel{
//             "server",
//             "age",
//             false,
//             {},
//             {},
//             {},
//             {{retina1face, -1}}};
//         retina1face->downstreams.push_back({age, -1});

//         PipelineModel *gender = new PipelineModel{
//             "server",
//             "gender",
//             false,
//             {},
//             {},
//             {},
//             {{retina1face, -1}}};
//         retina1face->downstreams.push_back({gender, -1});

//         PipelineModel *arcface = new PipelineModel{
//             "server",
//             "arcface",
//             false,
//             {},
//             {},
//             {},
//             {{retina1face, -1}}};
//         retina1face->downstreams.push_back({arcface, -1});

//         PipelineModel *sink = new PipelineModel{
//             "server",
//             "sink",
//             false,
//             {},
//             {},
//             {},
//             {{emotionnet, -1}, {age, -1}, {gender, -1}, {arcface, -1}}};
//         emotionnet->downstreams.push_back({sink, -1});
//         age->downstreams.push_back({sink, -1});
//         gender->downstreams.push_back({sink, -1});
//         arcface->downstreams.push_back({sink, -1});

//         return {datasource, retina1face, emotionnet, age, gender, arcface, sink};
//     }
//     default:
//         return {};
//     }
// }

// /**
//  * @brief Recursively traverse the model tree and try shifting models to edge devices
//  *
//  * @param models
//  * @param slo
//  */
// void Controller::shiftModelToEdge(Controller::Pipeline &models, const ModelType &currModel, uint64_t slo)
// {
// }

// /**
//  * @brief
//  *
//  * @param models
//  * @param slo
//  * @param nObjects
//  * @return std::map<ModelType, int>
//  */
// void Controller::getInitialBatchSizes(
//     Controller::Pipeline &models, uint64_t slo,
//     int nObjects)
// {

//     // for (auto &m: models) {
//     //     ModelType modelType  = std::get<0>(m);
//     //     m.second.batchSize = 1;
//     //     m.second.numReplicas = 1;
//     // }

//     // // DFS-style recursively estimate the latency of a pipeline from source to sin
//     // estimatePipelineLatency(models, models.begin()->first, 0);

//     // uint64_t expectedE2ELatency = models.at(ModelType::Sink).expectedStart2HereLatency;

//     // if (slo < expectedE2ELatency) {
//     //     spdlog::info("SLO is too low for the pipeline to meet. Expected E2E latency: {0:d}, SLO: {1:d}", expectedE2ELatency, slo);
//     // }

//     // // Increase number of replicas to avoid bottlenecks
//     // for (auto &m: models) {
//     //     incNumReplicas(m.second, m.second.device);
//     // }

//     // // Find near-optimal batch sizes
//     // auto foundBest = true;
//     // while (foundBest) {
//     //     foundBest = false;
//     //     uint64_t bestCost = models.at(ModelType::Sink).estimatedStart2HereCost;
//     //     PipelineModelListType tmp_models = models;
//     //     for (auto &m: tmp_models) {
//     //         m.second.batchSize *= 2;
//     //         estimatePipelineLatency(tmp_models, tmp_models.begin()->first, 0);
//     //         expectedE2ELatency = tmp_models.at(ModelType::Sink).expectedStart2HereLatency;
//     //         if (expectedE2ELatency < slo) {
//     //             // If increasing the batch size of model `m` creates a pipeline that meets the SLO, we should keep it
//     //             uint64_t estimatedE2Ecost = tmp_models.at(ModelType::Sink).estimatedStart2HereCost;
//     //             // Unless the estimated E2E cost is better than the best cost, we should not consider it as a candidate
//     //             if (estimatedE2Ecost < bestCost) {
//     //                 bestCost = estimatedE2Ecost;
//     //                 models = tmp_models;
//     //                 foundBest = true;
//     //             }
//     //             if (!foundBest) {
//     //                 continue;
//     //             }
//     //             // If increasing the batch size meets the SLO, we can try decreasing the number of replicas
//     //             decNumReplicas(m.second, m.second.device);
//     //             estimatedE2Ecost = tmp_models.at(ModelType::Sink).estimatedStart2HereCost;
//     //             if (estimatedE2Ecost < bestCost) {
//     //                 models = tmp_models;
//     //                 foundBest = true;
//     //             }
//     //         } else {
//     //             m.second.batchSize /= 2;
//     //         }
//     //     }
//     // }
// }

// /**
//  * @brief estimate the different types of latency, in microseconds
//  * Due to batch inference's nature, the queries that come later has to wait for more time both in preprocessor and postprocessor.
//  *
//  * @param model infomation about the model
//  * @param modelType
//  */
// void Controller::estimateModelLatency(PipelineModel *currModel, const std::string &deviceName)
// {
//     ModelProfile profile = currModel->processProfiles[deviceName];
//     uint64_t preprocessLatency = profile.p95prepLat;
//     BatchSizeType batchSize = currModel->batchSize;
//     uint64_t inferLatency = profile.batchInfer[batchSize].p95inferLat;
//     uint64_t postprocessLatency = profile.p95postLat;
//     float preprocessRate = 1000000.f / preprocessLatency;

//     currModel->expectedQueueingLatency = calculateQueuingLatency(currModel->arrivalProfiles.arrivalRates, preprocessRate);
//     currModel->expectedAvgPerQueryLatency = preprocessLatency + inferLatency * batchSize + postprocessLatency;
//     currModel->expectedMaxProcessLatency = preprocessLatency * batchSize + inferLatency * batchSize + postprocessLatency * batchSize;
//     currModel->estimatedPerQueryCost = currModel->expectedAvgPerQueryLatency + currModel->expectedQueueingLatency + currModel->expectedTransferLatency;
// }

// /**
//  * @brief DFS-style recursively estimate the latency of a pipeline from source to sink
//  *
//  * @param pipeline provides all information about the pipeline needed for scheduling
//  * @param currModel
//  */
// void Controller::estimatePipelineLatency(PipelineModel *currModel, const uint64_t start2HereLatency)
// {
//     // estimateModelLatency(currModel, currModel->device);

//     // Update the expected latency to reach the current model
//     // In case a model has multiple upstreams, the expected latency to reach the model is the maximum of the expected latency
//     // to reach from each upstream.
//     currModel->expectedStart2HereLatency = std::max(
//         currModel->expectedStart2HereLatency,
//         start2HereLatency + currModel->expectedMaxProcessLatency + currModel->expectedTransferLatency + currModel->expectedQueueingLatency);

//     // Cost of the pipeline until the current model
//     currModel->estimatedStart2HereCost += currModel->estimatedPerQueryCost;

//     std::vector<std::pair<PipelineModel *, int>> downstreams = currModel->downstreams;
//     for (const auto &d : downstreams)
//     {
//         estimatePipelineLatency(d.first, currModel->expectedStart2HereLatency);
//     }

//     if (currModel->downstreams.size() == 0)
//     {
//         return;
//     }
// }

// /**
//  * @brief Attempts to increase the number of replicas to meet the arrival rate
//  *
//  * @param model the model to be scaled
//  * @param deviceName
//  * @return uint8_t The number of replicas to be added
//  */
// uint8_t Controller::incNumReplicas(const PipelineModel *model)
// {
//     uint8_t numReplicas = model->numReplicas;
//     std::string deviceTypeName = model->deviceTypeName;
//     ModelProfile profile = model->processProfiles.at(deviceTypeName);
//     uint64_t inferenceLatency = profile.batchInfer.at(model->batchSize).p95inferLat;
//     float indiProcessRate = 1 / (inferenceLatency + profile.p95prepLat + profile.p95postLat);
//     float processRate = indiProcessRate * numReplicas;
//     while (processRate < model->arrivalProfiles.arrivalRates)
//     {
//         numReplicas++;
//         processRate = indiProcessRate * numReplicas;
//     }
//     return numReplicas - model->numReplicas;
// }

// /**
//  * @brief Decrease the number of replicas as long as it is possible to meet the arrival rate
//  *
//  * @param model
//  * @return uint8_t The number of replicas to be removed
//  */
// uint8_t Controller::decNumReplicas(const PipelineModel *model)
// {
//     uint8_t numReplicas = model->numReplicas;
//     std::string deviceTypeName = model->deviceTypeName;
//     ModelProfile profile = model->processProfiles.at(deviceTypeName);
//     uint64_t inferenceLatency = profile.batchInfer.at(model->batchSize).p95inferLat;
//     float indiProcessRate = 1 / (inferenceLatency + profile.p95prepLat + profile.p95postLat);
//     float processRate = indiProcessRate * numReplicas;
//     while (numReplicas > 1)
//     {
//         numReplicas--;
//         processRate = indiProcessRate * numReplicas;
//         // If the number of replicas is no longer enough to meet the arrival rate, we should not decrease the number of replicas anymore.
//         if (processRate < model->arrivalProfiles.arrivalRates)
//         {
//             numReplicas++;
//             break;
//         }
//     }
//     return model->numReplicas - numReplicas;
// }

// /**
//  * @brief Calculate queueing latency for each query coming to the preprocessor's queue, in microseconds
//  * Queue type is expected to be M/D/1
//  *
//  * @param arrival_rate
//  * @param preprocess_rate
//  * @return uint64_t
//  */
// uint64_t Controller::calculateQueuingLatency(const float &arrival_rate, const float &preprocess_rate)
// {
//     float rho = arrival_rate / preprocess_rate;
//     float numQueriesInSystem = rho / (1 - rho);
//     float averageQueueLength = rho * rho / (1 - rho);
//     return (uint64_t)(averageQueueLength / arrival_rate * 1000000);
// }

// // --------------------------------------------------------------------------------------------------------
// //                                      implementation
// // --------------------------------------------------------------------------------------------------------

// ModelInfoJF::ModelInfoJF(int bs, float il, int w, int h, std::string n, float acc, PipelineModel *m)
// {
//     batch_size = bs;

//     // the inference_latency is us
//     inference_latency = il;

//     // throughput is req/s
//     throughput = (int(bs / (il * 1e-6)) / 10) * 10; // round it to be devidisble by 10 for better dp computing
//     width = w;
//     height = h;
//     name = n;
//     accuracy = acc;
//     model = m;
// }

// ClientInfoJF::ClientInfoJF(std::string _ip, float _budget, int _req_rate, PipelineModel *_model)
// {
//     ip = _ip;
//     budget = _budget;
//     // image_shape = std::make_tuple(_width, _height);
//     req_rate = _req_rate;
//     model = _model;
// }

// // /**
// //  * @brief change the bandwidth in runtime
// //  *
// //  * @param bw
// //  */
// // void ClientInfoJF::set_bandwidth(float bw)
// // {
// //     this->bandwidth = bw;
// // }

// void ClientInfoJF::set_transmission_latency(int lat)
// {
//     this->transmission_latency = lat;
// }

// bool ModelSetCompare::operator()(
//     const std::tuple<std::string, float> &lhs,
//     const std::tuple<std::string, float> &rhs) const
// {
//     return std::get<1>(lhs) < std::get<1>(rhs);
// }

// // -------------------------------------------------------------------------------------------
// //                               implementation of ModelProfilesJF
// // -------------------------------------------------------------------------------------------

// /**
//  * @brief add profiled information of model
//  *
//  * @param model_type
//  * @param accuracy
//  * @param batch_size
//  * @param inference_latency
//  * @param throughput
//  */
// void ModelProfilesJF::add(std::string name, float accuracy, int batch_size, float inference_latency, int width, int height, PipelineModel *m)
// {
//     auto key = std::tuple<std::string, float>{name, accuracy};
//     ModelInfoJF value(batch_size, inference_latency, width, height, name, accuracy, m);
//     infos[key].push_back(value);
// }

// void ModelProfilesJF::add(const ModelInfoJF &model_info)
// {
//     auto key =
//         std::tuple<std::string, float>{model_info.name, model_info.accuracy};
//     infos[key].push_back(model_info);
// }

// void ModelProfilesJF::debugging()
// {
//     std::cout << "======================ModelProfiles Debugging=======================" << std::endl;
//     for (auto it = infos.begin(); it != infos.end(); ++it)
//     {
//         auto key = it->first;
//         auto profilings = it->second;
//         std::cout << "*********************************************" << std::endl;
//         std::cout << "Model: " << std::get<0>(key) << ", Accuracy: " << std::get<1>(key) << std::endl;
//         for (const auto &model_info : profilings)
//         {
//             std::cout << "batch size: " << model_info.batch_size << ", latency: " << model_info.inference_latency
//                       << ", width: " << model_info.width << ", height: " << model_info.height << std::endl;
//         }
//         std::cout << "*********************************************" << std::endl;
//     }
// }

// // -------------------------------------------------------------------------------------------
// //                               implementation of ClientProfilesJF
// // -------------------------------------------------------------------------------------------

// /**
//  * @brief sort the budget which equals (SLO - networking time)
//  *
//  * @param clients
//  */
// void ClientProfilesJF::sortBudgetDescending(std::vector<ClientInfoJF> &clients)
// {
//     std::sort(clients.begin(), clients.end(),
//               [](const ClientInfoJF &a, const ClientInfoJF &b)
//               {
//                   // FIXME: reduce networking latency here
//                   return a.budget - a.transmission_latency > b.budget - b.transmission_latency;
//               });
// }

// void ClientProfilesJF::add(const std::string &ip, float budget, int req_rate, PipelineModel *model)
// {
//     infos.push_back(ClientInfoJF(ip, budget, req_rate, model));
// }

// void ClientProfilesJF::debugging()
// {
//     std::cout << "===================================ClientProfiles Debugging==========================" << std::endl;
//     for (const auto &client_info : infos)
//     {
//         std::cout << "Unique id: " << client_info.ip << ", buget: " << client_info.budget << ", req_rate: " << client_info.req_rate << std::endl;
//     }
// }

// // -------------------------------------------------------------------------------------------
// //                               implementation of scheduling algorithms
// // -------------------------------------------------------------------------------------------

// std::vector<ClientInfoJF> findOptimalClients(const std::vector<ModelInfoJF> &models,
//                                              std::vector<ClientInfoJF> &clients)
// {
//     // sort clients
//     ClientProfilesJF::sortBudgetDescending(clients);
//     std::cout << "findOptimal start" << std::endl;
//     std::cout << "available sorted clients: " << std::endl;
//     for (auto &client : clients)
//     {
//         std::cout << client.ip << " " << client.budget << " " << client.req_rate
//                   << std::endl;
//     }
//     std::cout << "available models: " << std::endl;
//     for (auto &model : models)
//     {
//         std::cout << model.name << " " << model.accuracy << " " << model.batch_size << " " << model.throughput << " " << model.inference_latency << std::endl;
//     }
//     std::tuple<int, int> best_cell;
//     int best_value = 0;

//     // dp
//     auto [max_batch_size, max_index] = findMaxBatchSize(models, clients[0]);

//     std::cout << "max batch size: " << max_batch_size
//               << " and index: " << max_index << std::endl;

//     assert(max_batch_size > 0);

//     // construct the dp matrix
//     int rows = clients.size() + 1;
//     int h = 10; // assume gcd of all clients' req rate
//     // find max throughput
//     int max_throughput = 0;
//     for (auto &model : models)
//     {
//         if (model.throughput > max_throughput)
//         {
//             max_throughput = model.throughput;
//         }
//     }
//     // init matrix
//     int cols = max_throughput / h + 1;
//     std::cout << "max_throughput: " << max_throughput << std::endl;
//     std::cout << "row: " << rows << " cols: " << cols << std::endl;
//     std::vector<std::vector<int>> dp_mat(rows, std::vector<int>(cols, 0));
//     // iterating
//     for (int client_index = 1; client_index < clients.size(); client_index++)
//     {
//         auto &client = clients[client_index];
//         auto result = findMaxBatchSize(models, client, max_batch_size);
//         max_batch_size = std::get<0>(result);
//         max_index = std::get<1>(result);
//         std::cout << "client ip: " << client.ip << ", max_batch_size: " << max_batch_size << ", max_index: "
//                   << max_index << std::endl;
//         if (max_batch_size <= 0)
//         {
//             break;
//         }
//         int cols_upperbound = int(models[max_index].throughput / h);
//         int lambda_i = client.req_rate;
//         int v_i = client.req_rate;
//         std::cout << "cols_up " << cols_upperbound << ", req " << lambda_i
//                   << std::endl;
//         for (int k = 1; k <= cols_upperbound; k++)
//         {

//             int w_k = k * h;
//             if (lambda_i <= w_k)
//             {
//                 int k_prime = (w_k - lambda_i) / h;
//                 int v = v_i + dp_mat[client_index - 1][k_prime];
//                 if (v > dp_mat[client_index - 1][k])
//                 {
//                     dp_mat[client_index][k] = v;
//                 }
//                 if (v > best_value)
//                 {
//                     best_cell = std::make_tuple(client_index, k);
//                     best_value = v;
//                 }
//             }
//             else
//             {
//                 dp_mat[client_index][k] = dp_mat[client_index - 1][k];
//             }
//         }
//     }

//     std::cout << "updated dp_mat" << std::endl;
//     for (auto &row : dp_mat)
//     {
//         for (auto &v : row)
//         {
//             std::cout << v << " ";
//         }
//         std::cout << std::endl;
//     }

//     // perform backtracing from (row, col)
//     // using dp_mat, best_cell, best_value

//     std::vector<ClientInfoJF> selected_clients;

//     auto [row, col] = best_cell;

//     std::cout << "best cell: " << row << " " << col << std::endl;
//     int w = dp_mat[row][col];
//     while (row > 0 && col > 0)
//     {
//         std::cout << row << " " << col << std::endl;
//         if (dp_mat[row][col] == dp_mat[row - 1][col])
//         {
//             row--;
//         }
//         else
//         {
//             auto c = clients[row - 1];
//             int w_i = c.req_rate;
//             row = row - 1;
//             col = int((w - w_i) / h);
//             w = col * h;
//             assert(w == dp_mat[row][col]);
//             selected_clients.push_back(c);
//         }
//     }

//     std::cout << "findOptimal end" << std::endl;
//     std::cout << "selected clients" << std::endl;
//     for (auto &sc : selected_clients)
//     {
//         std::cout << sc.ip << " " << sc.budget << " " << sc.req_rate << std::endl;
//     }

//     return selected_clients;
// }

// /**
//  * @brief client dnn mapping algorithm strictly following the paper jellyfish's Algo1
//  *
//  * @param client_profile
//  * @param model_profiles
//  * @return a vector of [ (model_name, accuracy), vec[clients], batch_size ]
//  */
// std::vector<
//     std::tuple<std::tuple<std::string, float>, std::vector<ClientInfoJF>, int>>
// mapClient(ClientProfilesJF client_profile, ModelProfilesJF model_profiles)
// {
//     std::cout << " ======================= mapClient ==========================" << std::endl;

//     std::vector<
//         std::tuple<std::tuple<std::string, float>, std::vector<ClientInfoJF>, int>>
//         mappings;
//     std::vector<ClientInfoJF> clients = client_profile.infos;

//     int map_size = model_profiles.infos.size();
//     int key_index = 0;
//     for (auto it = model_profiles.infos.begin(); it != model_profiles.infos.end();
//          ++it)
//     {
//         key_index++;
//         std::cout << "before filtering" << std::endl;
//         for (auto &c : clients)
//         {
//             std::cout << c.ip << " " << c.budget << " " << c.req_rate << std::endl;
//         }

//         auto selected_clients = findOptimalClients(it->second, clients);

//         // tradeoff:
//         // assign all left clients to the last available model
//         if (key_index == map_size)
//         {
//             std::cout << "assign all rest clients" << std::endl;
//             selected_clients = clients;
//             clients.clear();
//             std::cout << "selected clients assgined" << std::endl;
//             for (auto &c : selected_clients)
//             {
//                 std::cout << c.ip << " " << c.budget << " " << c.req_rate << std::endl;
//             }
//             assert(clients.size() == 0);
//         }

//         int batch_size = check_and_assign(it->second, selected_clients);

//         std::cout << "model throughput: " << it->second[0].throughput << std::endl;
//         std::cout << "batch size: " << batch_size << std::endl;

//         mappings.push_back(
//             std::make_tuple(it->first, selected_clients, batch_size));
//         std::cout << "start removing collected clients" << std::endl;
//         differenceClients(clients, selected_clients);
//         std::cout << "after filtering" << std::endl;
//         for (auto &c : clients)
//         {
//             std::cout << c.ip << " " << c.budget << " " << c.req_rate << std::endl;
//         }
//         if (clients.size() == 0)
//         {
//             break;
//         }
//     }

//     std::cout << "mapping relation" << std::endl;
//     for (auto &t : mappings)
//     {
//         std::cout << "======================" << std::endl;
//         auto [model_info, clients, batch_size] = t;
//         std::cout << std::get<0>(model_info) << " " << std::get<1>(model_info)
//                   << " " << batch_size << std::endl;
//         for (auto &client : clients)
//         {
//             std::cout << "client name: " << client.ip << ", req rate: " << client.req_rate << ", budget-lat: " << client.budget << std::endl;
//         }
//         std::cout << "======================" << std::endl;
//     }
//     std::cout << "======================= End mapClient =======================" << std::endl;
//     return mappings;
// }

// /**
//  * @brief find the max available batch size for the associated clients of
//  * corresponding model
//  *
//  * @param model
//  * @param selected_clients
//  * @return int
//  */
// int check_and_assign(std::vector<ModelInfoJF> &model,
//                      std::vector<ClientInfoJF> &selected_clients)
// {
//     int total_req_rate = 0;
//     // sum all selected req rate
//     for (auto &client : selected_clients)
//     {
//         total_req_rate += client.req_rate;
//     }
//     int max_batch_size = 1;

//     for (auto &model_info : model)
//     {
//         if (model_info.throughput > total_req_rate &&
//             max_batch_size < model_info.batch_size)
//         {
//             max_batch_size = model_info.batch_size;
//         }
//     }
//     return max_batch_size;
// }

// // ====================== helper functions implementation ============================

// /**
//  * @brief find the maximum batch size for the client, the model vector is the set of model only different in batch size
//  *
//  * @param models
//  * @param budget
//  * @return max_batch_size, index
//  */
// std::tuple<int, int> findMaxBatchSize(const std::vector<ModelInfoJF> &models,
//                                       const ClientInfoJF &client, int max_available_batch_size)
// {
//     int max_batch_size = 0;
//     float budget = client.budget;
//     int index = 0;
//     int max_index = 0;
//     for (const auto &model : models)
//     {
//         if (model.inference_latency * 2.0 < client.budget &&
//             model.batch_size > max_batch_size && model.batch_size <= max_available_batch_size)
//         {
//             max_batch_size = model.batch_size;
//             max_index = index;
//         }
//         index++;
//     }
//     return std::make_tuple(max_batch_size, max_index);
// }

// /**
//  * @brief remove the selected clients
//  *
//  * @param src
//  * @param diff
//  */
// void differenceClients(std::vector<ClientInfoJF> &src,
//                        const std::vector<ClientInfoJF> &diff)
// {
//     auto is_in_diff = [&diff](const ClientInfoJF &client)
//     {
//         return std::find(diff.begin(), diff.end(), client) != diff.end();
//     };
//     src.erase(std::remove_if(src.begin(), src.end(), is_in_diff), src.end());
// }

// // ====================================================================================

// // void Controller::Scheduling(ClientProfilesJF &client_profiles_jf, ModelProfilesJF &model_profiles_jf)
// // {
// //     // TODO: please out your scheduling loop inside of here
// //     while (running)
// //     {
// //         // use list of devices, tasks and containers to schedule depending on your algorithm
// //         // put helper functions as a private member function of the controller and write them at the bottom of this file.
// //         std::this_thread::sleep_for(std::chrono::milliseconds(
// //             5000)); // sleep time can be adjusted to your algorithm or just left at 5 seconds for now

// //         // update the networking time for each client-server pair first

// //         for (auto &client : client_profiles_jf.infos)
// //         {
// //             // NetworkProfile test = queryNetworkProfile(
// //             //     *ctrl_metricsServerConn,
// //             //     ctrl_experimentName,
// //             //     ctrl_systemName,
// //             //     t.name,
// //             //     t.source,
// //             //     ctrl_containerLib[containerName].taskName,
// //             //     ctrl_containerLib[containerName].modelName,
// //             //     pair.first,
// //             //     pair.second,
// //             //     possibleNetworkEntryPairs[pair]);
// //             // model->arrivalProfiles.d2dNetworkProfile[std::make_pair(pair.first, pair.second)] = test;
// //         }
// //     }
// // }