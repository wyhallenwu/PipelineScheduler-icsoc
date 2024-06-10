#include "container_agent.h"

ABSL_FLAG(std::string, name, "", "base name of container");
ABSL_FLAG(std::optional<std::string>, json, std::nullopt, "configurations for microservices as json");
ABSL_FLAG(std::optional<std::string>, json_path, std::nullopt, "json for configuration inside a file");
ABSL_FLAG(std::optional<std::string>, trt_json, std::nullopt, "optional json for TRTConfiguration");
ABSL_FLAG(std::optional<std::string>, trt_json_path, std::nullopt, "json for TRTConfiguration");
ABSL_FLAG(uint16_t, port, 0, "server port for the service");
ABSL_FLAG(int16_t, device, 0, "Index of GPU device");
ABSL_FLAG(uint16_t, verbose, 2, "verbose level 0:trace, 1:debug, 2:info, 3:warn, 4:error, 5:critical, 6:off");
ABSL_FLAG(uint16_t, logging_mode, 0, "0:stdout, 1:file, 2:both");
ABSL_FLAG(std::string, log_dir, "../logs", "Log path for the container");
ABSL_FLAG(uint16_t, profiling_mode, 0,
          "flag to make the model running in profiling mode 0:deployment, 1:profiling, 2:empty_profiling");


void addProfileConfigs(json &msvcConfigs, const json &profileConfigs) {
    msvcConfigs["profile_inputRandomizeScheme"] = profileConfigs.at("profile_inputRandomizeScheme");
    msvcConfigs["profile_stepMode"] = profileConfigs.at("profile_stepMode");
    msvcConfigs["profile_step"] = profileConfigs.at("profile_step");
    msvcConfigs["profile_numProfileReqs"] = profileConfigs.at("profile_numProfileReqs");
    msvcConfigs["msvc_idealBatchSize"] = profileConfigs.at("profile_minBatch");
    msvcConfigs["profile_numWarmUpBatches"] = profileConfigs.at("profile_numWarmUpBatches");
    msvcConfigs["profile_maxBatch"] = profileConfigs.at("profile_maxBatch");
    msvcConfigs["profile_minBatch"] = profileConfigs.at("profile_minBatch");
}

json loadRunArgs(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);

    std::string name = absl::GetFlag(FLAGS_name);
    int8_t device = (int8_t) absl::GetFlag(FLAGS_device);
    uint16_t logLevel = absl::GetFlag(FLAGS_verbose);
    uint16_t loggingMode = absl::GetFlag(FLAGS_logging_mode);
    std::string logPath = absl::GetFlag(FLAGS_log_dir);
    uint16_t profiling_mode = absl::GetFlag(FLAGS_profiling_mode);

    RUNMODE runmode = static_cast<RUNMODE>(profiling_mode);

    std::tuple<json, json> configs = msvcconfigs::loadJson();
    json containerConfigs = std::get<0>(configs);
    json profilingConfigs = std::get<1>(configs);

    BatchSizeType minBatch = profilingConfigs.at("profile_minBatch");
    std::string templateModelPath = profilingConfigs.at("profile_templateModelPath");

    /**
     * @brief     If this is profiling, set configurations to the first batch size that should be profiled
     * This includes
     * 1. Setting its name based on the template model path    
     * 2. Setting the batch size to the smallest profile batch size
     * 
     */
    if (profiling_mode == 2) {

        name = removeSubstring(templateModelPath, ".engine");
        name = replaceSubstring(name, "[batch]", std::to_string(minBatch));
        name = splitString(name, "/").back();
        logPath = "../model_profiles";
    }

    logPath += "/" + containerConfigs["cont_experimentName"].get<std::string>();
    std::filesystem::create_directory(
        std::filesystem::path(logPath)
    );

    logPath += "/" + containerConfigs["cont_systemName"].get<std::string>();
    std::filesystem::create_directory(
        std::filesystem::path(logPath)
    );

    logPath += "/" + containerConfigs["cont_pipeName"].get<std::string>() + "_" + name;
    std::filesystem::create_directory(
        std::filesystem::path(logPath)
    );

    containerConfigs["cont_device"] = device;
    containerConfigs["cont_name"] = name;
    containerConfigs["cont_logLevel"] = logLevel;
    containerConfigs["cont_loggingMode"] = loggingMode;
    containerConfigs["cont_logPath"] = logPath;
    containerConfigs["cont_RUNMODE"] = runmode;
    containerConfigs["cont_port"] = absl::GetFlag(FLAGS_port);

    std::ifstream metricsServerCfgsFile = std::ifstream(containerConfigs.at("cont_metricServerConfigs"));
    json metricsServerConfigs = json::parse(metricsServerCfgsFile);

    containerConfigs["cont_metricsServerConfigs"] = metricsServerConfigs;
    if (containerConfigs["cont_taskName"] != "datasource") {
        containerConfigs["cont_inferModelName"] = splitString(containerConfigs.at("cont_pipeline")[2]["path"], "/").back();
        containerConfigs["cont_inferModelName"] = splitString(containerConfigs["cont_inferModelName"], ".").front();
        // The maximum batch size supported by the model (for TensorRT)
        std::vector<std::string> modelOptions = splitString(containerConfigs["cont_inferModelName"], "_");
        BatchSizeType maxModelBatchSize = std::stoull(modelOptions[modelOptions.size() - 2]);
        if (static_cast<RUNMODE>(runmode) == RUNMODE::PROFILING) {
            containerConfigs["cont_maxBatchSize"] = std::min((BatchSizeType)profilingConfigs["profile_maxBatch"], maxModelBatchSize);
        } else if (static_cast<RUNMODE>(runmode) == RUNMODE::DEPLOYMENT) {
            containerConfigs["cont_maxBatchSize"] = maxModelBatchSize;
        }
    }

    for (uint16_t i = 0; i < containerConfigs["cont_pipeline"].size(); i++) {
        containerConfigs.at("cont_pipeline")[i]["msvc_experimentName"] = containerConfigs["cont_experimentName"];
        containerConfigs.at("cont_pipeline")[i]["msvc_systemName"] = containerConfigs["cont_systemName"];
        containerConfigs.at("cont_pipeline")[i]["msvc_contName"] = name;
        containerConfigs.at("cont_pipeline")[i]["msvc_pipelineName"] = containerConfigs["cont_pipeName"];
        containerConfigs.at("cont_pipeline")[i]["msvc_taskName"] = containerConfigs["cont_taskName"];
        containerConfigs.at("cont_pipeline")[i]["msvc_hostDevice"] = containerConfigs["cont_hostDevice"];
        containerConfigs.at("cont_pipeline")[i]["msvc_deviceIndex"] = device;
        containerConfigs.at("cont_pipeline")[i]["msvc_containerLogPath"] = containerConfigs["cont_logPath"].get<std::string>() + "/" + name;
        containerConfigs.at("cont_pipeline")[i]["msvc_RUNMODE"] = runmode;
        containerConfigs.at(
                "cont_pipeline")[i]["cont_metricsScrapeIntervalMillisec"] = metricsServerConfigs["metricsServer_metricsReportIntervalMillisec"];
        containerConfigs.at("cont_pipeline")[i]["msvc_numWarmUpBatches"] = containerConfigs.at("cont_numWarmUpBatches");
        if (containerConfigs["cont_taskName"] != "datasource") {
            containerConfigs.at("cont_pipeline")[i]["msvc_maxBatchSize"] = containerConfigs.at("cont_maxBatchSize");
            containerConfigs.at("cont_pipeline")[i]["msvc_allocationMode"] = containerConfigs.at("cont_allocationMode");
        }

        /**
         * @brief     If this is profiling, set configurations to the first batch size that should be profiled
         * This includes
         * 1. Setting its profile dir whose name is based on the template model path    
         * 2. Setting the batch size to the smallest profile batch size
         * 
         */
        if (profiling_mode == 1) {
            addProfileConfigs(containerConfigs.at("cont_pipeline")[i], profilingConfigs);
            
        } else if (profiling_mode == 2) {
            containerConfigs.at("cont_pipeline")[i].at("msvc_idealBatchSize") = minBatch;
            if (i == 0) {
                addProfileConfigs(containerConfigs.at("cont_pipeline")[i], profilingConfigs);
            } else if (i == 2) {
                // Set the path to the engine
                containerConfigs.at("cont_pipeline")[i].at("path") = replaceSubstring(templateModelPath, "[batch]",
                                                                                      std::to_string(minBatch));
            }
        }
    }

    json finalConfigs;
    finalConfigs["container"] = containerConfigs;
    finalConfigs["profiling"] = profilingConfigs;

    if (containerConfigs["cont_taskName"] != "datasource") {
        checkCudaErrorCode(cudaSetDevice(device), __func__);
    }

    return finalConfigs;
};

std::vector<BaseMicroserviceConfigs> msvcconfigs::LoadFromJson() {
    if (!absl::GetFlag(FLAGS_json).has_value()) {
        spdlog::trace("{0:s} attempts to parse Microservice Configs from command line.", __func__);
        if (absl::GetFlag(FLAGS_json_path).has_value()) {
            std::ifstream file(absl::GetFlag(FLAGS_json_path).value());
            spdlog::trace("{0:s} finished parsing Microservice Configs from command line.", __func__);
            return json::parse(file).get<std::vector<BaseMicroserviceConfigs>>();
        } else {
            spdlog::error("No Configurations found. Please provide configuration either as json or file.");
            exit(1);
        }
    } else {
        spdlog::trace("{0:s} attempts to parse Microservice Configs from file.", __func__);
        if (absl::GetFlag(FLAGS_json_path).has_value()) {
            spdlog::error("No Configurations found. Please provide configuration either as json or file.");
            exit(1);
        } else {
            spdlog::trace("{0:s} finished parsing Microservice Configs from file.", __func__);
            return json::parse(absl::GetFlag(FLAGS_json).value()).get<std::vector<BaseMicroserviceConfigs>>();
        }
    }
}

std::tuple<json, json> msvcconfigs::loadJson() {
    json containerConfigs, profilingConfigs;
    if (!absl::GetFlag(FLAGS_json).has_value()) {
        spdlog::trace("{0:s} attempts to load Json Configs from file.", __func__);
        if (absl::GetFlag(FLAGS_json_path).has_value()) {
            std::ifstream file(absl::GetFlag(FLAGS_json_path).value());
            auto json_file = json::parse(file);
            containerConfigs = json_file.at("container");
            try {
                profilingConfigs = json_file.at("profiling");
            } catch (json::out_of_range &e) {
                spdlog::trace("{0:s} No profiling configurations found.", __func__);
            } catch (json::parse_error &e) {
                spdlog::error("{0:s} Error parsing json file.", __func__);
            }
            spdlog::trace("{0:s} finished loading Json Configs from file.", __func__);
            return std::make_tuple(containerConfigs, profilingConfigs);
        } else {
            spdlog::error("No Configurations found. Please provide configuration either as json or file.");
            exit(1);
        }
    } else {
        spdlog::trace("{0:s} attempts to load Json Configs from commandline.", __func__);
        if (absl::GetFlag(FLAGS_json_path).has_value()) {
            spdlog::error("Two Configurations found. Please provide configuration either as json or file.");
            exit(1);
        } else {
            auto json_file = json::parse(absl::GetFlag(FLAGS_json).value());
            containerConfigs = json_file.at("container");
            try {
                profilingConfigs = json_file.at("profiling");
            } catch (json::out_of_range &e) {
                spdlog::trace("{0:s} No profiling configurations found.", __func__);
            }
            spdlog::trace("{0:s} finished loading Json Configs from command line.", __func__);
            return std::make_tuple(containerConfigs, profilingConfigs);
        }
    }
}

void ContainerAgent::profiling(const json &pipeConfigs, const json &profileConfigs) {

    json pipelineConfigs = pipeConfigs;

    BatchSizeType minBatch = profileConfigs.at("profile_minBatch");
    BatchSizeType maxBatch = profileConfigs.at("profile_maxBatch");
    uint8_t stepMode = profileConfigs.at("profile_stepMode");
    uint8_t step = profileConfigs.at("profile_step");
    std::string templateModelPath = profileConfigs.at("profile_templateModelPath");

    std::thread metricsThread(&ContainerAgent::collectRuntimeMetrics, this);
    metricsThread.detach();

    this->dispatchMicroservices();

    for (BatchSizeType batch = minBatch; batch <= maxBatch;) {
        spdlog::get("container_agent")->trace("{0:s} model with a max batch of {1:d}.", __func__, batch);
        if (batch != minBatch) {
            std::string profileDirPath, cont_name;

            cont_name = removeSubstring(templateModelPath, ".engine");
            cont_name = replaceSubstring(cont_name, "[batch]", std::to_string(batch));
            cont_name = splitString(cont_name, "/").back();

            profileDirPath = cont_logDir + "/" + cont_name;
            std::filesystem::create_directory(
                    std::filesystem::path(profileDirPath)
            );

            // Making sure all the microservices are paused before reloading and reallocating resources
            // this is essential to avoiding runtime memory errors
            for (uint8_t i = 0; i < cont_msvcsList.size(); i++) {
                cont_msvcsList[i]->pauseThread();
            }
            waitPause();

            // Reload the configurations and dynamic allocation based on the new configurations
            for (uint8_t i = 0; i < cont_msvcsList.size(); i++) {
                pipelineConfigs[i].at("msvc_idealBatchSize") = batch;
                pipelineConfigs[i].at("msvc_containerLogPath") = profileDirPath;
                pipelineConfigs[i].at("msvc_deviceIndex") = cont_deviceIndex;
                pipelineConfigs[i].at("msvc_contName") = cont_name;
                // Set the path to the engine
                if (i == 2) {
                    pipelineConfigs[i].at("path") = replaceSubstring(templateModelPath, "[batch]",
                                                                     std::to_string(batch));
                }
                cont_msvcsList[i]->loadConfigs(pipelineConfigs[i], false);
                cont_msvcsList[i]->setRELOAD();
            }

        }

        this->waitReady();
        this->PROFILING_START(batch);

        for (int i = 1; i <= batch; i *= 2) {
            for (auto msvc: cont_msvcsList) {
                msvc->msvc_idealBatchSize = i;
            }
            while (true) {
                spdlog::get("container_agent")->info("{0:s} waiting for profiling of model with a max batch of {1:d} and real batch of {2:d}.",
                             __func__, batch, i);
                if (cont_msvcsList[0]->checkPause()) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
        }

        spdlog::get("container_agent")->info("===============================================================================================");
        if (stepMode == 0) {
            batch += step;
        } else {
            batch *= 2;
        }
    }
}

ContainerAgent::ContainerAgent(const json &configs) {

    json containerConfigs = configs["container"];
    //std::cout << containerConfigs.dump(4) << std::endl;

    cont_experimentName = abbreviate(containerConfigs["cont_experimentName"].get<std::string>());
    cont_name = abbreviate(containerConfigs["cont_name"].get<std::string>());
    cont_pipeName = abbreviate(containerConfigs["cont_pipeName"].get<std::string>());
    cont_taskName = abbreviate(containerConfigs["cont_taskName"].get<std::string>());
    cont_hostDevice = abbreviate(containerConfigs["cont_hostDevice"].get<std::string>());
    cont_systemName = containerConfigs["cont_systemName"].get<std::string>();

    cont_deviceIndex = containerConfigs["cont_device"];

    cont_RUNMODE = containerConfigs["cont_RUNMODE"];

    setupLogger(
        cont_logDir,
        cont_name,
        containerConfigs["cont_loggingMode"],
        containerConfigs["cont_logLevel"],
        cont_loggerSinks,
        cont_logger
    );

    // if (cont_RUNMODE == RUNMODE::EMPTY_PROFILING) {
    //     // Create the logDir for this container
    //     cont_logDir = (std::string) containerConfigs.at("cont_logPath");
    //     std::filesystem::create_directory(
    //             std::filesystem::path(cont_logDir)
    //     );
    // } else {
    //     cont_logDir = (std::string) containerConfigs["cont_logPath"];
    // }

    arrivalRate = 0;

    if (cont_taskName != "datasource") {
        cont_inferModel = abbreviate(containerConfigs["cont_inferModelName"].get<std::string>());
        cont_metricsServerConfigs.from_json(containerConfigs["cont_metricsServerConfigs"]);
        cont_metricsServerConfigs.schema = cont_experimentName + "_" + cont_systemName;
        cont_metricsServerConfigs.user = "container_agent";
        cont_metricsServerConfigs.password = "agent";

        cont_metricsServerConn = connectToMetricsServer(cont_metricsServerConfigs, cont_name);
        // Create arrival table
        std::string sql_statement;

        sql_statement = absl::StrFormat("CREATE SCHEMA IF NOT EXISTS %s;", cont_metricsServerConfigs.schema);
        executeSQL(*cont_metricsServerConn, sql_statement);

        if (cont_RUNMODE == RUNMODE::DEPLOYMENT) {
            cont_arrivalTableName = cont_metricsServerConfigs.schema + "." + cont_experimentName + "_" +  cont_pipeName + "_" + cont_taskName + "_arr";
            cont_processTableName = cont_metricsServerConfigs.schema + "." + cont_experimentName + "_" +  cont_pipeName + "__" + cont_inferModel + "__" + cont_hostDevice + "_proc";
            cont_hwMetricsTableName = cont_metricsServerConfigs.schema + "." + cont_experimentName + "_" +  cont_pipeName + "__" + cont_inferModel + "__" + cont_hostDevice + "_hw";
        } else if (cont_RUNMODE == RUNMODE::PROFILING) {
            cont_arrivalTableName = cont_experimentName + "_" +  cont_pipeName + "_" + cont_taskName + "_arrival_table";
            cont_processTableName = cont_experimentName + "__" + cont_inferModel + "__" + cont_hostDevice + "_proc";
            cont_hwMetricsTableName =
                    cont_experimentName + "__" + cont_inferModel + "__" + cont_hostDevice + "_hw";

            sql_statement = "DROP TABLE IF EXISTS " + cont_arrivalTableName + ";";
            executeSQL(*cont_metricsServerConn, sql_statement);
        }

        if (!tableExists(*cont_metricsServerConn, cont_arrivalTableName)) {
            sql_statement = "CREATE TABLE IF NOT EXISTS " + cont_arrivalTableName + " ("
                                                                                "arrival_timestamps BIGINT NOT NULL, "
                                                                                "stream TEXT NOT NULL, "
                                                                                "sender_host TEXT NOT NULL, "
                                                                                "receiver_host TEXT NOT NULL, "
                                                                                "transfer_duration BIGINT NOT NULL, "
                                                                                "full_transfer_duration BIGINT NOT NULL, "
                                                                                "rpc_batch_size INT2 NOT NULL, "
                                                                                "total_package_size INTEGER NOT NULL, "
                                                                                "request_size INTEGER NOT NULL, "
                                                                                "request_num INTEGER NOT NULL)";

            executeSQL(*cont_metricsServerConn, sql_statement);

            sql_statement = "SELECT create_hypertable('" + cont_arrivalTableName + "', 'arrival_timestamps', if_not_exists => TRUE);";
            
            executeSQL(*cont_metricsServerConn, sql_statement);

            sql_statement = "CREATE INDEX ON " + cont_arrivalTableName + " (stream);";
            sql_statement += "CREATE INDEX ON " + cont_arrivalTableName + " (sender_host);";
            sql_statement += "CREATE INDEX ON " + cont_arrivalTableName + " (receiver_host);";
            
            executeSQL(*cont_metricsServerConn, sql_statement);
        }

        if (!tableExists(*cont_metricsServerConn, cont_processTableName)) {
            sql_statement = "CREATE TABLE IF NOT EXISTS " + cont_processTableName + " ("
                                                                                "postprocess_timestamps BIGINT NOT NULL, "
                                                                                "stream TEXT NOT NULL, "
                                                                                "host TEXT NOT NULL, "
                                                                                "prep_duration INTEGER NOT NULL, "
                                                                                "batch_duration INTEGER NOT NULL, "
                                                                                "infer_duration INTEGER NOT NULL, "
                                                                                "post_duration INTEGER NOT NULL, "
                                                                                "infer_batch_size INT2 NOT NULL, "
                                                                                "input_size INTEGER NOT NULL, "
                                                                                "output_size INTEGER NOT NULL, "
                                                                                "request_num INTEGER NOT NULL)";

            executeSQL(*cont_metricsServerConn, sql_statement);

            sql_statement = "SELECT create_hypertable('" + cont_processTableName + "', 'postprocess_timestamps', if_not_exists => TRUE);";
            executeSQL(*cont_metricsServerConn, sql_statement);

            sql_statement = "CREATE INDEX ON " + cont_processTableName + " (stream);";
            sql_statement += "CREATE INDEX ON " + cont_processTableName + " (host);";
            executeSQL(*cont_metricsServerConn, sql_statement);
        }

        if (cont_RUNMODE == RUNMODE::PROFILING) {
            if (!tableExists(*cont_metricsServerConn, cont_hwMetricsTableName)) {
                sql_statement = "CREATE TABLE IF NOT EXISTS " + cont_hwMetricsTableName + " ("
                                                                                    "   timestamps BIGINT NOT NULL,"
                                                                                    "   batch_size INT2 NOT NULL,"
                                                                                    "   cpu_usage INT2 NOT NULL," // percentage (1-100)
                                                                                    "   mem_usage INT NOT NULL," // Megabytes
                                                                                    "   gpu_usage INT2 NOT NULL," // percentage (1-100)
                                                                                    "   gpu_mem_usage INT NOT NULL," // Megabytes
                                                                                    "   PRIMARY KEY (timestamps)"
                                                                                    ");";
                executeSQL(*cont_metricsServerConn, sql_statement);

                sql_statement = "SELECT create_hypertable('" + cont_hwMetricsTableName + "', 'timestamps', if_not_exists => TRUE);";
                executeSQL(*cont_metricsServerConn, sql_statement);

                sql_statement += "CREATE INDEX ON " + cont_hwMetricsTableName + " (batch_size);";
                executeSQL(*cont_metricsServerConn, sql_statement);
            }
        }
        spdlog::get("container_agent")->info("{0:s} created arrival table and process table.", cont_name);
        spdlog::get("container_agent")->flush();
    }


    int own_port = containerConfigs.at("cont_port");

    std::string server_address = absl::StrFormat("%s:%d", "localhost", own_port);
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    server_cq = builder.AddCompletionQueue();
    server = builder.BuildAndStart();

    stub = InDeviceCommunication::NewStub(grpc::CreateChannel("localhost:60003", grpc::InsecureChannelCredentials()));
    sender_cq = new CompletionQueue();

    run = true;
    reportHwMetrics = false;
    profiler = nullptr;
    std::thread receiver(&ContainerAgent::HandleRecvRpcs, this);
    receiver.detach();
}

void ContainerAgent::ReportStart() {
    ProcessData request;
    request.set_msvc_name(cont_name);
    ProcessData reply;
    ClientContext context;
    std::unique_ptr<ClientAsyncResponseReader<ProcessData>> rpc(
            stub->AsyncReportMsvcStart(&context, request, sender_cq));
    Status status;
    rpc->Finish(&reply, &status, (void *) 1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(sender_cq->Next(&got_tag, &ok));
    GPR_ASSERT(ok);
    pid = reply.pid();
    spdlog::get("container_agent")->info("Container Agent started with pid: {0:d}", pid);
    if (cont_taskName != "datasource" && cont_taskName != "sink") {
        profiler = new Profiler({pid});
        reportHwMetrics = true;
    }
}


void ContainerAgent::runService(const json &pipeConfigs, const json &configs) {
    if (configs["container"]["cont_RUNMODE"] == RUNMODE::EMPTY_PROFILING) {
        profiling(pipeConfigs, configs["profiling"]);
    } else {
        this->dispatchMicroservices();

        this->waitReady();
        this->START();

        collectRuntimeMetrics();
    }
}


void ContainerAgent::collectRuntimeMetrics() {
    std::vector<int> queueSizes;
    int lateCount;
    ArrivalRecordType arrivalRecords;
    ProcessRecordType processRecords;
    std::string sql;
    while (run) {
        if (cont_taskName == "datasource") {
            std::this_thread::sleep_for(std::chrono::milliseconds(10000));
            continue;
        }
        auto metricsStopwatch = Stopwatch();
        metricsStopwatch.start();
        auto startTime = metricsStopwatch.getStartTime();
        if (startTime >= cont_metricsServerConfigs.nextHwMetricsScrapeTime) {
            if (reportHwMetrics && pid > 0) {
                Profiler::sysStats stats = profiler->reportAtRuntime(getpid(), pid);
                HardwareMetrics hwMetrics = {startTime, stats.cpuUsage, stats.memoryUsage, stats.gpuUtilization,
                                             stats.gpuMemoryUsage};
                cont_hwMetrics.emplace_back(hwMetrics);
                cont_metricsServerConfigs.nextHwMetricsScrapeTime += std::chrono::milliseconds(
                        cont_metricsServerConfigs.hwMetricsScrapeIntervalMillisec);
            }
        }

        metricsStopwatch.reset();
        auto scrapeLatency = metricsStopwatch.elapsed_seconds();

        metricsStopwatch.start();
        startTime = metricsStopwatch.getStartTime();
        if (startTime >= cont_metricsServerConfigs.nextMetricsReportTime) {
            for (auto msvc: cont_msvcsList) {
                queueSizes.push_back(msvc->GetOutQueueSize(0));
                lateCount = cont_msvcsList[1]->GetDroppedReqCount();
            }

            pqxx::work session(*cont_metricsServerConn);
            std::string modelName = cont_msvcsList[2]->getModelName();
            if (cont_RUNMODE == RUNMODE::PROFILING) {
                if (reportHwMetrics && !cont_hwMetrics.empty()) {
                    sql = "INSERT INTO " + cont_hwMetricsTableName +
                        " (timestamps, batch_size, cpu_usage, mem_usage, gpu_usage, gpu_mem_usage) VALUES ";
                    for (const auto &record: cont_hwMetrics) {
                        sql += "(" + timePointToEpochString(record.timestamp) + ", ";
                        sql += std::to_string(cont_msvcsList[1]->msvc_idealBatchSize) + ", ";
                        sql += std::to_string(record.cpuUsage) + ", ";
                        sql += std::to_string(record.memUsage) + ", ";
                        sql += std::to_string(record.gpuUsage) + ", ";
                        sql += std::to_string(record.gpuMemUsage) + ")";
                        if (&record != &cont_hwMetrics.back()) {
                            sql += ", ";
                        }
                    }
                    sql += ";";
                    session.exec(sql.c_str());
                    cont_hwMetrics.clear();
                }
                if (cont_msvcsList[0]->STOP_THREADS) {
                    for (auto msvc: cont_msvcsList) {
                        msvc->STOP_THREADS = true;
                    }
                    run = false;
                }
            }
            arrivalRecords = cont_msvcsList[1]->getArrivalRecords();
            if (!arrivalRecords.empty()) {

                sql = "INSERT INTO " + cont_arrivalTableName +
                      "(arrival_timestamps, stream, sender_host, receiver_host, transfer_duration, "
                      "full_transfer_duration, rpc_batch_size, total_package_size, request_size, request_num) "
                      "VALUES ";
                for (auto &record: arrivalRecords) {
                    sql += "(" + timePointToEpochString(record.arrivalTime) + ", ";
                    sql += "'" + record.reqOriginStream + "'" + ", ";
                    sql += "'" + record.originDevice + "'" + ", ";
                    sql += "'" + cont_hostDevice + "'" + ", ";
                    sql += std::to_string(std::chrono::duration_cast<TimePrecisionType>(
                            record.arrivalTime - record.prevSenderTime).count()) + ", ";
                    sql += std::to_string(std::chrono::duration_cast<TimePrecisionType>(
                            record.arrivalTime - record.prevPostProcTime).count()) + ", ";
                    sql += std::to_string(record.rpcBatchSize) + ", ";
                    sql += std::to_string(record.totalPkgSize) + ", ";
                    sql += std::to_string(record.reqSize) + ", ";
                    sql += std::to_string(record.reqNum) + ")";

                    if (&record != &arrivalRecords.back()) {
                        sql += ", ";
                    } else {
                        sql += ";";
                    }
                }
                session.exec(sql.c_str());
                arrivalRecords.clear();
            }

            processRecords = cont_msvcsList[3]->getProcessRecords();
            if (!processRecords.empty()) {

                sql = "INSERT INTO " + cont_processTableName +
                      "(postprocess_timestamps, stream, host, prep_duration, "
                      "batch_duration, infer_duration, post_duration, infer_batch_size, input_size, "
                      "output_size, request_num) "
                      "VALUES ";
                for (auto &record: processRecords) {
                    sql += "(" + timePointToEpochString(record.postEndTime) + ", ";
                    sql += "'" + record.reqOriginStream + "'" + ", ";
                    sql += "'" + cont_hostDevice + "'" + ", ";
                    sql += std::to_string(std::chrono::duration_cast<TimePrecisionType>(
                            record.preEndTime - record.preStartTime).count()) + ", ";
                    sql += std::to_string(std::chrono::duration_cast<TimePrecisionType>(
                            record.batchingEndTime - record.preEndTime).count()) + ", ";
                    sql += std::to_string(std::chrono::duration_cast<TimePrecisionType>(
                            record.batchInferenceTime - record.batchingEndTime).count()) + ", ";
                    sql += std::to_string(std::chrono::duration_cast<TimePrecisionType>(
                            record.postEndTime - record.postStartTime).count()) + ", ";
                    sql += std::to_string(record.inferBatchSize) + ", ";
                    sql += std::to_string(record.inputSize) + ", ";
                    sql += std::to_string(record.outputSize) + ", ";
                    sql += std::to_string(record.reqNum) + ")";
                    if (&record != &processRecords.back()) {
                        sql += ", ";
                    } else {
                        sql += ";";
                    }
                }
                session.exec(sql.c_str());
                processRecords.clear();
            }
            session.commit();
            cont_metricsServerConfigs.nextMetricsReportTime += std::chrono::milliseconds(
                    cont_metricsServerConfigs.metricsReportIntervalMillisec);
        }
        metricsStopwatch.stop();
        auto reportLatency = metricsStopwatch.elapsed_seconds();

        std::chrono::milliseconds sleepPeriod(
                cont_metricsServerConfigs.metricsReportIntervalMillisec - (scrapeLatency + reportLatency) / 1000);
        std::this_thread::sleep_for(sleepPeriod);
    }
}

void ContainerAgent::HandleRecvRpcs() {
    new StopRequestHandler(&service, server_cq.get(), &run);
    new UpdateSenderRequestHandler(&service, server_cq.get(), &cont_msvcsList);
    new UpdateBatchSizeRequestHandler(&service, server_cq.get(), &cont_msvcsList);
    void *tag;
    bool ok;
    while (run) {
        GPR_ASSERT(server_cq->Next(&tag, &ok));
        GPR_ASSERT(ok);
        static_cast<RequestHandler *>(tag)->Proceed();
    }
}

void ContainerAgent::StopRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestStopExecution(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        *run = false;
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void ContainerAgent::UpdateSenderRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestUpdateSender(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new UpdateSenderRequestHandler(service, cq, msvcs);
        // TODO: Handle reconfiguration by restarting sender
        // pause processing except senders to clear out the queues
        for (auto msvc: *msvcs) {
            if (msvc->dnstreamMicroserviceList[0].name == request.name()) {
                continue;
            }
            msvc->pauseThread();
        }
        json config;
        std::vector<ThreadSafeFixSizedDoubleQueue *> inqueue;
        for (auto msvc: *msvcs) {
            if (msvc->dnstreamMicroserviceList[0].name == request.name()) {
                config = msvc->msvc_configs;
                config["msvc_dnstreamMicroservices"][0]["nb_link"][0] = absl::StrFormat("%s:%d", request.ip(),
                                                                                        request.port());
                inqueue = msvc->GetInQueue();
                msvc->stopThread();
                msvcs->erase(std::remove(msvcs->begin(), msvcs->end(), msvc), msvcs->end());
                break;
            }
        }
//        if (request.ip() == "localhost") {
//            // change postprocessing to keep the data on gpu
//
//            // start new GPU sender
//            msvcs->push_back(new GPUSender(config));
//        } else {
        // change postprocessing to offload data from gpu

        // start new serialized sender
        msvcs->push_back(new RemoteCPUSender(config));
//        }
        // align the data queue from postprocessor to new sender
        msvcs->back()->SetInQueue(inqueue);
        //start the new sender
        msvcs->back()->dispatchThread();
        for (auto msvc: *msvcs) {
            if (msvc->dnstreamMicroserviceList[0].name == request.name()) {
                continue;
            }
            msvc->unpauseThread();
        }

        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void ContainerAgent::UpdateBatchSizeRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestUpdateBatchSize(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new UpdateBatchSizeRequestHandler(service, cq, msvcs);
        // adjust batch size
//        for (auto msvc : *msvcs) {
//            msvc->setBatchSize(request.batch_size());
//        }
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

/**
 * @brief Check if all the microservices are paused
 * 
 * @return true 
 * @return false 
 */
bool ContainerAgent::checkPause() {
    for (auto msvc: cont_msvcsList) {
        if (msvc->checkPause()) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Wait for all the microservices to be paused
 * 
 */
void ContainerAgent::waitPause() {
    bool paused = false;
    while (true) {
        paused = true;
        spdlog::get("container_agent")->trace("{0:s} waiting for all microservices to be paused.", __func__);
        for (auto msvc: cont_msvcsList) {
            if (!msvc->checkPause()) {
                paused = false;
                break;
            }
        }
        if (paused) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

}

/**
 * @brief Check if all the microservices are ready
 * 
 * @return true 
 * @return false 
 */
bool ContainerAgent::checkReady() {
    for (auto msvc: cont_msvcsList) {
        if (!msvc->checkReady()) {
            return true;
        }
    }
    return true;

}

/**
 * @brief Wait for all the microservices to be ready
 * 
 */
void ContainerAgent::waitReady() {
    ReportStart();
    bool ready = false;
    while (!ready) {
        ready = true;

        spdlog::get("container_agent")->info("{0:s} waiting for all microservices to be ready.", __func__);
        for (auto msvc: cont_msvcsList) {
            if (!msvc->checkReady()) {
                ready = false;
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
}
