#include "misc.h"

/**
 * @brief Estimate network latency of a package of size `totalPkgSize` using linear interpolation
 * 
 * @param res contains packge size and latency data of sample packages in res[:]["p95_total_package_size_b"] and res[:]["p95_transfer_duration_us"]
 * @param totalPkgSize 
 * @return uint64_t 
 */
uint64_t estimateNetworkLatency(pqxx::result &res, const uint32_t &totalPkgSize) {
    if (res.empty()) {
        throw std::invalid_argument("The result set is empty.");
    }
    for (size_t i = 0; i < res.size() - 1; ++i) {
        uint32_t pkgSize1 = res[i]["p95_total_package_size_b"].as<uint32_t>();
        uint32_t pkgSize2 = res[i + 1]["p95_total_package_size_b"].as<uint32_t>();
        uint64_t latency1 = res[i]["p95_transfer_duration_us"].as<uint64_t>();
        uint64_t latency2 = res[i + 1]["p95_transfer_duration_us"].as<uint64_t>();

        if (totalPkgSize >= pkgSize1 && totalPkgSize <= pkgSize2) {
            // Linear interpolation formula
            double t = (double)(totalPkgSize - pkgSize1) / (pkgSize2 - pkgSize1);
            return latency1 + t * (latency2 - latency1);
        }
    }
}

/**
 * @brief 
 * 
 * @param pipelineName 
 * @param streamName 
 * @param taskName 
 * @param periods 
 * @return float 
 */
ModelArrivalProfile queryModelArrivalProfile(
    pqxx::connection &metricsConn,
    const std::string &experimentName,
    const std::string &systemName,
    const std::string &pipelineName,
    const std::string &streamName,
    const std::string &taskName,
    const std::string &modelName,
    const std::string &senderHost,
    const std::string &receiverHost,
    const std::vector<uint8_t> &periods //seconds
) {
    ModelArrivalProfile arrivalProfile;

    std::string senderHostAbbr = abbreviate(senderHost);
    std::string receiverHostAbbr = abbreviate(receiverHost);

    std::string schemaName = abbreviate(experimentName + "_" + systemName);
    std::string tableName = abbreviate(experimentName + "_" + pipelineName + "_" + taskName + "_arr");
   
    std::string periodQuery;
    for (const auto &period: periods) {
        periodQuery += absl::StrFormat("recent_data.arrival_rate_%ds,", period);
    }
    periodQuery.pop_back();

    std::string query = "WITH recent_data AS ("
                        "   SELECT * "
                        "   FROM %s "
                        "   WHERE timestamps >= (EXTRACT(EPOCH FROM NOW()) * 1000000 - 120 * 1000000)"
                        "   LIMIT 1"
                        "), "
                        "arrival_rate AS ("
                        "  SELECT GREATEST(%s) AS max_rate "
                        "  FROM recent_data "
                        "  WHERE stream = '%s' "
                        ") "
                        "SELECT MAX(max_rate) AS max_arrival_rate "
                        "FROM arrival_rate;";
    query = absl::StrFormat(query.c_str(), schemaName + "." + tableName, periodQuery, streamName);
    std::cout << query << std::endl;
    
    pqxx::result res = pullSQL(metricsConn, query);
    if (res[0][0].is_null()) {
        // If there is no historical data, we look for the rate of the most recent profiled data
        std::string profileTableName = abbreviate("prof_" + taskName + "_arr");
        query = "WITH recent_data AS ("
                "   SELECT * "
                "   FROM %s "
                "   LIMIT 10 "
                "), "
                "arrival_rate AS ("
                "  SELECT GREATEST(%s) AS max_rate "
                "  FROM recent_data "
                ") "
                "SELECT MAX(max_rate) AS max_arrival_rate "
                "FROM arrival_rate;";
        query = absl::StrFormat(query.c_str(), profileTableName, periodQuery);
        res = pullSQL(metricsConn, query);
    }
    arrivalProfile.arrivalRates = res[0]["max_arrival_rate"].as<float>();

    NetworkProfile *d2dNetworkProfile = &(arrivalProfile.d2dNetworkProfile[{senderHostAbbr, receiverHostAbbr}]);

    /**
     * @brief Querying for the network profile from the data in the last 120 seconds.
     * 
     */
    query = "WITH recent_data AS ("
            "   SELECT p95_out_queueing_duration_us, p95_transfer_duration_us, p95_queueing_duration_us, p95_total_package_size_b "
            "   FROM %s "
            "   WHERE stream = '%s' AND sender_host = '%s' AND receiver_host = '%s' AND timestamps >= (EXTRACT(EPOCH FROM NOW()) * 1000000 - 120 * 1000000)"
            "   LIMIT 100"
            ") "
            "SELECT "
            "   MAX(p95_out_queueing_duration_us) AS p95_out_queueing_duration_us, "
            "   MAX(p95_transfer_duration_us) AS p95_transfer_duration_us, "
            "   MAX(p95_queueing_duration_us) AS p95_queuing_duration_us, "
            "   MAX(p95_total_package_size_b) AS p95_total_package_size_b "
            "FROM recent_data;";
    query = absl::StrFormat(query.c_str(), schemaName + "." + tableName, streamName, senderHostAbbr, receiverHostAbbr);
    res = pullSQL(metricsConn, query);

    // if there are most current entries, then great, we update the profile and that's that
    if (!res[0]["p95_transfer_duration_us"].is_null()) {
        d2dNetworkProfile->p95OutQueueingDuration = res[0]["p95_out_queueing_duration_us"].as<uint64_t>();
        d2dNetworkProfile->p95QueueingDuration = res[0]["p95_queuing_duration_us"].as<uint64_t>();
        d2dNetworkProfile->p95PackageSize = res[0]["p95_total_package_size_b"].as<uint32_t>();
        d2dNetworkProfile->p95TransferDuration = res[0]["p95_transfer_duration_us"].as<uint64_t>();
    // If there is no historical data, we look for the rate of the most recent profiled data
    } else {
        std::string profileTableName = abbreviate("prof_" + taskName + "_arr");
        query = "WITH recent_data AS ("
        "   SELECT p95_out_queueing_duration_us, p95_queueing_duration_us, p95_total_package_size_b "
        "   FROM %s "
        "   WHERE stream = '%s' AND sender_host = '%s' AND receiver_host = '%s'"
        "   LIMIT 100"
        ") "
        "SELECT "
        "   MAX(p95_out_queueing_duration_us) AS p95_out_queueing_duration_us, "
        "   MAX(p95_queueing_duration_us) AS p95_queuing_duration_us, "
        "   MAX(p95_total_package_size_b) AS p95_total_package_size_b "
        "FROM recent_data;";
        query = absl::StrFormat(query.c_str(), profileTableName, streamName, senderHostAbbr, receiverHostAbbr);
        res = pullSQL(metricsConn, query);

        d2dNetworkProfile->p95OutQueueingDuration = res[0]["p95_out_queueing_duration_us"].as<uint64_t>();
        d2dNetworkProfile->p95QueueingDuration = res[0]["p95_queuing_duration_us"].as<uint64_t>();
        d2dNetworkProfile->p95PackageSize = res[0]["p95_total_package_size_b"].as<uint32_t>();

        std::string networkTableName = abbreviate("prof_" + receiverHost + "_netw");
        query = "SELECT p95_transfer_duration_us, p95_total_package_size_b "
                "FROM %s "
                "WHERE sender_host = '%s' "
                "LIMIT 100;";
        query = absl::StrFormat(query.c_str(), networkTableName, senderHostAbbr);
        res = pullSQL(metricsConn, query);
        // Estimate the upperbound of the transfer duration
        d2dNetworkProfile->p95TransferDuration = estimateNetworkLatency(res, d2dNetworkProfile->p95PackageSize);
    }
    return arrivalProfile;
}

/**
 * @brief Query pre and post processing latency
 * 
 * @param metricsConn 
 * @param tableName 
 * @param streamName 
 * @param deviceName 
 * @param modelName 
 * @param profile 
 */
void queryPrePostLatency(
    pqxx::connection &metricsConn,
    const std::string &experimentName,
    const std::string &systemName,
    const std::string &pipelineName,
    const std::string &streamName,
    const std::string &deviceName,
    const std::string &modelName,
    ModelProfile &profile
) {
    std::string schemaName = abbreviate(experimentName + "_" + systemName);
    std::string tableName = schemaName + "." + abbreviate(experimentName + "_" + pipelineName + "__" + modelName + "__" + deviceName + "_proc");
    std::string query = absl::StrFormat("WITH recent_data AS ("
            "SELECT p95_prep_duration_us, p95_post_duration_us, p95_input_size_b, p95_output_size_b "
            "FROM %s "
            "WHERE timestamps >= (EXTRACT(EPOCH FROM NOW()) * 1000000 - 120 * 1000000) AND stream = '%s' "
            ")"
            "SELECT "
            "   MAX(p95_prep_duration_us) AS p95_prep_duration_us_all, "
            "   MAX(p95_post_duration_us) AS p95_post_duration_us_all, "
            "   MAX(p95_input_size_b) AS p95_input_size_b_all, "
            "   MAX(p95_output_size_b) AS p95_output_size_b_all "
            "FROM recent_data;", tableName, streamName);


    pqxx::result res = pullSQL(metricsConn, query);
    // If most current historical data is not available, we query profiled data
    if (res[0][0].is_null()) {
        std::string profileTableName = abbreviate("prof__" + modelName +  "__" + deviceName + "_proc");
        query = absl::StrFormat("WITH recent_data AS ("
                                "SELECT p95_prep_duration_us, p95_post_duration_us, p95_input_size_b, p95_output_size_b "
                                "FROM %s "
                                "LIMIT 100 "
                                ") "
                                "SELECT "
                                "   MAX(p95_prep_duration_us) AS p95_prep_duration_us_all, "
                                "   MAX(p95_post_duration_us) AS p95_post_duration_us_all, "
                                "   MAX(p95_input_size_b) AS p95_input_size_b_all, "
                                "   MAX(p95_output_size_b) AS p95_output_size_b_all "
                                "FROM recent_data;", profileTableName);
        res = pullSQL(metricsConn, query);
    }
    for (const auto& row : res) {
        profile.p95prepLat = (uint64_t) row[0].as<double>();
        profile.p95postLat = (uint64_t) row[1].as<double>();
        profile.p95InputSize = (uint32_t) row[2].as<float>();
        profile.p95OutputSize = (uint32_t) row[3].as<float>();
    }
}

/**
 * @brief Query batch inference latency
 * 
 * @param metricsConn 
 * @param tableName 
 * @param streamName 
 * @param deviceName 
 * @param modelName 
 * @param modelProfile 
 */
void queryBatchInferLatency(
    pqxx::connection &metricsConn,
    const std::string &experimentName,
    const std::string &systemName,
    const std::string &pipelineName,
    const std::string &streamName,
    const std::string &deviceName,
    const std::string &modelName,
    ModelProfile &profile
) {
    std::string schemaName = abbreviate(experimentName + "_" + systemName);
    std::string tableName = schemaName + "." + abbreviate(experimentName + "_" + pipelineName + "__" + modelName + "__" + deviceName)  + "_batch";
    std::string query = absl::StrFormat("SELECT infer_batch_size, MAX(p95_infer_duration_us) "
                            "FROM %s "
                            "WHERE timestamps >= (EXTRACT(EPOCH FROM NOW()) * 1000000 - 120 * 1000000) AND stream = '%s' "
                            "GROUP BY infer_batch_size;", tableName, streamName);
    
    pqxx::result res = pullSQL(metricsConn, query);
    if (res[0][0].is_null()) {
        std::string profileTableName = abbreviate("prof__" + modelName + "__" + deviceName) + "_batch";
        query = absl::StrFormat("SELECT infer_batch_size, MAX(p95_infer_duration_us) "
                                "FROM %s "
                                "GROUP BY infer_batch_size", profileTableName);
        res = pullSQL(metricsConn, query);
    }
    for (const auto& row : res) {
        BatchSizeType batchSize = row[0].as<BatchSizeType>();
        profile.batchInfer[batchSize].p95inferLat = row[1].as<uint64_t>() / batchSize;
    }
}

/**
 * @brief 
 * 
 * @param metricsConn 
 * @param tableName 
 * @param streamName 
 * @param deviceName 
 * @param modelName 
 * @param profile 
 */
void queryResourceRequirements(
    pqxx::connection &metricsConn,
    const std::string &deviceName,
    const std::string &modelName,
    ModelProfile &profile
) {
    std::string tableName = abbreviate("prof__" + modelName + "__" + deviceName + "_hw");
    std::string query = absl::StrFormat("SELECT batch_size, MAX(cpu_usage), MAX(mem_usage), MAX(rss_mem_usage), MAX(gpu_usage), MAX(gpu_mem_usage) "
                            "FROM %s "
                            "GROUP BY batch_size;", tableName);

    pqxx::result res = pullSQL(metricsConn, query);
    for (const auto& row : res) {
        BatchSizeType batchSize = row[0].as<BatchSizeType>();
        profile.batchInfer[batchSize].cpuUtil = row[1].as<CpuUtilType>();
        profile.batchInfer[batchSize].memUsage = row[2].as<MemUsageType>();
        profile.batchInfer[batchSize].rssMemUsage = row[3].as<MemUsageType>();
        profile.batchInfer[batchSize].gpuUtil = row[4].as<GpuUtilType>();
        profile.batchInfer[batchSize].gpuMemUsage = row[5].as<GpuMemUsageType>();
    }
}


/**
 * @brief 
 * 
 * @param experimentName 
 * @param pipelineName 
 * @param streamName 
 * @param deviceName 
 * @param modelName 
 * @return ModelProfile 
 */
ModelProfile queryModelProfile(
    pqxx::connection &metricsConn,
    const std::string &experimentName,
    const std::string &systemName,
    const std::string &pipelineName,
    const std::string &streamName,
    const std::string &deviceName,
    const std::string &modelName
) {
    ModelProfile profile;

    // // Query batch inference profilectrl_systemName;
    // std::string query = absl::StrFormat(
    //     "SELECT infer_batch_size, p95_infer_duration, cpu_usage, mem_usage, rss_mem_usage, gpu_usage, gpu_mem_usage "
    //     "FROM %s;", tableName
    // );
    // pqxx::result res = pullSQL(*ctrl_metricsServerConn, query);

    // for (const auto& row : res) {
    //     BatchSizeType batchSize = row[0].as<BatchSizeType>();
    //     profile.batchInfer[batchSize].p95inferLat = row[1].as<uint64_t>() / batchSize;
    //     profile.batchInfer[batchSize].cpuUtil = row[2].as<CpuUtilType>();
    //     profile.batchInfer[batchSize].memUsage = row[3].as<MemUsageType>();
    //     profile.batchInfer[batchSize].rssMemUsage = row[4].as<MemUsageType>();
    //     profile.batchInfer[batchSize].gpuUtil = row[5].as<GpuUtilType>();
    //     profile.batchInfer[batchSize].gpuMemUsage = row[6].as<GpuMemUsageType>();
    // }

    /**
     * @brief Query pre, post processing profile
     * 
     */
    queryPrePostLatency(metricsConn, experimentName, systemName, pipelineName, streamName, deviceName, modelName, profile);

    /**
     * @brief Query the batch inference profile
     * 
     */
    queryBatchInferLatency(metricsConn, experimentName, systemName, pipelineName, streamName, deviceName, modelName, profile);

    /**
     * @brief Query the batch resource consumptions
     * 
     */
    queryResourceRequirements(metricsConn, deviceName, modelName, profile);
    return profile;
}

void trt::to_json(nlohmann::json &j, const trt::TRTConfigs &val) {
    j["path"] = val.path;
    j["prec"] = val.precision;
    j["calibration"] = val.calibrationDataDirectoryPath;
    j["cbs"] = val.calibrationBatchSize;
    j["obs"] = val.optBatchSize;
    j["mbs"] = val.maxBatchSize;
    j["di"] = val.deviceIndex;
    j["normalize"] = val.normalize;
}

void trt::from_json(const nlohmann::json &j, trt::TRTConfigs &val) {
    j.at("path").get_to(val.path);
    j.at("prec").get_to(val.precision);
    j.at("calibration").get_to(val.calibrationDataDirectoryPath);
    j.at("cbs").get_to(val.calibrationBatchSize);
    j.at("obs").get_to(val.optBatchSize);
    j.at("mbs").get_to(val.maxBatchSize);
    j.at("di").get_to(val.deviceIndex);
    j.at("normalize").get_to(val.normalize);
}

void saveGPUAsImg(const cv::cuda::GpuMat &img, std::string name, float scale) {

    cv::Mat cpuImg;
    cv::cuda::GpuMat tempImg;
    img.convertTo(tempImg, CV_8UC3, scale);
    tempImg.download(cpuImg);
    cv::imwrite(name, cpuImg);
}

void saveCPUAsImg(const cv::Mat &img, std::string name, float scale) {
    cv::Mat cpuImg;
    img.convertTo(cpuImg, CV_8UC3, scale);
    cv::imwrite(name, img);
}

float fractionToFloat(const std::string& fraction) {
    std::istringstream iss(fraction);
    std::string numerator, denominator;

    // Extract the numerator and denominator
    std::getline(iss, numerator, '/');
    std::getline(iss, denominator);

    // Convert the numerator and denominator to float
    float num = std::stof(numerator);
    float den = std::stof(denominator);

    // Check for division by zero
    if (den == 0) {
        return 0.0f; // or any other desired value for division by zero
    }

    // Calculate and return the result
    return num / den;
}

std::string removeSubstring(const std::string& str, const std::string& substring) {
    std::string result = str;
    size_t pos = result.find(substring);

    while (pos != std::string::npos) {
        result.erase(pos, substring.length());
        pos = result.find(substring);
    }

    return result;
}


std::string timePointToEpochString(const std::chrono::system_clock::time_point& tp) {
    // Convert time_point to microseconds
    TimePrecisionType ns = std::chrono::duration_cast<TimePrecisionType>(tp.time_since_epoch());

    // Convert microseconds to string
    std::stringstream ss;
    ss << ns.count();
    return ss.str();
}

std::string replaceSubstring(const std::string& input, const std::string& toReplace, const std::string& replacement) {
    std::string result = input;
    std::size_t pos = 0;

    while ((pos = result.find(toReplace, pos)) != std::string::npos) {
        result.replace(pos, toReplace.length(), replacement);
        pos += replacement.length();
    }

    return result;
}

std::vector<std::string> splitString(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> result;
    size_t start = 0, end = str.find(delimiter);

    while (end != std::string::npos) {
        result.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }

    result.push_back(str.substr(start));
    return result;
}

/**
 * @brief Get the current timestamp in the format of a string
 * 
 * @return std::string 
 */
std::string getTimestampString() {
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char buffer[80];
    std::strftime(buffer, sizeof(buffer), "%d_%H-%M-%S", std::localtime(&now));
    std::string timestamp(buffer);
    timestamp.erase(timestamp.length() - 1); // Remove newline character
    return timestamp;
}

uint64_t getTimestamp() {
    return std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
}

void setupLogger(
    const std::string &logPath,
    const std::string &loggerName,
    uint16_t loggingMode,
    uint16_t verboseLevel,
    std::vector<spdlog::sink_ptr> &loggerSinks,
    std::shared_ptr<spdlog::logger> &logger
) {
    std::string path = logPath + "/" + loggerName + ".log";



    if (loggingMode == 0 || loggingMode == 2) {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        loggerSinks.emplace_back(console_sink);
    }
    bool auto_flush = true;
    if (loggingMode == 1 || loggingMode == 2) {
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(path, auto_flush);
        loggerSinks.emplace_back(file_sink);
    }

    logger = std::make_shared<spdlog::logger>("container_agent", loggerSinks.begin(), loggerSinks.end());
    spdlog::register_logger(logger);

    spdlog::get("container_agent")->set_pattern("[%C-%m-%d %H:%M:%S.%f] [%l] %v");
    spdlog::get("container_agent")->set_level(spdlog::level::level_enum(verboseLevel));
}


std::unique_ptr<pqxx::connection> connectToMetricsServer(MetricsServerConfigs &metricsServerConfigs, const std::string &name) {
    try {
        std::string conn_statement = absl::StrFormat(
            "host=%s port=%d user=%s password=%s dbname=%s",
            metricsServerConfigs.ip, metricsServerConfigs.port,
            metricsServerConfigs.user, metricsServerConfigs.password, metricsServerConfigs.DBName
        );
        std::unique_ptr<pqxx::connection> metricsServerConn = std::make_unique<pqxx::connection>(conn_statement);

        if (metricsServerConn->is_open()) {
            spdlog::info("{0:s} connected to database successfully: {1:s}", name, metricsServerConn->dbname());
        } else {
            spdlog::get("container_agent")->error("Metrics Server is not open.");
        }

        return metricsServerConn;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
}

pqxx::result pushSQL(pqxx::connection &conn, const std::string &sql) {

    pqxx::work session(conn);
    pqxx::result res;
    try {
        res = session.exec(sql.c_str());
        session.commit();
        return res;
    } catch (const pqxx::sql_error &e) {
        spdlog::get("container_agent")->error("{0:s} SQL Error: {1:s}", __func__, e.what());
        exit(1);
    }
}

pqxx::result pullSQL(pqxx::connection &conn, const std::string &sql) {
    pqxx::nontransaction session(conn);
    pqxx::result res;
    try {
        res = session.exec(sql.c_str());
        return res;
    } catch (const pqxx::undefined_table &e) {
        spdlog::get("container_agent")->error("{0:s} Undefined table {1:s}", __func__, e.what());
        return {};
    } catch (const pqxx::sql_error &e) {
        spdlog::get("container_agent")->error("{0:s} SQL Error: {1:s}", __func__, e.what());
        exit(1);
    }
}


bool isHypertable(pqxx::connection &conn, const std::string &tableName) {
    pqxx::work txn(conn);
    std::string query = "SELECT EXISTS (SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = '" + tableName + "');";
    pqxx::result r = txn.exec(query);
    return r[0][0].as<bool>();
}

bool tableExists(pqxx::connection &conn, const std::string &schemaName, const std::string &tableName) {
    pqxx::work txn(conn);
    std::string name = splitString(tableName, ".").back();
    std::string query = 
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = " + txn.quote(schemaName) + 
        " AND table_name = " + txn.quote(name) + ");";
    pqxx::result r = txn.exec(query);
    return r[0][0].as<bool>();
}

/**
 * @brief Abbreviate a keyphrase using a predefined map of abbreviations
 * If a word is not found in the map, only the first 4 characters of the word are accepted
 * 
 * @param keyphrase 
 * @return std::string 
 */
std::string abbreviate(const std::string &keyphrase) {
    std::vector<std::string> words = splitString(keyphrase, "_");
    std::string abbr = "";
    for (const auto &word : words) {
        try {
            abbr += keywordAbbrs.at(word);
        } catch (const std::out_of_range &e) {
            abbr += word.substr(0, 4);
        }
        if (word != words.back()) {
            abbr += "_";
        }
    }
    return abbr;
}

bool confirmIntention(const std::string &message, const std::string &magicPhrase) {

    std::cout << message << std::endl;
    std::cout << "Please enter \"" << magicPhrase << "\" to confirm, or \"exit\" to cancel: ";

    std::string userInput;

    while (true) {
        std::getline(std::cin, userInput);

        if (userInput == magicPhrase) {
            std::cout << "Correct phrase entered. Proceeding...\n";
            break;
        } else if (userInput == "exit") {
            std::cout << "Exiting...\n";
            return false;
        } else {
            std::cout << "Incorrect phrase. Please try again: ";
        }
    }

    return true;
}