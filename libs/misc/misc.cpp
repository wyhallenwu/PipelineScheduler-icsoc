#include "misc.h"

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
    // Convert time_point to nanoseconds
    TimePrecisionType ns = std::chrono::duration_cast<TimePrecisionType>(tp.time_since_epoch());

    // Convert nanoseconds to string
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
            spdlog::get("container_agent")->info("{0:s} connected to database successfully: {1:s}", name, metricsServerConn->dbname());
        } else {
            spdlog::get("container_agent")->error("Metrics Server is not open.");
        }

        return metricsServerConn;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
}

void executeSQL(pqxx::connection &conn, const std::string &sql) {
    pqxx::work session(conn);
    try {
        session.exec(sql.c_str());
        session.commit();
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

bool tableExists(pqxx::connection &conn, const std::string &tableName) {
    pqxx::work txn(conn);
    std::string query = "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '" + tableName + "');";
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