#include "misc.h"

using json = nlohmann::json;

uint64_t calculateP95(std::vector<uint64_t> &values) {
    if (values.empty()) {
        throw std::invalid_argument("Values set is empty.");
    }
    std::sort(values.begin(), values.end());
    size_t index = static_cast<size_t>(std::ceil(0.95 * values.size())) - 1;
    return values[index];
}

NetworkEntryType aggregateNetworkEntries(const NetworkEntryType &res) {
    std::map<uint32_t, std::vector<uint64_t>> groupedEntries;

    // Group entries by package size
    for (const auto entry : res) {
        if (entry.second > 10000000) {
            continue;
        }
        groupedEntries[entry.first].push_back(entry.second);
    }

    // Calculate the p95 value for each group and create a new NetworkEntryType vector
    NetworkEntryType uniqueEntries;
    for (auto &group : groupedEntries) {
        uint64_t p95 = calculateP95(group.second);
        uniqueEntries.emplace_back(group.first, p95);
    }

    // Sort the uniqueEntries by package size
    std::sort(uniqueEntries.begin(), uniqueEntries.end(), [](const auto &a, const auto &b) {
        return a.first < b.first;
    });

    return uniqueEntries;
}

/**
 * @brief Estimate network latency of a package of size `totalPkgSize` using linear interpolation
 * 
 * @param res contains packge size and latency data of sample packages
 * @param totalPkgSize 
 * @return uint64_t 
 */
uint64_t estimateNetworkLatency(const NetworkEntryType& res, const uint32_t &totalPkgSize) {
    if (res.empty()) {
        throw std::invalid_argument("The result set is empty.");
    }

    // Handle case where there is only one entry left
    if (res.size() == 1) {
        return res[0].second; // Directly return the latency of the single entry
    }

    // Handle case where totalPkgSize is smaller than the smallest package size in res
    if (totalPkgSize <= res.front().first) {
        return res.front().second;
    }

    // Handle case where totalPkgSize is larger than the largest package size in res
    if (totalPkgSize >= res.back().first) {
        return res.back().second;
    }

    // Perform linear interpolation within the range
    for (size_t i = 0; i < res.size() - 1; ++i) {
        uint32_t pkgSize1 = res[i].first;
        uint32_t pkgSize2 = res[i + 1].first;
        uint64_t latency1 = res[i].second;
        uint64_t latency2 = res[i + 1].second;

        if (totalPkgSize >= pkgSize1 && totalPkgSize <= pkgSize2) {
            if (pkgSize1 == pkgSize2) {
                return latency1; // If sizes are the same, return latency1
            }

            return std::max(latency2, latency1);
        }
    }

    // Should not reach here if the data is consistent
    throw std::runtime_error("Failed to estimate network latency due to unexpected data range.");
}

// ================================================================== Queries functions ==================================================================
// =======================================================================================================================================================
// =======================================================================================================================================================
// =======================================================================================================================================================

float queryArrivalRate(
    pqxx::connection &metricsConn,
    const std::string &experimentName,
    const std::string &systemName,
    const std::string &pipelineName,
    const std::string &streamName,
    const std::string &taskName,
    const std::string &modelName,
    const uint16_t systemFPS,
    const std::vector<uint8_t> &periods //seconds
) {
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
                        "  WHERE stream = '%s'"
                        ") "
                        "SELECT MAX(max_rate) AS max_arrival_rate "
                        "FROM arrival_rate;";
    query = absl::StrFormat(query.c_str(), schemaName + "." + tableName, periodQuery, streamName);
    pqxx::result res = pullSQL(metricsConn, query);

    std::string modelNameAbbr = abbreviate(splitString(modelName, ".").front());

    if (res[0][0].is_null()) {
        // If there is no historical data, we look for the rate of the most recent profiled data
        std::string profileTableName = abbreviate("pf" + std::to_string(systemFPS) + "_" + taskName + "_arr");
        query = "WITH recent_data AS ("
                "   SELECT * "
                "   FROM %s "
                "   WHERE model_name = '%s' "
                "   LIMIT 10 "
                "), "
                "arrival_rate AS ("
                "  SELECT GREATEST(%s) AS max_rate "
                "  FROM recent_data "
                ") "
                "SELECT MAX(max_rate) AS max_arrival_rate "
                "FROM arrival_rate;";
        query = absl::StrFormat(query.c_str(), profileTableName, modelNameAbbr, periodQuery);
        res = pullSQL(metricsConn, query);
    }
    return res[0]["max_arrival_rate"].as<float>();
}

NetworkProfile queryNetworkProfile(
    pqxx::connection &metricsConn,
    const std::string &experimentName,
    const std::string &systemName,
    const std::string &pipelineName,
    const std::string &streamName,
    const std::string &taskName,
    const std::string &modelName,
    const std::string &senderHost,
    const std::string &senderDeviceType,
    const std::string &receiverHost,
    const std::string &receiverDeviceType,
    const NetworkEntryType &networkEntries,
    uint16_t systemFPS
) {
    std::string senderHostAbbr = abbreviate(senderHost);
    std::string receiverHostAbbr = abbreviate(receiverHost);

    std::string senderDeviceTypeAbbr = abbreviate(senderDeviceType);
    std::string receiverDeviceTypeAbbr = abbreviate(receiverDeviceType);    

    std::string schemaName = abbreviate(experimentName + "_" + systemName);
    std::string tableName = abbreviate(experimentName + "_" + pipelineName + "_" + taskName + "_arr");

    NetworkProfile d2dNetworkProfile;

    /**
     * @brief Querying for the network profile from the data in the last 120 seconds.
     * 
     */
    std::string query = "WITH recent_data AS ("
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
    pqxx::result res = pullSQL(metricsConn, query);

    // if there are most current entries, then great, we update the profile and that's that
    if (!res[0][0].is_null()) {
        d2dNetworkProfile.p95OutQueueingDuration = res[0]["p95_out_queueing_duration_us"].as<uint64_t>();
        d2dNetworkProfile.p95QueueingDuration = res[0]["p95_queuing_duration_us"].as<uint64_t>();
        d2dNetworkProfile.p95PackageSize = res[0]["p95_total_package_size_b"].as<uint32_t>();
        d2dNetworkProfile.p95TransferDuration = res[0]["p95_transfer_duration_us"].as<uint64_t>();

        return d2dNetworkProfile;
    }

    std::string modelNameAbbr = abbreviate(splitString(modelName, ".").front());

    // If there is no historical data, we look for the rate of the most recent profiled data
    std::string profileTableName = abbreviate("pf" + std::to_string(systemFPS) + "_" + taskName + "_arr");
    query = "WITH recent_data AS ("
    "   SELECT p95_out_queueing_duration_us, p95_queueing_duration_us, p95_total_package_size_b "
    "   FROM %s "
    "   WHERE receiver_host = '%s' AND model_name = '%s'"
    "   LIMIT 100"
    ") "
    "SELECT "
    "   MAX(p95_out_queueing_duration_us) AS p95_out_queueing_duration_us, "
    "   MAX(p95_queueing_duration_us) AS p95_queuing_duration_us, "
    "   MAX(p95_total_package_size_b) AS p95_total_package_size_b "
    "FROM recent_data;";
    query = absl::StrFormat(query.c_str(), profileTableName, receiverDeviceTypeAbbr, abbreviate(modelNameAbbr));
    res = pullSQL(metricsConn, query);

    d2dNetworkProfile.p95OutQueueingDuration = res[0]["p95_out_queueing_duration_us"].as<uint64_t>();
    d2dNetworkProfile.p95QueueingDuration = res[0]["p95_queuing_duration_us"].as<uint64_t>();
    d2dNetworkProfile.p95PackageSize = res[0]["p95_total_package_size_b"].as<uint32_t>();

    if ((senderHost != "server") && (senderHost == receiverHost) && (taskName.find("yolo") != std::string::npos)) {
        d2dNetworkProfile.p95PackageSize = 0;
        return d2dNetworkProfile;
    }

    // For network transfer duration, we estimate the latency using linear interpolation based on the package size
    // The network entries are updated in a separate thread
    d2dNetworkProfile.p95TransferDuration = estimateNetworkLatency(networkEntries, d2dNetworkProfile.p95PackageSize);

    return d2dNetworkProfile;
}

/**
 * @brief query the rates, network profile of a model
 * 
 * @param metricsConn 
 * @param experimentName 
 * @param systemName 
 * @param pipelineName 
 * @param streamName 
 * @param taskName 
 * @param modelName 
 * @param senderHost 
 * @param networkEntries The latest update network entries between the sender host and the receiver host
 *                                 Ideally the specific data of this task should be queried, but if thats not available,
 *                                 the latest per-device data will be used. These entries are updated in a separate thread.
 * @param receiverHost 
 * @param periods 
 * @return ModelArrivalProfile 
 */
ModelArrivalProfile queryModelArrivalProfile(
    pqxx::connection &metricsConn,
    const std::string &experimentName,
    const std::string &systemName,
    const std::string &pipelineName,
    const std::string &streamName,
    const std::string &taskName,
    const std::string &modelName,
    const std::vector<std::pair<std::string, std::string>> &commPairs,
    const std::map<std::pair<std::string, std::string>, NetworkEntryType> &networkEntries,
    const uint16_t systemFPS,
    const std::vector<uint8_t> &periods //seconds
) {
    ModelArrivalProfile arrivalProfile;

    std::string schemaName = abbreviate(experimentName + "_" + systemName);
    std::string tableName = abbreviate(experimentName + "_" + pipelineName + "_" + taskName + "_arr");

    std::string modelNameAbbr = abbreviate(splitString(modelName, ".").front());

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
        std::string profileTableName = abbreviate("pf" + std::to_string(systemFPS) + "_" + taskName + "_arr");
        query = "WITH recent_data AS ("
                "   SELECT * "
                "   FROM %s "
                "   WHERE model_name = '%s' "
                "   LIMIT 10 "
                "), "
                "arrival_rate AS ("
                "  SELECT GREATEST(%s) AS max_rate "
                "  FROM recent_data "
                ") "
                "SELECT MAX(max_rate) AS max_arrival_rate "
                "FROM arrival_rate;";
        query = absl::StrFormat(query.c_str(), profileTableName, modelNameAbbr, periodQuery);
        res = pullSQL(metricsConn, query);
    }
    arrivalProfile.arrivalRates = res[0]["max_arrival_rate"].as<float>();

    for (const auto &commPair : commPairs) {

        std::string senderHostAbbr = abbreviate(commPair.first);
        std::string receiverHostAbbr = abbreviate(commPair.second);

        NetworkProfile *d2dNetworkProfile = &(arrivalProfile.d2dNetworkProfile[std::make_pair(senderHostAbbr, receiverHostAbbr)]);

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
                "   percentile_disc(0.95) WITHIN GROUP (ORDER BY p95_out_queueing_duration_us) AS p95_out_queueing_duration_us, "
                "   percentile_disc(0.95) WITHIN GROUP (ORDER BY p95_transfer_duration_us) AS p95_transfer_duration_us, "
                "   percentile_disc(0.95) WITHIN GROUP (ORDER BY p95_queueing_duration_us) AS p95_queueing_duration_us, "
                "   percentile_disc(0.95) WITHIN GROUP (ORDER BY p95_total_package_size_b) AS p95_total_package_size_b "
                "FROM recent_data;";
        query = absl::StrFormat(query.c_str(), schemaName + "." + tableName, streamName, senderHostAbbr, receiverHostAbbr);
        res = pullSQL(metricsConn, query);

        // if there are most current entries, then great, we update the profile and that's that
        if (!res[0][0].is_null()) {
            d2dNetworkProfile->p95OutQueueingDuration = res[0]["p95_out_queueing_duration_us"].as<uint64_t>();
            d2dNetworkProfile->p95QueueingDuration = res[0]["p95_queuing_duration_us"].as<uint64_t>();
            d2dNetworkProfile->p95PackageSize = res[0]["p95_total_package_size_b"].as<uint32_t>();
            d2dNetworkProfile->p95TransferDuration = res[0]["p95_transfer_duration_us"].as<uint64_t>();

            continue;
        }

        // If there is no historical data, we look for the rate of the most recent profiled data
        std::string profileTableName = abbreviate("pf" + std::to_string(systemFPS) + "_" + taskName + "_arr");
        query = "WITH recent_data AS ("
        "   SELECT p95_out_queueing_duration_us, p95_queueing_duration_us, p95_total_package_size_b "
        "   FROM %s "
        "   WHERE receiver_host = '%s' AND model_name = '%s'"
        "   LIMIT 100"
        ") "
        "SELECT "
        "   percentile_disc(0.95) WITHIN GROUP (ORDER BY p95_out_queueing_duration_us) AS p95_out_queueing_duration_us, "
        "   percentile_disc(0.95) WITHIN GROUP (ORDER BY p95_queueing_duration_us) AS p95_queueing_duration_us, "
        "   percentile_disc(0.95) WITHIN GROUP (ORDER BY p95_total_package_size_b) AS p95_total_package_size_b "
        "FROM recent_data;";
        query = absl::StrFormat(query.c_str(), profileTableName, receiverHostAbbr, modelNameAbbr);
        res = pullSQL(metricsConn, query);

        d2dNetworkProfile->p95OutQueueingDuration = res[0]["p95_out_queueing_duration_us"].as<uint64_t>();
        d2dNetworkProfile->p95QueueingDuration = res[0]["p95_queuing_duration_us"].as<uint64_t>();
        d2dNetworkProfile->p95PackageSize = res[0]["p95_total_package_size_b"].as<uint32_t>();

        // For network transfer duration, we estimate the latency using linear interpolation based on the package size
        // The network entries are updated in a separate thread
        d2dNetworkProfile->p95TransferDuration = estimateNetworkLatency(networkEntries.at(commPair), d2dNetworkProfile->p95PackageSize);
    }
    return arrivalProfile;
}

/**
 * @brief Query pre, post processing latency as well as input and output sizes
 * 
 * @param metricsConn 
 * @param experimentName 
 * @param systemName 
 * @param pipelineName 
 * @param streamName 
 * @param deviceName 
 * @param modelName 
 * @param profile this will be updated
 */
void queryPrePostLatency(
    pqxx::connection &metricsConn,
    const std::string &experimentName,
    const std::string &systemName,
    const std::string &pipelineName,
    const std::string &streamName,
    const std::string &deviceName,
    const std::string &deviceTypeName,
    const std::string &modelName,
    ModelProfile &profile,
    const uint16_t systemFPS
) {
    std::string schemaName = abbreviate(experimentName + "_" + systemName);
    std::string modelNameAbbr = abbreviate(splitString(modelName, ".").front()); 
    std::string tableName = schemaName + "." + abbreviate(experimentName + "_" + pipelineName + "__" + modelNameAbbr + "__" + deviceName + "_proc");
    std::string query = absl::StrFormat(
        "WITH recent_data AS ("
        "    SELECT infer_batch_size, p95_prep_duration_us, p95_infer_duration_us, p95_post_duration_us, p95_input_size_b, p95_output_size_b "
        "    FROM %s "
        "    WHERE timestamps >= (EXTRACT(EPOCH FROM NOW()) * 1000000 - 120 * 1000000) AND stream = '%s' "
        ") "
        "SELECT "
        "    infer_batch_size, "
        "    COUNT (*) AS entry_count,"
        "    percentile_disc(0.95) WITHIN GROUP (ORDER BY p95_prep_duration_us) AS p95_prep_duration_us_all, "
        "    percentile_disc(0.95) WITHIN GROUP (ORDER BY p95_infer_duration_us) AS p95_infer_duration_us_all, "
        "    percentile_disc(0.95) WITHIN GROUP (ORDER BY p95_post_duration_us) AS p95_post_duration_us_all, "
        "    percentile_disc(0.95) WITHIN GROUP (ORDER BY p95_input_size_b) AS p95_input_size_b_all, "
        "    percentile_disc(0.95) WITHIN GROUP (ORDER BY p95_output_size_b) AS p95_output_size_b_all "
        "FROM recent_data "
        "GROUP BY infer_batch_size;", 
        tableName, streamName
    );

    pqxx::result res = pullSQL(metricsConn, query);
    std::vector<BatchSizeType> retrievedBatchSizes;
    for (const auto& row : res) {
        if (row["entry_count"].as<uint16_t>() < 3 && row[0].as<uint16_t>() > 0){
            continue;
        }
        BatchSizeType batchSize = row["infer_batch_size"].as<BatchSizeType>();
        retrievedBatchSizes.push_back(batchSize);
        profile.batchInfer[batchSize].p95prepLat = (uint64_t) row["p95_prep_duration_us_all"].as<double>();
        profile.batchInfer[batchSize].p95inferLat = (uint64_t) row["p95_infer_duration_us_all"].as<double>();
        profile.batchInfer[batchSize].p95postLat = (uint64_t) row["p95_post_duration_us_all"].as<double>();
        profile.p95InputSize = (uint32_t) row["p95_input_size_b_all"].as<float>();
        profile.p95OutputSize = (uint32_t) row["p95_output_size_b_all"].as<float>();
    }

    // If most current historical data is not available for some batch sizes not specified in retrievedBatchSizes, we query profiled data
    std::string profileTableName = abbreviate("pf" + std::to_string(systemFPS) + "__" + modelNameAbbr +  "__" + deviceTypeName + "_proc");
    query = absl::StrFormat("WITH recent_data AS ("
                            "SELECT infer_batch_size, p95_prep_duration_us, p95_infer_duration_us, p95_post_duration_us, p95_input_size_b, p95_output_size_b "
                            "FROM %s "
                            "LIMIT 100 "
                            ") "
                            "SELECT "
                            "    infer_batch_size, "
                            "    percentile_disc(0.95) WITHIN GROUP (ORDER BY p95_prep_duration_us) AS p95_prep_duration_us_all, "
                            "    percentile_disc(0.95) WITHIN GROUP (ORDER BY p95_infer_duration_us) AS p95_infer_duration_us_all, "
                            "    percentile_disc(0.95) WITHIN GROUP (ORDER BY p95_post_duration_us) AS p95_post_duration_us_all, "
                            "    percentile_disc(0.95) WITHIN GROUP (ORDER BY p95_input_size_b) AS p95_input_size_b_all, "
                            "    percentile_disc(0.95) WITHIN GROUP (ORDER BY p95_output_size_b) AS p95_output_size_b_all "
                            "FROM recent_data;", profileTableName);
    res = pullSQL(metricsConn, query);
    for (const auto& row : res) {
        BatchSizeType batchSize = row["infer_batch_size"].as<BatchSizeType>();
        if (std::find(retrievedBatchSizes.begin(), retrievedBatchSizes.end(), batchSize) != retrievedBatchSizes.end()) {
            continue;
        }
        profile.batchInfer[batchSize].p95prepLat = (uint64_t) row["p95_prep_duration_us_all"].as<double>();
        profile.batchInfer[batchSize].p95inferLat = (uint64_t) row["p95_infer_duration_us_all"].as<double>();
        profile.batchInfer[batchSize].p95postLat = (uint64_t) row["p95_post_duration_us_all"].as<double>();
        profile.p95InputSize = (uint32_t) row["p95_input_size_b_all"].as<float>();
        profile.p95OutputSize = (uint32_t) row["p95_output_size_b_all"].as<float>();
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
    const std::string &deviceTypeName,
    const std::string &modelName,
    ModelProfile &profile,
    const uint16_t systemFPS
) {
    BatchInferProfileListType batchInferProfile;
    std::string modelNameAbbr = abbreviate(splitString(modelName, ".").front());
    std::string schemaName = abbreviate(experimentName + "_" + systemName);
    std::string tableName = schemaName + "." + abbreviate(experimentName + "_" + pipelineName + "__" + modelNameAbbr + "__" + deviceName)  + "_batch";
    std::string query = absl::StrFormat("SELECT infer_batch_size, percentile_disc(0.95) WITHIN GROUP (ORDER BY p95_infer_duration_us) AS p95_infer_duration_us "
                            "FROM %s "
                            "WHERE timestamps >= (EXTRACT(EPOCH FROM NOW()) * 1000000 - 120 * 1000000) AND stream = '%s' "
                            "GROUP BY infer_batch_size;", tableName, streamName);

    pqxx::result res = pullSQL(metricsConn, query);
    if (res[0][0].is_null()) {
        std::string profileTableName = abbreviate("pf" + std::to_string(systemFPS) + "__" + modelNameAbbr + "__" + deviceTypeName) + "_batch";
        query = absl::StrFormat("SELECT infer_batch_size, percentile_disc(0.95) WITHIN GROUP (ORDER BY p95_infer_duration_us) AS p95_infer_duration_us "
                                "FROM %s "
                                "GROUP BY infer_batch_size", profileTableName);
        res = pullSQL(metricsConn, query);
    }
    for (const auto& row : res) {
        BatchSizeType batchSize = row[0].as<BatchSizeType>();
        batchInferProfile[batchSize].p95inferLat = row[1].as<uint64_t>() / batchSize;
    }
    profile.batchInfer = batchInferProfile;
}

BatchInferProfileListType queryBatchInferLatency(
    pqxx::connection &metricsConn,
    const std::string &experimentName,
    const std::string &systemName,
    const std::string &pipelineName,
    const std::string &streamName,
    const std::string &deviceName,
    const std::string &deviceTypeName,
    const std::string &modelName,
    const uint16_t systemFPS
) {
    ModelProfile modelProfile;
    queryBatchInferLatency(
        metricsConn,
        experimentName,
        systemName,
        pipelineName,
        streamName,
        deviceName,
        deviceTypeName,
        modelName,
        modelProfile
    );
    return modelProfile.batchInfer;
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
    const std::string &deviceTypeName,
    const std::string &modelName,
    ModelProfile &profile,
    const uint16_t systemFPS
) {
    std::string modelNameAbbr = abbreviate(splitString(modelName, ".").front());
    std::string tableName = abbreviate("pf" + std::to_string(systemFPS) + "__" + modelNameAbbr + "__" + deviceTypeName + "_hw");
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
    const std::string &deviceTypeName,
    const std::string &modelName,
    uint16_t systemFPS
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
    queryPrePostLatency(metricsConn, experimentName, systemName, pipelineName, streamName, deviceName, deviceTypeName, modelName, profile);

    /**
     * @brief Query the batch inference profile
     * 
     */
    queryBatchInferLatency(metricsConn, experimentName, systemName, pipelineName, streamName, deviceName, deviceTypeName, modelName, profile);

    /**
     * @brief Query the batch resource consumptions
     * 
     */
    queryResourceRequirements(metricsConn, deviceTypeName, modelName, profile);
    return profile;
}

// =======================================================================================================================================================
// =======================================================================================================================================================
// =======================================================================================================================================================
// =======================================================================================================================================================

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
    size_t start = 0;
    size_t end = str.find_first_of(delimiter, start);

    while (end != std::string::npos) {
        result.push_back(str.substr(start, end - start));
        start = end + 1;
        end = str.find_first_of(delimiter, start);
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
        return nullptr;
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
std::string abbreviate(const std::string &keyphrase, const std::string delimiter) {
    std::vector<std::string> words = splitString(keyphrase, delimiter);
    std::string abbr = "";
    for (const auto &word : words) {
        if (word.find("-") != std::string::npos) {
            abbr += abbreviate(word, "-");
            if (word != words.back()) {
                abbr += delimiter;
            }
            continue;
        }
        try {
            abbr += keywordAbbrs.at(word);
        } catch (const std::out_of_range &e) {
            abbr += word.substr(0, 4);
        }
        if (word != words.back()) {
            abbr += delimiter;
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

std::map<std::string, std::string> keywordAbbrs = {
    {"batch", "batch"},
    {"server", "serv"},
    {"agxavier", "agx"},
    {"agxavier1", "agx1"},
    {"orinano", "orn"},
    {"orinano1", "orn1"},
    {"orinano2", "orn2"},
    {"orinano3", "orn3"},
    {"nxavier", "nx"},
    {"nxavier1", "nx1"},
    {"nxxavier", "nx2"},
    {"nxavier3", "nx3"},
    {"nxavier4", "nx4"},
    {"nxavier5", "nx5"},
    {"datasource", "dsrc"},
    {"traffic", "trfc"},
    {"building", "bldg"},
    {"yolov5", "y5"},
    {"yolov5n", "y5n"},
    {"yolov5s", "y5s"},
    {"yolov5m", "y5m"},
    {"yolov5l", "y5l"},
    {"yolov5x", "y5x"},
    {"yolov5ndsrc", "y5nd"},
    {"retina1face", "rt1f"},
    {"retina1facedsrc", "rt1fd"},
    {"age", "age"},
    {"arcface", "arcf"},
    {"carbrand", "cbrd"},
    {"gender", "gndr"},
    {"emotion", "emtn"},
    {"emotionnet", "emtn"},
    {"platedet", "pldt"},
    {"dynamic", "dyn"},
    {"movenet", "move"},
    {"3090", "39"}, // GPU name
    {"fp32", "32"},
    {"fp16", "16"},
    {"int8", "8"}
};

std::map<SystemDeviceType, std::string> SystemDeviceTypeList = {
    {Server, "server"},
    {NXXavier, "nxavier"},
    {AGXXavier, "agxavier"},
    {OrinNano, "orinano"}
};

// Reverse map for SystemDeviceTypeList
std::map<std::string, SystemDeviceType> SystemDeviceTypeReverseList = {
    {"server", Server},
    {"serv", Server},
    {"nxavier", NXXavier},
    {"nx", NXXavier},
    {"agxavier", AGXXavier},
    {"agx", AGXXavier},
    {"orinano", OrinNano},
    {"orn", OrinNano}
};

std::map<ModelType, std::string> ModelTypeList = {
    {DataSource, "datasource"},
    {Sink, "sink"},
    {Yolov5n, "yolov5n"},
    {Yolov5s, "yolov5s"},
    {Yolov5m, "yolov5m"},
    {Yolov5nDsrc, "yolov5ndsrc"},
    {Arcface, "arcface"},
    {Retinaface, "retina1face"},
    {RetinafaceDsrc, "retina1facedsrc"},
    {PlateDet, "platedet"},
    {Movenet, "movenet"},
    {Emotionnet, "emotionnet"},
    {Gender, "gender"},
    {Age, "age"},
    {CarBrand, "carbrand"}
};

std::map<std::string, ModelType> ModelTypeReverseList = {
    {"datasource", DataSource},
    {"dsrc", DataSource},
    {"sink", Sink},
    {"yolov5n", Yolov5n},
    {"y5n", Yolov5n},
    {"yolov5s", Yolov5s},
    {"y5s", Yolov5s},
    {"yolov5m", Yolov5m},
    {"y5m", Yolov5m},
    {"yolov5ndsrc", Yolov5nDsrc},
    {"y5nd", Yolov5nDsrc},
    {"arcface", Arcface},
    {"arcf", Arcface},
    {"retina1face", Retinaface},
    {"rt1f", Retinaface},
    {"retina1facedsrc", RetinafaceDsrc},
    {"rt1fd", RetinafaceDsrc},
    {"plateDet", PlateDet},
    {"pldt", PlateDet},
    {"emotionnet", Emotionnet},
    {"emtn", Emotionnet},
    {"yolov5ndsrc", Yolov5nDsrc},
    {"y5nd", Yolov5nDsrc},
    {"arcface", Arcface},
    {"arcf", Arcface},
    {"platedet", PlateDet},
    {"pldt", PlateDet},
    {"movenet", Movenet},
    {"move", Movenet},
    {"gender", Gender},
    {"gndr", Gender},
    {"age", Age},
    {"carbrand", CarBrand},
    {"cbrd", CarBrand}  
};

bool isFileEmpty(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return true; // Treat as empty if the file cannot be opened
    }

    std::streamsize size = file.tellg();
    file.close();

    return size == 0;
}

std::string getContainerName(const std::string& deviceTypeName, const std::string& modelName) {
    return modelName + "-" + deviceTypeName;
}

std::string getContainerName(const SystemDeviceType& deviceType, const ModelType& modelType) {
    std::string deviceAbbr = SystemDeviceTypeList.at(deviceType);
    std::string modelAbbr = ModelTypeList.at(modelType);
    return modelAbbr + "-" + deviceAbbr;
}

std::string getDeviceTypeAbbr(const SystemDeviceType &deviceType) {
    return abbreviate(SystemDeviceTypeList.at(deviceType));
}

ContainerLibType getContainerLib(const std::string& deviceType) {
    ContainerLibType containerLib;
    std::ifstream file("../jsons/container_lib.json");
    json j = json::parse(file);
    file.close();
    for (const auto item : j.items()) {
        std::string containerName = item.key();
        if (containerName.find(deviceType) == std::string::npos && deviceType != "all") {
            continue;
        }
        try {
            containerLib[containerName].taskName = j[containerName]["taskName"];
            std::string templatePath = j[containerName]["templateConfigPath"].get<std::string>();
            if (!templatePath.empty() && !isFileEmpty(templatePath)) {
                std::ifstream file = std::ifstream(templatePath);
                containerLib[containerName].templateConfig = json::parse(file);
            } else {
                spdlog::get("container_agent")->error("Template config file for {0:s} is empty or does not exist.", containerName);
            }
            containerLib[containerName].runCommand = j[containerName]["runCommand"];
            containerLib[containerName].modelPath = j[containerName]["modelPath"];
            containerLib[containerName].modelName = splitString(containerLib[containerName].modelPath, "/").back();
        } catch (json::exception &e) {
            spdlog::get("container_agent")->error("Error parsing template config file for {0:s}: {1:}", containerName, e.what());
            containerLib.erase(containerName);
        }
    }
    spdlog::get("container_agent")->info("Container Library Loaded");
    return containerLib;
}

std::string getDeviceTypeName(SystemDeviceType deviceType) {
    return SystemDeviceTypeList[deviceType];
}