#ifndef MISC_H
#define MISC_H

#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <iostream>
#include "../json/json.h"
#include "spdlog/spdlog.h"
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "opencv2/opencv.hpp"
#include <unordered_set>
#include <pqxx/pqxx>
#include "absl/strings/str_format.h"
#include "absl/flags/parse.h"
#include "absl/flags/flag.h"
#include <fstream>


typedef uint16_t NumQueuesType;
typedef uint16_t QueueLengthType;
typedef uint32_t MsvcSLOType;
typedef std::vector<uint32_t> RequestSLOType;
typedef std::vector<std::string> RequestPathType;
typedef uint16_t NumMscvType;
typedef std::chrono::high_resolution_clock::time_point ClockType;
typedef std::vector<ClockType> RequestTimeType;
typedef std::vector<RequestTimeType> BatchTimeType;
const uint8_t CUDA_IPC_HANDLE_LENGTH = 64; // bytes
typedef const char *InterConGPUReqDataType;
typedef std::vector<int32_t> RequestDataShapeType;
typedef std::vector<std::vector<int32_t>> RequestShapeType;
typedef cv::cuda::GpuMat LocalGPUReqDataType;
typedef cv::Mat LocalCPUReqDataType;
typedef uint16_t BatchSizeType;
typedef uint32_t RequestMemSizeType;

// Hw Metrics
typedef int CpuUtilType;
typedef unsigned int GpuUtilType;
typedef int MemUsageType;
typedef unsigned int GpuMemUsageType;

struct BatchInferProfile {
    uint64_t p95inferLat;
    
    CpuUtilType cpuUtil;
    MemUsageType memUsage;
    MemUsageType rssMemUsage;
    GpuUtilType gpuUtil;
    GpuMemUsageType gpuMemUsage;
};

typedef std::map<BatchSizeType, BatchInferProfile> BatchInferProfileListType;

struct ArrivalRecord {
    ClockType prevPostProcTime;
    ClockType prevSenderTime;
    ClockType arrivalTime;
    uint32_t rpcBatchSize;
    uint32_t totalPkgSize;
    uint32_t reqSize;
    uint32_t reqNum;
    std::string reqOriginStream;
    std::string originDevice;
};
typedef std::vector<ArrivalRecord> ArrivalRecordType;

struct ProcessRecord {
    ClockType preStartTime;
    ClockType preEndTime;
    ClockType batchingEndTime;
    ClockType batchInferenceTime;
    ClockType postStartTime;
    ClockType postEndTime;
    uint32_t inferBatchSize;
    uint32_t inputSize;
    uint32_t outputSize;
    uint32_t reqNum;
    std::string reqOriginStream;
};
typedef std::vector<ProcessRecord> ProcessRecordType;
typedef std::chrono::microseconds TimePrecisionType;

const std::unordered_set<uint16_t> GRAYSCALE_CONVERSION_CODES = {6, 7, 10, 11};


void saveGPUAsImg(const cv::cuda::GpuMat &img, std::string name = "test.jpg", float scale = 1.f);

void saveCPUAsImg(const cv::Mat &img, std::string name = "test.jpg", float scale = 1.f);

const std::vector<std::string> cocoClassNames = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
};

const std::map<std::string, std::string> keywordAbbrs {
    {"server", "serv"},
    {"agx-xavier1", "agx1"},
    {"orinano1", "orn1"},
    {"orinano2", "orn2"},
    {"orinano3", "orn3"},
    {"xavier-nx1", "xnx1"},
    {"xavier-nx2", "xnx2"},
    {"xavier-nx3", "xnx3"},
    {"xavier-nx4", "xnx4"},
    {"xavier-nx5", "xnx5"},
    {"datasource", "dsrc"},
    {"traffic", "trfc"},
    {"building", "bldg"},
    {"yolov5", "y5"},
    {"yolov5n", "y5n"},
    {"yolov5s", "y5s"},
    {"yolov5m", "y5m"},
    {"yolov5l", "y5l"},
    {"yolov5x", "y5x"},
    {"retina1face", "rt1f"},
    {"arcface", "arcf"},
    {"carbrand", "cbrd"},
    {"gender", "gndr"},
    {"emotion", "emtn"},
    {"platedet", "pldt"},
    {"dynamic", "dyn"},
    {"3090", "39"}, // GPU name
    {"fp32", "32"},
    {"fp16", "16"},
    {"int8", "8"}
};

struct MetricsServerConfigs {
    std::string ip = "localhost";
    uint64_t port = 60004;
    std::string DBName = "pipeline";
    std::string schema = "public";
    std::string user = "container_agent";
    std::string password = "pipe";
    uint64_t hwMetricsScrapeIntervalMillisec = 50;
    uint64_t metricsReportIntervalMillisec = 60000;
    ClockType nextHwMetricsScrapeTime;
    ClockType nextMetricsReportTime;

    MetricsServerConfigs(const std::string &path) {
        std::ifstream file(path);
        nlohmann::json j = nlohmann::json::parse(file);
        from_json(j);

        nextHwMetricsScrapeTime = std::chrono::high_resolution_clock::now() + std::chrono::milliseconds(4 * hwMetricsScrapeIntervalMillisec);
        nextMetricsReportTime = std::chrono::high_resolution_clock::now() + std::chrono::milliseconds(metricsReportIntervalMillisec);

    }

    MetricsServerConfigs() = default;

    void from_json(const nlohmann::json &j) {
        j.at("metricsServer_ip").get_to(ip);
        j.at("metricsServer_port").get_to(port);
        j.at("metricsServer_DBName").get_to(DBName);
        j.at("metricsServer_user").get_to(user);
        j.at("metricsServer_password").get_to(password);
        j.at("metricsServer_hwMetricsScrapeIntervalMillisec").get_to(hwMetricsScrapeIntervalMillisec);
        j.at("metricsServer_metricsReportIntervalMillisec").get_to(metricsReportIntervalMillisec);
    }
};

std::unique_ptr<pqxx::connection> connectToMetricsServer(MetricsServerConfigs &metricsServerConfigs, const std::string &name);

enum MODEL_DATA_TYPE {
    int8 = sizeof(uint8_t),
    fp16 = int(sizeof(float) / 2),
    fp32 = sizeof(float)
};

inline void checkCudaErrorCode(cudaError_t code, std::string func_name) {
    if (code != 0) {
        std::string errMsg = "At " + func_name + "CUDA operation failed with code: " + std::to_string(code) + "(" + cudaGetErrorName(code) +
                             "), with message: " + cudaGetErrorString(code);
        std::cout << errMsg << std::endl;
        throw std::runtime_error(errMsg);
    }
}

namespace trt {
    // TRTConfigs for the network
    struct TRTConfigs {
        // Path to the engine or onnx file
        std::string path = "";
        // Path to save the engine in the case of conversion from onnx
        std::string storePath = "";
        // Precision to use for GPU inference.
        MODEL_DATA_TYPE precision = MODEL_DATA_TYPE::fp32;
        // If INT8 precision is selected, must provide path to calibration dataset directory.
        std::string calibrationDataDirectoryPath;
        // The batch size to be used when computing calibration data for INT8 inference.
        // Should be set to as large a batch number as your GPU will support.
        int32_t calibrationBatchSize = 128;
        // The batch size which should be optimized for.
        int32_t optBatchSize = 1;
        // Maximum batch size  we want to use for inference
        // this will be compared with the maximum batch size set when the engine model was created min(maxBatchSize, engine_max)
        // This determines the GPU memory buffer sizes allocated upon model loading so CANNOT BE CHANGE DURING RUNTIME.
        int32_t maxBatchSize = 128;
        // GPU device index
        int8_t deviceIndex = 0;

        size_t maxWorkspaceSize = 1 << 30;
        bool normalize = false;
        std::array<float, 3> subVals{0.f, 0.f, 0.f};
        std::array<float, 3> divVals{1.f, 1.f, 1.f};
    };

    void to_json(nlohmann::json &j, const TRTConfigs &val);

    void from_json(const nlohmann::json &j, TRTConfigs &val);
}

class Stopwatch {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point stop_time;
    bool running;

public:
    Stopwatch() : running(false) {}

    void start() {
        if (!running) {
            start_time = std::chrono::high_resolution_clock::now();
            running = true;
        }
    }

    void stop() {
        if (running) {
            stop_time = std::chrono::high_resolution_clock::now();
            running = false;
        }
    }

    void reset() {
        running = false;
    }

    uint64_t elapsed_seconds() const {
        if (running) {
            return std::chrono::duration_cast<TimePrecisionType>(std::chrono::high_resolution_clock::now() - start_time).count();
        } else {
            return std::chrono::duration_cast<TimePrecisionType>(stop_time - start_time).count();
        }
    }

    ClockType getStartTime() {
        return start_time;
    }
};

void setupLogger(
    const std::string &logPath,
    const std::string &loggerName,
    uint16_t loggingMode,
    uint16_t verboseLevel,
    std::vector<spdlog::sink_ptr> &loggerSinks,
    std::shared_ptr<spdlog::logger> &logger
);

float fractionToFloat(const std::string& fraction);

std::string removeSubstring(const std::string& str, const std::string& substring);

std::string timePointToEpochString(const std::chrono::system_clock::time_point& tp);

std::string replaceSubstring(const std::string& input, const std::string& toReplace, const std::string& replacement);

std::vector<std::string> splitString(const std::string& str, const std::string& delimiter) ;

std::string getTimestampString();

uint64_t getTimestamp();

pqxx::result pushSQL(pqxx::connection &conn, const std::string &sql);

pqxx::result pullSQL(pqxx::connection &conn, const std::string &sql);

bool isHypertable(pqxx::connection &conn, const std::string &tableName);

bool tableExists(pqxx::connection &conn, const std::string &schemaName, const std::string &tableName);

std::string abbreviate(const std::string &keyphrase);

#endif