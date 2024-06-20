#include "data_reader.h"

inline cv::Mat resizePadRightBottom(
        const cv::Mat &input,
        const size_t height,
        const size_t width,
        const std::vector<float> &bgcolor,
        uint8_t RESIZE_INTERPOL_TYPE
) {
    spdlog::get("container_agent")->trace("Going into {0:s}", __func__);

    float r = std::min(width / (input.cols * 1.0), height / (input.rows * 1.0));
    int unpad_w = r * input.cols;
    int unpad_h = r * input.rows;
    // Create a new Mat
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(unpad_w, unpad_h), 0, 0, RESIZE_INTERPOL_TYPE);
    cv::Mat out = cv::Mat(height, width, input.type(), cv::Scalar(bgcolor[0], bgcolor[1], bgcolor[2]));
    // Copy resized image to output Mat
    resized.copyTo(out(cv::Rect(0, 0, resized.cols, resized.rows)));

    spdlog::get("container_agent")->trace("Finished {0:s}", __func__);

    return out;
}

DataReader::DataReader(const json &jsonConfigs) : Microservice(jsonConfigs) {
    loadConfigs(jsonConfigs, true);
};

void DataReader::loadConfigs(const json &jsonConfigs, bool isConstructing) {
    if (!isConstructing) {
        Microservice::loadConfigs(jsonConfigs);
    }

    link = jsonConfigs.at("msvc_upstreamMicroservices")[0].at("nb_link")[0];
    source = cv::VideoCapture(link);
    wait_time_ms = 1000 / jsonConfigs.at("msvc_idealBatchSize").get<int>();
    frame_count = (jsonConfigs.at("msvc_idealBatchSize").get<int>() == 30) ? 1 :
            30 / (30 - jsonConfigs.at("msvc_idealBatchSize").get<int>());
    link = link.substr(link.find_last_of('/') + 1);
};

void DataReader::Process() {
    int i = 1;
    while (true) {
        ClockType time = std::chrono::system_clock::now();
        cv::Mat frame;
        source >> frame;
        if (frame.empty()) {
            std::cout << "No more frames to read, exiting Video Processing." << std::endl;
            return;
        }
        if (frame_count > 1 && i++ >= frame_count) {
            i = 1;
        } else {
            // two `time`s is not necessary, but it follows the format set for the downstreams.
            int frameNum = (int) source.get(cv::CAP_PROP_POS_FRAMES);
            std::cout << "Frame Number: " << frameNum << std::endl;
            frame = resizePadRightBottom(frame, msvc_dataShape[0][1], msvc_dataShape[0][2],
                                         {128, 128, 128}, cv::INTER_AREA);
            RequestMemSizeType frameMemSize = frame.channels() * frame.rows * frame.cols * CV_ELEM_SIZE1(frame.type());
            Request<LocalCPUReqDataType> req = {{{time, time}}, {msvc_svcLevelObjLatency},
                                                {"[" + msvc_hostDevice + "|" + link + "|" +
                                                 std::to_string(frameNum) + 
                                                 "|1|1|" + std::to_string(frameMemSize)  + "]"}, 1,
                                                {RequestData<LocalCPUReqDataType>{{frame.dims, frame.rows, frame.cols},
                                                                                  frame}}};
            for (auto q: msvc_OutQueue) {
                q->emplace(req);
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(wait_time_ms));
    }
};