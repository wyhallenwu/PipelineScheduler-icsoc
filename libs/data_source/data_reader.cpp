#include "data_reader.h"

DataReader::DataReader(const json &jsonConfigs) : Microservice(jsonConfigs) {
    loadConfigs(jsonConfigs, true);
};

void DataReader::loadConfigs(const json &jsonConfigs, bool isConstructing) {
    if (!isConstructing) {
        Microservice::loadConfigs(jsonConfigs);
    }

    std::string link = jsonConfigs.at("msvc_upstreamMicroservices")[0].at("nb_link")[0];
    source = cv::VideoCapture(link);
    wait_time_ms = jsonConfigs.at("msvc_idealBatchSize");
};

void DataReader::Process() {
    while (true) {
        ClockType time = std::chrono::system_clock::now();
        cv::Mat frame;
        source >> frame;
        if (frame.empty()) {
            source.set(cv::CAP_PROP_POS_FRAMES, 0); // retry to get the frame by modifying source
            source >> frame;
            if (frame.empty()) {
                std::cout << "No more frames to read" << std::endl;
                return;
            }
        }
        Request<LocalCPUReqDataType> req = {{{time}}, {msvc_svcLevelObjLatency}, {msvc_name}, 1,
                                            {RequestData<LocalCPUReqDataType>{{frame.dims, frame.rows, frame.cols}, frame}}};
        msvc_OutQueue[0]->emplace(req);
        std::this_thread::sleep_for(std::chrono::milliseconds(wait_time_ms));
    }
};