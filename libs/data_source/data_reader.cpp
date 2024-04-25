#include "data_reader.h"

DataReader::DataReader(const BaseMicroserviceConfigs &configs) : Microservice(configs), source(cv::VideoCapture(
        configs.msvc_upstreamMicroservices.front().link[0])), wait_time_ms(configs.msvc_idealBatchSize) {};

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
        Request<LocalCPUReqDataType> req = {{time}, {msvc_svcLevelObjLatency}, {msvc_name}, 1,
                                            {RequestData<LocalCPUReqDataType>{{frame.cols, frame.rows}, frame}}};
        msvc_OutQueue[0]->emplace(req);
        std::this_thread::sleep_for(std::chrono::milliseconds(wait_time_ms));
    }
};