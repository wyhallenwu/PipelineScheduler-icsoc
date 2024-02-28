#include <opencv2/opencv.hpp>
#include <list>
#include <thread>

#include "microservice.h"

#ifndef COMMUNICATOR_H

#include "../communicator/sender.cpp"

#define COMMUNICATOR_H
#endif

using namespace cv;

class DataReader : public SerDataMicroservice<void> {
public:
    DataReader(const BaseMicroserviceConfigs &configs, std::string &datapath) : SerDataMicroservice<void>(configs) {
        source = VideoCapture(datapath);
    };

    ~DataReader() {
        source.release();
    };

    void Schedule() override {
        int64_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        Mat frame;
        source >> frame;
        if (frame.empty()) {
            source.set(CAP_PROP_POS_FRAMES, 0);
            source >> frame;
            if (frame.empty()) {
                std::cout << "No more frames to read" << std::endl;
                return;
            }
        }
        DataRequest<LocalCPUDataType> req = {time, msvc_svcLevelObjLatency, msvc_name, 1,
                                             {Data<LocalCPUDataType>{{frame.cols, frame.rows}, frame}}};
        OutQueue->emplace(req);
    };

private:
    VideoCapture source;
};

class DataSource {
public:
    DataSource(std::string &datapath, std::vector<int32_t> dataShape, uint32_t slo) {
        NeighborMicroserviceConfigs neighbor_reader_configs = {"cam1::source", CommMethod::localQueue, {""},
                                                               QueueType::cpuDataQueue, 30, -2, {dataShape}};
        NeighborMicroserviceConfigs neighbor_sender_configs = {"cam1::sender", CommMethod::localQueue, {""},
                                                               QueueType::cpuDataQueue, 30, -1, {dataShape}};
        BaseMicroserviceConfigs reader_configs = {"cam1::source", MicroserviceType::Postprocessor, slo, 1, {dataShape},
                                                  std::list<NeighborMicroserviceConfigs>(), {neighbor_sender_configs}};
        BaseMicroserviceConfigs sender_configs = {"cam1::sender", MicroserviceType::Sender, slo, 1, {dataShape},
                                                  {neighbor_reader_configs}, std::list<NeighborMicroserviceConfigs>()};
        reader = new DataReader(reader_configs, datapath);
        std::string connection = "localhost:50000";
        sender = new LocalCPUSender(sender_configs, connection);
    };

    ~DataSource() {
        delete reader;
        delete sender;
    };

    void Run() {
        // Start the reader thread and pop according to fps

        // Start the sender thread to consume the reader output
    }

private:
    DataReader *reader;
    LocalCPUSender *sender;
};