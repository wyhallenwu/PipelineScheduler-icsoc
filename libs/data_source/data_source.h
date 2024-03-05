#ifndef PIPEPLUSPLUS_DATA_SOURCE_H
#define PIPEPLUSPLUS_DATA_SOURCE_H

#include <opencv2/opencv.hpp>
#include <list>
#include <thread>
#include "container_agent.h"
#include "microservice.h"
#include "sender.h"

using namespace cv;

class DataReader : public SerDataMicroservice<void> {
public:
    DataReader(const BaseMicroserviceConfigs &configs, std::string &datapath);

    ~DataReader() {
        source.release();
    };

    void Schedule() override;

private:
    VideoCapture source;
};

class DataSource {
public:
    DataSource(std::string &datapath, std::vector<int32_t> dataShape, uint32_t slo);

    ~DataSource() {
        delete reader;
        delete sender;
    };

    void Run();

private:
    DataReader *reader;
    LocalCPUSender *sender;
};

class DataSourceAgent : public ContainerAgent {
public:
    DataSourceAgent(const std::string &name, uint16_t device_port, uint16_t own_port,
                    std::vector<BaseMicroserviceConfigs> &msvc_configs);
};


#endif //PIPEPLUSPLUS_DATA_SOURCE_H
