#ifndef PIPEPLUSPLUS_DATA_SOURCE_H
#define PIPEPLUSPLUS_DATA_SOURCE_H

#include <opencv2/opencv.hpp>
#include <list>
#include <thread>
#include "container_agent.h"
#include "microservice.h"
#include "sender.h"

using namespace cv;

class DataReader : public Microservice {
public:
    DataReader(const BaseMicroserviceConfigs &configs, std::string &datapath);

    ~DataReader() {
        source.release();
    };

    void Process();

private:
    VideoCapture source;
};

class DataSourceAgent : public ContainerAgent {
public:
    DataSourceAgent(const std::string &name, uint16_t device_port, uint16_t own_port,
                    std::vector<BaseMicroserviceConfigs> &msvc_configs);
};


#endif //PIPEPLUSPLUS_DATA_SOURCE_H
