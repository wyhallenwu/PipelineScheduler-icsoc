#ifndef PIPEPLUSPLUS_DATA_READER_H
#define PIPEPLUSPLUS_DATA_READER_H

#include <opencv2/opencv.hpp>
#include <thread>
#include "microservice.h"

class DataReader : public Microservice
{
public:
    // msvc_idealBatchSize is used for the wait time, where 33 equals ~30.3 fps
    // the link to the upstream microservice is used to specify the file location
    DataReader(const json &jsonConfigs);

    ~DataReader() override
    {
        source.release();
    };

    void dispatchThread() override
    {
        std::thread handler(&DataReader::Process, this);
        READY = true;
        handler.detach();
    }

    virtual void loadConfigs(const json &jsonConfigs, bool isConstructing) override;

private:
    void Process();

    std::string link;
    cv::VideoCapture source;
    int wait_time_ms;
    int frame_count;
    // image size
    int image_size_idx;
};

#endif // PIPEPLUSPLUS_DATA_READER_H
