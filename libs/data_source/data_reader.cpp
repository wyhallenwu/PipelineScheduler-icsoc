#include "data_reader.h"

DataReader::DataReader(const json &jsonConfigs) : Microservice(jsonConfigs)
{
    loadConfigs(jsonConfigs, true);
};

void DataReader::loadConfigs(const json &jsonConfigs, bool isConstructing)
{
    if (!isConstructing)
    {
        Microservice::loadConfigs(jsonConfigs);
    }

    link = jsonConfigs.at("msvc_upstreamMicroservices")[0].at("nb_link")[0];
    source = cv::VideoCapture(link);
    wait_time_ms = 1000 / jsonConfigs.at("msvc_idealBatchSize").get<int>();
    frame_count = (jsonConfigs.at("msvc_idealBatchSize").get<int>() == 30) ? 1 :
            30 / (30 - jsonConfigs.at("msvc_idealBatchSize").get<int>());
    link = link.substr(link.find_last_of('/') + 1);

    // init resolution for image
    width = 640;
    height = 640;
};

/**
 * @brief  for jellyfish, data reader performs image resizing. On the server side, preprocessing does nothing
 *
 */
void DataReader::Process()
{
    int i = 1;
    while (true)
    {
        ClockType time = std::chrono::system_clock::now();
        cv::Mat frame;
        source >> frame;

        if (frame.empty())
        {
            std::cout << "No more frames to read" << std::endl;
            return;
        }
        if (frame_count > 1 && i++ >= frame_count) {
            i = 1;
        } else {
            // ============================= added ==============================
            // resize the image
            cv::Mat resized_frame;
            cv::resize(frame, resized_frame, cv::Size(width, height));
            Request<LocalCPUReqDataType>
                req = {{{time}}, {msvc_svcLevelObjLatency}, {"[" + link + "_" + std::to_string((int)source.get(cv::CAP_PROP_POS_FRAMES)) + "]"}, 1, {RequestData<LocalCPUReqDataType>{{resized_frame.dims, resized_frame.rows, resized_frame.cols},
                                                                                  resized_frame}}};
            msvc_OutQueue[0]->emplace(req);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(wait_time_ms));
    }
};