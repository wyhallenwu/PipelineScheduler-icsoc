#include "data_source.h"

DataReader::DataReader(const BaseMicroserviceConfigs &configs) : Microservice(configs), source(VideoCapture(
        configs.upstreamMicroservices.front().link[0])) {};

void DataReader::Process(int wait_time_ms) {
    while (true) {
        ClockType time = std::chrono::system_clock::now();
        Mat frame;
        source >> frame;
        if (frame.empty()) {
            source.set(CAP_PROP_POS_FRAMES, 0); // retry to get the frame by modifying source
            source >> frame;
            if (frame.empty()) {
                std::cout << "No more frames to read" << std::endl;
                return;
            }
        }
        Request<LocalCPUReqDataType> req = {time, msvc_svcLevelObjLatency, msvc_name, 1,
                                            {RequestData<LocalCPUReqDataType>{{frame.cols, frame.rows}, frame}}};
        msvc_OutQueue[0]->emplace(req);
        std::this_thread::sleep_for(std::chrono::milliseconds(wait_time_ms));
    }
};

DataSourceAgent::DataSourceAgent(const std::string &name, uint16_t own_port,
                                 std::vector<BaseMicroserviceConfigs> &msvc_configs) : ContainerAgent(name, own_port) {
    msvcs.push_back(reinterpret_cast<Microservice *const>(new DataReader(msvc_configs[0])));
    msvcs.push_back(reinterpret_cast<Microservice *const>(new RemoteCPUSender(msvc_configs[1])));
    msvcs[1]->SetInQueue(msvcs[0]->GetOutQueue());
    std::thread processor(&DataReader::Process, dynamic_cast<DataReader *>(msvcs[0]), 33); // ~30.3 fps
    processor.detach();
    std::thread sender(&RemoteCPUSender::Process, dynamic_cast<RemoteCPUSender *>(msvcs[1]));
    sender.detach();
}

int main(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);
    std::vector<BaseMicroserviceConfigs> msvc_configs = msvcconfigs::LoadFromJson();
    std::string name = absl::GetFlag(FLAGS_name);
    ContainerAgent *agent = new DataSourceAgent(name, absl::GetFlag(FLAGS_port), msvc_configs);
    while (agent->running()) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        agent->SendQueueLengths();
    }
    delete agent;
    return 0;
}