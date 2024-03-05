#include "data_source.h"

DataReader::DataReader(const BaseMicroserviceConfigs &configs, std::string &datapath) : Microservice(
        configs) {
    source = VideoCapture(datapath);
};

void DataReader::Process() {
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
    Request<LocalCPUReqDataType> req = {time, msvc_svcLevelObjLatency, msvc_name, 1,
                                         {RequestData<LocalCPUReqDataType>{{frame.cols, frame.rows}, frame}}};
    msvc_OutQueue[0]->emplace(req);
};

DataSourceAgent::DataSourceAgent(const std::string &name, uint16_t device_port, uint16_t own_port,
                    std::vector<BaseMicroserviceConfigs> &msvc_configs) : ContainerAgent(name, device_port, own_port) {
        msvcs.push_back(reinterpret_cast<Microservice* const>(new DataReader(msvc_configs[0],
                                                                                   msvc_configs[0].upstreamMicroservices.front().link[0])));
        msvcs.push_back(reinterpret_cast<Microservice* const>(new LocalCPUSender(msvc_configs[1],
                                                                                       msvc_configs[1].dnstreamMicroservices.front().link[0])));
        msvcs[1]->SetInQueue(msvcs[0]->GetOutQueue());
        std::thread processor(&DataReader::Process, dynamic_cast<DataReader*>(msvcs[0]));
        processor.detach();
        std::thread sender(&LocalCPUSender::Process, dynamic_cast<LocalCPUSender*>(msvcs[1]));
        sender.detach();
    }

int main(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);
    auto msvc_configs = json::parse(absl::GetFlag(FLAGS_json)).get<std::vector<BaseMicroserviceConfigs>>();
    std::string name = absl::GetFlag(FLAGS_name);
    ContainerAgent *agent = new DataSourceAgent(name, 2000, absl::GetFlag(FLAGS_port), msvc_configs);
    while (agent->running()) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        agent->SendQueueLengths();
    }
    delete agent;
    return 0;
}