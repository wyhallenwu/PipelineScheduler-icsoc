#include "data_source.h"

DataReader::DataReader(const BaseMicroserviceConfigs &configs, std::string &datapath) : SerDataMicroservice<void>(
        configs) {
    source = VideoCapture(datapath);
};

void DataReader::Schedule() {
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

DataSource::DataSource(std::string &datapath, std::vector<int32_t> dataShape, uint32_t slo) {
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


void DataSource::Run() {
    // Start the reader thread and pop according to fps

    // Start the sender thread to consume the reader output
}

DataSourceAgent::DataSourceAgent(const std::string &name, uint16_t device_port, uint16_t own_port,
                    std::vector<BaseMicroserviceConfigs> &msvc_configs) : ContainerAgent(name, device_port, own_port) {
        msvcs.push_back(reinterpret_cast<Microservice<void> *const>(new DataReader(msvc_configs[0],
                                                                                   msvc_configs[0].upstreamMicroservices.front().link[0])));
        msvcs.push_back(reinterpret_cast<Microservice<void> *const>(new LocalCPUSender(msvc_configs[1],
                                                                                       msvc_configs[1].dnstreamMicroservices.front().link[0])));
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