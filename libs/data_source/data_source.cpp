#include "data_source.h"

DataSourceAgent::DataSourceAgent(
    const std::string &name,
    uint16_t own_port,
    int8_t devIndex,
    const std::string& logPath,
    std::vector<BaseMicroserviceConfigs> &msvc_configs
) : ContainerAgent(name, own_port, devIndex, logPath) {
    msvcs.push_back(reinterpret_cast<Microservice *const>(new DataReader(msvc_configs[0])));
    msvcs.push_back(reinterpret_cast<Microservice *const>(new RemoteCPUSender(msvc_configs[1])));
    msvcs[1]->SetInQueue(msvcs[0]->GetOutQueue());

    dynamic_cast<DataReader *>(msvcs[0])->dispatchThread();
    std::thread sender(&RemoteCPUSender::Process, dynamic_cast<RemoteCPUSender *>(msvcs[1]));
    sender.detach();
}

int main(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);
    std::vector<BaseMicroserviceConfigs> msvc_configs = msvcconfigs::LoadFromJson();
    std::string name = absl::GetFlag(FLAGS_name);
    std::string logPath = absl::GetFlag(FLAGS_log_dir);
    ContainerAgent *agent = new DataSourceAgent(name, absl::GetFlag(FLAGS_port), -1, logPath, msvc_configs);
    while (agent->running()) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        agent->SendState();
    }
    delete agent;
    return 0;
}