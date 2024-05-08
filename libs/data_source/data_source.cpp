#include "data_source.h"

// DataSourceAgent::DataSourceAgent(
//         const std::string &name,
//         uint16_t own_port,
//         int8_t devIndex,
//         std::string logPath,
//         RUNMODE runmode,
//         const json &profiling_configs,
//         const json &cont_configs
// ) : ContainerAgent(name, own_port, devIndex, logPath, runmode, profiling_configs) {

// }

DataSourceAgent::DataSourceAgent(
    const json &configs
) : ContainerAgent(configs) {

    json pipeConfigs = configs["container"]["cont_pipeline"];
    msvcs.push_back(new DataReader(pipeConfigs[0]));
    msvcs.push_back(new RemoteCPUSender(pipeConfigs[1]));
    msvcs[1]->SetInQueue(msvcs[0]->GetOutQueue());

    for (auto &msvc : msvcs) {
        msvc->dispatchThread();
        msvc->PAUSE_THREADS = false;
    }
}

int main(int argc, char **argv) {
    json configs = loadRunArgs(argc, argv);
    ContainerAgent *agent = new DataSourceAgent(configs);
    while (agent->running()) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        agent->SendState();
    }
    delete agent;
    return 0;
}