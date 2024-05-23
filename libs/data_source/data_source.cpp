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
    cont_msvcsList.push_back(new DataReader(pipeConfigs[0]));
    cont_msvcsList.push_back(new RemoteCPUSender(pipeConfigs[1]));
    cont_msvcsList[1]->SetInQueue(cont_msvcsList[0]->GetOutQueue());
}

int main(int argc, char **argv) {
    json configs = loadRunArgs(argc, argv);
    ContainerAgent *agent = new DataSourceAgent(configs);
    agent->runService("", configs);
    delete agent;
    return 0;
}