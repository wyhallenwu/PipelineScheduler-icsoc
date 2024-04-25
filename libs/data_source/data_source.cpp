#include "data_source.h"

DataSourceAgent::DataSourceAgent(
    const std::string &name,
    uint16_t own_port,
    int8_t devIndex,
    std::string logPath,
    RUNMODE runmode,
    const json &profiling_configs,
    const json &cont_configs
) : ContainerAgent(name, own_port, devIndex, logPath, runmode, profiling_configs) {
    msvcs.push_back(new DataReader(cont_configs[0]));
    msvcs.push_back(new RemoteCPUSender(cont_configs[1]));
    msvcs[1]->SetInQueue(msvcs[0]->GetOutQueue());

    dynamic_cast<DataReader *>(msvcs[0])->dispatchThread();
    std::thread sender(&RemoteCPUSender::Process, dynamic_cast<RemoteCPUSender *>(msvcs[1]));
    sender.detach();
}

int main(int argc, char **argv) {
    contRunArgs cont_args = loadRunArgs(argc, argv);
    ContainerAgent *agent = new DataSourceAgent(cont_args.cont_name, cont_args.cont_port, cont_args.cont_devIndex, cont_args.cont_logPath, cont_args.cont_runmode, cont_args.cont_profilingConfigs, cont_args.cont_pipeConfigs);
    while (agent->running()) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        agent->SendState();
    }
    delete agent;
    return 0;
}