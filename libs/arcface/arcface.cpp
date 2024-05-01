#include "arcface.h"

#include <utility>

ArcFaceAgent::ArcFaceAgent(
    const std::string &name,
    uint16_t own_port,
    int8_t devIndex,
    std::string logPath,
    RUNMODE runmode,
    const json &profiling_configs
) : ContainerAgent(name, own_port, devIndex, logPath, runmode, profiling_configs) {
    // for (uint16_t i = 4; i < msvcs.size(); i++) {
    //     msvcs[i]->dispatchThread();
    // }
}

ArcFaceDataSource::ArcFaceDataSource(
    const std::string &name,
    uint16_t own_port,
    int8_t devIndex,
    std::string logPath,
    RUNMODE runmode,
    const json &profiling_configs
) : ContainerAgent(name, own_port, devIndex, logPath, runmode, profiling_configs) {

    // msvcs = std::move(services);
    // dynamic_cast<DataReader*>(msvcs[0])->dispatchThread();
    // dynamic_cast<BaseReqBatcher*>(msvcs[1])->dispatchThread();
    // dynamic_cast<BaseBatchInferencer*>(msvcs[2])->dispatchThread();
    // dynamic_cast<BaseBBoxCropper*>(msvcs[3])->dispatchThread();
    // for (uint16_t i = 4; i < msvcs.size(); i++) {
    //     std::thread sender(&Sender::Process, dynamic_cast<Sender*>(msvcs[i]));
    //     sender.detach();
    // }
}

int main(int argc, char **argv) {

    contRunArgs cont_args = loadRunArgs(argc, argv);

    ContainerAgent *agent;

    if (cont_args.cont_pipeConfigs[0].at("msvc_type") == MicroserviceType::DataSource) {
        agent = new ArcFaceDataSource(cont_args.cont_name, cont_args.cont_port, cont_args.cont_devIndex, cont_args.cont_logPath, cont_args.cont_runmode, cont_args.cont_profilingConfigs);
    } else {
        agent = new ArcFaceAgent(cont_args.cont_name, cont_args.cont_port, cont_args.cont_devIndex, cont_args.cont_logPath, cont_args.cont_runmode, cont_args.cont_profilingConfigs);
    }

    std::vector<Microservice*> msvcsList;
    if (cont_args.cont_pipeConfigs[0].at("msvc_type") == MicroserviceType::DataSource) {
        msvcsList.push_back(new DataReader(cont_args.cont_pipeConfigs[0]));
    } else {
        msvcsList.push_back(new Receiver(cont_args.cont_pipeConfigs[0]));
    }
    msvcsList.push_back(new BaseReqBatcher(cont_args.cont_pipeConfigs[1]));
    msvcsList[1]->SetInQueue(msvcsList[0]->GetOutQueue());
    msvcsList.push_back(new BaseBatchInferencer(cont_args.cont_pipeConfigs[2]));
    msvcsList[2]->SetInQueue(msvcsList[1]->GetOutQueue());
    msvcsList.push_back(new BaseClassifier(cont_args.cont_pipeConfigs[3]));
    msvcsList[3]->SetInQueue(msvcsList[2]->GetOutQueue());
    if (cont_args.cont_runmode == RUNMODE::PROFILING) {
        msvcsList[0]->SetInQueue(msvcsList[3]->GetOutQueue());
    } else {
        for (uint16_t i = 4; i < cont_args.cont_pipeConfigs.size(); i++) {
            if (cont_args.cont_pipeConfigs[i].at("msvc_dnstreamMicroservices")[i-4].at("nb_commMethod") == CommMethod::localGPU) {
                msvcsList.push_back(new GPUSender(cont_args.cont_pipeConfigs[i]));
            } else if (cont_args.cont_pipeConfigs[i].at("msvc_dnstreamMicroservices")[i-4].at("nb_commMethod") == CommMethod::sharedMemory) {
                msvcsList.push_back(new LocalCPUSender(cont_args.cont_pipeConfigs[i]));
            } else if (cont_args.cont_pipeConfigs[i].at("msvc_dnstreamMicroservices")[i-4].at("nb_commMethod") == CommMethod::serialized) {
                msvcsList.push_back(new RemoteCPUSender(cont_args.cont_pipeConfigs[i]));
            }
            msvcsList[i]->SetInQueue({msvcsList[3]->GetOutQueue(cont_args.cont_pipeConfigs[3].at("msvc_dnstreamMicroservices")[i-4].at("nb_classOfInterest"))});
        }
    }

    agent->addMicroservice(msvcsList);
    agent->dispatchMicroservices();

    agent->checkReady();
    
    while (agent->running()) {
        std::this_thread::sleep_for(std::chrono::seconds(4));
        agent->SendState();
    }
    delete agent;
    return 0;
}