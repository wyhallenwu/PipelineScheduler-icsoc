#include "retinaface.h"

#include <utility>

RetinaFaceAgent::RetinaFaceAgent(
        const json &configs
) : ContainerAgent(configs) {
    // for (uint16_t i = 4; i < msvcs.size(); i++) {
    //     msvcs[i]->dispatchThread();
    // }
}

RetinaFaceDataSource::RetinaFaceDataSource(
        const json &configs
) : ContainerAgent(configs) {

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

    json configs = loadRunArgs(argc, argv);

    ContainerAgent *agent;

    json pipeConfigs = configs["container"]["cont_pipeline"];

    if (pipeConfigs[0].at("msvc_type") == MicroserviceType::DataSource) {
        agent = new RetinaFaceDataSource(configs);
    } else {
        agent = new RetinaFaceAgent(configs);
    }

    std::vector<Microservice *> msvcsList;
    if (pipeConfigs[0].at("msvc_type") == MicroserviceType::DataSource) {
        msvcsList.push_back(new DataReader(pipeConfigs[0]));
    } else {
        msvcsList.push_back(new Receiver(pipeConfigs[0]));
    }
    msvcsList.push_back(new BaseReqBatcher(pipeConfigs[1]));
    msvcsList[1]->SetInQueue(msvcsList[0]->GetOutQueue());
    msvcsList.push_back(new BaseBatchInferencer(pipeConfigs[2]));
    msvcsList[2]->SetInQueue(msvcsList[1]->GetOutQueue());
    msvcsList.push_back(new BaseBBoxCropperAugmentation(pipeConfigs[3]));
    msvcsList[3]->SetInQueue(msvcsList[2]->GetOutQueue());
    // dynamic_cast<BaseBBoxCropper*>(msvcsList[3])->setInferenceShape(dynamic_cast<BaseBatchInferencer*>(msvcsList[2])->getInputShapeVector());
    if (configs["container"]["cont_RUNMODE"] == RUNMODE::PROFILING) {
        msvcsList.push_back(new BaseSink(pipeConfigs[4]));
        msvcsList[4]->SetInQueue(msvcsList[3]->GetOutQueue());
    } else {
        for (uint16_t i = 4; i < pipeConfigs.size(); i++) {
            if (pipeConfigs[i].at("msvc_dnstreamMicroservices")[0].at("nb_commMethod") == CommMethod::localGPU) {
                spdlog::get("container_agent")->info("Local GPU Sender");
                msvcsList.push_back(new GPUSender(pipeConfigs[i]));
            } else if (pipeConfigs[i].at("msvc_dnstreamMicroservices")[0].at("nb_commMethod") == CommMethod::sharedMemory) {
                spdlog::get("container_agent")->info("Local CPU Sender");
                msvcsList.push_back(new LocalCPUSender(pipeConfigs[i]));
            } else if (pipeConfigs[i].at("msvc_dnstreamMicroservices")[0].at("nb_commMethod") == CommMethod::serialized) {
                spdlog::get("container_agent")->info("Remote CPU Sender");
                msvcsList.push_back(new RemoteCPUSender(pipeConfigs[i]));
            }
            msvcsList[i]->SetInQueue({msvcsList[3]->GetOutQueue(pipeConfigs[3].at("msvc_dnstreamMicroservices")[i-4].at("nb_classOfInterest"))});
        }
    }
    agent->addMicroservice(msvcsList);

    agent->runService(pipeConfigs, configs);
    delete agent;
    return 0;
}