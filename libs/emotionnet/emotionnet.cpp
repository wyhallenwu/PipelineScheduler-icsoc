#include "emotionnet.h"

#include <utility>

EmotionNetAgent::EmotionNetAgent(
    const std::string &name,
    uint16_t own_port,
    int8_t devIndex,
    std::string logPath,
    std::vector<Microservice*> services
) : ContainerAgent(name, own_port, devIndex, logPath) {

    msvcs = std::move(services);
    dynamic_cast<Receiver*>(msvcs[0])->dispatchThread();
    dynamic_cast<BaseReqBatcher*>(msvcs[1])->dispatchThread();
    dynamic_cast<BaseBatchInferencer*>(msvcs[2])->dispatchThread();
    dynamic_cast<BaseClassifier*>(msvcs[3])->dispatchThread();
    for (uint16_t i = 4; i < msvcs.size(); i++) {
        std::thread sender(&Sender::Process, dynamic_cast<Sender*>(msvcs[i]));
        sender.detach();
    }
}


int main(int argc, char **argv){
    spdlog::set_pattern("[%C-%m-%d %H:%M:%S.%f] [%l] %v");

    absl::ParseCommandLine(argc, argv);

    int8_t device = absl::GetFlag(FLAGS_device);
    std::string name = absl::GetFlag(FLAGS_name);
    uint16_t logLevel = absl::GetFlag(FLAGS_verbose);
    std::string logPath = absl::GetFlag(FLAGS_log_dir);
    bool profiling_mode = absl::GetFlag(FLAGS_profile_mode);

    checkCudaErrorCode(cudaSetDevice(device), __func__);
    std::vector<BaseMicroserviceConfigs> msvc_configs = msvcconfigs::LoadFromJson();
    for (uint8_t i = 0; i < msvc_configs.size(); i++) {
        msvc_configs[i].msvc_deviceIndex = device;
        msvc_configs[i].msvc_containerLogPath = logPath + "/" + name;
        msvc_configs[i].msvc_RUNMODE = (profiling_mode) ? RUNMODE::PROFILING : RUNMODE::DEPLOYMENT;
    }

    spdlog::set_level(spdlog::level::level_enum(logLevel));

    std::vector<Microservice*> msvcsList;
    msvcsList.push_back(new Receiver(msvc_configs[0]));
     msvcsList.push_back(new BaseReqBatcher(msvc_configs[1]));
    msvcsList[1]->SetInQueue(msvcsList[0]->GetOutQueue());
    msvcsList.push_back(new BaseBatchInferencer(msvc_configs[2]));
    msvcsList[2]->SetInQueue(msvcsList[1]->GetOutQueue());
    msvcsList.push_back(new BaseClassifier(msvc_configs[3]));
    msvcsList[3]->SetInQueue(msvcsList[2]->GetOutQueue());
    if (msvc_configs[0].msvc_RUNMODE == RUNMODE::PROFILING) {
        msvcsList[0]->SetInQueue(msvcsList[3]->GetOutQueue());
    } else {
        for (uint16_t i = 4; i < msvc_configs.size(); i++) {
            if (msvc_configs[i].msvc_dnstreamMicroservices.front().commMethod == CommMethod::localGPU) {
                msvcsList.push_back(new GPUSender(msvc_configs[i]));
            } else if (msvc_configs[i].msvc_dnstreamMicroservices.front().commMethod == CommMethod::sharedMemory) {
                msvcsList.push_back(new LocalCPUSender(msvc_configs[i]));
            } else if (msvc_configs[i].msvc_dnstreamMicroservices.front().commMethod == CommMethod::sharedMemory) {
                msvcsList.push_back(new RemoteCPUSender(msvc_configs[i]));
            }
            msvcsList[i]->SetInQueue(msvcsList[i - 1]->GetOutQueue());
        }
    }

    for (auto msvc : msvcsList) {
        msvc->msvc_containerName = name;
    }

    ContainerAgent *agent = new EmotionNetAgent(name, absl::GetFlag(FLAGS_port), device, logPath, msvcsList);

    agent->checkReady();
    
    while (agent->running()) {
        std::this_thread::sleep_for(std::chrono::seconds(4));
        agent->SendState();
    }
    delete agent;
    return 0;

}