#include "yolov5.h"

#include <utility>

YoloV5Agent::YoloV5Agent(const std::string &name, uint16_t own_port, std::vector<Microservice*> services)
        : ContainerAgent(name, own_port) {
    msvcs = std::move(services);
    std::thread preprocessor(&YoloV5Preprocessor::batchRequests, dynamic_cast<YoloV5Preprocessor*>(msvcs[1]));
    preprocessor.detach();
    std::thread inference(&YoloV5Inference::inference, dynamic_cast<YoloV5Inference*>(msvcs[2]));
    inference.detach();
    std::thread postprocessor(&YoloV5Postprocessor::postProcessing, dynamic_cast<YoloV5Postprocessor*>(msvcs[3]));
    postprocessor.detach();
    for (int i = 4; i < msvcs.size(); i++) {
        std::thread sender(&Sender::Process, dynamic_cast<Sender*>(msvcs[i]));
        sender.detach();
    }
}

int main(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);
    std::vector<BaseMicroserviceConfigs> msvc_configs = msvcconfigs::LoadFromJson();
    TRTConfigs yoloConfigs = json::parse(absl::GetFlag(FLAGS_trt_json).value()).get<TRTConfigs>();
    std::string name = absl::GetFlag(FLAGS_name);
    std::vector<Microservice*> msvcs;
    msvcs.push_back(new Receiver(msvc_configs[0], CommMethod::localGPU));
    msvcs.push_back(new YoloV5Preprocessor(msvc_configs[1]));
    msvcs[1]->SetInQueue(msvcs[0]->GetOutQueue());
    msvcs.push_back(new YoloV5Inference(msvc_configs[2], yoloConfigs));
    msvcs[2]->SetInQueue(msvcs[1]->GetOutQueue());
    msvcs.push_back(new YoloV5Postprocessor(msvc_configs[3]));
    msvcs[3]->SetInQueue(msvcs[2]->GetOutQueue());
    for (int i = 4; i < msvc_configs.size(); i++) {
        if (msvc_configs[i].dnstreamMicroservices.front().commMethod == CommMethod::localGPU) {
            msvcs.push_back(new GPUSender(msvc_configs[i]));
        } else if (msvc_configs[i].dnstreamMicroservices.front().commMethod == CommMethod::sharedMemory) {
            msvcs.push_back(new LocalCPUSender(msvc_configs[i]));
        } else if (msvc_configs[i].dnstreamMicroservices.front().commMethod == CommMethod::sharedMemory) {
            msvcs.push_back(new RemoteCPUSender(msvc_configs[i]));
        }
        msvcs[i]->SetInQueue(msvcs[i - 1]->GetOutQueue());
    }
    ContainerAgent *agent = new YoloV5Agent(name, absl::GetFlag(FLAGS_port), msvcs);
    while (agent->running()) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        agent->SendQueueLengths();
    }
    delete agent;
    return 0;
}