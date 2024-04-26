#include "controller.h"

Controller::Controller() {
    running = true;
    devices = std::map<std::string, NodeHandle>();
    tasks = std::map<std::string, TaskHandle>();
    microservices = std::map<std::string, MicroserviceHandle>();

    std::string server_address = absl::StrFormat("%s:%d", "localhost", 60002);
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    cq = builder.AddCompletionQueue();
    server = builder.BuildAndStart();
}

void Controller::HandleRecvRpcs() {
    new LightMetricsRequestHandler(&service, cq.get(), this);
    new FullMetricsRequestHandler(&service, cq.get(), this);
    while (running) {
        void *tag;
        bool ok;
        if (!cq->Next(&tag, &ok)) {
            break;
        }
        static_cast<RequestHandler *>(tag)->Proceed();
    }
}

void Controller::AddTask(std::string name, int slo, PipelineType type, std::string source, std::string device_name) {
    tasks.insert({name, {slo, type, {}}});
    TaskHandle *task = &tasks[name];
    NodeHandle *device = &devices[device_name];
    std::vector<std::string> models = getModelsByPipelineType(type);
    microservices.insert({name.append(":datasource"), {DataSource, device, task, {}, {}}});
    task->subtasks.insert({name.append(":datasource"), &microservices[name.append(":datasource")]});
    device->microservices.insert({name.append(":datasource"), &microservices[name.append(":datasource")]});
    device = &devices["server"];
    for(const auto& m : models) {
        std::string tmp = name.append(m);
        microservices.insert({tmp, {MODEL_TYPES[m], device, task, {}, {}}});
        task->subtasks.insert({tmp, &microservices[tmp]});
        device->microservices.insert({tmp, &microservices[tmp]});
    }
    switch (type) {
        case PipelineType::Traffic:
            //more processing
            break;
        case PipelineType::Video_Call:
            //more processing
            break;
        case PipelineType::Building_Security:
            //more processing
            break;
    }


    // void *got_tag;
    // bool ok = false;
    // GPR_ASSERT(device->cq->Next(&got_tag, &ok));
    // GPR_ASSERT(ok);
}

void Controller::UpdateLightMetrics(google::protobuf::RepeatedPtrField<LightMetrics> metrics) {
    for (auto metric: metrics) {
        microservices[metric.name()].queue_lengths = metric.queue_size();
        microservices[metric.name()].metrics.requestRate = metric.request_rate();
    }
}

void Controller::UpdateFullMetrics(google::protobuf::RepeatedPtrField<FullMetrics> metrics) {
    for (auto metric: metrics) {
        microservices[metric.name()].queue_lengths = metric.queue_size();
        Metrics *m = &microservices[metric.name()].metrics;
        m->requestRate = metric.request_rate();
        m->cpuUsage = metric.cpu_usage();
        m->memUsage = metric.mem_usage();
        m->gpuUsage = metric.gpu_usage();
        m->gpuMemUsage = metric.gpu_mem_usage();
    }
}

void Controller::LightMetricsRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestSendLightMetrics(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new LightMetricsRequestHandler(service, cq, controller);
        controller->UpdateLightMetrics(request.metrics());
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void Controller::FullMetricsRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestSendFullMetrics(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new FullMetricsRequestHandler(service, cq, controller);
        controller->UpdateFullMetrics(request.metrics());
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void Controller::DeviseAdvertisementHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestAdvertiseToController(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new DeviseAdvertisementHandler(service, cq, controller);
        std::string target_str = absl::StrFormat("%s:%d", request.ip_address(), 60002);
        controller->devices.insert({request.device_name(),
                                    {ControlCommunication::NewStub(
                                            grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials())),
                                     new CompletionQueue(),
                                     static_cast<DeviceType>(request.device_type()),
                                     request.processors(),
                                     request.memory(), {}}});
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

std::vector<std::string> Controller::getModelsByPipelineType(PipelineType type) {
    switch (type) {
        case PipelineType::Traffic:
            return {":yolov5", ":retinaface", ":arcface", ":cartype", ":plate"};
        case PipelineType::Video_Call:
            return {":retinaface", ":gender", ":age", ":emotion", ":arcface"};
        case PipelineType::Building_Security:
            return {":yolov5", ":retinaface", ":movenet", "gender", ":age"};
        default:
            return {};
    }
}

int main() {
    auto controller = new Controller();
    std::thread receiver_thread(&Controller::HandleRecvRpcs, controller);
    while (controller->isRunning()) {
        std::string command;
        PipelineType type;
        std::cout << "Enter command {Traffic, Video_Call, People, exit): ";
        std::cin >> command;
        if (command == "exit") {
            controller->Stop();
            continue;
        } else if (command == "Traffic") {
            type = PipelineType::Traffic;
        } else if (command == "Video_Call") {
            type = PipelineType::Video_Call;
        } else if (command == "People") {
            type = PipelineType::Building_Security;
        } else {
            std::cout << "Invalid command" << std::endl;
            continue;
        }
        std::string name;
        int slo;
        std::string path;
        std::string device;
        std::cout << "Enter name of task: ";
        std::cin >> name;
        std::cout << "Enter SLO in ms: ";
        std::cin >> slo;
        std::cout << "Enter total path to source file: ";
        std::cin >> path;
        std::cout << "Enter name of source device: ";
        std::cin >> device;
        controller->AddTask(name, 0, type, path, device);
    }
    delete controller;
    return 0;
}
