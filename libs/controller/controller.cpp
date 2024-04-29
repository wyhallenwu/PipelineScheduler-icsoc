#include "controller.h"

void TaskDescription::to_json(json &j, const TaskDescription::TaskStruct &val) {
    j = json{{"name", val.name},
             {"slo", val.slo},
             {"type", val.type},
             {"source", val.source},
             {"device", val.device}};
}

void TaskDescription::from_json(const nlohmann::json &j, TaskDescription::TaskStruct &val) {
    j.at("name").get_to(val.name);
    j.at("slo").get_to(val.slo);
    j.at("type").get_to(val.type);
    j.at("source").get_to(val.source);
    j.at("device").get_to(val.device);
}

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

void Controller::AddTask(const TaskDescription::TaskStruct &t) {
    tasks.insert({t.name, {t.slo, t.type, {}}});
    TaskHandle *task = &tasks[t.name];
    NodeHandle *device = &devices[t.device];
    auto models = getModelsByPipelineType(t.type);
    // TODO: get initial batch sizes based on TaskDescription and Devices
    std::map<std::string, int> batch_sizes = {};
    std::string tmp = t.name;
    microservices.insert({tmp.append(":datasource"), {tmp, DataSource, device, task, {}, {}}});
    task->subtasks.insert({tmp, &microservices[tmp]});
    task->subtasks[tmp]->recv_port = device->next_free_port++;
    device->microservices.insert({tmp, task->subtasks[tmp]});
    device = &devices["server"];
    for(const auto& m : models) {
        tmp = t.name;
        microservices.insert({tmp.append(m.first), {tmp, MODEL_TYPES[m.first], device, task, {}, {}}});
        task->subtasks.insert({tmp, &microservices[tmp]});
        task->subtasks[tmp]->recv_port = device->next_free_port++;
        device->microservices.insert({tmp, task->subtasks[tmp]});
    }
    for(const auto& m : models) {
        for(const auto& d : m.second) {
            tmp = t.name;
            task->subtasks[tmp.append(d.first)]->class_of_interest = d.second;
            task->subtasks[tmp]->upstreams.push_back(task->subtasks[t.name + m.first]);
            task->subtasks[t.name + m.first]->downstreams.push_back(task->subtasks[tmp]);
        }
    }
    MicroserviceConfig request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    for (auto msvc: task->subtasks) {
        request.set_name(msvc.first);
        request.set_model(msvc.second->model);
        request.set_bath_size(batch_sizes[msvc.first]);
        request.set_recv_port(msvc.second->recv_port);
        request.set_slo(task->slo);
        for (auto dwnstr: msvc.second->downstreams) {
            Neighbor *dwn = request.add_downstream();
            dwn->set_name(dwnstr->name);
            dwn->set_ip(dwnstr->device_agent->ip);
            dwn->set_class_of_interest(dwnstr->class_of_interest);
        }
        if (msvc.second->model == DataSource) {
            Neighbor *up = request.add_upstream();
            up->set_name("video_source");
            up->set_ip(t.source);
            up->set_class_of_interest(-1);
        } else {
            for (auto upstr: msvc.second->upstreams) {
                Neighbor *up = request.add_upstream();
                up->set_name(upstr->name);
                up->set_ip(absl::StrFormat("%s:%d", upstr->device_agent->ip, upstr->recv_port));
                up->set_class_of_interest(-2);
            }
        }
        std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
                msvc.second->device_agent->stub->AsyncStartMicroservice(&context, request, msvc.second->device_agent->cq));
        rpc->Finish(&reply, &status, (void *) 1);
        void *got_tag;
        bool ok = false;
        GPR_ASSERT(msvc.second->device_agent->cq->Next(&got_tag, &ok));
        GPR_ASSERT(ok);
    }
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
                                    {request.ip_address(),
                                     ControlCommunication::NewStub(
                                            grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials())),
                                     new CompletionQueue(),
                                     static_cast<DeviceType>(request.device_type()),
                                     request.processors(),
                                     request.memory(), {}, 55001}});
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

std::vector<std::pair<std::string, std::vector<std::pair<std::string, int>>>> Controller::getModelsByPipelineType(PipelineType type) {
    switch (type) {
        case PipelineType::Traffic:
            return {{":yolov5", {{":retinaface", 0}, {":cartype", 2}, {":plate", 2}}},
                    {":retinaface", {{":arcface", -1}}},
                    {":arcface", {}},
                    {":cartype", {}},
                    {":plate", {}}};
        case PipelineType::Video_Call:
            return {{":retinaface", {}},
                    {":gender", {}},
                    {":age", {}},
                    {":emotion", {}},
                    {":arcface", {}}};
        case PipelineType::Building_Security:
            return {{":yolov5", {}},
                    {":retinaface", {}},
                    {":movenet", {}},
                    {"gender", {}},
                    {":age", {}}};
        default:
            return {};
    }
}

int main() {
    auto controller = new Controller();
    std::thread receiver_thread(&Controller::HandleRecvRpcs, controller);
    std::ifstream file("../jsons/experiment.json");
    std::vector<TaskDescription::TaskStruct> tasks = json::parse(file);
    while (controller->isRunning()) {
        TaskDescription::TaskStruct task;
        std::string command;
        std::cout << "Enter command {Traffic, Video_Call, People, exit): ";
        std::cin >> command;
        if (command == "exit") {
            controller->Stop();
            continue;
        } else if (command == "Traffic") {
            task.type = PipelineType::Traffic;
        } else if (command == "Video_Call") {
            task.type = PipelineType::Video_Call;
        } else if (command == "People") {
            task.type = PipelineType::Building_Security;
        } else {
            std::cout << "Invalid command" << std::endl;
            continue;
        }
        std::cout << "Enter name of task: ";
        std::cin >> task.name;
        std::cout << "Enter SLO in ms: ";
        std::cin >> task.slo;
        std::cout << "Enter total path to source file: ";
        std::cin >> task.source;
        std::cout << "Enter name of source device: ";
        std::cin >> task.device;
        controller->AddTask(task);
    }
    delete controller;
    return 0;
}
