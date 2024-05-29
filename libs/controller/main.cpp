#include "controller.h"

int main() {
    auto controller = new Controller();
    std::thread receiver_thread(&Controller::HandleRecvRpcs, controller);
    receiver_thread.detach();
    std::ifstream file("../jsons/experiment.json");
    std::vector<TaskDescription::TaskStruct> tasks = json::parse(file);
    std::string command;

    while (controller->isRunning()) {
        TaskDescription::TaskStruct task;
        std::cout << "Enter command {init, traffic, video_call, people, exit): ";
        std::cin >> command;
        if (command == "exit") {
            controller->Stop();
            break;
        } else if (command == "init") {
            for (auto &t: tasks) {
                controller->AddTask(t);
            }
            continue;
        } else if (command == "traffic") {
            task.type = PipelineType::Traffic;
        } else if (command == "video_call") {
            task.type = PipelineType::Video_Call;
        } else if (command == "people") {
            task.type = PipelineType::Building_Security;
        } else {
            std::cout << "Invalid command" << std::endl;
            continue;
        }
        std::cout << "Enter name of task: ";
        std::cin >> task.name;
        std::cout << "Enter SLO in ns: ";
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
