#include "controller.h"

Controller::Controller() {
    running = true;
    devices = std::vector<NodeHandle*>();
    tasks = std::vector<TaskHandle*>();
    microservices = std::vector<MicroserviceHandle*>();
}

void Controller::Proceed() {

}

void Controller::AddTask(std::string name, int slo, PipelineType type, std::string source, std::string device) {

}

int main() {
    auto controller = new Controller();
    std::thread receiver_thread(&Controller::Proceed, controller);
    while (controller->isRunning()){
        std::string command;
        PipelineType type;
        std::cout << "Enter command: ";
        std::cin >> command;
        if (command == "exit") {
            controller->Stop();
            continue;
        } else if (command == "traffic") {
            type = PipelineType::Traffic;
        }
        std::string name;
        int slo;
        std::string path;
        std::string device;
        std::cout << "Enter name of task: ";
        std::cin >> name;
        std::cout << "Enter SLO in ms: ";
        std::cin >> slo;
        std::cout << "Enter path to source file: ";
        std::cin >> path;
        std::cout << "Enter name of device: ";
        std::cin >> device;
        controller->AddTask(name, 0, type, path, device);
    }
    delete controller;
    return 0;
}