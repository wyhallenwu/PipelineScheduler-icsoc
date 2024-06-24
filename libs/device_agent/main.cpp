#include "device_agent.h"

int main(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);

    std::string name = absl::GetFlag(FLAGS_name);
    std::string type = absl::GetFlag(FLAGS_device_type);
    std::string controller_url = absl::GetFlag(FLAGS_controller_url);
    SystemDeviceType deviceType;
    if (type == "server") {
        deviceType = SystemDeviceType::Server;
    } else if (type == "NXXavier") {
        deviceType = SystemDeviceType::NXXavier;
    } else if (type == "AGXXavier") {
        deviceType = SystemDeviceType::AGXXavier;
    } else if (type == "OrinNano") {
        deviceType = SystemDeviceType::OrinNano;
    }
    else {
        std::cerr << "Invalid device type" << std::endl;
        exit(1);
    }

    auto *agent = new DeviceAgent(controller_url, name, deviceType);
    while (agent->isRunning()) {
        std::string command;
        std::cin >> command;
        if (command == "exit") {
            break;
        }
    }
    delete agent;
    return 0;
}