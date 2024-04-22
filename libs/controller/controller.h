#ifndef PIPEPLUSPLUS_CONTROLLER_H
#define PIPEPLUSPLUS_CONTROLLER_H
#include "microservice.h"
#include "device_agent.h"
#include "controlcommunication.grpc.pb.h"

struct JobHandle;
struct DeviceHandle {
    std::shared_ptr<ControlCommunication::Stub> stub;
    std::vector<JobHandle*> jobs;
};
struct JobHandle {
    std::string name;
    DeviceHandle *device_agent;
};


class Controller {
public:
    Controller();
    ~Controller() = default;
private:
    std::vector<DeviceHandle*> devices;
    std::vector<JobHandle*> jobs;
};


#endif //PIPEPLUSPLUS_CONTROLLER_H
