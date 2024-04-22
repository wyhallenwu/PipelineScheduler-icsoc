#ifndef PIPEPLUSPLUS_CONTROLLER_H
#define PIPEPLUSPLUS_CONTROLLER_H
#include "microservice.h"
#include "device_agent.h"
#include <thread>
#include "controlcommunication.grpc.pb.h"

enum DeviceType {
    Server,
    JetsonNano,
    JetsonNX,
    JetsonAGX
};

enum ModelType {
    Yolov5,
    Arcface,
    Retinaface,
    Yolov5_Plate
};

enum PipelineType {
    Traffic,
    Video_Call,
    Buildiung_Security
};

struct MicroserviceHandle;
struct NodeHandle {
    std::string name;
    std::shared_ptr<ControlCommunication::Stub> stub;
    DeviceType type;
    int num_processors; // number of processing units, CPU cores for Jetson or GPUs for server
    long memory;
    float utilization;
    std::vector<MicroserviceHandle*> microservices;
};

struct TaskHandle {
    std::string name;
    int slo;
    std::vector<MicroserviceHandle*> subtasks;
};

struct MicroserviceHandle {
    std::string name;
    ModelType model;
    NodeHandle *device_agent;
    TaskHandle *task;
};


class Controller {
public:
    Controller();
    ~Controller() = default;

    void Proceed();
    void AddTask(std::string name, int slo, PipelineType type, std::string source, std::string device);

    bool isRunning() { return running; };
    void Stop() { running = false; };

private:

    bool running;
    std::vector<NodeHandle*> devices;
    std::vector<TaskHandle*> tasks;
    std::vector<MicroserviceHandle*> microservices;
};


#endif //PIPEPLUSPLUS_CONTROLLER_H
