#include "controller.h"

Controller::Controller() {
    devices = std::vector<DeviceHandle*>();
    jobs = std::vector<JobHandle*>();
}

int main() {
    Controller* controller = new Controller();
    delete controller;
    return 0;
}