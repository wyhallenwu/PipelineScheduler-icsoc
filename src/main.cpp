#include "receiver.h"

int main() {
    auto r = new Receiver({"dummy_receiver_0", MicroserviceType::Receiver, 0, 1, {}, {{"datasource_0", CommMethod::sharedMemory, {"localhost:5000"}, 0, -2, {{0,0}}}},{{"dummy", CommMethod::localCPU, {""}, 0, -1, {{0,0}}}}}, CommMethod::localCPU);
    while (true){
        std::cout << "QueueSize = " << r->GetOutQueueSize(0) << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return 0;
}