#include "receiver.h"

int main() {
    auto r = new Receiver(
            {"dummy_receiver_0", MicroserviceType::Receiver, "", 0, 1, {},
             {{"dummy_0", CommMethod::sharedMemory, {"localhost:55002"}, 0, -2, {{0,0}}}},
             {{"downstream", CommMethod::localCPU, {""}, 0, -1, {{0,0}}}}});
    while (true){
        std::cout << "QueueSize = " << r->GetOutQueueSize(0) << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return 0;
}
