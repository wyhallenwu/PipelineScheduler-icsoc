#include "receiver.h"
#include "basesink.cpp"

int main() {
    Microservice* receiver = new Receiver(json::parse("{\"msvc_contName\": \"dataSink\", \"msvc_deviceIndex\": 0, "
                                                "\"msvc_RUNMODE\": 0, \"msvc_name\": \"receiver\", \"msvc_type\": 0, "
                                                "\"msvc_appLvlConfigs\":\"\", \"msvc_svcLevelObjLatency\": 1, "
                                                "\"msvc_idealBatchSize\": 1, \"msvc_dataShape\": [[0, 0]], "
                                                "\"msvc_maxQueueSize\": 100, \"msvc_dnstreamMicroservices\": [{"
                                                "\"nb_name\": \"::data_sink\", \"nb_commMethod\": 4, \"nb_link\": [\"\"], "
                                                "\"nb_classOfInterest\": -1, \"nb_maxQueueSize\": 10, "
                                                "\"nb_expectedShape\": [[-1, -1]]}], \"msvc_upstreamMicroservices\": [{"
                                                "\"nb_name\": \"various\", \"nb_commMethod\": 2, "
                                                "\"nb_link\": [\"0.0.0.0:55020\"], \"nb_classOfInterest\": -2, "
                                                "\"nb_maxQueueSize\": 10, \"nb_expectedShape\": [[-1, -1]]}],"
                                                "\"msvc_containerLogPath\": \".\"}"));
    Microservice* sink = new BaseSink(json::parse("{\"msvc_contName\": \"dataSink\", \"msvc_deviceIndex\": 0, "
                                      "\"msvc_RUNMODE\": 0, \"msvc_name\": \"::data_sink\", \"msvc_type\": 502, "
                                      "\"msvc_appLvlConfigs\":\"\", \"msvc_svcLevelObjLatency\": 1, "
                                      "\"msvc_idealBatchSize\": 1, \"msvc_dataShape\": [[0, 0]], "
                                      "\"msvc_maxQueueSize\": 100, \"msvc_dnstreamMicroservices\": [], "
                                      "\"msvc_upstreamMicroservices\": [{\"nb_name\": \"::receiver\", "
                                      "\"nb_commMethod\": 2, \"nb_link\": [\"\"], \"nb_classOfInterest\": -2, "
                                      "\"nb_maxQueueSize\": 10, \"nb_expectedShape\": [[-1, -1]]}],"
                                      "\"msvc_containerLogPath\": \".\"}"));
    sink->SetInQueue(receiver->GetOutQueue());
    receiver->dispatchThread();
    sink->dispatchThread();
    receiver->unpauseThread();
    sink->unpauseThread();
    std::cout << "Start Running" << std::endl;

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    return 0;
}
