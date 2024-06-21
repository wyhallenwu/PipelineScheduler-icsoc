#include "data_source.h"

DataSourceAgent::DataSourceAgent(
        const json &configs
) : ContainerAgent(configs) {
    json pipeConfigs = configs["container"]["cont_pipeline"];
    cont_msvcsList.push_back(new DataReader(pipeConfigs[0]));
    for (int i = 1; i < pipeConfigs.size(); i++) {
        cont_msvcsList.push_back(new RemoteCPUSender(pipeConfigs[i]));
        cont_msvcsList[i]->SetInQueue(cont_msvcsList[0]->GetOutQueue());
    }
}

void DataSourceAgent::runService(const json &pipeConfigs, const json &configs) {
    if (configs["cont_allocationMode"] == 0) {
        while (!cont_msvcsList[0]->checkReady()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000 / pipeConfigs[0]["msvc_idealBatchSize"].get<int>()));
        }
    } else {
        cont_msvcsList[0]->setReady();
    }
    this->dispatchMicroservices();

    this->waitReady();
    this->START();
    collectRuntimeMetrics();

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    exit(0);
}


void DataSourceAgent::SetStartFrameRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestSetStartFrame(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        data_reader->SetCurrFrameID(request.value() - 1);
        data_reader->setReady();
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void DataSourceAgent::HandleRecvRpcs() {
    new SetStartFrameRequestHandler(&service, server_cq.get(), cont_msvcsList[0]);
    ContainerAgent::HandleRecvRpcs();

}