#ifndef PIPEPLUSPLUS_DATA_SOURCE_H
#define PIPEPLUSPLUS_DATA_SOURCE_H

#include <list>
#include <thread>
#include "container_agent.h"
#include "data_reader.h"

class DataSourceAgent : public ContainerAgent {
public:
    DataSourceAgent(
        const json &configs
    );

    void runService(const json &pipeConfigs, const json &configs) override;

    class SetStartFrameRequestHandler : public RequestHandler {
    public:
        SetStartFrameRequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                    Microservice *data_reader)
                : RequestHandler(service, cq), data_reader(data_reader) {
            Proceed();
        }

        void Proceed() final;

    private:
        indevicecommunication::Int32 request;
        Microservice *data_reader;
    };

    void HandleRecvRpcs() override;
};

#endif //PIPEPLUSPLUS_DATA_SOURCE_H
