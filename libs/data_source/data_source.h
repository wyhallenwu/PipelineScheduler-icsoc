#ifndef PIPEPLUSPLUS_DATA_SOURCE_H
#define PIPEPLUSPLUS_DATA_SOURCE_H

#include <list>
#include <thread>
#include "container_agent.h"
#include "sender.h"
#include "data_reader.h"

class DataSourceAgent : public ContainerAgent {
public:
    DataSourceAgent(
        const std::string &name,
        uint16_t own_port,
        int8_t devIndex,
        std::string logPath,
        RUNMODE runmode,
        const json &profiling_configs,
        const json &cont_configs
    );
};


#endif //PIPEPLUSPLUS_DATA_SOURCE_H
