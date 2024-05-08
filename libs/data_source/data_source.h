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
};


#endif //PIPEPLUSPLUS_DATA_SOURCE_H
