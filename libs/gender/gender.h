#include <baseprocessor.h>
#include <trtengine.h>
#include <misc.h>
#include "container_agent.h"
#include "receiver.h"


class GenderAgent: public ContainerAgent{
public:
    GenderAgent(
        const std::string &name,
        uint16_t own_port,
        int8_t devIndex,
        std::string logPath,
        RUNMODE runmode,
        const json &profiling_configs
    );
};