#include <baseprocessor.h>
#include <trtengine.h>
#include <misc.h>
#include "container_agent.h"
#include "receiver.h"
#include "data_reader.h"

using namespace spdlog;


class ArcFaceAgent : public ContainerAgent {
public:
    ArcFaceAgent(
        const std::string &name,
        uint16_t own_port,
        int8_t devIndex,
        std::string logPath,
        RUNMODE runmode,
        const json &profiling_configs
    );
};

class ArcFaceDataSource : public ContainerAgent {
public:
    ArcFaceDataSource(
        const std::string &name,
        uint16_t own_port,
        int8_t devIndex,
        std::string logPath,
        RUNMODE runmode,
        const json &profiling_configs
    );
};