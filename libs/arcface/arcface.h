#include <baseprocessor.h>
#include <trtengine.h>
#include <misc.h>
#include "container_agent.h"
#include "receiver.h"
#include "profilegenerator.h"
#include "data_reader.h"

using namespace spdlog;


class ArcFaceAgent : public ContainerAgent {
public:
    ArcFaceAgent(
        const json &configs
    );
};

class ArcFaceDataSource : public ContainerAgent {
public:
    ArcFaceDataSource(
        const json &configs
    );
};