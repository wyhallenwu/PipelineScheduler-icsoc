#include <baseprocessor.h>
#include <trtengine.h>
#include <misc.h>
#include "container_agent.h"
#include "receiver.h"


class AgeAgent: public ContainerAgent{
public:
    AgeAgent(
        const json &configs
    );
};