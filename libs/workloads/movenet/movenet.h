#include <baseprocessor.h>
#include <trtengine.h>
#include <misc.h>
#include "container_agent.h"
#include "receiver.h"


class MoveNetAgent: public ContainerAgent{
public:
    MoveNetAgent(
        const json& configs
    );
};