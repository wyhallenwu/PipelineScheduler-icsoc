#include <baseprocessor.h>
#include <trtengine.h>
#include <misc.h>
#include "container_agent.h"
#include "receiver.h"


class GenderAgent: public ContainerAgent{
public:
    GenderAgent(
        const json &configs
    );
};