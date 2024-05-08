#include <baseprocessor.h>
#include <trtengine.h>
#include <misc.h>
#include "container_agent.h"
#include "profilegenerator.h"
#include "receiver.h"


class carbrandAgent: public ContainerAgent{
public:
    carbrandAgent(
        const json &configs
    );
};