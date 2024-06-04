#include "baseprocessor.h"
#include <trtengine.h>
#include <misc.h>
#include "container_agent.h"
#include "receiver.h"


class EmotionNetAgent: public ContainerAgent{
public:
    EmotionNetAgent(
        const json& configs
    );
};