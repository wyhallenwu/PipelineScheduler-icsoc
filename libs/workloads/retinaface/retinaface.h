#include <baseprocessor.h>
#include <trtengine.h>
#include <misc.h>
#include "container_agent.h"
#include "data_reader.h"
#include "data_source.h"
#include "receiver.h"

class RetinaFaceAgent : public ContainerAgent {
public:
    RetinaFaceAgent(const json &configs): ContainerAgent(configs) {}
};

class RetinaFaceDataSource : public DataSourceAgent {
public:
    RetinaFaceDataSource(const json &configs): DataSourceAgent(configs) {}
};