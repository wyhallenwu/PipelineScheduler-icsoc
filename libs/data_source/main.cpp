#include "data_source.h"

int main(int argc, char **argv) {
    json configs = loadRunArgs(argc, argv);
    ContainerAgent *agent = new DataSourceAgent(configs);
    agent->runService("", configs);
    delete agent;
    return 0;
}
