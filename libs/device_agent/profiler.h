#ifndef PIPEPLUSPLUS_PROFILER_H
#define PIPEPLUSPLUS_PROFILER_H
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <nvml.h>

class Profiler {
public:
    Profiler(std::vector<unsigned int> pids);
    ~Profiler();

    struct sysStats {
        uint64_t timestamp;
        double cpuUsage;
        long memoryUsage;
        unsigned int gpuUtilization;
        unsigned int gpuMemoryUsage;
        long maxGpuMemoryUsage;
        unsigned int pcieThroughput;
    };

    void run();
    void stop();

    std::vector<sysStats> getStats(unsigned int pid) const;
    std::vector<sysStats> popStats(unsigned int pid);

private:
    void collectStats();

    bool initializeNVML();
    bool setAccounting(nvmlDevice_t device);
    std::vector<nvmlDevice_t> getDevices();
    bool cleanupNVML();

    double getCPUInfo(unsigned int pid);
    long getMemoryInfo(unsigned int pid);
    nvmlAccountingStats_t getGPUInfo(unsigned int pid, nvmlDevice_t device);
    unsigned int getPcieInfo(nvmlDevice_t device);

    bool nvmlInitialized;
    bool running;
    std::thread profilerThread;
    std::map<unsigned int, nvmlDevice_t> pidOnDevices;
    std::map<unsigned int, std::vector<sysStats>> stats;
};


#endif //PIPEPLUSPLUS_PROFILER_H
