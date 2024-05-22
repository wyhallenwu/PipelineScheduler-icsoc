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
    Profiler(const std::vector<unsigned int> &pids);
    ~Profiler();

    void run();
    void stop();

    void updatePids(const std::vector<unsigned int> &pids);
    void addPid(unsigned int pid);
    void removePid(unsigned int pid);

    struct sysStats {
        uint64_t timestamp = 0;
        double cpuUsage = 0;
        long memoryUsage = 0;
        unsigned int gpuUtilization = 0;
        unsigned int gpuMemoryUsage = 0;
        long maxGpuMemoryUsage = 0;
        unsigned int pcieThroughput = 0;
    };

    static int getGpuCount();
    static std::vector<long> getGpuMemory(int device_count);
    [[nodiscard]] std::vector<sysStats> getStats(unsigned int pid) const;
    std::vector<sysStats> popStats(unsigned int pid);
    sysStats reportAtRuntime(unsigned int pid);

private:
    void collectStats();

    bool initializeNVML();
    static bool setAccounting(nvmlDevice_t device);
    static std::vector<nvmlDevice_t> getDevices();
    void setPidOnDevices(unsigned int pid, std::vector<nvmlDevice_t> devices);
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
