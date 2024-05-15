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
#include <Python.h>

class Profiler {
public:
    Profiler(const std::vector<unsigned int> &pids);
    ~Profiler();

    void run();
    void stop();
    void addPid(unsigned int pid);

    struct sysStats {
        uint64_t timestamp = 0;
        double cpuUsage = 0;
        long memoryUsage = 0;
        unsigned int gpuUtilization = 0;
        long gpuMemoryUsage = 0;
    };

    static int getGpuCount();
    uint64_t getGpuMemory(int device_count);
    [[nodiscard]] std::vector<sysStats> getStats(unsigned int pid) const;
    std::vector<sysStats> popStats(unsigned int pid);
    sysStats reportAtRuntime(unsigned int pid);

private:
    void collectStats();

    sysStats getStatsJetson(unsigned int pid);
    long getMemoryUsageForPID(unsigned int pid);

    void initializeJtop();

    bool running;
    std::thread profilerThread;
    std::map<unsigned int, std::vector<sysStats>> stats;
    PyObject *pName, *pModule, *pFunc, *pArgs, *pValue;
};


#endif //PIPEPLUSPLUS_PROFILER_H
