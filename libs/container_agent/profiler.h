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
#include <mutex>
#include <condition_variable>
#include <termios.h>
#include <sys/select.h>

class Profiler {
public:
    Profiler(const std::vector<unsigned int> &pids);
    ~Profiler();

    void run();
    void addPid(unsigned int pid);

    struct sysStats {
        uint64_t timestamp = 0;
        double cpuUtilization = 0;
        double processMemoryUsage = 0;
        double processGpuMemoryUsage = 0;
        double gpuUtilization = 0;
        double totalGpuRamUsage = 0;
        double totalCpuRamUsage = 0;
        double maxRamCapacity = 0;
    };

    static int getGpuCount();
    std::vector<unsigned int> getGpuMemory(int processing_units);
    sysStats reportAtRuntime(unsigned int pid);

private:
    void collectStats();

    bool running;
    std::map<unsigned int, std::vector<sysStats>> stats;
};


#endif //PIPEPLUSPLUS_PROFILER_H
