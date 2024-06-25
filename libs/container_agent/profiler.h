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
    Profiler(const std::vector<unsigned int> &pids, std::string mode);
    ~Profiler();

    void run();

    struct sysStats {
        int cpuUsage = 0;
        int processMemoryUsage = 0;
        int deviceMemoryUsage = 0;
        unsigned int gpuUtilization = 0;
        unsigned int gpuMemoryUsage = 0;
    };

    static int getGpuCount() { return 1; };
    void addPid(unsigned int pid) { stats[pid] = sysStats(); };
    static std::vector<unsigned int> getGpuMemory() { return {0}; };

    sysStats reportAtRuntime(unsigned int pid) {
        std::lock_guard<std::mutex> lock(m); return stats[pid]; };

private:
    void jtop(const std::string &cmd);
    std::thread t;
    std::mutex m;
    std::map<unsigned int, sysStats> stats;
};


#endif //PIPEPLUSPLUS_PROFILER_H
