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

    struct sysStats {
        uint64_t timestamp = 0;
        int cpuUsage = 0;
        int memoryUsage = 0;
        int rssMemory = 0;
        unsigned int gpuUtilization = 0;
        unsigned int gpuMemoryUsage = 0;
    };

    static int getGpuCount() { return 1; };
    static std::vector<unsigned int> getGpuMemory() { return {0}; };
    sysStats reportAtRuntime(unsigned int pid) {
        std::lock_guard<std::mutex> lock(m); return stats; };

private:
    void jtop(const std::string &cmd);
    std::thread t;
    std::mutex m;
    sysStats stats;
};


#endif //PIPEPLUSPLUS_PROFILER_H
