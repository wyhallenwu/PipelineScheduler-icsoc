#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <unistd.h>
#include <sys/sysinfo.h>
#include <nvml.h>

struct sysStats {
    double cpuUsage;
    double memoryUsage;
    int gpuUtilization;
    double gpuMemoryUsage;
};

std::vector<std::string> CpuUtilization() {
    std::vector<std::string> timers;
    std::string timer;
    std::string line;
    std::string skip;
    std::ifstream stream("/proc/stat");
    if (stream.is_open()) {
        std::getline(stream, line);
        std::istringstream linestream(line);
        linestream >> skip;
        for(int i = 0; i < 10; ++i) {
            linestream >> timer;
            timers.push_back(timer);
        }
    }
    stream.close();
    std::cout << "timers: " << timers[0] << std::endl;
    return timers;
}

long Jiffies() {
    std::vector<std::string> jiffies = CpuUtilization();
    long total = 0;
    for(std::string jiffie : jiffies) {
        total += std::stol(jiffie);
    }
    long idle = std::stol(jiffies[3]);
    long iowait = std::stol(jiffies[4]);
    return total - idle - iowait;
}


double getCurrentCpuUsage(int pid) {
    std::string utime;
    std::string stime;
    std::string line;
    std::string skip;
    long total_active = Jiffies();
    std::ifstream stream("/proc/"+ std::to_string(pid) + "/stat");
    if (stream.is_open()) {
        std::getline(stream, line);
        std::istringstream linestream(line);
        for(int i = 0; i < 13; ++i) {
            linestream >> skip;
        }
        linestream >> utime >> stime;
    }
    long process_active = std::stol(utime) + std::stol(stime);
    return 100* (double)process_active / total_active;


//    std::ifstream file("/proc/"+ std::to_string(pid) + "/stat");
//    std::string line;
//    std::getline(file, line);
//    std::istringstream ss(line);
//    for (int i = 0; i < 13; i++) {
//        ss.ignore();
//    }
//    long utime, stime;
//    ss >> utime >> stime;
//    long totalTicks = utime + stime;
//    long clockTicksPerSecond = sysconf(_SC_CLK_TCK);
//    double elapsedTime = (totalTicks / clockTicksPerSecond) - previousCPU;
//    std::cout << "previousCPU: " << previousCPU << " | totalClicks: " << totalTicks << " | elapsedTime: " << elapsedTime << std::endl;
//    previousCPU = totalTime;
//    return elapsedTime / 1;
}

int getCurrentMemoryUsage(int pid) {
    std::string value = "0";
    bool search = true;
    std::string line;
    std::string tmp;
    std::ifstream stream("/proc/" + std::to_string(pid) + "/status");
    if(stream.is_open()) {
        while(search == true && stream.peek() != EOF) {
            std::getline(stream, line);
            std::istringstream linestream(line);
            linestream >> tmp;
            if(tmp == "VmSize:") {
                linestream >> tmp;
                value = tmp;
                search = false;
            }
        }
    }
    return std::atoi(value.c_str());
}

//Section for NVML functions
nvmlReturn_t initializeNVML() {
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
        exit(1);
    }
    return result;
}

void cleanupNVML() {
    nvmlReturn_t result = nvmlShutdown();
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to shutdown NVML: " << nvmlErrorString(result) << std::endl;
    }
}

nvmlUtilization_st getGpuUtilization() {
    nvmlDevice_t device;
    nvmlReturn_t result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get device handle: " << nvmlErrorString(result) << std::endl;
        exit(1);
    }

    nvmlUtilization_st utilization;
    result = nvmlDeviceGetUtilizationRates(device, &utilization);
//    unsigned int processes;
//    nvmlProcessUtilizationSample_st utilizationSt;
//    nvmlDeviceGetProcessUtilization(device, &utilizationSt, &processes, 0);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get GPU utilization: " << nvmlErrorString(result) << std::endl;
        exit(1);
    }

    return utilization;
}

sysStats getSystemInfo(int pid) {
    sysStats info;
    info.cpuUsage = getCurrentCpuUsage(pid);
    info.memoryUsage = getCurrentMemoryUsage(pid) / 1000; // Convert to MB
    nvmlUtilization_st gpuUtilization = getGpuUtilization();
    info.gpuUtilization = gpuUtilization.gpu;
    info.gpuMemoryUsage = gpuUtilization.memory;
    return info;
}

int main() {
    int pid;
    initializeNVML();
    std::time_t currentTime;
    sysStats systemInfo;
    std::cin >> pid;
    while (true) {
        currentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

        systemInfo = getSystemInfo(pid);

        // Output statistics to stdout
        std::cout << "Timestamp: " << std::put_time(std::localtime(&currentTime), "%H:%M:%S") << " | ";
        std::cout << "CPU Usage: " << systemInfo.cpuUsage << " % | ";
        std::cout << "Memory Usage: " << systemInfo.memoryUsage << " MB | ";
        std::cout << "GPU Utilization: " << systemInfo.gpuUtilization << " % | ";
        std::cout << "GPU Memory Usage: " << systemInfo.gpuMemoryUsage << " % " << std::endl;

        usleep(1000000);
    }

    cleanupNVML();
    return 0;
}
