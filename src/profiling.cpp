#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <sys/sysinfo.h>
#include <nvml.h>

struct sysStats {
    double cpuUsage;
    double memoryUsage;
    int gpuUtilization;
    int gpuMemoryUsage;
    double maxGpuMemoryUsage;
    unsigned int pcieThroughput;
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

    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get device handle: " << nvmlErrorString(result) << std::endl;
        exit(1);
    }

    nvmlEnableState_t state;
    result = nvmlDeviceGetAccountingMode(device, &state);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get accounting mode: " << nvmlErrorString(result) << std::endl;
        exit(1);
    }
    if (state == NVML_FEATURE_DISABLED) {
        std::cerr << "Enabling accounting mode" << std::endl;
        result = nvmlDeviceSetAccountingMode(device, NVML_FEATURE_ENABLED);
        if (result != NVML_SUCCESS) {
            std::cerr << "Failed to enable accounting mode: " << nvmlErrorString(result) << std::endl;
            exit(1);
        }
    }

    return result;
}

void cleanupNVML() {
    nvmlDevice_t device;
    nvmlReturn_t result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get device handle: " << nvmlErrorString(result) << std::endl;
        exit(1);
    }

    nvmlEnableState_t state;
    result = nvmlDeviceGetAccountingMode(device, &state);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get accounting mode: " << nvmlErrorString(result) << std::endl;
        exit(1);
    }
    if (state == NVML_FEATURE_ENABLED) {
        std::cerr << "Disabling accounting mode" << std::endl;
        result = nvmlDeviceSetAccountingMode(device, NVML_FEATURE_DISABLED);
        if (result != NVML_SUCCESS) {
            std::cerr << "Failed to disable accounting mode: " << nvmlErrorString(result) << std::endl;
            exit(1);
        }
    }

    result = nvmlShutdown();
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to shutdown NVML: " << nvmlErrorString(result) << std::endl;
    }
}

std::pair<nvmlAccountingStats_t, unsigned int> getGpuUtilization(int pid) {
    nvmlDevice_t device;
    nvmlReturn_t result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get device handle: " << nvmlErrorString(result) << std::endl;
        exit(1);
    }

    nvmlAccountingStats_t gpu;
    result = nvmlDeviceGetAccountingStats(device, pid, &gpu);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get GPU Accounting Stats: " << nvmlErrorString(result) << std::endl;
        gpu.gpuUtilization = 0;
        gpu.memoryUtilization = 0;
        gpu.maxMemoryUsage = 0;
    }
    unsigned int pcie;
    result = nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_COUNT, &pcie);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get PCIe throughput: " << nvmlErrorString(result) << std::endl;
        pcie = 0;
    }

    return std::pair<nvmlAccountingStats_t, unsigned int>(gpu, pcie);
}

sysStats getSystemInfo(int pid) {
    sysStats info;
    info.cpuUsage = getCurrentCpuUsage(pid);
    info.memoryUsage = getCurrentMemoryUsage(pid) / 1000; // Convert to MB
    std::pair<nvmlAccountingStats_t, unsigned int> gpu = getGpuUtilization(pid);
    info.gpuUtilization = gpu.first.gpuUtilization;
    info.gpuMemoryUsage = gpu.first.memoryUtilization;
    info.maxGpuMemoryUsage = gpu.first.maxMemoryUsage / 1000000; // Convert to MB
    info.pcieThroughput = gpu.second / 1000; // Convert to MB/s
    return info;
}

void mainLoop(bool &running, int pid) {
    std::time_t currentTime;
    sysStats systemInfo;
    while (running) {
        currentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        systemInfo = getSystemInfo(pid);

        std::cout << "Timestamp: " << std::put_time(std::localtime(&currentTime), "%H:%M:%S") << " | ";
        std::cout << "CPU Usage: " << systemInfo.cpuUsage << " % | ";
        std::cout << "Memory Usage: " << systemInfo.memoryUsage << " MB | ";
        std::cout << "GPU Utilization: " << systemInfo.gpuUtilization << " % | ";
        std::cout << "GPU Memory Usage: " << systemInfo.gpuMemoryUsage << " % ";
        std::cout << "Max GPU Memory Usage: " << systemInfo.maxGpuMemoryUsage << " MB | ";
        std::cout << "PCIe Throughput: " << systemInfo.pcieThroughput << " MB/s" << std::endl;

        sleep(1);
    }
    return;
}

int main() {
    initializeNVML();
    std::cout << "Enter PID of process to monitor: ";
    int pid;
    std::cin >> pid;
    std::cout << "Monitoring process with PID: " << pid << " Press q + Enter to exit." << std::endl;
    bool running = true;
    std::thread mainThread(mainLoop, std::ref(running), pid);
    char input;
    while (running) {
        std::cin >> input;
        if (input == 'q') {
            running = false;
        }
    }
    std::cout << "Exiting..." << std::endl;
    mainThread.join();

    cleanupNVML();
    return 0;
}
