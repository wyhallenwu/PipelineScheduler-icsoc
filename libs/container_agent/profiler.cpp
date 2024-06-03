#include "profiler.h"

Profiler::Profiler(const std::vector<unsigned int> &pids) {
    if (!initializeNVML()) {
        std::cerr << "Failed to initialize NVML" << std::endl;
        return;
    }
    pidOnDevices = std::map<unsigned int, nvmlDevice_t>();
    if (!pids.empty()) {
        auto devices = getDevices();
        for (const auto &pid: pids) {
            setPidOnDevices(pid, devices);
        }
    }
    nvmlInitialized = true;
    running = false;
}

Profiler::~Profiler() {

    if (nvmlInitialized) {
        if (!cleanupNVML()) {
            std::cerr << "Failed to shutdown NVML" << std::endl;
        }
    }
}

void Profiler::run() {
    running = true;
    profilerThread = std::thread(&Profiler::collectStats, this);
    profilerThread.detach();
}


void Profiler::stop() {
    running = false;
    profilerThread.join();
}

void Profiler::updatePids(const std::vector<unsigned int> &pids) {
    bool restart = false;
    if (running) {
        stop();
        restart = true;
    }
    pidOnDevices.clear();
    stats.clear();
    auto devices = getDevices();
    for (const auto &pid: pids) {
        setPidOnDevices(pid, devices);
    }
    if (restart) {
        run();
    }
}

void Profiler::addPid(unsigned int pid) {
    setPidOnDevices(pid, getDevices());
}

void Profiler::removePid(unsigned int pid) {
    pidOnDevices.erase(pid);
    stats.erase(pid);
}

int Profiler::getGpuCount() {
    unsigned int device_count;
    nvmlReturn_t result = nvmlDeviceGetCount(&device_count);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get device count: " << nvmlErrorString(result) << std::endl;
        return -1;
    }
    return device_count;
}

std::vector<long> Profiler::getGpuMemory(int device_count) {
    std::vector<long> totalMemory;
    for (int i = 0; i < device_count; i++) {
        nvmlDevice_t device;
        nvmlReturn_t result = nvmlDeviceGetHandleByIndex(i, &device);
        if (result != NVML_SUCCESS) {
            std::cerr << "Failed to get handle for device " << i << ": " << nvmlErrorString(result) << std::endl;
            return {-1};
        }
        nvmlMemory_t memory;
        result = nvmlDeviceGetMemoryInfo(device, &memory);
        if (result != NVML_SUCCESS) {
            std::cerr << "Failed to get memory info for device " << i << ": " << nvmlErrorString(result) << std::endl;
            return {-1};
        }
        totalMemory.push_back((long) memory.total / 1000000); // convert to MB
    }
    return totalMemory;
}

std::vector<Profiler::sysStats> Profiler::getStats(unsigned int pid) const {
    return stats.at(pid);
}

std::vector<Profiler::sysStats> Profiler::popStats(unsigned int pid) {
    std::vector<Profiler::sysStats> statsCopy = stats[pid];
    stats[pid] = std::vector<sysStats>();
    return statsCopy;
}

Profiler::sysStats Profiler::reportAtRuntime(unsigned int cpu_pid, unsigned int gpu_pid) {
    sysStats value{};
    value.cpuUsage = getCPUInfo(cpu_pid);
    value.memoryUsage = getMemoryInfo(cpu_pid) / 1000; // convert to MB
    auto gpu = getGPUInfo(gpu_pid, pidOnDevices[gpu_pid]);
    value.gpuUtilization = gpu.gpuUtilization;
    value.gpuMemoryUsage = gpu.memoryUtilization;
    return value;
}

void Profiler::collectStats() {
    uint64_t currentTime;
    while (running) {
        for (const auto &[pid, device]: pidOnDevices) {
            currentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            auto cpu = getCPUInfo(pid);
            auto memory = getMemoryInfo(pid);
            auto gpu = getGPUInfo(pid, device);
            auto pcie = getPcieInfo(device);
            sysStats systemInfo{
                    currentTime,
                    cpu,
                    memory / 1000, // convert to MB
                    gpu.gpuUtilization,
                    gpu.memoryUtilization,
                    (long) gpu.maxMemoryUsage / 1000000, // convert to MB
                    pcie / 1000 // convert to MB/s
            };
            stats[pid].push_back(systemInfo);
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

bool Profiler::initializeNVML() {
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
        return false;
    }
    return true;
}

bool Profiler::setAccounting(nvmlDevice_t device) {
    nvmlEnableState_t state;
    nvmlReturn_t result = nvmlDeviceGetAccountingMode(device, &state);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get accounting mode: " << nvmlErrorString(result) << std::endl;
        return false;
    }
    if (state == NVML_FEATURE_DISABLED) {
        result = nvmlDeviceSetAccountingMode(device, NVML_FEATURE_ENABLED);
        if (result != NVML_SUCCESS) {
            std::cerr << "Failed to enable accounting mode: " << nvmlErrorString(result) << std::endl;
            return false;
        }
    }
    return true;
}

std::vector<nvmlDevice_t> Profiler::getDevices() {
    std::vector<nvmlDevice_t> devices;
    unsigned int deviceCount;
    nvmlReturn_t result = nvmlDeviceGetCount(&deviceCount);
    if (NVML_SUCCESS != result) {
        std::cerr << "Failed to query device count: " << nvmlErrorString(result) << std::endl;
        return std::vector<nvmlDevice_t> ();
    }
    for (unsigned int i = 0; i < deviceCount; i++) {
        nvmlDevice_t device;
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if (NVML_SUCCESS != result) {
            std::cerr << "Failed to get handle for device " << i << ": " << nvmlErrorString(result) << std::endl;
            return std::vector<nvmlDevice_t>();
        }
        setAccounting(device);
        devices.push_back(device);
    }
    return devices;
}

void Profiler::setPidOnDevices(unsigned int pid, std::vector<nvmlDevice_t> devices) {
    for (const auto &device: devices) {
        nvmlProcessInfo_t processes[64];
        unsigned int infoCount = 64;
        nvmlReturn_t result = nvmlDeviceGetComputeRunningProcesses(device, &infoCount, processes);
        if (NVML_SUCCESS != result) {
            std::cerr << "Failed to get compute running processes for a device: " << nvmlErrorString(result) << std::endl;
            break;
        }
        for (unsigned int j = 0; j < infoCount; j++) {
            if (processes[j].pid == pid) {
                pidOnDevices[pid] = device;
                stats[pid] = std::vector<sysStats>();
                break;
            }
        }
    }
}

bool Profiler::cleanupNVML() {
    for (const auto &[pid, device]: pidOnDevices) {
        nvmlReturn_t result = nvmlDeviceClearAccountingPids(device);
        if (result != NVML_SUCCESS) {
            std::cerr << "Failed to clear accounting PIDs: " << nvmlErrorString(result) << std::endl;
            return false;
        }
    }
    return true;
}

double Profiler::getCPUInfo(unsigned int pid) {
    std::vector<std::string> timers;
    std::string timer, line, skip, utime, stime;
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
    long total_active = 0;
    for(unsigned int i = 0; i < timers.size(); ++i) {
        if(i != 3 && i != 4) total_active += std::stol(timers[i]);
    }
    stream = std::ifstream("/proc/"+ std::to_string(pid) + "/stat");
    if (stream.is_open()) {
        std::getline(stream, line);
        std::istringstream linestream(line);
        for(int i = 0; i < 13; ++i) linestream >> skip;
        linestream >> utime >> stime;
    }
    long process_active = 0;
    try {
        process_active = std::stol(utime) + std::stol(stime);
    } catch (const std::invalid_argument& ia) {
    }

    double cpuUsage = 0.0;
    if (prevCpuTimes.count(pid) > 0) {
        long prev_process_active = prevCpuTimes[pid].first;
        long prev_total_active = prevCpuTimes[pid].second;
        cpuUsage = 100.0 * (process_active - prev_process_active) / (total_active - prev_total_active);
    }
    prevCpuTimes[pid] = std::make_pair(process_active, total_active);

    return cpuUsage;
}

long Profiler::getMemoryInfo(unsigned int pid) {
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

nvmlAccountingStats_t Profiler::getGPUInfo(unsigned int pid, nvmlDevice_t device) {
    nvmlAccountingStats_t gpu;
    nvmlReturn_t result = nvmlDeviceGetAccountingStats(device, pid, &gpu);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get GPU Accounting Stats: " << nvmlErrorString(result) << std::endl;
        gpu.gpuUtilization = 0;
        gpu.memoryUtilization = 0;
        gpu.maxMemoryUsage = 0;
    }
    return gpu;
}

unsigned int Profiler::getPcieInfo(nvmlDevice_t device) {
    unsigned int pcie;
    nvmlReturn_t result = nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_COUNT, &pcie);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get PCIe throughput: " << nvmlErrorString(result) << std::endl;
        pcie = 0;
    }
    return pcie;
}
