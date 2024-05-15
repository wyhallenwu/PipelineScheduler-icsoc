#include "profiler.h"

Profiler::Profiler(const std::vector<unsigned int> &pids) {
    // Initialize Python and jtop
    Py_Initialize();
    initializeJtop();
    running = false;
}

Profiler::~Profiler() {
    Py_XDECREF(pFunc);
    Py_DECREF(pModule);
    Py_FinalizeEx();
}

void Profiler::initializeJtop() {
    pName = PyUnicode_DecodeFSDefault("jtop");
    pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, "jtop");
        if (pFunc && PyCallable_Check(pFunc)) {
            pArgs = PyTuple_New(0);
            pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);
            if (pValue != NULL) {
                std::cout << "jtop object created" << std::endl;
            } else {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                std::cerr << "Call failed" << std::endl;
            }
        } else {
            if (PyErr_Occurred())
                PyErr_Print();
            std::cerr << "Cannot find function" << std::endl;
        }
    } else {
        PyErr_Print();
        std::cerr << "Failed to load module" << std::endl;
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

void addPid(unsigned int pid){
    
}

std::vector<Profiler::sysStats> Profiler::getStats(unsigned int pid) const {
    if (stats.count(pid) > 0) {
        return stats.at(pid);
    }
    return {};
}

std::vector<Profiler::sysStats> Profiler::popStats(unsigned int pid) {
    if (stats.count(pid) > 0) {
        std::vector<Profiler::sysStats> statsCopy = stats[pid];
        stats[pid].clear();
        return statsCopy;
    }
    return {};
}

Profiler::sysStats Profiler::reportAtRuntime(unsigned int pid) {
    if (!running || stats.empty() || stats[0].empty()) {
        return sysStats{};
    }
    return stats[0].back();
}

void Profiler::collectStats() {
    while (running) {
        for (const auto &[pid, _] : stats) {
            sysStats stats = getStatsJetson(pid);
            this->stats[pid].push_back(stats);
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

Profiler::sysStats Profiler::getStatsJetson(unsigned int pid) {
    sysStats stats{};
    stats.timestamp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    PyObject* pStats = PyObject_GetAttrString(pValue, "stats");
    if (pStats && PyDict_Check(pStats)) {
        PyObject* pCPU = PyDict_GetItemString(pStats, "CPU");
        PyObject* pGPU = PyDict_GetItemString(pStats, "GPU");
        PyObject* pRAM = PyDict_GetItemString(pStats, "RAM");

        if (pCPU && PyDict_Check(pCPU)) {
            PyObject* pCPUVal = PyDict_GetItemString(pCPU, "val");
            if (pCPUVal && PyLong_Check(pCPUVal)) {
                stats.cpuUsage = PyLong_AsLong(pCPUVal);
            }
        }

        if (pGPU && PyDict_Check(pGPU)) {
            PyObject* pGPUVal = PyDict_GetItemString(pGPU, "val");
            if (pGPUVal && PyLong_Check(pGPUVal)) {
                stats.gpuUtilization = PyLong_AsLong(pGPUVal);
            }
        }
    }

    // Get memory usage for the specific PID
    stats.memoryUsage = getMemoryUsageForPID(pid);
    stats.gpuMemoryUsage = stats.memoryUsage;

    return stats;
}

long Profiler::getMemoryUsageForPID(unsigned int pid) {
    std::string value = "0";
    bool search = true;
    std::string line;
    std::string tmp;
    std::ifstream stream("/proc/" + std::to_string(pid) + "/status");
    if (stream.is_open()) {
        while (search && stream.peek() != EOF) {
            std::getline(stream, line);
            std::istringstream linestream(line);
            linestream >> tmp;
            if (tmp == "VmSize:") {
                linestream >> tmp;
                value = tmp;
                search = false;
            }
        }
    }
    return std::atol(value.c_str());
}

int Profiler::getGpuCount() {
    return 1;
}

uint64_t Profiler::getGpuMemory(int device_count) {
    uint64_t totalMemory = 0;
    PyObject* pStats = PyObject_GetAttrString(pValue, "stats");
    if (pStats && PyDict_Check(pStats)) {
        PyObject* pRAM = PyDict_GetItemString(pStats, "RAM");
        if (pRAM && PyDict_Check(pRAM)) {
            PyObject* pRAMTotal = PyDict_GetItemString(pRAM, "total");
            if (pRAMTotal && PyLong_Check(pRAMTotal)) {
                totalMemory += PyLong_AsUnsignedLongLong(pRAMTotal);
            }
        }
    }
    return totalMemory;
}

int main() {
    std::cout << "Enter PID of process to monitor: ";
    int pid;
    std::cin >> pid;
    Profiler *profiler = new Profiler({pid});
    
    while (true) {
        Profiler::sysStats stats = profiler->reportAtRuntime(pid);
        std::cout << stats.cpuUsage << " | " << stats.memoryUsage << " | " << stats.gpuUtilization << " | " << stats.gpuMemoryUsage << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    return 0;
}
