#include "profiler.h"

// Shared buffer and synchronization primitives
std::mutex mtx;
std::condition_variable cv;
std::string sharedBuffer;
bool dataReady = false;

// Function to execute a command and capture the output asynchronously
void execCommandAsync(const char* cmd) {
    std::array<char, 128> buffer;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        std::string result(buffer.data());

        // Lock the mutex and update the shared buffer
        std::lock_guard<std::mutex> lock(mtx);
        sharedBuffer += result;
        dataReady = true;
        cv.notify_one(); // Notify the main thread
    }
}

// Function to parse the delimited output and populate sysStats
void parseDelimited(const std::string& data, Profiler::sysStats *stats) {
    std::cout << "Start parsing\n";
    std::istringstream ss(data);
    std::string line;

    while (std::getline(ss, line)) {
        std::istringstream lineStream(line);
        std::string token;
        std::vector<std::string> tokens;

        // Check if line contains GPU Load or RAM information
        if (line.find("GPU Load") != std::string::npos) {
            stats->gpuUtilization = std::stod(line.substr(line.find(":") + 1));
            continue;
        }

        if (line.find("RAM:") != std::string::npos) {
            std::string ramData = line.substr(line.find(":") + 1);
            std::istringstream ramStream(ramData);
            std::string ramToken;
            std::vector<std::string> ramTokens;
            while (std::getline(ramStream, ramToken, ',')) {
                ramTokens.push_back(ramToken);
            }
            for (const auto& token : ramTokens) {
                if (token.find("GPU") != std::string::npos) {
                    stats->totalGpuRamUsage = std::stod(token.substr(token.find("=") + 1));
                } else if (token.find("CPU") != std::string::npos) {
                    stats->totalCpuRamUsage = std::stod(token.substr(token.find("=") + 1));
                } else if (token.find("TOTAL") != std::string::npos) {
                    stats->maxRamCapacity = std::stod(token.substr(token.find("=") + 1));
                }
            }
            continue;
        }

        // Parse delimited tokens
        while (std::getline(lineStream, token, '|')) {
            tokens.push_back(token);
        }

        if (tokens.size() == 5) {
            stats->timestamp = std::stoull(tokens[0]);
            stats->cpuUtilization = std::stod(tokens[2]);
            stats->processMemoryUsage = std::stod(tokens[3]);
            stats->processGpuMemoryUsage = std::stod(tokens[4]);
        } else {
            std::cerr << "Unexpected token size: " << tokens.size() << std::endl;
        }
    }
}

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
    std::cout << "Starting profiler..." << std::endl;
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

profilerThread.join();
    delete profiler;
    return 0;
}