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
    std::cout << "Initializing Python..." << std::endl;
    Py_Initialize();
    if (!Py_IsInitialized()) {
        std::cerr << "Failed to initialize Python" << std::endl;
        return;
    }
    std::cout << "Python initialized." << std::endl;

    running = false;
    for (const auto &pid : pids) {
        stats[pid] = std::vector<sysStats>();
    }
    std::cout << "Profiler initialized with PIDs." << std::endl;
}

Profiler::~Profiler() {
    if (Py_IsInitialized()) {
        Py_Finalize();
        std::cout << "Python finalized." << std::endl;
    }
}

void Profiler::run() {
    std::cout << "Starting profiler..." << std::endl;
    running = true;
    std::thread collectingStats(&Profiler::collectStats, this);
    collectingStats.detach();
    std::cout << "Exitting collecting stats..." << std::endl;
}

void Profiler::addPid(unsigned int pid) {
    // stats[pid] = std::vector<sysStats>();
    std::cout << "Added PID: " << pid << std::endl;
}

int Profiler::getGpuCount() {
    return 1; 
}

std::vector<unsigned int> Profiler::getGpuMemory(int processing_units) {
    return std::vector<unsigned int>(); 
}

Profiler::sysStats Profiler::reportAtRuntime(unsigned int pid) {
    sysStats value{};
    if (!running) {
        value.timestamp = 1;
        return value;
    }
    if (stats.find(pid) != stats.end() && !stats[pid].empty()) {
        value = stats[pid].back();
    }
    return value;
}

void Profiler::collectStats() {
    std::cout << "Entering collectStats..." << std::endl;
    // const char* pythonScriptPath = "get_jetson_stats.py"; // Replace with actual path
    // std::string command = std::string("python3 ") + pythonScriptPath;
    // Start the Python script in a separate thread
    std::thread pythonThread(execCommandAsync, "python3 get_jetson_stats.py");
    std::cout << "Python thread started." << std::endl;

    while (running) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [] { return dataReady; }); // Wait until data is ready

        std::cout << "Data ready, processing..." << std::endl;

        for (auto &entry : stats) {
            unsigned int pid = entry.first;
            Profiler::sysStats *stat= new Profiler::sysStats;
            parseDelimited(sharedBuffer, stat);
            entry.second.push_back(*stat);
        }

        sharedBuffer.clear(); // Clear the buffer after processing
        dataReady = false; // Reset the flag
        std::cout << "Data processed." << std::endl;
    }

    pythonThread.join(); // Ensure the Python thread is finished
    std::cout << "Exiting collectStats..." << std::endl;
}

// Main function to interactively input PIDs
int main() {
    // Vector to store PIDs
    std::vector<unsigned int> pids;
    unsigned int pid;

    std::cout << "Enter PIDs to monitor (enter 0 to finish): " << std::endl;
    while (true) {
        std::cin >> pid;
        if (pid == 0) break;
        pids.push_back(pid);
    }

     // Create the profiler
    Profiler *profiler = new Profiler(pids);

    // Start the profiler on a separate thread
    std::thread profilerThread(&Profiler::run, profiler);

    // Monitor the stats
    while (true) {
        for (const auto& pid : pids) {
            Profiler::sysStats stats = profiler->reportAtRuntime(pid);
            std::cout << "PID: " << pid << "\n"
                    << "Timestamp: " << stats.timestamp << "\n"
                    << "CPU Usage: " << stats.cpuUtilization << "%\n"
                    << "Process RAM Usage: " << stats.processMemoryUsage << "MB\n"
                    << "Process GPU Memory Usage: " << stats.processGpuMemoryUsage << "MB\n"
                    << "GPU Load: " << stats.gpuUtilization << "%\n"
                    << "Total GPU RAM Usage: " << stats.totalGpuRamUsage << "MB\n"
                    << "Total CPU RAM Usage: " << stats.totalCpuRamUsage << "MB\n"
                    << "Max RAM: " << stats.maxRamCapacity << "MB\n"
                    << std::endl;
        }
        // Sleep for a short interval to avoid consuming too much CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    profilerThread.join();
    delete profiler;
    return 0;
}