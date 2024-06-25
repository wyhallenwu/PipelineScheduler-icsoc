#include "profiler.h"

Profiler::Profiler(const std::vector<unsigned int> &pids, std::string mode) {
    if (!pids.empty()) {
        int pid = pids[0];
        std::string command = "python3 ../jetson_profiler.py " + mode + " " + std::to_string(pid);
        t = std::thread(&Profiler::jtop, this, command);
        t.detach();
    }
}

Profiler::~Profiler() {
    if (t.joinable()) {
        t.join();
    }
}

void Profiler::jtop(const std::string &cmd) {
    std::array<char, 128> buffer;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    std::vector<std::string> result;
    std::string token;
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        std::stringstream ss(buffer.data());
        while (std::getline(ss, token, '|')) {
            result.push_back(token);
        }
        std::lock_guard<std::mutex> lock(m);
        stats[std::stoi(result[0])] = {std::stoi(result[1]), std::stoi(result[2]), std::stoi(result[5]), std::stoi(result[4]),
                 std::stoi(result[3])};
        m.unlock();
        result = {};
    }
}

