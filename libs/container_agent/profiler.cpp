#include "profiler.h"

// Function to execute a command and capture the output asynchronously
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
        stats = {0, std::stoi(result[0]), std::stoi(result[1]), std::stoi(result[1]), std::stoi(result[3]),
                 std::stoi(result[2])};
        m.unlock();
        result = {};
    }
}

Profiler::Profiler(const std::vector<unsigned int> &pids) {
//    Py_Initialize();
//    if (!Py_IsInitialized()) {
//        std::cerr << "Failed to initialize Python" << std::endl;
//        return;
//    }
//    PyRun_SimpleString("import sys");
//    PyRun_SimpleString("sys.path.append(\"..\")");
//    std::cout << "Python initialized." << std::endl;
//    PyObject *pName = PyUnicode_DecodeFSDefault("jetson_profiler");
//    pModule = PyImport_Import(pName);
//    Py_DECREF(pName);
//    if (pModule != nullptr) {
//        // Get the function from the module
//        pFunc = PyObject_GetAttrString(pModule, "get_jetson_stats");
//        if (pFunc && PyCallable_Check(pFunc)) {
//            std::cout << "Python module loaded." << std::endl;
//        } else {
//            if (PyErr_Occurred()) PyErr_Print();
//            std::cerr << "Cannot find function 'get_jetson_stats'" << std::endl;
//            Py_DECREF(pModule);
//            Py_Finalize();
//        }
//    }

    if (!pids.empty()) {
        int pid = pids[0];
        std::string command = "python3 ../jetson_profiler.py " + std::to_string(pid);
        t = std::thread(&Profiler::jtop, this, command);
        t.detach();
    }
}

Profiler::~Profiler() {
//    if (Py_IsInitialized()) return;
//    Py_XDECREF(pFunc);
//    Py_DECREF(pModule);
//    Py_Finalize();
    t.join();
}