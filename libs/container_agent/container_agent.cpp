#include "container_agent.h"

ABSL_FLAG(std::string, name, "", "base name of container");
ABSL_FLAG(std::optional<std::string>, json, std::nullopt, "configurations for microservices as json");
ABSL_FLAG(std::optional<std::string>, json_path, std::nullopt, "json for configuration inside a file");
ABSL_FLAG(std::optional<std::string>, trt_json, std::nullopt, "optional json for TRTConfiguration");
ABSL_FLAG(std::optional<std::string>, trt_json_path, std::nullopt, "json for TRTConfiguration");
ABSL_FLAG(uint16_t, port, 0, "server port for the service");
ABSL_FLAG(int16_t, device, 0, "Index of GPU device");
ABSL_FLAG(uint16_t, verbose, 2, "verbose level 0:trace, 1:debug, 2:info, 3:warn, 4:error, 5:critical, 6:off");
ABSL_FLAG(std::string, log_dir, "../logs", "Log path for the container");
ABSL_FLAG(bool, profiling_mode, false, "flag to make the model running in profiling mode.");


void addProfileConfigs(json &msvcConfigs, const json &profileConfigs) {
    msvcConfigs["profile_numWarmUpBatches"] = profileConfigs.at("profile_numWarmUpBatches");
    msvcConfigs["profile_numProfileBatches"] = profileConfigs.at("profile_numProfileBatches");
    msvcConfigs["profile_inputRandomizeScheme"] = profileConfigs.at("profile_inputRandomizeScheme");
    msvcConfigs["profile_stepMode"] = profileConfigs.at("profile_stepMode");
    msvcConfigs["profile_step"] = profileConfigs.at("profile_step");
}

contRunArgs loadRunArgs(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);

    std::string name = absl::GetFlag(FLAGS_name);
    int8_t device = absl::GetFlag(FLAGS_device);
    uint16_t logLevel = absl::GetFlag(FLAGS_verbose);
    std::string logPath = absl::GetFlag(FLAGS_log_dir);
    bool profiling_mode = absl::GetFlag(FLAGS_profiling_mode);

    RUNMODE runmode = profiling_mode ? RUNMODE::PROFILING : RUNMODE::DEPLOYMENT;

    spdlog::set_pattern("[%C-%m-%d %H:%M:%S.%f] [%l] %v");
    spdlog::set_level(spdlog::level::level_enum(logLevel));

    std::tuple<json, json> configs = msvcconfigs::loadJson();
    json pipeConfigs = std::get<0>(configs);
    json profilingConfigs = std::get<1>(configs);

    BatchSizeType minBatch =  profilingConfigs.at("profile_minBatch");
    std::string templateModelPath = profilingConfigs.at("profile_templateModelPath");

    /**
     * @brief     If this is profiling, set configurations to the first batch size that should be profiled
     * This includes
     * 1. Setting its name based on the template model path    
     * 2. Setting the batch size to the smallest profile batch size
     * 
     */
    if (profiling_mode) {
    
        name = removeSubstring(templateModelPath, ".engine");
        name = replaceSubstring(name, "[batch]", std::to_string(minBatch));
        name = splitString(name, '/').back();
        logPath = "../model_profiles";
    }

    for (auto i = 0; i < pipeConfigs.size(); i++) {
        pipeConfigs[i]["msvc_contName"] = name;
        pipeConfigs[i]["msvc_deviceIndex"] = device;
        pipeConfigs[i]["msvc_containerLogPath"] = logPath + "/" + name;
        pipeConfigs[i]["msvc_RUNMODE"] = runmode;

        /**
         * @brief     If this is profiling, set configurations to the first batch size that should be profiled
         * This includes
         * 1. Setting its profile dir whose name is based on the template model path    
         * 2. Setting the batch size to the smallest profile batch size
         * 
         */
        if (profiling_mode) {
            pipeConfigs[i].at("msvc_idealBatchSize") = minBatch;
            if (i == 0) {
                addProfileConfigs(pipeConfigs[i], profilingConfigs);
            } else if (i == 2) {
                // Set the path to the engine
                pipeConfigs[i].at("path") = replaceSubstring(templateModelPath, "[batch]", std::to_string(minBatch));
            }
        }
    }

    checkCudaErrorCode(cudaSetDevice(device), __func__);

    return {name, absl::GetFlag(FLAGS_port), device, logPath, runmode, pipeConfigs, profilingConfigs};
};

std::vector<BaseMicroserviceConfigs> msvcconfigs::LoadFromJson() {
    if (!absl::GetFlag(FLAGS_json).has_value()) {
        spdlog::trace("{0:s} attempts to parse Microservice Configs from command line.", __func__);
        if (absl::GetFlag(FLAGS_json_path).has_value()) {
            std::ifstream file(absl::GetFlag(FLAGS_json_path).value());
            spdlog::trace("{0:s} finished parsing Microservice Configs from command line.", __func__);
            return json::parse(file).get<std::vector<BaseMicroserviceConfigs>>();
        } else {
            spdlog::error("No Configurations found. Please provide configuration either as json or file.");
            exit(1);
        }
    } else {
        spdlog::trace("{0:s} attempts to parse Microservice Configs from file.", __func__);
        if (absl::GetFlag(FLAGS_json_path).has_value()) {
            spdlog::error("No Configurations found. Please provide configuration either as json or file.");
            exit(1);
        } else {
            spdlog::trace("{0:s} finished parsing Microservice Configs from file.", __func__);
            return json::parse(absl::GetFlag(FLAGS_json).value()).get<std::vector<BaseMicroserviceConfigs>>();
        }
    }
}

std::tuple<json, json> msvcconfigs::loadJson() {
    json pipeConfigs, profilingConfigs;
    if (!absl::GetFlag(FLAGS_json).has_value()) {
        spdlog::trace("{0:s} attempts to load Json Configs from command line.", __func__);
        if (absl::GetFlag(FLAGS_json_path).has_value()) {
            std::ifstream file(absl::GetFlag(FLAGS_json_path).value());
            auto json_file = json::parse(file);
            pipeConfigs = json_file.at("pipeline");
            try {
                profilingConfigs = json_file.at("profiling");
            } catch (json::out_of_range &e) {
                spdlog::trace("{0:s} No profiling configurations found.", __func__);
            } catch (json::parse_error &e) {
                spdlog::error("{0:s} Error parsing json file.", __func__);
            }
            spdlog::trace("{0:s} finished loading Json Configs from command line.", __func__);
            return std::make_tuple(pipeConfigs, profilingConfigs);
        } else {
            spdlog::error("No Configurations found. Please provide configuration either as json or file.");
            exit(1);
        }
    } else {
        spdlog::trace("{0:s} attempts to load Json Configs from file.", __func__);
        if (absl::GetFlag(FLAGS_json_path).has_value()) {
            spdlog::error("No Configurations found. Please provide configuration either as json or file.");
            exit(1);
        } else {
            spdlog::trace("{0:s} finished loading Json Configs from file.", __func__);
            std::ifstream file(absl::GetFlag(FLAGS_json_path).value());
            auto json_file = json::parse(file);
            pipeConfigs = json_file.at("pipeline");
            try {
                profilingConfigs = json_file.at("profiling");
            } catch (json::out_of_range &e) {
                spdlog::trace("{0:s} No profiling configurations found.", __func__);
            }
            spdlog::trace("{0:s} finished loading Json Configs from command line.", __func__);
            return std::make_tuple(pipeConfigs, profilingConfigs);
        }
    }
}

void ContainerAgent::profiling(const json &pipeConfigs, const json &profileConfigs) {

    json pipelineConfigs = pipeConfigs;

    BatchSizeType minBatch =  profileConfigs.at("profile_minBatch");
    BatchSizeType maxBatch =  profileConfigs.at("profile_maxBatch");
    uint8_t stepMode = profileConfigs.at("profile_stepMode");
    uint8_t step = profileConfigs.at("profile_step");
    std::string templateModelPath = profileConfigs.at("profile_templateModelPath");

    this->dispatchMicroservices();

    for (BatchSizeType batch = minBatch; batch <= maxBatch;) {
        if (batch != minBatch) {
            // cudaDeviceReset();
            // checkCudaErrorCode(cudaSetDevice(cont_deviceIndex), __func__);
            // cont_deviceIndex = ++cont_deviceIndex % 4;

            std::string profileDirPath, name;
            
            name = removeSubstring(templateModelPath, ".engine");
            name = replaceSubstring(name, "[batch]", std::to_string(batch));
            name = splitString(name, '/').back();

            profileDirPath = cont_logDir + "/" + name;
            std::filesystem::create_directory(
                std::filesystem::path(profileDirPath)
            );
            
            // Making sure all the microservices are paused before reloading and reallocating resources
            // this is essential to avoiding runtime memory errors
            for (uint8_t i = 0; i < pipelineConfigs.size(); i++) {
                msvcs[i]->pauseThread();
            }
            waitPause();

            // Reload the configurations and dynamic allocation based on the new configurations
            for (uint8_t i = 0; i < pipelineConfigs.size(); i++) {
                pipelineConfigs[i].at("msvc_idealBatchSize") = batch;
                pipelineConfigs[i].at("msvc_containerLogPath") = profileDirPath;
                pipelineConfigs[i].at("msvc_deviceIndex") = cont_deviceIndex;
                pipelineConfigs[i].at("msvc_contName") = name;
                // Set the path to the engine
                if (i == 2) {
                    pipelineConfigs[i].at("path") = replaceSubstring(templateModelPath, "[batch]", std::to_string(batch));
                }
                msvcs[i]->loadConfigs(pipelineConfigs[i], false);
                msvcs[i]->setRELOAD();
            }

        }

        this->waitReady();
        this->PROFILING_START(batch);


        while (true) {
            spdlog::info("{0:s} waiting for profiling of model with a max batch of {1:d}.", __func__, batch);
            if (msvcs[0]->checkPause()) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }

        // spdlog::info("========================= Batch {0:d}/{1:d} =====================", );
        spdlog::info("====================================================================================================");
        if (stepMode == 0) {
            batch += step;
        } else {
            batch *= 2;
        }
    }
}

ContainerAgent::ContainerAgent(
    const std::string &name,
    uint16_t own_port,
    int8_t devIndex,
    const std::string &logPath,
    RUNMODE runmode,
    const json &profiling_configs
) : ContainerAgent(name, own_port, devIndex, logPath) {

    cont_RUNMODE = runmode;
    cont_deviceIndex = devIndex;

    if (cont_RUNMODE == RUNMODE::PROFILING) {
        // Create the logDir for this container
        cont_logDir = logPath + "/" + name;
        std::filesystem::create_directory(
            std::filesystem::path(cont_logDir)
        );
    } else {
        cont_logDir = logPath;
    }
}

ContainerAgent::ContainerAgent(
    const std::string &name,
    uint16_t own_port,
    int8_t devIndex,
    const std::string &logPath
) : name(name) {
    arrivalRate = 0;

    std::string server_address = absl::StrFormat("%s:%d", "localhost", own_port);
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    server_cq = builder.AddCompletionQueue();
    server = builder.BuildAndStart();

    stub = InDeviceCommunication::NewStub(grpc::CreateChannel("localhost:60003", grpc::InsecureChannelCredentials()));
    sender_cq = new CompletionQueue();

    run = true;
    std::thread receiver(&ContainerAgent::HandleRecvRpcs, this);
    receiver.detach();
}

void ContainerAgent::ReportStart() {
    ProcessData request;
    request.set_msvc_name(name);
    request.set_pid(getpid());
    EmptyMessage reply;
    ClientContext context;
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            stub->AsyncReportMsvcStart(&context, request, sender_cq));
    Status status;
    rpc->Finish(&reply, &status, (void *) 1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(sender_cq->Next(&got_tag, &ok));
    GPR_ASSERT(ok);
}

void ContainerAgent::SendState() {
    State request;
    request.set_name(name);
    request.set_arrival_rate(arrivalRate);
    for (auto msvc: msvcs) {
        request.add_queue_size(msvc->GetOutQueueSize(0));
        spdlog::info("{0:s} Length of queue is {1:d}", msvc->msvc_name, msvc->GetOutQueueSize(0));
    }
    EmptyMessage reply;
    ClientContext context;
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            stub->AsyncSendState(&context, request, sender_cq));
    Status status;
    rpc->Finish(&reply, &status, (void *) 1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(sender_cq->Next(&got_tag, &ok));
    GPR_ASSERT(ok);
}

void ContainerAgent::HandleRecvRpcs() {
    new StopRequestHandler(&service, server_cq.get(), &run);
    void *tag;
    bool ok;
    while (run) {
        GPR_ASSERT(server_cq->Next(&tag, &ok));
        GPR_ASSERT(ok);
        static_cast<RequestHandler *>(tag)->Proceed();
    }
}

void ContainerAgent::StopRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestStopExecution(&ctx, &request, &responder, cq, cq,
                                      this);
    } else if (status == PROCESS) {
        *run = false;
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void ContainerAgent::UpdateSenderRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestUpdateSender(&ctx, &request, &responder, cq, cq,
                                      this);
    } else if (status == PROCESS) {
        // TODO: Handle reconfiguration by restarting sender
        // pause processing except senders to clear out the queues

        // adjust json for configuration
//        json config = this->request;
        // stop the old sender
//        for (auto msvc : *msvcs) {
//            if (msvc->downstream[0].name == request.name()) {
//                msvc->stopThread();
//                msvcs->erase(std::remove(msvcs->begin(), msvcs->end(), msvc), msvcs->end());
//                break;
//            }
//        }
        if (request.ip() == "localhost") {
            // change postprocessing to keep the data on gpu

            // start new GPU sender
//            msvcs->push_back(new GPUSender(config));
        } else {
            // change postprocessing to offload data from gpu

            // start new serialized sender
//            msvcs->push_back(new RemoteCPUSender(config));
        }
        // align the data queue from postprocessor to new sender
//        msvcs->back()->SetInQueue(msvcs[3]->GetOutQueue());
        //start the new sender
//        msvcs->back()->startThread();

        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

/**
 * @brief Check if all the microservices are paused
 * 
 * @return true 
 * @return false 
 */
bool ContainerAgent::checkPause() {
    for (auto msvc : msvcs) {
        if (msvc->checkPause()) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Wait for all the microservices to be paused
 * 
 */
void ContainerAgent::waitPause() {
    bool paused = false;
    while (true) {
        paused = true;
        spdlog::trace("{0:s} waiting for all microservices to be paused.", __func__);
        for (auto msvc : msvcs) {
            if (!msvc->checkPause()) {
                paused = false;
                break;
            }
        }
        if (paused) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

}

/**
 * @brief Check if all the microservices are ready
 * 
 * @return true 
 * @return false 
 */
bool ContainerAgent::checkReady() {
    for (auto msvc : msvcs) {
        if (!msvc->checkReady()) {
            return true;
        }
    }
    return true;

}

/**
 * @brief Wait for all the microservices to be ready
 * 
 */
void ContainerAgent::waitReady() {
    ReportStart();
    bool ready = false;
    while (!ready) {
        ready = true;
        
        spdlog::info("{0:s} waiting for all microservices to be ready.", __func__);
        for (auto msvc : msvcs) {
            if (!msvc->checkReady()) {
                ready = false;
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
}
