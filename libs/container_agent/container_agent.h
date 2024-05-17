#ifndef CONTAINER_AGENT_H
#define CONTAINER_AGENT_H

#include <vector>
#include <thread>
#include <fstream>
#include "absl/strings/str_format.h"
#include "absl/flags/parse.h"
#include "absl/flags/flag.h"
#include <grpcpp/grpcpp.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/health_check_service_interface.h>
#include <google/protobuf/empty.pb.h>
#include <filesystem>
#include "receiver.h"
#include <pqxx/pqxx>

#include "microservice.h"
#include "sender.h"
#include "indevicecommunication.grpc.pb.h"

ABSL_DECLARE_FLAG(std::string, name);
ABSL_DECLARE_FLAG(std::optional<std::string>, json);
ABSL_DECLARE_FLAG(std::optional<std::string>, json_path);
ABSL_DECLARE_FLAG(std::optional<std::string>, trt_json);
ABSL_DECLARE_FLAG(std::optional<std::string>, trt_json_path);
ABSL_DECLARE_FLAG(uint16_t, port);
ABSL_DECLARE_FLAG(int16_t, device);
ABSL_DECLARE_FLAG(uint16_t, verbose);
ABSL_DECLARE_FLAG(std::string, log_dir);
ABSL_DECLARE_FLAG(bool, profiling_mode);

using json = nlohmann::ordered_json;

using grpc::Status;
using grpc::CompletionQueue;
using grpc::ClientContext;
using grpc::ClientAsyncResponseReader;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using indevicecommunication::InDeviceCommunication;
using indevicecommunication::State;
using indevicecommunication::Signal;
using indevicecommunication::Connection;
using indevicecommunication::ProcessData;
using EmptyMessage = google::protobuf::Empty;

struct MetricsServerConfigs {
    std::string ip = "localhost";
    uint64_t port = 60004;
    std::string DBName = "pipeline";
    std::string user = "container_agent";
    std::string password = "pipe";
    uint64_t scrapeIntervalMilisec = 60000;
};

enum TransferMethod {
    LocalCPU,
    RemoteCPU,
    GPU
};

namespace msvcconfigs {

    std::tuple<json, json> loadJson();

    std::vector<BaseMicroserviceConfigs> LoadFromJson();
}

struct contRunArgs {
    std::string cont_name;
    uint16_t cont_port;
    int8_t cont_devIndex;
    std::string cont_logPath;
    RUNMODE cont_runmode;
    json cont_pipeConfigs;
    json cont_profilingConfigs;
};

json loadRunArgs(int argc, char **argv);

void addProfileConfigs(json &msvcConfigs, const json &profileConfigs);

class ContainerAgent {
public:
    ContainerAgent(const json &configs);

    ~ContainerAgent() {
        for (auto msvc: cont_msvcsList) {
            delete msvc;
        }
        server->Shutdown();
        server_cq->Shutdown();
        sender_cq->Shutdown();
    };

    [[nodiscard]] bool running() const {
        return run;
    }

    void SendState();

    void START() {
        for (auto msvc: cont_msvcsList) {
            msvc->unpauseThread();
        }
        spdlog::info("=========================================== STARTS ===========================================");
    }

    void PROFILING_START(BatchSizeType batch) {
        for (auto msvc: cont_msvcsList) {
            msvc->unpauseThread();
        }

        spdlog::info(
                "======================================= PROFILING MODEL BATCH {0:d} =======================================",
                batch);
    }

    void waitReady();

    bool checkReady();

    void waitPause();

    bool checkPause();

    void addMicroservice(std::vector<Microservice *> msvcs) {
        this->cont_msvcsList = msvcs;
    }

    void dispatchMicroservices() {
        for (auto msvc: cont_msvcsList) {
            msvc->dispatchThread();
        }
    }

    void profiling(const json &pipeConfigs, const json &profileConfigs);

    void loadProfilingConfigs();

    void connectToMetricsServer();

protected:
    uint8_t deviceIndex = -1;

    void ReportStart();

    class RequestHandler {
    public:
        RequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq)
                : service(service), cq(cq), status(CREATE), responder(&ctx) {};

        virtual ~RequestHandler() = default;

        virtual void Proceed() = 0;

    protected:
        enum CallStatus {
            CREATE, PROCESS, FINISH
        };

        InDeviceCommunication::AsyncService *service;
        ServerCompletionQueue *cq;
        ServerContext ctx;
        CallStatus status;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
    };

    class StopRequestHandler : public RequestHandler {
    public:
        StopRequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq,
                           std::atomic<bool> *run)
                : RequestHandler(service, cq), run(run) {
            Proceed();
        }

        void Proceed() final;

    private:
        Signal request;
        std::atomic<bool> *run;
    };

    class UpdateSenderRequestHandler : public RequestHandler {
    public:
        UpdateSenderRequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                   std::vector<Microservice *> *msvcs)
                : RequestHandler(service, cq), msvcs(msvcs) {
            Proceed();
        }

        void Proceed() final;

    private:
        Connection request;
        std::vector<Microservice *> *msvcs;
    };

    class UpdateBatchSizeRequestHandler : public RequestHandler {
    public:
        UpdateBatchSizeRequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                      std::vector<Microservice *> *msvcs)
                : RequestHandler(service, cq), msvcs(msvcs) {
            Proceed();
        }

        void Proceed() final;

    private:
        indevicecommunication::BatchSize request;
        std::vector<Microservice *> *msvcs;
    };

    void HandleRecvRpcs();

    std::string cont_name;
    std::vector<Microservice *> cont_msvcsList;
    float arrivalRate;
    std::unique_ptr<ServerCompletionQueue> server_cq;
    CompletionQueue *sender_cq;
    InDeviceCommunication::AsyncService service;
    std::unique_ptr<grpc::Server> server;
    std::unique_ptr<InDeviceCommunication::Stub> stub;
    std::atomic<bool> run;

    std::string cont_logDir;
    RUNMODE cont_RUNMODE;
    uint8_t cont_deviceIndex;
    MetricsServerConfigs cont_metricsServerConfigs;
};

#endif //CONTAINER_AGENT_H