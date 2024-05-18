#include "profilegenerator.h"

using namespace spdlog;

void ProfileGenerator::loadConfigs(const json &jsonConfigs, bool isConstructing) {
    spdlog::trace("{0:s} is LOADING configs...", __func__);
    if (!isConstructing) {
        Receiver::loadConfigs(jsonConfigs, isConstructing);
    }
    
    msvc_numWarmUpBatches = jsonConfigs.at("profile_numWarmUpBatches");
    msvc_numProfileBatches = jsonConfigs.at("profile_numProfileBatches");
    msvc_inputRandomizeScheme = jsonConfigs.at("profile_inputRandomizeScheme");
    msvc_stepMode = jsonConfigs.at("profile_stepMode");
    msvc_step = jsonConfigs.at("profile_step");
}

ProfileGenerator::ProfileGenerator(const json &jsonConfigs) : Receiver(jsonConfigs) {
    loadConfigs(jsonConfigs, true);
    info("{0:s} is created.", __func__);
}

void ProfileGenerator::profileDataGenerator() {
    msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);

    // Since we dont know the shape of data before hand, we would choose a few potential shapes and choose randomly amongst them
    // during profiling
    uint8_t randomShapeIndex;
    std::uniform_int_distribution<> dis(0, msvc_dataShape.size() - 1);
    std::mt19937 gen(2024);

    std::vector<RequestData<LocalCPUReqDataType>> requestData;
    RequestData<LocalCPUReqDataType> data;
    Request<LocalCPUReqDataType> request;
    RequestDataShapeType shape;
    cv::Mat img;
    std::string requestPath;
    if (msvc_OutQueue[0]->getActiveQueueIndex() != 1) msvc_OutQueue[0]->setActiveQueueIndex(1);
    msvc_OutQueue[0]->setQueueSize(1000);

    Request<LocalCPUReqDataType> startNextBatchReq;

    auto numWarmUpBatches = msvc_numWarmUpBatches;
    auto numProfileBatches = msvc_numProfileBatches;
    BatchSizeType batchSize = 1;
    BatchSizeType batchNum = 1;
    msvc_InQueue.at(0)->setActiveQueueIndex(1);

    int keepProfiling = 1;

    while (true) {
        if (this->STOP_THREADS) {
            spdlog::info("{0:s} STOPS.", msvc_name);
            break;
        }
        else if (this->PAUSE_THREADS) {
            std::cout << "";
            if (RELOADING) {
                setDevice();
                numWarmUpBatches = msvc_numWarmUpBatches;
                numProfileBatches = msvc_numProfileBatches;
                batchSize = 1;
                batchNum = 1;
                msvc_inReqCount = 0;
                keepProfiling = 1;
                spdlog::info("{0:s} is (RE)LOADED.", msvc_name);
                READY = true;
                RELOADING = false;
            }
            ///spdlog::info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }
        if (msvc_inReqCount > 0) {
            startNextBatchReq = msvc_InQueue.at(0)->pop1(500);
            // Meaning the the timeout in pop() has been reached and no request was actually popped
            if (strcmp(startNextBatchReq.req_travelPath[0].c_str(), "empty") == 0) {
                continue;
            }

            spdlog::info("Finished {0:s}.", startNextBatchReq.req_travelPath[0]);

            if (startNextBatchReq.req_data[0].data.at<uint8_t>(0,0) == 1) {
                keepProfiling = 1;
            } else {
                keepProfiling = 0;
                this->pauseThread();
            }
        }
        if (batchNum <= (numProfileBatches + numWarmUpBatches) && (keepProfiling == 1)) {
            /**
             * @brief Warming up to avoid cold start effects.
             * During warming up, we use inference `numBatches` batches of requests.
             * 
             */
            for (BatchSizeType i = 1; i <= batchSize; i++) { // Filling up the batch
                randomShapeIndex = dis(gen);
                shape = msvc_dataShape[randomShapeIndex];
                img = cv::Mat(shape[1], shape[2], CV_8UC3);
                data = {
                    shape,
                    img
                };
                requestData.emplace_back(data);
                // For bookkeeping, we add a certain pattern into the `requestPath` field.
                // [batchSize, batchNum, i]
                requestPath = std::to_string(msvc_idealBatchSize) + "," + std::to_string(batchSize) + "," + std::to_string(batchNum) + "," + std::to_string(i);

                // The very last batch of this profiling session is marked with "END" in the `requestPath` field.
                if ((batchNum == (numProfileBatches + numWarmUpBatches)) && (i == msvc_idealBatchSize)) {
                    requestPath = requestPath + "BATCH_ENDS";
                }

                if (batchNum <= numWarmUpBatches) {
                    requestPath += "WARMUP";
                }
                
                request = {
                    {{std::chrono::_V2::system_clock::now()}}, // FIRST_TIMESTAMP
                    {9999},
                    {requestPath},
                    1,
                    requestData
                };
                msvc_OutQueue[0]->emplace(request);
                msvc_inReqCount++;
            }

            batchNum++;
            requestData.clear();
        } 
        if (batchNum > (numProfileBatches + numWarmUpBatches)) {
            if (msvc_stepMode == 0) {
                batchSize += msvc_step;
            } else {
                batchSize *= 2;
            }
            batchNum = 1;
        }
    }
    msvc_logFile.close();
}
