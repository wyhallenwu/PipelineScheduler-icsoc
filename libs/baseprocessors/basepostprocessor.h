#include "microservice.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <cuda_runtime.h>

template <typename InType>
class BasePostprocessor : public Microservice<InType> {
public:
    BasePostprocessor(const BaseMicroserviceConfigs &configs);
    ~BasePostprocessor();

    std::vector<ThreadSafeFixSizedDoubleQueue<LocalGPUReqDataType, LocalCPUDataType>> *getOutQueue() {
        return OutQueue;
    }

    QueueLengthType GetOutQueueSize() {
        return OutQueue->size();
    }

protected:
    std::vector<ThreadSafeFixSizedDoubleQueue<LocalGPUReqDataType, LocalCPUDataType>> *OutQueue;
};