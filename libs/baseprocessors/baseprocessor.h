#include <microservice.h>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>

typedef uint16_t BatchSizeType;

template<typename InType>
class BaseProcessor : public LocalGPUDataMicroservice<InType> {
public:
    BaseProcessor(const BaseMicroserviceConfigs &configs);
    ~BaseProcessor();
protected:
    BatchSizeType msvc_onBufferBatchSize;
    bool checkReqEligibility(ClockTypeTemp currReq_genTime) override;
};
