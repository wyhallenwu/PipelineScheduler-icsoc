#include <microservice.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>

typedef uint16_t BatchSizeType;

cv::cuda::GpuMat resizePadRightBottom(
    const cv::cuda::GpuMat &input,
    size_t height,
    size_t width,
    const cv::Scalar &bgcolor
);

void normalize(
    cv::cuda::GpuMat &input,
    const std::array<float, 3>& subVals,
    const std::array<float, 3>& divVals
);

template<class InType>
class BasePreprocessor : public LocalGPUDataMicroservice<InType> {
public:
    BasePreprocessor(const BaseMicroserviceConfigs &configs);
    ~BasePreprocessor();
protected:
    BatchSizeType msvc_idealBatchSize;
    BatchSizeType msvc_onBufferBatchSize;
    std::vector<cv::cuda::GpuMat> msvc_batchBuffer;
    bool isTimeToBatch() override;
};
