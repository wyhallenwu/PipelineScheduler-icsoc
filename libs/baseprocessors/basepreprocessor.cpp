#include "basepreprocessor.h"

/**
 * @brief 
 * 
 * @param input 
 * @param height 
 * @param width 
 * @param bgcolor 
 * @return cv::cuda::GpuMat 
 */
cv::cuda::GpuMat resizePadRightBottom(
    const cv::cuda::GpuMat &input,
    size_t height,
    size_t width,
    const cv::Scalar &bgcolor
) {
    float r = std::min(width / (input.cols * 1.0), height / (input.rows * 1.0));
    int unpad_w = r * input.cols;
    int unpad_h = r * input.rows;
    //Create a new GPU Mat 
    cv::cuda::GpuMat re(unpad_h, unpad_w, CV_8UC3);
    cv::cuda::resize(input, re, re.size());
    cv::cuda::GpuMat out(height, width, CV_8UC3, bgcolor);
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

/**
 * @brief 
 * 
 * @param input 
 * @param subVals 
 * @param divVals 
 */
void normalize(
    cv::cuda::GpuMat &input,
    const std::array<float, 3>& subVals = {0.f, 0.f, 0.f},
    const std::array<float, 3>& divVals = {1.f, 1.f, 1.f}
) {
    cv::cuda::subtract(input, cv::Scalar(subVals[0], subVals[1], subVals[2]), input, cv::noArray(), -1);
    cv::cuda::divide(input, cv::Scalar(divVals[0], divVals[1], divVals[2]), input, 1, -1);
    input.convertTo(input, CV_32FC3, 1.f / 255.f);
}

template<class InType>
BasePreprocessor<InType>::BasePreprocessor(const BaseMicroserviceConfigs &configs) : LocalCPUDataMicroservice<InType>(configs){
    msvc_idealBatchSize = configs.msvc_dataShape[0];
    for (BatchSizeType i = 0; i < msvc_idealBatchSize; ++i) {
        msvc_batchBuffer.emplace_back(cv::cuda::GpuMat(configs.msvc_dataShape[2], configs.msvc_dataShape[3], CV_32FC3));
    }
}

/**
 * @brief Simplest function to decide if the requests should be batched and sent to the main processor.
 * 
 * @tparam InType 
 * @return true True if its time to batch
 * @return false if otherwise
 */
template<class InType>
bool BasePreprocessor<InType>::isTimeToBatch() {
    if (msvc_onBufferBatchSize == msvc_idealBatchSize) {
        return true;
    }
}