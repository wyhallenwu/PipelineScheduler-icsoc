#include "microservice.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <cuda_runtime.h>

/**
 * @brief crop from input image all bounding boxes whose coordinates are provided by `bbox_coorList`
 * 
 * @param image 
 * @param infer_h the height of the image during inference, used to scale to bounding boxes to their original coordinates for cropping
 * @param infer_w the height of the image during inference, used to scale to bounding boxes to their original coordinates for cropping
 * @param bbox_coorList pointer to a 2d `float` array of bounding box coordinates of size (TopK, 4). The box format is 
 *                      [x1, y1, x2, y2] (e.g., [0, 266, 260, 447])
 * @return cv::cuda::GpuMat
 */
void crop(
    const cv::cuda::GpuMat &image,
    uint16_t infer_h,
    uint16_t infer_w,
    RequestShapeType numDetections,
    float *bbox_coorList,
    std::vector<cv::cuda::GpuMat> &croppedBBoxes
);

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