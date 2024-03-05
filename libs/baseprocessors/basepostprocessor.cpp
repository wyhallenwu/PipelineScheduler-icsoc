#include "baseprocessor.h"

/**
 * @brief Scale the bounding box coordinates to the original aspect ratio of the image
 * 
 * @param orig_h Original height of the image
 * @param orig_w Original width of the image
 * @param infer_h Height of the image used for inference
 * @param infer_w Width of the image used for inference
 * @param bbox_coors [x1, y1, x2, y2]
 */
void scaleBBox(
    uint16_t orig_h,
    uint16_t orig_w,
    uint16_t infer_h,
    uint16_t infer_w,
    const float &infer_bboxCoors,
    float &orig_bboxCoors
) {
    // TO BE IMPLEMENTED
}

void crop(
    const cv::cuda::GpuMat &image,
    uint16_t infer_h,
    uint16_t infer_w,
    uint8_t numDetections,
    const float &bbox_coorList,
    std::vector<cv::cuda::GpuMat> &croppedBBoxes
) {
    uint16_t orig_h, orig_w;
    orig_h = image.rows;
    orig_w = image.cols;
    float orig_bboxCoors[4];
    for (uint8_t i = 0; i < numDetections; ++i) {
        scaleBBox(orig_h, orig_w, infer_h, infer_w, bbox_coorList + i * 4, *orig_bboxCoors);
        cv::cuda::GpuMat croppedBBox = image(cv::Range(orig_bboxCoors[0], orig_bboxCoors[2]), cv::Range(orig_bboxCoors[1], orig_bboxCoors[3])).clone();
        croppedBBoxes.emplace_back(croppedBBox);
    }
}

BasePostprocessor::BasePostprocessor(const BaseMicroserviceConfigs &configs) : Microservice(configs) {
}