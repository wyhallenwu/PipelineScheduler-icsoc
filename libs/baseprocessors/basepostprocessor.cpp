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
    int orig_h,
    int orig_w,
    int infer_h,
    int infer_w,
    const float *infer_bboxCoors,
    int * orig_bboxCoors
) {
    float ratio = std::min(1.f * infer_h / orig_h, 1.f * infer_w / orig_w);
    infer_h = (int) (ratio * orig_h);
    infer_w = (int) (ratio * orig_w);

    // TO BE IMPLEMENTED
    float coor[4];
    for (uint8_t i = 0; i < 4; ++i) {
        coor[i] = (*(infer_bboxCoors + i));
    }

    float gain = std::min(1.f * infer_h / orig_h, 1.f * infer_w / orig_w);

    float pad_w = (1.f * infer_w - orig_w * gain) / 2.f;
    float pad_h = (1.f * infer_h - orig_h * gain) / 2.f;

    coor[0] -= pad_w;
    coor[1] -= pad_h;
    coor[2] -= pad_w;
    coor[3] -= pad_h;

    // if (scale_h > scale_w) {
    //     coor[1]= coor[1] / scale_w;
    //     coor[3]= coor[3] / scale_w;
    //     coor[0]= (coor[0] - (infer_h - scale_w * orig_h) / 2) / scale_w;
    //     coor[2]= (coor[2] - (infer_h - scale_w * orig_h) / 2) / scale_w;
    // } else {
    //     coor[1]= (coor[1] - (infer_w - scale_h * orig_w) / 2) / scale_h;
    //     coor[3]= (coor[3] - (infer_w - scale_h * orig_w) / 2) / scale_h;
    //     coor[0]= coor[0] / scale_h;
    //     coor[2]= coor[2] / scale_h;
    // }

    for (uint8_t i = 0; i < 4; ++i) {
        coor[i] /= gain;
        int maxcoor = (i % 2 == 0) ? orig_w : orig_h;
        if (coor[i] >= maxcoor) {
            coor[i] = maxcoor - 1;
        }
        if (coor[i] < 0) {
            coor[i] = 0;
        }
        *(orig_bboxCoors + i) = (int)coor[i];
    }
}

/**
 * @brief Cropping multiple boxes from 1 picture
 * 
 * @param image 
 * @param infer_h 
 * @param infer_w 
 * @param numDetections 
 * @param bbox_coorList 
 * @param croppedBBoxes 
 */
void crop(
    const cv::cuda::GpuMat &image,
    int orig_h,
    int orig_w,
    int infer_h,
    int infer_w,
    int numDetections,
    const float *bbox_coorList,
    std::vector<cv::cuda::GpuMat> &croppedBBoxes
) {
    int orig_bboxCoors[4];
    for (uint16_t i = 0; i < numDetections; ++i) {
        scaleBBox(orig_h, orig_w, infer_h, infer_w, bbox_coorList + i * 4, orig_bboxCoors);
        std::cout << (int)orig_bboxCoors[0] << " " << (int)orig_bboxCoors[1] << " " << (int)orig_bboxCoors[2] << " " << (int)orig_bboxCoors[3] << std::endl;
        cv::cuda::GpuMat croppedBBox = image(cv::Range((int)orig_bboxCoors[1], (int)orig_bboxCoors[3]), cv::Range((int)orig_bboxCoors[0], (int)orig_bboxCoors[2])).clone();
        croppedBBoxes.emplace_back(croppedBBox);
    }
}

/**
 * @brief Cropping 1 box from 1 picture
 * 
 * @param image 
 * @param infer_h 
 * @param infer_w 
 * @param numDetections 
 * @param bbox_coorList 
 * @param croppedBBoxes 
 */
void cropOneBox(
    const cv::cuda::GpuMat &image,
    int infer_h,
    int infer_w,
    int numDetections,
    const float *bbox_coorList,
    cv::cuda::GpuMat &croppedBBoxes
) {
    int orig_h, orig_w;
    orig_h = image.rows;
    orig_w = image.cols;
    int orig_bboxCoors[4];
    scaleBBox(orig_h, orig_w, infer_h, infer_w, bbox_coorList, orig_bboxCoors);
    cv::cuda::GpuMat croppedBBox = image(cv::Range((int)orig_bboxCoors[0], (int)orig_bboxCoors[2]), cv::Range((int)orig_bboxCoors[1], (int)orig_bboxCoors[3])).clone();
    croppedBBoxes = croppedBBox;
}

BasePostprocessor::BasePostprocessor(const BaseMicroserviceConfigs &configs) : Microservice(configs) {
}