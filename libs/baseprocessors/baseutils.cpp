#include "baseprocessor.h"

using namespace spdlog;

inline cv::Scalar vectorToScalar(const std::vector<float>& vec) {
    // Ensure the vector has exactly 4 elements
    if (vec.size() == 1) {
        return cv::Scalar(vec[0]);
    } else if (vec.size() == 3) {
        return cv::Scalar(vec[0], vec[1], vec[2]);
    } else {
        throw std::runtime_error("Unsupported number of channels");
    }
}

/**
 * @brief normalize the input data by subtracting and dividing values from the original pixesl
 * 
 * @param input the input data
 * @param subVals values to be subtracted
 * @param divVals values to be dividing by
 * @param stream an opencv stream for asynchronous operation on cuda
 */
cv::cuda::GpuMat normalize(
    cv::cuda::GpuMat &input,
    cv::cuda::Stream &stream,
    const std::vector<float>& subVals,
    const std::vector<float>& divVals,
    const float normalized_scale
) {
    trace("Going into {0:s}", __func__);
    cv::cuda::GpuMat normalized;
    if (input.channels() == 1) {
        input.convertTo(normalized, CV_32FC1, normalized_scale, stream);
        cv::cuda::subtract(normalized, vectorToScalar(subVals), normalized, cv::noArray(), -1, stream);
        cv::cuda::divide(normalized, vectorToScalar(divVals), normalized, 1, -1, stream);
    } else if (input.channels() == 3) {
        input.convertTo(normalized, CV_32FC3, normalized_scale, stream);    
        cv::cuda::subtract(normalized, vectorToScalar(subVals), normalized, cv::noArray(), -1, stream);
        cv::cuda::divide(normalized, vectorToScalar(divVals), normalized, 1, -1, stream);
    } else {
        throw std::runtime_error("Unsupported number of channels");
    }

    stream.waitForCompletion();
    trace("Finished {0:s}", __func__);

    return normalized;
}

cv::cuda::GpuMat cvtHWCToCHW(
    cv::cuda::GpuMat &input,
    cv::cuda::Stream &stream,
    uint8_t IMG_TYPE
) {

    trace("Going into {0:s}", __func__);
    uint16_t height = input.rows;
    uint16_t width = input.cols;
    /**
     * @brief TODOs
     * This is the correct way but since we haven't figured out how to carry to image to be cropped
     * it screws things up a bit.
     * cv::cuda::GpuMat transposed(1, height * width, CV_8UC3);
     */
    // cv::cuda::GpuMat transposed(height, width, CV_8UC3);
    cv::cuda::GpuMat transposed(1, height * width, IMG_TYPE);
    std::vector<cv::cuda::GpuMat> channels;
    if (input.channels() == 1) {
        uint8_t IMG_SINGLE_CHANNEL_TYPE = IMG_TYPE;
        channels = {
            cv::cuda::GpuMat(height, width, IMG_SINGLE_CHANNEL_TYPE, &(transposed.ptr()[0]))
        };
    } else if (input.channels() == 3) {
        uint8_t IMG_SINGLE_CHANNEL_TYPE = IMG_TYPE ^ 16;
        size_t channel_mem_width = height * width;
        
        channels = {
            cv::cuda::GpuMat(height, width, IMG_SINGLE_CHANNEL_TYPE, &(transposed.ptr()[0])),
            cv::cuda::GpuMat(height, width, IMG_SINGLE_CHANNEL_TYPE, &(transposed.ptr()[channel_mem_width])),
            cv::cuda::GpuMat(height, width, IMG_SINGLE_CHANNEL_TYPE, &(transposed.ptr()[channel_mem_width * 2]))
        };
    } else {
        throw std::runtime_error("Unsupported number of channels");
    }
    cv::cuda::split(input, channels, stream);

    stream.waitForCompletion();    

    trace("Finished {0:s}", __func__);

    return transposed;
}

/**
 * @brief resize the input data without changing the aspect ratio and pad the empty area with a designated color
 * 
 * @param input the input data
 * @param height the expected height after processing
 * @param width  the expect width after processing
 * @param bgcolor color to pad the empty area with
 * @return cv::cuda::GpuMat 
 */
cv::cuda::GpuMat resizePadRightBottom(
    cv::cuda::GpuMat &input,
    const size_t height,
    const size_t width,
    const std::vector<float> &bgcolor,
    cv::cuda::Stream &stream,
    uint8_t IMG_TYPE,
    uint8_t COLOR_CVT_TYPE,
    uint8_t RESIZE_INTERPOL_TYPE

) {
    trace("Going into {0:s}", __func__);

    cv::cuda::GpuMat rgb_img(input.rows, input.cols, IMG_TYPE);
    cv::cuda::cvtColor(input, rgb_img, COLOR_CVT_TYPE, 0, stream);

    float r = std::min(width / (input.cols * 1.0), height / (input.rows * 1.0));
    int unpad_w = r * input.cols;
    int unpad_h = r * input.rows;
    //Create a new GPU Mat 
    cv::cuda::GpuMat resized(unpad_h, unpad_w, IMG_TYPE);
    cv::cuda::resize(rgb_img, resized, resized.size(), 0, 0, RESIZE_INTERPOL_TYPE, stream);
    cv::cuda::GpuMat out(height, width, IMG_TYPE, vectorToScalar(bgcolor));
    // Creating an opencv stream for asynchronous operation on cuda
    resized.copyTo(out(cv::Rect(0, 0, resized.cols, resized.rows)), stream);

    stream.waitForCompletion();
    trace("Finished {0:s}", __func__);

    return out;
}

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
        // std::cout << (int)orig_bboxCoors[0] << " " << (int)orig_bboxCoors[1] << " " << (int)orig_bboxCoors[2] << " " << (int)orig_bboxCoors[3] << std::endl;
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