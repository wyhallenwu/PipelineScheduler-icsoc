#include "controller.h"

// the accuracy value here is dummy, just use for ranking models
const std::map<std::string, float> ACC_LEVEL_MAP = {
        {"yolov5n320", 0.30},
        {"yolov5n512", 0.40},
        {"yolov5n640", 0.50},
        {"yolov5s640", 0.55},
        {"yolov5m640", 0.60},
};
