#include "misc.h"

void trt::to_json(nlohmann::json &j, const trt::TRTConfigs &val) {
    j["path"] = val.path;
    j["prec"] = val.precision;
    j["calibration"] = val.calibrationDataDirectoryPath;
    j["cbs"] = val.calibrationBatchSize;
    j["obs"] = val.optBatchSize;
    j["mbs"] = val.maxBatchSize;
    j["di"] = val.deviceIndex;
    j["normalize"] = val.normalize;
}

void trt::from_json(const nlohmann::json &j, trt::TRTConfigs &val) {
    j.at("path").get_to(val.path);
    j.at("prec").get_to(val.precision);
    j.at("calibration").get_to(val.calibrationDataDirectoryPath);
    j.at("cbs").get_to(val.calibrationBatchSize);
    j.at("obs").get_to(val.optBatchSize);
    j.at("mbs").get_to(val.maxBatchSize);
    j.at("di").get_to(val.deviceIndex);
    j.at("normalize").get_to(val.normalize);
}

void saveGPUAsImg(const cv::cuda::GpuMat &img, std::string name, float scale) {

    cv::Mat cpuImg;
    cv::cuda::GpuMat tempImg;
    img.convertTo(tempImg, CV_8UC3, scale);
    tempImg.download(cpuImg);
    cv::imwrite(name, cpuImg);
}

void saveCPUAsImg(const cv::Mat &img, std::string name, float scale) {
    cv::Mat cpuImg;
    img.convertTo(cpuImg, CV_8UC3, scale);
    cv::imwrite(name, img);
}

