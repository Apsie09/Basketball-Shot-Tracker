#pragma once
#include <memory>
#include <string>
#include <opencv4/opencv2/dnn.hpp>
#include <opencv4/opencv2/opencv.hpp>

namespace bbst {

// RAII wrapper for model loading (Topic 48-50)
class ModelResource {
    std::unique_ptr<cv::dnn::Net> net_;
    std::string model_path_;
    
public:
    explicit ModelResource(const std::string& path) 
        : model_path_(path) {
        net_ = std::make_unique<cv::dnn::Net>(
            cv::dnn::readNetFromONNX(path)
        );
    }
    
    // Move semantics (Topic 17)
    ModelResource(ModelResource&&) noexcept = default;
    ModelResource& operator=(ModelResource&&) noexcept = default;
    
    // Deleted copy (Topic 13, 20)
    ModelResource(const ModelResource&) = delete;
    ModelResource& operator=(const ModelResource&) = delete;
    
    ~ModelResource() {
        // Automatic cleanup
    }
    
    cv::dnn::Net& get() { return *net_; }
    const cv::dnn::Net& get() const { return *net_; }
};

} // namespace bbst