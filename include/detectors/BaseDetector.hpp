#pragma once
#include "core/IDetector.hpp"
#include <opencv2/opencv.hpp>

namespace bbst {

// Abstract base with virtual methods (Topic 17)
class BaseDetector : public IDetector<Detection<>> {
protected:
    float confidence_threshold_;
    cv::Size input_size_;
    
    // Template method pattern (Topic 26)
    virtual cv::Mat preProcess(const cv::Mat& frame) = 0;  // Returns processed blob
    virtual std::vector<Detection<>> postProcess(const std::vector<cv::Mat>& outputs, const cv::Mat& original_frame) = 0;  // Returns detections
    
public:
    explicit BaseDetector(float threshold = 0.25f) 
        : confidence_threshold_(threshold) {}
    
    virtual ~BaseDetector() = default;
    
    // Final implementation (Topic 17)
    std::vector<Detection<>> detect(const cv::Mat& frame) = 0;
    
    void setConfidenceThreshold(float threshold) override {
        confidence_threshold_ = threshold;
    }
};

} // namespace bbst