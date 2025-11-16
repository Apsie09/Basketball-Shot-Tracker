#pragma once
#include "BaseDetector.hpp"
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>
#include <memory>

namespace bbst {

// YOLO-specific configuration
struct YoloConfig {
    float input_width = 640.0f;
    float input_height = 640.0f;
    float score_threshold = 0.25f;
    float nms_threshold = 0.45f;
    float confidence_threshold = 0.25f;
};

class YoloDetector : public BaseDetector {
private:
    std::unique_ptr<cv::dnn::Net> net_;
    YoloConfig config_;
    std::vector<std::string> class_names_;
    
    // Helper methods
    cv::Mat formatYoloInput(const cv::Mat& source);
    std::vector<Detection<>> parseYoloOutput(const std::vector<cv::Mat>& outputs,
                                              const cv::Mat& original_image);
    std::vector<int> performNMS(const std::vector<cv::Rect>& boxes,
                                const std::vector<float>& confidences);
    
protected:
    // Override virtual methods from BaseDetector
    cv::Mat preProcess(const cv::Mat& frame) override;
    std::vector<Detection<>> postProcess(const std::vector<cv::Mat>& outputs,
                                         const cv::Mat& original_frame) override;
    
public:
    // Constructor with all parameters
    explicit YoloDetector(const std::string& model_path,
                         const std::string& class_names_path = "",
                         const YoloConfig& config = YoloConfig());
    
    // Destructor
    ~YoloDetector() override = default;
    
    // Delete copy operations (Topic 20)
    YoloDetector(const YoloDetector&) = delete;
    YoloDetector& operator=(const YoloDetector&) = delete;
    
    // Move operations (Topic 17)
    YoloDetector(YoloDetector&&) noexcept = default;
    YoloDetector& operator=(YoloDetector&&) noexcept = default;
    
    // Override detect from BaseDetector
    std::vector<Detection<>> detect(const cv::Mat& frame) override;
    
    // Additional YOLO-specific methods
    void loadClassNames(const std::string& path);
    const std::vector<std::string>& getClassNames() const { return class_names_; }
};

} // namespace bbst