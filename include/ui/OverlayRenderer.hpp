#pragma once
#include "core/Trajectory.hpp"
#include "core/IDetector.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <map>

namespace bbst::ui {

// Color scheme (Topic 10)
struct ColorScheme {
    cv::Scalar trajectory = cv::Scalar(255, 0, 255);  // Magenta
    cv::Scalar bbox = cv::Scalar(0, 255, 0);           // Green
    cv::Scalar text = cv::Scalar(255, 255, 255);       // White
    cv::Scalar background = cv::Scalar(0, 0, 0);       // Black
    
    // Class-specific colors
    std::map<int, cv::Scalar> class_colors = {
        {0, cv::Scalar(0, 165, 255)},  // Basketball - Orange
        {1, cv::Scalar(0, 255, 0)},     // Rim - Green
        {2, cv::Scalar(255, 0, 255)}    // Sports ball - Purple
    };
    
    cv::Scalar getClassColor(int class_id) const {
        auto it = class_colors.find(class_id);
        return it != class_colors.end() ? it->second : bbox;
    }
};

// Renderer for overlays (Topic 12-14)
class OverlayRenderer {
private:
    ColorScheme colors_;
    int trajectory_thickness_;
    float font_scale_;
    int font_face_;
    
    void drawLabel(cv::Mat& image, const std::string& label,
                  int left, int top, const cv::Scalar& color);
    
public:
    explicit OverlayRenderer(const ColorScheme& colors = ColorScheme(),
                            int trajectory_thickness = 2,
                            float font_scale = 0.5);
    
    // Draw trajectory (Topic 15: passing by reference)
    void drawTrajectory(cv::Mat& image, const Trajectory& trajectory);
    
    // Draw bounding box with label
    void drawDetection(cv::Mat& image, const Detection<>& det,
                      const std::string& label = "");
    
    // Draw multiple detections (Topic 7: iterators)
    template<typename Iterator>
    void drawDetections(cv::Mat& image, Iterator begin, Iterator end,
                       const std::vector<std::string>& class_names = {});
    
    // Draw info overlay
    void drawInfo(cv::Mat& image, const std::string& info,
                 const cv::Point& position = cv::Point(10, 30));
    
    // Set colors
    void setColorScheme(const ColorScheme& colors) { colors_ = colors; }
};

// Template implementation
template<typename Iterator>
void OverlayRenderer::drawDetections(cv::Mat& image, Iterator begin, Iterator end,
                                    const std::vector<std::string>& class_names) {
    for (auto it = begin; it != end; ++it) {
        std::string label;
        if (!class_names.empty() && it->class_id < class_names.size()) {
            label = class_names[it->class_id] + ": " + 
                   std::to_string(static_cast<int>(it->confidence * 100)) + "%";
        }
        drawDetection(image, *it, label);
    }
}

} // namespace bbst::ui