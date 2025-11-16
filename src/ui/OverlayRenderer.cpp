#include "ui/OverlayRenderer.hpp"
#include <iomanip>
#include <sstream>

namespace bbst::ui {

// Color definitions
static const cv::Scalar ORANGE = cv::Scalar(0, 165, 255);
static const cv::Scalar GREEN = cv::Scalar(0, 255, 0);
static const cv::Scalar PURPLE = cv::Scalar(255, 0, 255);
static const cv::Scalar BLUE = cv::Scalar(255, 178, 50);
static const cv::Scalar BLACK = cv::Scalar(0, 0, 0);
static const cv::Scalar CYAN = cv::Scalar(255, 255, 0);

// Helper function to get class-specific color
static cv::Scalar getClassColor(int class_id) {
    switch(class_id) {
        case 0: return ORANGE;  // basketball
        case 1: return GREEN;   // rim
        case 2: return PURPLE;  // sports ball
        default: return BLUE;
    }
}

OverlayRenderer::OverlayRenderer(const ColorScheme& colors,
                                 int trajectory_thickness,
                                 float font_scale)
    : colors_(colors)
    , trajectory_thickness_(trajectory_thickness)
    , font_scale_(font_scale)
    , font_face_(cv::FONT_HERSHEY_SIMPLEX)
{
}

void OverlayRenderer::drawTrajectory(cv::Mat& image, const Trajectory& trajectory) {
    if (trajectory.size() < 2) return;
    
    // Draw trajectory with gradient (older points are more transparent)
    for (size_t i = 1; i < trajectory.size(); i++) {
        cv::Point2f p1 = trajectory[i - 1];
        cv::Point2f p2 = trajectory[i];
        
        // Calculate alpha based on position in trajectory (newer = more opaque)
        float alpha = static_cast<float>(i) / trajectory.size();
        
        // Gradient color from red (old) to cyan (new)
        cv::Scalar color(
            255 * (1 - alpha),  // Blue component
            255 * alpha,         // Green component
            255                  // Red component
        );
        
        int thickness = std::max(1, static_cast<int>(trajectory_thickness_ * alpha));
        
        cv::line(image, p1, p2, color, thickness, cv::LINE_AA);
        
        // Draw circle at current position
        if (i == trajectory.size() - 1) {
            cv::circle(image, p2, 5, CYAN, -1, cv::LINE_AA);
        }
    }
}

void OverlayRenderer::drawLabel(cv::Mat& image, const std::string& label,
                                int left, int top, const cv::Scalar& color) {
    int baseLine;
    cv::Size label_size = cv::getTextSize(label, font_face_, font_scale_, 
                                          1, &baseLine);
    
    int label_top;
    if (top - label_size.height - baseLine - 5 < 0) {
        label_top = top + baseLine + 5;
    } else {
        label_top = top - label_size.height - baseLine - 5;
    }
    
    cv::Point tlc(left, label_top);
    cv::Point brc(left + label_size.width, 
                  label_top + label_size.height + baseLine);
    
    // Draw semi-transparent background
    cv::Mat overlay = image.clone();
    cv::rectangle(overlay, tlc, brc, BLACK, cv::FILLED);
    cv::addWeighted(overlay, 0.6, image, 0.4, 0, image);
    
    // Draw text
    cv::putText(image, label, cv::Point(left, label_top + label_size.height),
                font_face_, font_scale_, color, 1);
}

void OverlayRenderer::drawDetection(cv::Mat& image, const Detection<>& det,
                                   const std::string& label) {
    // Get class-specific color
    cv::Scalar color = getClassColor(det.class_id);
    
    // Draw bounding box
    cv::rectangle(image, det.box, color, 2);
    
    // Format label with class name and confidence
    std::stringstream ss;
    ss << label << " " << std::fixed << std::setprecision(2) << det.confidence;
    
    // Draw label
    drawLabel(image, ss.str(), det.box.x, det.box.y, color);
}

void OverlayRenderer::drawInfo(cv::Mat& image, const std::string& info,
                              const cv::Point& position) {
    // Draw semi-transparent background
    cv::Mat overlay = image.clone();
    cv::rectangle(overlay, cv::Point(position.x - 5, position.y - 20),
                 cv::Point(position.x + 600, position.y + 5),
                 BLACK, cv::FILLED);
    cv::addWeighted(overlay, 0.6, image, 0.4, 0, image);
    
    // Draw text
    cv::putText(image, info, position, font_face_, 0.4, GREEN, 1);
}

} // namespace bbst::ui