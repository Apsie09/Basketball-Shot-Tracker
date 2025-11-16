#include "detectors/BaseDetector.hpp"

namespace bbst {

std::vector<Detection<>> BaseDetector::detect(const cv::Mat& frame) {
    // Preprocess the frame
    cv::Mat blob = preProcess(frame);
    
    // For YOLO, we need to run inference here
    // This is a bit of a design compromise - ideally inference would be in a separate method
    // But for now, we'll handle it in the derived class
    
    // This is a placeholder - actual implementation depends on detector type
    return {};
}

} // namespace bbst