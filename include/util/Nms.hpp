#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

namespace bbst::util {

// Non-Maximum Suppression utility (Topic 22: Static methods)
class NMS {
public:
    // Static method for NMS (Topic 22)
    static std::vector<int> apply(
        const std::vector<cv::Rect>& boxes,
        const std::vector<float>& scores,
        float nms_threshold = 0.45f,
        float score_threshold = 0.25f
    ) {
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, score_threshold, 
                         nms_threshold, indices);
        return indices;
    }
    
    // Soft-NMS variant
    static std::vector<int> applySoft(
        const std::vector<cv::Rect>& boxes,
        const std::vector<float>& scores,
        float sigma = 0.5f,
        float score_threshold = 0.25f
    );
};

} // namespace bbst::util