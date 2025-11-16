#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

namespace bbst {

// Empty struct for default metadata
struct EmptyMetadata {};

// Detection result with optional metadata (Topic 35-36: Templates)
template<typename MetaData = EmptyMetadata>
struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
    cv::Point2f center;
    MetaData metadata;  // Now defaults to EmptyMetadata instead of void
    
    Detection() 
        : class_id(-1)
        , confidence(0.0f)
        , box()
        , center()
        , metadata() 
    {}
};

// Detector interface (Topic 35: Generic programming)
template<typename DetectionType>
class IDetector {
public:
    virtual ~IDetector() = default;
    
    // Pure virtual detection method
    virtual std::vector<DetectionType> detect(const cv::Mat& frame) = 0;
    
    // Configuration
    virtual void setConfidenceThreshold(float threshold) = 0;
};

} // namespace bbst