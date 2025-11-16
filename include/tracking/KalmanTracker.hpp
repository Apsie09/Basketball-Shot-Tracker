#pragma once
#include "core/Trajectory.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <deque>
#include <memory>

namespace bbst::tracking {

// Configuration for tracker (Topic 12, 35)
struct TrackerConfig {
    float max_velocity = 70.0f;
    float min_ball_size = 5.0f;
    float max_ball_size = 120.0f;
    float min_aspect_ratio = 0.3f;
    float max_aspect_ratio = 3.0f;
    int max_frames_without_detection = 40;
    size_t max_trajectory_length = 50;
};

// Full Kalman-based ball tracker (Topics 12-14, 21)
class KalmanTracker {
private:
    // State variables
    std::unique_ptr<cv::KalmanFilter> kf_;  // Smart pointer (Topic 21)
    cv::Mat measurement_;
    Trajectory trajectory_;
    bool initialized_;
    int frames_without_detection_;
    cv::Point2f last_position_;
    float last_size_;
    int consecutive_good_detections_;
    int total_detections_;
    
    // Configuration
    TrackerConfig config_;
    
    // Private helper methods
    void initKalmanFilter();
    bool validateSize(float size) const;
    bool validateAspectRatio(float width, float height) const;
    bool validateVelocity(const cv::Point2f& new_point, const cv::Point2f& predicted) const;
    
public:
    // Constructor (Topic 13)
    explicit KalmanTracker(const TrackerConfig& config = TrackerConfig());
    
    // Destructor (Topic 13)
    ~KalmanTracker() = default;
    
    // Move semantics (Topic 17)
    KalmanTracker(KalmanTracker&&) noexcept = default;
    KalmanTracker& operator=(KalmanTracker&&) noexcept = default;
    
    // Delete copy (Topic 20)
    KalmanTracker(const KalmanTracker&) = delete;
    KalmanTracker& operator=(const KalmanTracker&) = delete;
    
    // Main interface
    void init(const cv::Point2f& initial_point, float size);
    cv::Point2f predict();
    bool isValidDetection(const cv::Point2f& point, float size, bool strict = false) const;
    cv::Point2f update(const cv::Point2f& measurement_point, float size);
    cv::Point2f updateWithoutMeasurement();
    
    // Getters (const methods - Topic 19)
    const Trajectory& getTrajectory() const { return trajectory_; }
    bool isActive() const;
    bool isStable() const;
    cv::Point2f getLastPosition() const { return last_position_; }
    int getTotalDetections() const { return total_detections_; }
    
    // Reset
    void reset();
};

} // namespace bbst::tracking