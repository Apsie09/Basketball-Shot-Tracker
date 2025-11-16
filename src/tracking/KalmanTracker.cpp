#include "tracking/KalmanTracker.hpp"
#include <cmath>

namespace bbst::tracking {

KalmanTracker::KalmanTracker(const TrackerConfig& config)
    : kf_(std::make_unique<cv::KalmanFilter>(4, 2, 0))  // Smart pointer initialization
    , trajectory_(config.max_trajectory_length)
    , initialized_(false)
    , frames_without_detection_(0)
    , last_size_(0.0f)
    , consecutive_good_detections_(0)
    , total_detections_(0)
    , config_(config)
{
    initKalmanFilter();
    measurement_ = cv::Mat::zeros(2, 1, CV_32F);
}

void KalmanTracker::initKalmanFilter() {
    // Transition matrix (constant velocity model)
    kf_->transitionMatrix = (cv::Mat_<float>(4, 4) << 
        1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1);
    
    // Measurement matrix
    kf_->measurementMatrix = (cv::Mat_<float>(2, 4) << 
        1, 0, 0, 0,
        0, 1, 0, 0);
    
    // Process noise
    cv::setIdentity(kf_->processNoiseCov, cv::Scalar::all(1e-1));
    
    // Measurement noise
    cv::setIdentity(kf_->measurementNoiseCov, cv::Scalar::all(2e-1));
    
    // Error covariance
    cv::setIdentity(kf_->errorCovPost, cv::Scalar::all(1));
}

void KalmanTracker::init(const cv::Point2f& initial_point, float size) {
    kf_->statePost = (cv::Mat_<float>(4, 1) << 
        initial_point.x, initial_point.y, 0, 0);
    
    initialized_ = true;
    frames_without_detection_ = 0;
    last_position_ = initial_point;
    last_size_ = size;
    consecutive_good_detections_ = 1;
    total_detections_ = 1;
    
    trajectory_ = Trajectory(config_.max_trajectory_length);  // Reset trajectory
    trajectory_ += initial_point;  // Use operator+= (Topic 23)
}

cv::Point2f KalmanTracker::predict() {
    if (!initialized_) return cv::Point2f(-1, -1);
    
    cv::Mat prediction = kf_->predict();
    return cv::Point2f(prediction.at<float>(0), prediction.at<float>(1));
}

bool KalmanTracker::validateSize(float size) const {
    return size >= config_.min_ball_size && size <= config_.max_ball_size;
}

bool KalmanTracker::validateAspectRatio(float width, float height) const {
    float aspect_ratio = width / height;
    return aspect_ratio >= config_.min_aspect_ratio && 
           aspect_ratio <= config_.max_aspect_ratio;
}

bool KalmanTracker::validateVelocity(const cv::Point2f& new_point, 
                                     const cv::Point2f& predicted) const {
    float distance = cv::norm(new_point - predicted);
    float max_allowed = config_.max_velocity * 2.0f;
    
    // Stricter when stable
    if (consecutive_good_detections_ > 20) {
        max_allowed = config_.max_velocity * 1.2f;
    }
    
    return distance < max_allowed;
}

bool KalmanTracker::isValidDetection(const cv::Point2f& measurement,
                                     float size,
                                     bool strict [[maybe_unused]]) const {
    // Add [[maybe_unused]] attribute to suppress warning
    // Or use it in the implementation if needed
    
    // Size validation
    if (size < config_.min_ball_size || size > config_.max_ball_size) {
        return false;
    }
    
    // Size consistency check
    if (last_size_ > 0) {
        float size_ratio = size / last_size_;
        if (size_ratio < 0.2f || size_ratio > 5.0f) return false;
    }
    
    // Velocity check
    cv::Point2f predicted = const_cast<KalmanTracker*>(this)->predict();
    if (frames_without_detection_ < 15 && !validateVelocity(measurement, predicted)) {
        return false;
    }
    
    return true;
}

cv::Point2f KalmanTracker::update(const cv::Point2f& measurement_point, float size) {
    if (!initialized_) {
        init(measurement_point, size);
        return measurement_point;
    }
    
    if (!isValidDetection(measurement_point, size, false)) {
        return updateWithoutMeasurement();
    }
    
    // Update Kalman filter
    measurement_.at<float>(0) = measurement_point.x;
    measurement_.at<float>(1) = measurement_point.y;
    
    cv::Mat estimated = kf_->correct(measurement_);
    cv::Point2f corrected_point(estimated.at<float>(0), estimated.at<float>(1));
    
    // Update trajectory using operator+= (Topic 23)
    trajectory_ += corrected_point;
    
    frames_without_detection_ = 0;
    consecutive_good_detections_++;
    total_detections_++;
    last_position_ = corrected_point;
    last_size_ = size;
    
    return corrected_point;
}

cv::Point2f KalmanTracker::updateWithoutMeasurement() {
    if (!initialized_) return cv::Point2f(-1, -1);
    
    frames_without_detection_++;
    
    if (consecutive_good_detections_ > 5 && frames_without_detection_ > 10) {
        consecutive_good_detections_--;
    }
    
    cv::Point2f predicted = predict();
    
    if (frames_without_detection_ <= config_.max_frames_without_detection) {
        trajectory_ += predicted;
        last_position_ = predicted;
    } else {
        reset();
    }
    
    return predicted;
}

bool KalmanTracker::isActive() const {
    return initialized_ && 
           frames_without_detection_ <= config_.max_frames_without_detection;
}

bool KalmanTracker::isStable() const {
    return total_detections_ >= 1;
}

void KalmanTracker::reset() {
    initialized_ = false;
    frames_without_detection_ = 0;
    consecutive_good_detections_ = 0;
    total_detections_ = 0;
    last_size_ = 0;
    trajectory_ = Trajectory(config_.max_trajectory_length);
}

} // namespace bbst::tracking