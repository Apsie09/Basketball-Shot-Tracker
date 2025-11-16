#pragma once
#include <deque>
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>

namespace bbst {

class Trajectory {
    std::deque<cv::Point2f> points_;
    size_t max_length_;
    
public:
    explicit Trajectory(size_t max_len = 50) : max_length_(max_len) {}
    
    // Operator overloading (Topic 23)
    Trajectory& operator+=(const cv::Point2f& point) {
        points_.push_back(point);
        if (points_.size() > max_length_) {
            points_.pop_front();
        }
        return *this;
    }
    
    cv::Point2f operator[](size_t idx) const {
        return points_[idx];
    }
    
    // Iterator support (Topic 7)
    auto begin() { return points_.begin(); }
    auto end() { return points_.end(); }
    auto begin() const { return points_.begin(); }
    auto end() const { return points_.end(); }
    
    size_t size() const { return points_.size(); }
    
    // Stream operator (Topic 23)
    friend std::ostream& operator<<(std::ostream& os, const Trajectory& traj) {
        os << "Trajectory[" << traj.size() << " points]";
        return os;
    }
};

} // namespace bbst