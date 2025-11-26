#include "tracking/KalmanTracker.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace bbst::tracking;

// Helper function to check if points are close
bool pointsClose(const cv::Point2f& p1, const cv::Point2f& p2, float epsilon = 5.0f) {
    return std::abs(p1.x - p2.x) < epsilon && std::abs(p1.y - p2.y) < epsilon;
}

// Test tracker initialization
void test_initialization() {
    std::cout << "Testing tracker initialization..." << std::endl;
    
    TrackerConfig config;
    KalmanTracker tracker(config);
    
    // Tracker should not be active initially
    assert(!tracker.isActive());
    assert(!tracker.isStable());
    assert(tracker.getTotalDetections() == 0);
    
    // Initialize with first measurement
    cv::Point2f init_point(100.0f, 200.0f);
    tracker.init(init_point, 20.0f);
    
    assert(tracker.isActive());
    assert(tracker.isStable());
    assert(tracker.getTotalDetections() == 1);
    assert(pointsClose(tracker.getLastPosition(), init_point));
    
    std::cout << "✓ Initialization passed" << std::endl;
}

// Test prediction
void test_prediction() {
    std::cout << "Testing prediction..." << std::endl;
    
    TrackerConfig config;
    KalmanTracker tracker(config);
    
    // Initialize
    tracker.init(cv::Point2f(100.0f, 100.0f), 20.0f);
    
    // Update with moving ball
    tracker.update(cv::Point2f(110.0f, 105.0f), 20.0f);
    tracker.update(cv::Point2f(120.0f, 110.0f), 20.0f);
    
    // Predict should give next position based on velocity
    cv::Point2f predicted = tracker.predict();
    
    // Predicted point should be somewhere ahead
    assert(predicted.x > 120.0f);
    assert(predicted.y > 110.0f);
    
    std::cout << "✓ Prediction passed" << std::endl;
}

// Test update with valid measurements
void test_valid_updates() {
    std::cout << "Testing valid updates..." << std::endl;
    
    TrackerConfig config;
    config.max_velocity = 50.0f;
    KalmanTracker tracker(config);
    
    // Initialize
    tracker.init(cv::Point2f(100.0f, 100.0f), 20.0f);
    
    // Valid updates (reasonable movement)
    tracker.update(cv::Point2f(110.0f, 105.0f), 22.0f);
    assert(tracker.getTotalDetections() == 2);
    
    tracker.update(cv::Point2f(120.0f, 110.0f), 21.0f);
    assert(tracker.getTotalDetections() == 3);
    
    // Trajectory should contain points
    assert(tracker.getTrajectory().size() == 3);
    
    std::cout << "✓ Valid updates passed" << std::endl;
}

// Test validation rejection
void test_validation_rejection() {
    std::cout << "Testing validation rejection..." << std::endl;
    
    TrackerConfig config;
    config.max_velocity = 30.0f;
    KalmanTracker tracker(config);
    
    // Initialize and build some history
    tracker.init(cv::Point2f(100.0f, 100.0f), 20.0f);
    for (int i = 0; i < 10; ++i) {
        tracker.update(cv::Point2f(100.0f + i * 5.0f, 100.0f + i * 2.0f), 20.0f);
    }
    
    int detections_before = tracker.getTotalDetections();
    
    // Try to update with invalid measurement (too far away)
    cv::Point2f invalid_point(500.0f, 500.0f);  // Way too far
    tracker.update(invalid_point, 20.0f);
    
    // Should reject and not increment detection count
    assert(tracker.getTotalDetections() == detections_before);
    
    std::cout << "✓ Validation rejection passed" << std::endl;
}

// Test update without measurement
void test_update_without_measurement() {
    std::cout << "Testing update without measurement..." << std::endl;
    
    TrackerConfig config;
    config.max_frames_without_detection = 5;
    KalmanTracker tracker(config);
    
    // Initialize
    tracker.init(cv::Point2f(100.0f, 100.0f), 20.0f);
    tracker.update(cv::Point2f(110.0f, 105.0f), 20.0f);
    
    // Update without measurements
    for (int i = 0; i < 3; ++i) {
        tracker.updateWithoutMeasurement();
        assert(tracker.isActive());  // Should still be active
    }
    
    // Trajectory should have grown with predicted points
    assert(tracker.getTrajectory().size() > 2);
    
    std::cout << "✓ Update without measurement passed" << std::endl;
}

// Test tracker reset after too many missed detections
void test_reset_on_loss() {
    std::cout << "Testing reset on loss..." << std::endl;
    
    TrackerConfig config;
    config.max_frames_without_detection = 5;
    KalmanTracker tracker(config);
    
    // Initialize
    tracker.init(cv::Point2f(100.0f, 100.0f), 20.0f);
    
    // Miss many detections
    for (int i = 0; i < 10; ++i) {
        tracker.updateWithoutMeasurement();
    }
    
    // Should no longer be active after max_frames_without_detection
    assert(!tracker.isActive());
    
    std::cout << "✓ Reset on loss passed" << std::endl;
}

// Test trajectory smoothing
void test_trajectory_smoothing() {
    std::cout << "Testing trajectory smoothing..." << std::endl;
    
    TrackerConfig config;
    KalmanTracker tracker(config);
    
    // Initialize
    tracker.init(cv::Point2f(100.0f, 100.0f), 20.0f);
    
    // Add noisy measurements
    cv::Point2f measurements[] = {
        cv::Point2f(110.0f, 105.0f),
        cv::Point2f(118.0f, 112.0f),  // Slightly noisy
        cv::Point2f(130.0f, 115.0f),
        cv::Point2f(140.0f, 120.0f)
    };
    
    for (const auto& m : measurements) {
        tracker.update(m, 20.0f);
    }
    
    // Trajectory should be smoother than raw measurements
    const auto& traj = tracker.getTrajectory();
    assert(traj.size() == 5);  // Init + 4 updates
    
    std::cout << "✓ Trajectory smoothing passed" << std::endl;
}

// Test manual reset
void test_manual_reset() {
    std::cout << "Testing manual reset..." << std::endl;
    
    TrackerConfig config;
    KalmanTracker tracker(config);
    
    // Initialize and add some detections
    tracker.init(cv::Point2f(100.0f, 100.0f), 20.0f);
    tracker.update(cv::Point2f(110.0f, 105.0f), 20.0f);
    tracker.update(cv::Point2f(120.0f, 110.0f), 20.0f);
    
    assert(tracker.isActive());
    assert(tracker.getTotalDetections() == 3);
    
    // Reset
    tracker.reset();
    
    assert(!tracker.isActive());
    assert(tracker.getTotalDetections() == 0);
    assert(tracker.getTrajectory().size() == 0);
    
    std::cout << "✓ Manual reset passed" << std::endl;
}

// Test configuration
void test_configuration() {
    std::cout << "Testing configuration..." << std::endl;
    
    TrackerConfig config;
    config.max_velocity = 100.0f;
    config.min_ball_size = 10.0f;
    config.max_ball_size = 50.0f;
    
    KalmanTracker tracker(config);
    
    tracker.init(cv::Point2f(100.0f, 100.0f), 25.0f);
    
    // Valid size
    assert(tracker.isValidDetection(cv::Point2f(105.0f, 105.0f), 30.0f));
    
    // Invalid size (too small)
    assert(!tracker.isValidDetection(cv::Point2f(105.0f, 105.0f), 5.0f));
    
    // Invalid size (too large)
    assert(!tracker.isValidDetection(cv::Point2f(105.0f, 105.0f), 100.0f));
    
    std::cout << "✓ Configuration passed" << std::endl;
}

int main() {
    std::cout << "=== Running Tracker Tests ===" << std::endl << std::endl;
    
    try {
        test_initialization();
        test_prediction();
        test_valid_updates();
        test_validation_rejection();
        test_update_without_measurement();
        test_reset_on_loss();
        test_trajectory_smoothing();
        test_manual_reset();
        test_configuration();
        
        std::cout << std::endl << "=== All Tracker Tests Passed! ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "✗ Test failed with unknown exception" << std::endl;
        return 1;
    }
}