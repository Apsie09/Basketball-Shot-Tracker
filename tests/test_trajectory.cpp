#include "core/Trajectory.hpp"
#include <iostream>
#include <cassert>
#include <opencv2/opencv.hpp>

using namespace bbst;

// Test basic trajectory operations
void test_basic_operations() {
    std::cout << "Testing basic trajectory operations..." << std::endl;
    
    Trajectory traj(10);  // Max 10 points
    
    // Test empty trajectory
    assert(traj.size() == 0);
    
    // Test adding points using operator+=
    cv::Point2f p1(10.0f, 20.0f);
    cv::Point2f p2(15.0f, 25.0f);
    cv::Point2f p3(20.0f, 30.0f);
    
    traj += p1;
    assert(traj.size() == 1);
    assert(traj[0] == p1);
    
    traj += p2;
    assert(traj.size() == 2);
    assert(traj[1] == p2);
    
    traj += p3;
    assert(traj.size() == 3);
    
    std::cout << "✓ Basic operations passed" << std::endl;
}

// Test max length constraint
void test_max_length() {
    std::cout << "Testing max length constraint..." << std::endl;
    
    Trajectory traj(5);  // Max 5 points
    
    // Add more than max points
    for (int i = 0; i < 10; ++i) {
        traj += cv::Point2f(i * 10.0f, i * 10.0f);
    }
    
    // Should only keep last 5 points
    assert(traj.size() == 5);
    
    // First point should be (5, 5) since (0,0) to (4,4) were removed
    assert(traj[0].x == 50.0f);
    assert(traj[0].y == 50.0f);
    
    std::cout << "✓ Max length constraint passed" << std::endl;
}

// Test iterator support
void test_iterators() {
    std::cout << "Testing iterators..." << std::endl;
    
    Trajectory traj(10);
    
    // Add some points
    for (int i = 0; i < 5; ++i) {
        traj += cv::Point2f(i * 1.0f, i * 2.0f);
    }
    
    // Test range-based for loop
    int count = 0;
    for (const auto& point : traj) {
        assert(point.x == count * 1.0f);
        assert(point.y == count * 2.0f);
        count++;
    }
    assert(count == 5);
    
    // Test manual iteration
    auto it = traj.begin();
    assert(it->x == 0.0f);
    ++it;
    assert(it->x == 1.0f);
    
    std::cout << "✓ Iterators passed" << std::endl;
}

// Test operator[] access
void test_operator_access() {
    std::cout << "Testing operator[] access..." << std::endl;
    
    Trajectory traj(10);
    
    for (int i = 0; i < 5; ++i) {
        traj += cv::Point2f(i * 5.0f, i * 10.0f);
    }
    
    // Test random access
    assert(traj[0].x == 0.0f);
    assert(traj[2].x == 10.0f);
    assert(traj[4].x == 20.0f);
    
    std::cout << "✓ Operator[] access passed" << std::endl;
}

// Test stream operator
void test_stream_operator() {
    std::cout << "Testing stream operator..." << std::endl;
    
    Trajectory traj(10);
    
    for (int i = 0; i < 3; ++i) {
        traj += cv::Point2f(i * 1.0f, i * 1.0f);
    }
    
    // Should print "Trajectory[3 points]"
    std::cout << "  Stream output: " << traj << std::endl;
    
    std::cout << "✓ Stream operator passed" << std::endl;
}

// Test copy and assignment
void test_copy_operations() {
    std::cout << "Testing copy operations..." << std::endl;
    
    Trajectory traj1(10);
    traj1 += cv::Point2f(1.0f, 2.0f);
    traj1 += cv::Point2f(3.0f, 4.0f);
    
    // Copy constructor
    Trajectory traj2 = traj1;
    assert(traj2.size() == traj1.size());
    assert(traj2[0] == traj1[0]);
    
    // Assignment operator
    Trajectory traj3(5);
    traj3 = traj1;
    assert(traj3.size() == traj1.size());
    assert(traj3[1] == traj1[1]);
    
    std::cout << "✓ Copy operations passed" << std::endl;
}

int main() {
    std::cout << "=== Running Trajectory Tests ===" << std::endl << std::endl;
    
    try {
        test_basic_operations();
        test_max_length();
        test_iterators();
        test_operator_access();
        test_stream_operator();
        test_copy_operations();
        
        std::cout << std::endl << "=== All Trajectory Tests Passed! ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "✗ Test failed with unknown exception" << std::endl;
        return 1;
    }
}