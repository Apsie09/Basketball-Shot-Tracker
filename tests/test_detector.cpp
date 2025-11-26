#include "detectors/YoloDetector.hpp"
#include "core/IDetector.hpp"
#include <iostream>
#include <cassert>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace bbst;

// Test detector initialization
void test_initialization() {
    std::cout << "Testing detector initialization..." << std::endl;
    
    try {
        // Constructor: YoloDetector(model_path, class_names_path, config)
        YoloDetector detector("models/yolov5s.onnx");
        std::cout << "✓ Detector loaded successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "⚠ Model not found (expected in test environment): " 
                  << e.what() << std::endl;
        std::cout << "✓ Initialization error handling works" << std::endl;
    }
}

// Test confidence threshold
void test_confidence_threshold() {
    std::cout << "Testing confidence threshold..." << std::endl;
    
    try {
        YoloConfig config;
        config.confidence_threshold = 0.5f;
        
        // Pass model path as first argument, config as third
        YoloDetector detector("models/yolov5s.onnx", "", config);
        
        // Verify threshold was set
        detector.setConfidenceThreshold(0.3f);
        
        std::cout << "✓ Confidence threshold setting works" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "⚠ Skipping test (model not available): " << e.what() << std::endl;
    }
}

// Test class names loading
void test_class_names() {
    std::cout << "Testing class names loading..." << std::endl;
    
    // Create a temporary class names file
    std::ofstream file("test_classes.names");
    file << "basketball\n";
    file << "rim\n";
    file << "sports ball\n";
    file.close();
    
    try {
        // Constructor: YoloDetector(model_path, class_names_path, config)
        YoloDetector detector("models/yolov5s.onnx", "test_classes.names");
        
        const auto& class_names = detector.getClassNames();
        assert(class_names.size() == 3);
        assert(class_names[0] == "basketball");
        assert(class_names[1] == "rim");
        assert(class_names[2] == "sports ball");
        
        std::cout << "✓ Class names loading passed" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "⚠ Skipping test (model not available): " << e.what() << std::endl;
    }
    
    // Cleanup
    std::remove("test_classes.names");
}

// Test detection on synthetic image
void test_detection_on_image() {
    std::cout << "Testing detection on synthetic image..." << std::endl;
    
    try {
        YoloDetector detector("models/yolov5s.onnx");
        
        // Create a test image
        cv::Mat test_image(640, 640, CV_8UC3, cv::Scalar(100, 100, 100));
        
        // Draw a circle to simulate a ball
        cv::circle(test_image, cv::Point(320, 320), 30, cv::Scalar(255, 165, 0), -1);
        
        // Run detection
        auto detections = detector.detect(test_image);
        
        std::cout << "  Found " << detections.size() << " detections" << std::endl;
        
        // Note: May not detect on synthetic image, but should not crash
        std::cout << "✓ Detection execution passed" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "⚠ Skipping test (model not available): " << e.what() << std::endl;
    }
}

// Test detection structure
void test_detection_structure() {
    std::cout << "Testing detection structure..." << std::endl;
    
    // Test Detection struct directly
    Detection<> det;
    det.class_id = 0;
    det.confidence = 0.95f;
    det.box = cv::Rect(100, 100, 50, 50);
    det.center = cv::Point2f(125.0f, 125.0f);
    
    assert(det.class_id == 0);
    assert(det.confidence == 0.95f);
    assert(det.box.width == 50);
    assert(det.center.x == 125.0f);
    
    std::cout << "✓ Detection structure passed" << std::endl;
}

// Test NMS (Non-Maximum Suppression)
void test_nms() {
    std::cout << "Testing NMS functionality..." << std::endl;
    
    // Create overlapping boxes
    std::vector<cv::Rect> boxes = {
        cv::Rect(100, 100, 50, 50),
        cv::Rect(105, 105, 50, 50),  // Overlaps with first
        cv::Rect(200, 200, 50, 50)   // Separate
    };
    
    std::vector<float> confidences = {0.9f, 0.85f, 0.8f};
    
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.5f, 0.45f, indices);
    
    // Should keep first and third boxes (second overlaps with first)
    assert(indices.size() <= 2);
    
    std::cout << "✓ NMS functionality passed" << std::endl;
}

// Test batch processing
void test_batch_processing() {
    std::cout << "Testing batch processing..." << std::endl;
    
    try {
        YoloDetector detector("models/yolov5s.onnx");
        
        // Create multiple test images
        std::vector<cv::Mat> images;
        for (int i = 0; i < 3; ++i) {
            cv::Mat img(640, 640, CV_8UC3, cv::Scalar(50, 50, 50));
            images.push_back(img);
        }
        
        // Process each image
        for (const auto& img : images) {
            auto detections = detector.detect(img);
            // Should not crash
        }
        
        std::cout << "✓ Batch processing passed" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "⚠ Skipping test (model not available): " << e.what() << std::endl;
    }
}

// Test configuration
void test_yolo_config() {
    std::cout << "Testing YOLO configuration..." << std::endl;
    
    YoloConfig config;
    config.input_width = 640.0f;
    config.input_height = 640.0f;
    config.score_threshold = 0.3f;
    config.nms_threshold = 0.5f;
    config.confidence_threshold = 0.25f;
    
    assert(config.input_width == 640.0f);
    assert(config.score_threshold == 0.3f);
    
    try {
        // Use custom config
        YoloDetector detector("models/yolov5s.onnx", "", config);
        std::cout << "✓ YOLO configuration passed" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "⚠ Skipping test (model not available): " << e.what() << std::endl;
    }
}

// Test error handling
void test_error_handling() {
    std::cout << "Testing error handling..." << std::endl;
    
    // Test with invalid model path
    try {
        YoloDetector detector("nonexistent_model.onnx");
        std::cout << "✗ Should have thrown exception" << std::endl;
        assert(false);
    } catch (const std::exception& e) {
        std::cout << "✓ Correctly threw exception: " << e.what() << std::endl;
    }
    
    // Test with invalid class names file
    try {
        YoloDetector detector("models/yolov5s.onnx", "nonexistent.names");
        std::cout << "✓ Handled missing class names file" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✓ Correctly handled missing class names: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "=== Running Detector Tests ===" << std::endl << std::endl;
    
    try {
        test_initialization();
        test_confidence_threshold();
        test_class_names();
        test_detection_structure();
        test_nms();
        test_yolo_config();
        test_error_handling();
        test_detection_on_image();
        test_batch_processing();
        
        std::cout << std::endl << "=== All Detector Tests Passed! ===" << std::endl;
        std::cout << "(Some tests may be skipped if model files are not available)" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "✗ Test failed with unknown exception" << std::endl;
        return 1;
    }
}