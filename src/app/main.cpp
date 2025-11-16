#include "detectors/YoloDetector.hpp"
#include "tracking/KalmanTracker.hpp"
#include "ui/OverlayRenderer.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>

using namespace bbst;
using namespace bbst::tracking;
using namespace bbst::ui;

int main(int argc, char** argv) {
    // Parse arguments
    std::string video_path = argc > 1 ? argv[1] : "data/videos/tyreseMaxey.mp4";
    std::string model_path = "models/basketball_model.onnx";
    std::string names_path = "models/basketball.names";
    std::string output_path = argc > 2 ? argv[2] : "output_tracked.mp4";
    
    try {
        // Initialize detector
        YoloConfig yolo_config;
        yolo_config.confidence_threshold = 0.25f;
        yolo_config.nms_threshold = 0.45f;
        yolo_config.score_threshold = 0.25f;
        
        YoloDetector detector(model_path, names_path, yolo_config);
        
        // Initialize tracker
        TrackerConfig tracker_config;
        tracker_config.max_trajectory_length = 50;
        tracker_config.min_ball_size = 5.0f;
        tracker_config.max_ball_size = 120.0f;
        tracker_config.max_velocity = 70.0f;
        tracker_config.min_aspect_ratio = 0.3f;
        tracker_config.max_aspect_ratio = 3.0f;
        tracker_config.max_frames_without_detection = 20;
        
        KalmanTracker ball_tracker(tracker_config);
        
        // Initialize renderer
        ColorScheme colors;
        OverlayRenderer renderer(colors, 3, 0.5f);
        
        // Open video
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open video " << video_path << std::endl;
            return -1;
        }
        
        // Get video properties
        int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        
        std::cout << "Video: " << frame_width << "x" << frame_height 
                  << " @ " << fps << "fps, " << total_frames << " frames" << std::endl;
        
        // Setup video writer for output
        cv::VideoWriter out;
        int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        out.open(output_path, codec, fps, cv::Size(frame_width, frame_height));
        
        if (!out.isOpened()) {
            std::cerr << "Warning: Could not create output video" << std::endl;
            return -1;
        }
        
        std::cout << "Output will be saved to: " << output_path << std::endl;
        std::cout << "Processing video... Press 'q' to quit" << std::endl;
        
        // Load class names
        std::vector<std::string> class_names;
        std::ifstream names_file(names_path);
        std::string line;
        while (std::getline(names_file, line)) {
            class_names.push_back(line);
        }
        
        int frame_count = 0;
        double total_inference_time = 0.0;
        
        cv::Mat frame;
        while (cap.read(frame)) {
            frame_count++;
            auto start = cv::getTickCount();
            
            // Predict ball position
            cv::Point2f predicted_pos = ball_tracker.predict();
            
            // Detect objects
            auto detections = detector.detect(frame);
            
            // Draw all detections with bounding boxes and labels
            for (const auto& det : detections) {
                std::string class_name = "Unknown";
                if (det.class_id < static_cast<int>(class_names.size())) {
                    class_name = class_names[det.class_id];
                }
                renderer.drawDetection(frame, det, class_name);
            }
            
            // Find best basketball detection
            Detection<>* best_ball = nullptr;
            float best_confidence = 0.0f;
            float best_distance = FLT_MAX;
            
            for (auto& det : detections) {
                if (det.class_id == 0 || det.class_id == 2) {  // basketball or sports ball
                    float size = (det.box.width + det.box.height) / 2.0f;
                    float aspect_ratio = static_cast<float>(det.box.width) / det.box.height;
                    
                    // Validate aspect ratio
                    if (aspect_ratio < tracker_config.min_aspect_ratio || 
                        aspect_ratio > tracker_config.max_aspect_ratio) {
                        continue;
                    }
                    
                    // Validate size
                    if (size < tracker_config.min_ball_size || 
                        size > tracker_config.max_ball_size) {
                        continue;
                    }
                    
                    if (ball_tracker.isActive()) {
                        float distance = cv::norm(det.center - predicted_pos);
                        float max_search_radius = tracker_config.max_velocity * 4.0f;
                        
                        if (distance < max_search_radius) {
                            float score = det.confidence * 100.0f - distance * 0.5f;
                            float current_best = best_confidence * 100.0f - best_distance * 0.5f;
                            
                            if (score > current_best) {
                                best_distance = distance;
                                best_confidence = det.confidence;
                                best_ball = &det;
                            }
                        }
                    } else {
                        if (det.confidence > best_confidence) {
                            best_confidence = det.confidence;
                            best_ball = &det;
                        }
                    }
                }
            }
            
            // Update tracker
            if (best_ball != nullptr) {
                float size = (best_ball->box.width + best_ball->box.height) / 2.0f;
                ball_tracker.update(best_ball->center, size);
            } else {
                ball_tracker.updateWithoutMeasurement();
            }
            
            // Draw trajectory if active and stable
            if (ball_tracker.isActive() && ball_tracker.isStable()) {
                renderer.drawTrajectory(frame, ball_tracker.getTrajectory());
            }
            
            // Calculate timing
            auto end = cv::getTickCount();
            double freq = cv::getTickFrequency() / 1000.0;
            double processing_time = (end - start) / freq;
            total_inference_time += processing_time;
            double processing_fps = 1000.0 / processing_time;
            
            // Draw info overlay
            std::stringstream info_ss;
            info_ss << "Frame: " << frame_count << "/" << total_frames 
                    << " | " << std::fixed << std::setprecision(1) 
                    << processing_time << "ms"
                    << " | " << processing_fps << "fps"
                    << " | Det: " << detections.size()
                    << " | Track: " << (ball_tracker.isActive() ? "Active" : "Lost");
            
            renderer.drawInfo(frame, info_ss.str(), cv::Point(10, 22));
            
            // Write frame to output video
            out.write(frame);
            
            // Display frame
            cv::imshow("Basketball Tracking", frame);
            
            if (cv::waitKey(1) == 'q') {
                std::cout << "\nStopped by user" << std::endl;
                break;
            }
            
            // Progress update
            if (frame_count % 30 == 0) {
                double progress = (static_cast<double>(frame_count) / total_frames) * 100.0;
                std::cout << "Progress: " << std::fixed << std::setprecision(1) 
                         << progress << "% (" << frame_count << "/" 
                         << total_frames << ")" << std::endl;
            }
        }
        
        // Cleanup
        cap.release();
        out.release();
        cv::destroyAllWindows();
        
        // Print statistics
        double avg_time = total_inference_time / frame_count;
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << std::setw(30) << "STATISTICS" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        std::cout << "Total frames: " << frame_count << "/" << total_frames << std::endl;
        std::cout << "Avg processing time: " << std::fixed << std::setprecision(2) 
                  << avg_time << "ms" << std::endl;
        std::cout << "Avg FPS: " << std::setprecision(1) 
                  << (1000.0 / avg_time) << std::endl;
        std::cout << "Output saved to: " << output_path << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}