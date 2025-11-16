#include "detectors/YoloDetector.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

namespace bbst {

// Add color definitions
static const cv::Scalar ORANGE = cv::Scalar(0, 165, 255);
static const cv::Scalar GREEN = cv::Scalar(0, 255, 0);
static const cv::Scalar PURPLE = cv::Scalar(255, 0, 255);
static const cv::Scalar BLUE = cv::Scalar(255, 178, 50);
static const cv::Scalar BLACK = cv::Scalar(0, 0, 0);

YoloDetector::YoloDetector(const std::string& model_path,
                           const std::string& class_names_path,
                           const YoloConfig& config)
    : BaseDetector(config.confidence_threshold)
    , config_(config)
{
    try {
        net_ = std::make_unique<cv::dnn::Net>(
            cv::dnn::readNetFromONNX(model_path)
        );
        
        // Use GPU if available
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            net_->setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net_->setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        } else {
            net_->setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_->setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
        
        if (!class_names_path.empty()) {
            loadClassNames(class_names_path);
        }
        
    } catch (const cv::Exception& e) {
        throw std::runtime_error("Failed to load YOLO model: " + std::string(e.what()));
    }
}

void YoloDetector::loadClassNames(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open class names file: " + path);
    }
    
    class_names_.clear();
    std::string line;
    while (std::getline(file, line)) {
        class_names_.push_back(line);
    }
}

cv::Mat YoloDetector::formatYoloInput(const cv::Mat& source) {
    cv::Mat blob;
    cv::dnn::blobFromImage(source, blob, 1.0/255.0, 
                          cv::Size(config_.input_width, config_.input_height),
                          cv::Scalar(), true, false);
    return blob;
}

cv::Mat YoloDetector::preProcess(const cv::Mat& frame) {
    return formatYoloInput(frame);
}

std::vector<Detection<>> YoloDetector::detect(const cv::Mat& frame) {
    // Preprocess
    cv::Mat blob = preProcess(frame);
    
    // Run inference
    net_->setInput(blob);
    std::vector<cv::Mat> outputs;
    net_->forward(outputs, net_->getUnconnectedOutLayersNames());
    
    // Postprocess
    return postProcess(outputs, frame);
}

std::vector<Detection<>> YoloDetector::postProcess(const std::vector<cv::Mat>& outputs,
                                                    const cv::Mat& original_frame) {
    return parseYoloOutput(outputs, original_frame);
}

std::vector<Detection<>> YoloDetector::parseYoloOutput(
    const std::vector<cv::Mat>& outputs,
    const cv::Mat& original_image) 
{
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<Detection<>> detections;
    
    if (outputs.empty()) {
        return detections;
    }
    
    cv::Mat output = outputs[0];
    
    // Reshape if needed
    if (output.dims == 3) {
        output = output.reshape(1, output.size[1]);
    }
    
    // Debug output shape
    std::cout << "Output shape: [" << output.rows << ", " << output.cols << "]" << std::endl;
    
    int num_classes = output.rows - 4;  // First 4 rows are bbox coords
    std::cout << "Detected " << num_classes << " classes in model" << std::endl;
    
    float x_factor = original_image.cols / static_cast<float>(config_.input_width);
    float y_factor = original_image.rows / static_cast<float>(config_.input_height);
    
    // Iterate through detections (columns)
    for (int i = 0; i < output.cols; ++i) {
        // Get bbox coordinates (first 4 rows)
        float cx = output.at<float>(0, i);
        float cy = output.at<float>(1, i);
        float w = output.at<float>(2, i);
        float h = output.at<float>(3, i);
        
        // Get class scores (remaining rows)
        float max_score = 0.0f;
        int max_class_id = -1;
        
        for (int c = 0; c < num_classes; ++c) {
            float score = output.at<float>(4 + c, i);
            if (score > max_score) {
                max_score = score;
                max_class_id = c;
            }
        }
        
        // Filter by confidence threshold
        if (max_score >= config_.confidence_threshold) {
            // Convert from center format to corner format
            float x = (cx - w / 2.0f) * x_factor;
            float y = (cy - h / 2.0f) * y_factor;
            float width = w * x_factor;
            float height = h * y_factor;
            
            boxes.push_back(cv::Rect(
                static_cast<int>(x),
                static_cast<int>(y),
                static_cast<int>(width),
                static_cast<int>(height)
            ));
            confidences.push_back(max_score);
            class_ids.push_back(max_class_id);
        }
    }
    
    std::cout << "Before NMS: " << boxes.size() << " detections" << std::endl;
    
    // Apply NMS
    std::vector<int> nms_result = performNMS(boxes, confidences);
    
    std::cout << "After NMS: " << nms_result.size() << " detections" << std::endl;
    
    // Create Detection objects
    for (int idx : nms_result) {
        Detection<> det;
        det.class_id = class_ids[idx];
        det.confidence = confidences[idx];
        det.box = boxes[idx];
        det.center = cv::Point2f(
            boxes[idx].x + boxes[idx].width / 2.0f,
            boxes[idx].y + boxes[idx].height / 2.0f
        );
        detections.push_back(det);
    }
    
    return detections;
}

std::vector<int> YoloDetector::performNMS(const std::vector<cv::Rect>& boxes,
                                          const std::vector<float>& confidences) {
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, config_.score_threshold, 
                     config_.nms_threshold, indices);
    return indices;
}

} // namespace bbst