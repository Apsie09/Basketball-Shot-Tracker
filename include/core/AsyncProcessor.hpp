#pragma once
#include <future>
#include <vector>
#include <functional>
#include <opencv4/opencv2/opencv.hpp>

namespace bbst {

class AsyncFrameProcessor {
    using FrameCallback = std::function<void(cv::Mat&)>;
    
    std::vector<std::future<void>> pending_tasks_;
    
public:
    // Async processing with std::async (Topic 45-46)
    template<typename Func>
    void processFrameAsync(cv::Mat frame, Func&& callback) {
        auto future = std::async(std::launch::async, 
            [frame = std::move(frame), 
             cb = std::forward<Func>(callback)]() mutable {
                cb(frame);
            });
        pending_tasks_.push_back(std::move(future));
    }
    
    // Wait for all tasks
    void waitAll() {
        for (auto& task : pending_tasks_) {
            task.wait();
        }
        pending_tasks_.clear();
    }
    
    // Parallel batch processing (Topic 40)
    template<typename Iterator, typename Func>
    void parallelForEach(Iterator begin, Iterator end, Func&& func) {
        std::vector<std::future<void>> futures;
        
        for (auto it = begin; it != end; ++it) {
            futures.push_back(
                std::async(std::launch::async, func, *it)
            );
        }
        
        for (auto& f : futures) {
            f.wait();
        }
    }
};

} // namespace bbst