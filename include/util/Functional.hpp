#pragma once
#include <functional>
#include <vector>
#include <algorithm>
#include <future>

namespace bbst::util {

// Lambda utilities (Topic 37-38)
template<typename T, typename Predicate>
std::vector<T> filter(const std::vector<T>& data, Predicate pred) {
    std::vector<T> result;
    std::copy_if(data.begin(), data.end(), 
                 std::back_inserter(result), pred);
    return result;
}

template<typename T, typename Transform>
auto map(const std::vector<T>& data, Transform func) 
    -> std::vector<decltype(func(std::declval<T>()))> {
    using ResultType = decltype(func(std::declval<T>()));
    std::vector<ResultType> result;
    result.reserve(data.size());
    std::transform(data.begin(), data.end(), 
                   std::back_inserter(result), func);
    return result;
}

// Async processing with futures (Topic 40-42, 45)
template<typename Func, typename... Args>
auto async_execute(Func&& f, Args&&... args) {
    return std::async(std::launch::async, 
                     std::forward<Func>(f), 
                     std::forward<Args>(args)...);
}

} // namespace bbst::util