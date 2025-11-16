#pragma once
#include <string>
#include <map>
#include <variant>
#include <any>

namespace bbst {

// Type-safe configuration (Topic 35)
class Config {
    using Value = std::variant<int, float, double, std::string, bool>;
    std::map<std::string, Value> values_;
    
public:
    // Variadic template for setting multiple values
    template<typename T>
    void set(const std::string& key, T&& value) {
        values_[key] = std::forward<T>(value);
    }
    
    // Template method with type checking
    template<typename T>
    T get(const std::string& key) const {
        auto it = values_.find(key);
        if (it == values_.end()) {
            throw std::runtime_error("Key not found: " + key);
        }
        return std::get<T>(it->second);
    }
    
    // Variadic builder pattern
    template<typename... Args>
    static Config build(Args&&... args) {
        Config config;
        (config.addPair(std::forward<Args>(args)), ...);
        return config;
    }
    
private:
    template<typename K, typename V>
    void addPair(const std::pair<K, V>& pair) {
        set(pair.first, pair.second);
    }
};

} // namespace bbst