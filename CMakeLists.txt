cmake_minimum_required(VERSION 3.15)
project(BasketballAnalyser VERSION 1.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -pedantic)
endif()

# Find packages
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Source files for library
set(LIB_SOURCES
    src/tracking/KalmanTracker.cpp
    src/detectors/YoloDetector.cpp
    src/ui/OverlayRenderer.cpp
)

# Create static library
add_library(bbst_lib STATIC ${LIB_SOURCES})
target_link_libraries(bbst_lib 
    PUBLIC ${OpenCV_LIBS}
    PUBLIC Threads::Threads
)

# Main executable
add_executable(basketball_tracker src/app/main.cpp)
target_link_libraries(basketball_tracker PRIVATE bbst_lib)

# Simple tracker executable (optional - comment out if file doesn't exist)
# add_executable(simple_tracker src/app/simple_tracker.cpp)
# target_link_libraries(simple_tracker PRIVATE ${OpenCV_LIBS})

# Tests
enable_testing()

# Test detector
add_executable(test_detector tests/test_detector.cpp)
target_link_libraries(test_detector PRIVATE bbst_lib)
add_test(NAME DetectorTest COMMAND test_detector)

# Test tracker
add_executable(test_tracker tests/test_tracker.cpp)
target_link_libraries(test_tracker PRIVATE bbst_lib)
add_test(NAME TrackerTest COMMAND test_tracker)

# Test model
add_executable(test_model src/app/test_model.cpp)
target_link_libraries(test_model PRIVATE bbst_lib)

# Test trajectory (only if file exists)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_trajectory.cpp")
    add_executable(test_trajectory tests/test_trajectory.cpp)
    target_link_libraries(test_trajectory PRIVATE bbst_lib)
    add_test(NAME TrajectoryTest COMMAND test_trajectory)
endif()

# Install
install(TARGETS basketball_tracker
        RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

# Print configuration
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")