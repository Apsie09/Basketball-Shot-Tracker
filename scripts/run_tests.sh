#!/bin/bash
# Run all tests

cd "$(dirname "$0")/.."

if [ ! -d "build" ]; then
    echo "Build directory not found. Running cmake..."
    mkdir build && cd build
    cmake ..
    make
    cd ..
fi

cd build

# Run tests
echo "Running detector tests..."
./test_detector

echo "Running tracker tests..."
./test_tracker

echo "Running trajectory tests..."
./test_trajectory

echo "All tests completed!"