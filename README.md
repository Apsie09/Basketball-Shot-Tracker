# Basketball Shot Tracker 🏀

[![Demo Video](https://img.shields.io/badge/Demo-YouTube-red)](https://www.youtube.com/watch?v=NlDKxWA4Llc)
[![Documentation](https://img.shields.io/badge/docs-Doxygen-blue)](docs/html/index.html)
[![Dataset](https://img.shields.io/badge/dataset-Roboflow-purple)](https://universe.roboflow.com/gaga-lala-7qi2v/basketball-ball-1ddrw)

Real-time basketball detection and tracking system using YOLO object detection and Kalman filtering. Developed as a course project for Programming Languages course at TU-Sofia (Year 3, Semester 5).

## 🎥 Demo

[![Basketball Tracking Demo](https://img.youtube.com/vi/NlDKxWA4Llc/maxresdefault.jpg)](https://www.youtube.com/watch?v=NlDKxWA4Llc)

*Click to watch the full demo on YouTube*

## ✨ Features

- **Real-time Detection**: YOLO-based basketball detection
- **Smart Tracking**: Kalman filter for smooth trajectory prediction
- **Visual Feedback**: Trajectory visualization 
- **Performance Metrics**: FPS counter and detection statistics
- **Video Output**: Save tracked videos with overlays
- **Custom YOLO Model**: Trained specifically for basketball detection

## 🏗️ Architecture

```
Basketball_Analyser/
├── include/
│   ├── detectors/     # YOLO detection implementation
│   ├── tracking/      # Kalman filter tracker
│   ├── ui/           # Rendering and visualization
│   └── util/         # Utility functions
├── src/              # Implementation files
├── models/           # YOLO model files (.onnx)
├── notebooks/        # Jupyter notebook for model training
└── docs/             # Doxygen documentation
```

## 🔧 Requirements

### C++ Application
- C++17 or later
- CMake 3.10+
- OpenCV 4.x
- ONNX Runtime

### Model Training
- Python 3.8+
- Ultralytics YOLOv8
- Roboflow
- CUDA-enabled GPU (recommended)

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    libonnxruntime-dev
```

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/basketball-shot-tracker.git
cd basketball-shot-tracker/Basketball_Analyser
```

### 2. Build the C++ application
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 3. Get the YOLO model

**Train your own model**
```bash
# Install Python dependencies
pip install ultralytics roboflow

# Open the training notebook
jupyter notebook notebooks/yolo_basketball_training.ipynb
```

## 🚀 Usage

### Basic Usage
```bash
./basketball_tracker [input_video.mp4] [output_video.mp4]
```

### Example
```bash
./basketball_tracker data/videos/tyreseMaxey.mp4 output_tracked.mp4
```

### Controls
- Press `q` to quit processing

## 🎓 Model Training

The YOLO model was trained using a custom basketball dataset with the following specifications:

- **Dataset**: [Basketball Ball Detection Dataset](https://universe.roboflow.com/gaga-lala-7qi2v/basketball-ball-1ddrw)
- **Model**: YOLOv8s
- **Image Size**: 960x960
- **Epochs**: 30
- **Batch Size**: 16
- **Augmentations**: 
  - Mosaic: 1.0
  - Mixup: 0.15
  - Copy-Paste: 0.30
  - Translation: 0.10
  - Scale: 0.70
  - Horizontal Flip: 0.50

### Training the Model

Follow the Jupyter notebook [`yolo_basketball_training.ipynb`](notebooks/yolo_basketball_training.ipynb) for step-by-step training:

1. **Setup Environment** (Google Colab recommended)
   ```python
   !pip install ultralytics==8.2.103
   ```

2. **Download Dataset**
   ```python
   from roboflow import Roboflow
   rf = Roboflow(api_key="YOUR_API_KEY")
   project = rf.workspace("basketballpe").project("basketball-ball-1ddrw-q7oyg")
   dataset = project.version(2).download("yolov8")
   ```

3. **Train Model**
   ```bash
   yolo detect train \
     data=basketball-ball-2/data.yaml \
     model=yolov8s.pt \
     imgsz=960 epochs=30 batch=16
   ```

4. **Export to ONNX**
   ```python
   from ultralytics import YOLO
   model = YOLO('runs/detect/train/weights/best.pt')
   model.export(format='onnx')
   ```

## 📊 Performance

- **Processing Speed**: ~30-60 FPS (depending on hardware)
- **Detection Accuracy**: Optimized for basketball detection
- **Model Size**: ~50MB (ONNX format)
- **Input Resolution**: 640x640 (configurable)

## Testing

Run the test suite:
```bash
cd build
./test_detector
./test_tracker
./test_trajectory
```

## 📚 Documentation

Full API documentation is available in the `docs/` folder. Generate or view:

```bash
# Generate documentation
doxygen DoxyFile

# Open in browser
xdg-open docs/html/index.html
```

## 🗂️ Project Structure

```
Basketball_Analyser/
├── CMakeLists.txt              # Build configuration
├── DoxyFile                     # Documentation config
├── README.md                    # This file
├── include/                     # Header files
│   ├── core/                   # Core utilities
│   ├── detectors/              # Detection interfaces
│   ├── tracking/               # Tracking algorithms
│   ├── ui/                     # UI components
│   └── util/                   # Helper functions
├── src/                        # Implementation files
│   ├── app/main.cpp           # Main application
│   ├── detectors/             # Detector implementations
│   ├── tracking/              # Tracker implementations
│   └── ui/                    # UI implementations
├── models/                     # YOLO models
│   ├── basketball_model.onnx  # Custom trained model
│   └── basketball.names       # Class names
├── notebooks/                  # Jupyter notebooks
│   └── yolo_basketball_training.ipynb
├── tests/                      # Unit tests
├── data/                       # Sample data
│   ├── images/
│   └── videos/
└── docs/                       # Generated documentation
```

## Course Project Information

**Course**: Programming Languages  
**Year**: 3, Semester 5  
**University**: Technical University of Sofia  
**Author**: Asen Popov

### Project Goals
- Implement real-time object detection in C++
- Apply Kalman filtering for trajectory tracking
- Train custom YOLO model for basketball detection
- Demonstrate modern C++ practices (C++17)
- Integrate computer vision libraries (OpenCV, ONNX)

### Technologies Used
- **C++17**: Modern C++ features and best practices
- **OpenCV**: Computer vision and image processing
- **ONNX Runtime**: Neural network inference
- **CMake**: Cross-platform build system
- **YOLOv8**: State-of-the-art object detection
- **Kalman Filter**: Optimal state estimation

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection framework
- [Roboflow](https://roboflow.com/) for dataset hosting and augmentation tools
- [Basketball Ball Detection Dataset](https://universe.roboflow.com/gaga-lala-7qi2v/basketball-ball-1ddrw) by basketballpe
- OpenCV community for computer vision tools
- Course instructors and peers

## 📧 Contact

Asen Popov - asepopov@tu-sofia.bg

---
