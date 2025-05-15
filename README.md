# Multi-Detection AI System

This application uses YOLOv11 models to perform multiple types of detection tasks including:

- Fire and smoke detection
- Fall detection
- Violence detection
- Choking detection

## Project Structure

```
project/
├── app.py                     # Main application entry point
├── run_with_reload.py         # Auto-reload server when code changes
├── processing/                # Processing modules
│   ├── __init__.py
│   ├── global_state.py        # Global state management
│   ├── image_processor.py     # Static image processing
│   ├── video_processor.py     # Video processing with detection filtering
│   └── webcam_processor.py    # Real-time webcam processing
├── ui/                        # User interface modules
│   ├── __init__.py
│   ├── main_interface.py      # Main demo interface
│   ├── static_image_tab.py    # Static image tab
│   ├── video_upload_tab.py    # Video upload tab
│   ├── webcam_tab.py          # Webcam/RTSP tab
│   └── training_tab.py        # Model training tab
├── training/                  # Training modules
│   ├── __init__.py
│   ├── dataset_manager.py     # Dataset creation and management
│   └── trainer.py             # Model training functionality
├── detectors/                 # Detector modules
│   ├── __init__.py            # Makes detectors a Python package
│   ├── base_detector.py       # Abstract base class for all detectors
│   ├── fire_smoke_detector.py # Fire and smoke detection module
│   ├── fall_detector.py       # Fall detection module
│   ├── violence_detector.py   # Violence detection module
│   └── choking_detector.py    # Choking detection module
├── models/                    # Model files
│   ├── fire_smoke.pt          # Fire and smoke model
│   ├── fall.pt                # Fall detection model
│   ├── vilonce1.pt            # Violence detection model
│   └── choking.pt             # Choking detection model
├── input/                     # Example images folder
├── datasets/                  # Training datasets folder (auto-created)
├── training_runs/             # Training output folder (auto-created)
└── utils/                     # Utility functions
    ├── __init__.py
    └── model_manager.py       # Handles model loading and switching
```

## Key Features

**Modular Architecture**

- Processing Layer: Separate modules for image, video, and webcam processing
- UI Layer: Individual components for each tab interface
- Training Layer: Complete training pipeline with dataset management
- Global State Management: Centralized state handling for real-time operations

## Enhanced Violence Detection

- Smart filtering: Only logs detections when "violence" is actually detected
- Reduces false positives in detection logs
- Provides more meaningful detection results
- Training Capabilities
- Create and manage datasets
- Upload training images
- Configure training parameters
- Start training with custom models

## Setup Instructions

**Install dependencies:**

```
pip install ultralytics gradio pillow opencv-python pandas pyyaml
```

## Prepare model files: Place your trained YOLOv11 models in the models/ folder:

```
models/
├── fire_smoke.pt
├── fall.pt
├── vilonce1.pt
└── choking.pt
```

Add example images (optional): Place some example images in the input/ folder for the interface examples.
Running the Application
Start the application with:

```
python run_with_reload.py app.py

```

**Or run directly:**

```
python app.py

```

The web interface will be accessible in your browser at:

```
http://127.0.0.1:7860/
```

## Using the Application

**Web Interface**
The application provides a user-friendly web interface for easy interaction. You can upload images, videos, or use your webcam for real-time detection.
The interface is built using Gradio, allowing for quick and responsive interactions.
**Real-time Processing**
The application supports real-time processing using your webcam or RTSP streams. You can adjust parameters like confidence threshold and IoU threshold for optimal performance.
**Training Capabilities**
The application allows you to create and manage training datasets. You can upload images, configure the dataset, and start training with custom parameters.
The training process is fully integrated into the application, making it easy to train new models based on your specific needs.

## Using the Interface

**The application has four main tabs:**

1. **Static Image**

- Upload an image to perform detection
- Select the detection type from the dropdown
- Adjust confidence threshold, IoU threshold, and image size
- Click "Run Inference" to process the image

2. **Video Upload**

- Upload a video file to process frame by frame
- Real-time processing with live frame display
- Detection results table with timestamps
- Note: Violence detection only logs actual violence detections

3. **RTSP (Real-time)**

- Use your webcam for real-time detection
- Live streaming with detection overlay
- Adjustable parameters for real-time tuning
- Performance indicators (FPS, status)

4. **Training**

- Create and manage training datasets
- Upload images for training
- Edit dataset configuration (data.yaml)
- Configure and start training with custom parameters
- Choose from available pre-trained models

## Detection Types

- Fire and Smoke Detection
- Detects fire and smoke in images/videos
- Optimized for early fire detection
- Fall Detection

Create detector class:

# detectors/your_detector.py

```
from detectors.base_detector import BaseDetector

class YourDetector(BaseDetector):
    def __init__(self):
        super().__init__(model_path="./models/your_model.pt", name="Your Detection")

    def get_description(self):
        return "Description of your detector"

    def get_model_info(self):
        return {
            "name": self.name,
            "path": self.model_path,
            "classes": self.class_names,
            "type": "YOLOv11 Object Detection"
        }
```

Add model file: Place your trained model in:
`models/your_model.pt`

- Update imports: Add your detector to detectors/**init**.py:

```
from detectors.your_detector import YourDetector

__all__ = [
    'FireSmokeDetector',
    'FallDetector',
    'ViolenceDetector',
    'ChokingDetector',
    'YourDetector'  # Add your detector here
]
```

- Update model manager: Add your detector to utils/model_manager.py:

```
from detectors import YourDetector

self.detectors = {
    "fire_smoke": FireSmokeDetector(),
    "fall": FallDetector(),
    "violence": ViolenceDetector(),
    "choking": ChokingDetector(),
    "your_detector": YourDetector()  # Add your detector here
}
```

## Dependencies

- ultralytics: YOLO model framework
- gradio: Web interface framework
- opencv-python: Image and video processing
- pillow: Image manipulation
- pandas: Data handling for results
- pyyaml: YAML configuration handling
