# Multi-Detection AI System

This application uses YOLOv11 models to perform multiple types of detection tasks including:

- Fire and smoke detection
- Fall detection
- Violence detection
- Choking detection

## Project Structure

```
project/
├── app.py                     # Main application file
├── run_with_reload.py         # When the code change, server will reload automatically.
├── detectors/                 # Detector modules folder
│   ├── __init__.py            # Makes detectors a Python package
│   ├── base_detector.py       # Abstract base class for all detectors
│   ├── fire_smoke_detector.py # Fire and smoke detection module
│   ├── fall_detector.py       # Fall detection module
│   ├── violence_detector.py   # Violence detection module
│   └── choking_detector.py    # Choking detection module
├── models/                    # Models folder
│   ├── fire_smoke.pt          # Fire and smoke model files
│   ├── fall.pt                # Fall detection model files
│   ├── violence1.pt           # Violence detection model files
│   └── choking.pt             # Choking detection model files
├── input/                     # Example images folder
└── utils/                     # Utility functions
    ├── __init__.py
    └── model_manager.py       # Handles model loading and switching
```

## Setup Instructions

1. **Install dependencies**:

   ```
   pip install ultralytics gradio pillow opencv-python
   ```

2. **Prepare model files**:
   Place your trained YOLOv11 models in the respective model folders:

   - `models/fire_smoke.pt` (already available)
   - `models/fall.pt`
   - `models/violence1.pt`
   - `models/choking.pt`

   You can train your own models or use pre-trained models for each detection type.

3. **Add example images**:
   Place some example images in the `input/` folder for the interface examples.

## Running the Application

Start the application with:

```
python run_with_reload app.py
```

The web interface will be accessible in your browser at `http://127.0.0.1:7860/`.

## Using the Interface

The application has three tabs:

1. **Static Image**: Upload an image to perform detection

   - Select the detection type from the dropdown
   - Adjust confidence threshold, IoU threshold, and image size
   - Click "Run Inference" to process the image

2. **Video Upload**: Upload a video file to process

   - Select the detection type
   - Configure detection parameters
   - Click "Process Video" to analyze the video frame by frame

3. **RTSP (Real-time)**: Use your webcam for real-time detection
   - Select the detection type
   - Adjust detection parameters
   - The system will automatically process the webcam feed

## Adding New Detection Types

To add a new detection type:

1. Create a new detector class in the `detectors/` folder by subclassing `BaseDetector`
2. Create a model folder in `models/` and add your trained YOLOv11 model
3. Update the `detectors/__init__.py` file to include your new detector
4. The system will automatically include your new detector in the UI

## Performance Notes

- Smaller image sizes will provide faster processing but may reduce detection accuracy
- Adjust the confidence threshold to reduce false positives/negatives
- Real-time processing performance depends on your hardware capabilities
