import os
import time
import gradio as gr
import PIL.Image as Image
import numpy as np
import cv2
import yaml
import threading
import subprocess
import shutil
from pathlib import Path
from utils.model_manager import ModelManager
import pandas as pd
from datetime import datetime

# Initialize the model manager
model_manager = ModelManager()
webcam_processing_flag = threading.Event()
current_detection_type = "fire_smoke"  # Default detection type

# Global variable to control video processing
video_processing_flag = threading.Event()
video_processing_flag.set()  # Initially set (not stopped)

# Global variable for detection results
detection_results = []
detection_results_lock = threading.Lock()

def toggle_webcam(is_streaming, detector_key):
    """Toggle webcam on/off and update detection type"""
    global current_detection_type
    
    if is_streaming:
        # Turn off webcam
        webcam_processing_flag.clear()
        current_detection_type = detector_key
        return False, "Turn On Camera", "Camera turned off"
    else:
        # Turn on webcam
        webcam_processing_flag.set()
        current_detection_type = detector_key
        return True, "Turn Off Camera", "Camera turned on"


def predict_image(img, detector_key, conf_threshold, iou_threshold, image_size):
    """Predicts objects in an image using the selected detector"""
    detector = model_manager.get_detector(detector_key)
    return detector.predict_image(img, conf_threshold, iou_threshold, image_size)

def predict_video(video_path, detector_key, conf_threshold, iou_threshold, image_size):
    """Generator function that processes video frames and yields them in real-time."""
    global detection_results
    detector = model_manager.get_detector(detector_key)
    video_processing_flag.set()  # Start processing
    
    # Clear previous results
    with detection_results_lock:
        detection_results = []
    
    try:
        # Status indicators
        frame_count = 0
        start_time = time.time()
        last_fps_update = start_time
        fps = 0
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            yield np.zeros((300, 400, 3), dtype=np.uint8), pd.DataFrame(columns=["Time", "Detection Type", "Classes", "Confidence", "Frame Size"])
            return
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Process frames
        while cap.isOpened() and video_processing_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            # Convert frame to RGB (YOLO expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run prediction on the frame
            annotated_frame, results = detector.predict_video_frame(frame, conf_threshold, iou_threshold, image_size)
            
            # Extract detection information
            current_time = frame_count / video_fps
            if results and len(results) > 0:
                for r in results:
                    boxes = r.boxes
                    if boxes is not None and len(boxes) > 0:
                        classes = boxes.cls.tolist()
                        confidences = boxes.conf.tolist()
                        class_names = [detector.class_names[int(cls)] if int(cls) < len(detector.class_names) else f"class_{int(cls)}" for cls in classes]
                        
                        # Add detection to results with better formatting
                        with detection_results_lock:
                            detection_results.append({
                                "Time": f"{current_time:.2f}s",
                                "Detection Type": detector.name,
                                "Classes": f"Detected { detector_key }",
                                "Confidence": f"{max(confidences):.2f}",
                                "Frame Size": f"{frame_width}x{frame_height}"
                            })
            
            # Calculate FPS every second
            current_time_real = time.time()
            if current_time_real - last_fps_update >= 1.0:  # Update FPS every second
                fps = frame_count / (current_time_real - start_time)
                last_fps_update = current_time_real
            
            # Add status info to the frame
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_count}/{total_frames} | FPS: {fps:.1f} | Detector: {detector.name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            
            # Add progress indicator
            progress_percent = min(100, int((frame_count / total_frames) * 100)) if total_frames > 0 else 0
            cv2.rectangle(annotated_frame, (10, 50), (10 + int(3.8 * progress_percent), 70), (0, 255, 0), -1)
            cv2.rectangle(annotated_frame, (10, 50), (390, 70), (255, 255, 255), 2)
            
            # Create a copy of detection results for yielding
            with detection_results_lock:
                df = pd.DataFrame(detection_results.copy())
            
            # Yield the processed frame and updated table
            yield annotated_frame, df
            
            # Small delay to allow UI to update
            time.sleep(0.01)
            
    finally:
        # Ensure resources are released
        if 'cap' in locals():
            cap.release()

def stop_video_processing():
    """Stop the video processing"""
    video_processing_flag.clear()
    return "Video processing stopped"

# Training-related functions
def get_available_models():
    """Get available models for training"""
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    
    model_files = list(models_dir.glob("*.pt"))
    return [f.stem for f in model_files]

def get_available_datasets():
    """Get available datasets"""
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        datasets_dir.mkdir()
        return []
    
    dataset_dirs = [d for d in datasets_dir.iterdir() if d.is_dir()]
    return [d.name for d in dataset_dirs]

def create_new_dataset(dataset_name):
    """Create a new dataset directory structure"""
    datasets_dir = Path("datasets")
    new_dataset_dir = datasets_dir / dataset_name
    
    if new_dataset_dir.exists():
        return f"Dataset '{dataset_name}' already exists!"
    
    # Create dataset directory structure
    (new_dataset_dir / "images" / "train").mkdir(parents=True)
    (new_dataset_dir / "images" / "val").mkdir(parents=True)
    (new_dataset_dir / "labels" / "train").mkdir(parents=True)
    (new_dataset_dir / "labels" / "val").mkdir(parents=True)
    
    # Create data.yaml file
    data_yaml = {
        'path': str(new_dataset_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'class1'  # Default class
        }
    }
    
    with open(new_dataset_dir / "data.yaml", 'w') as f:
        yaml.dump(data_yaml, f)
    
    return f"Dataset '{dataset_name}' created successfully!"

def upload_images_to_dataset(dataset_name, files):
    """Upload images to dataset"""
    if not files:
        return "No files selected"
    
    dataset_dir = Path("datasets") / dataset_name
    train_images_dir = dataset_dir / "images" / "train"
    
    if not train_images_dir.exists():
        return f"Dataset '{dataset_name}' not found"
    
    uploaded_count = 0
    for file in files:
        if file is not None:
            # Copy file to train images directory
            filename = os.path.basename(file.name)
            shutil.copy(file.name, train_images_dir / filename)
            uploaded_count += 1
    
    return f"Uploaded {uploaded_count} images to dataset '{dataset_name}'"

def get_dataset_images(dataset_name, split="train"):
    """Get images from dataset"""
    if not dataset_name:
        return []
    
    dataset_dir = Path("datasets") / dataset_name
    images_dir = dataset_dir / "images" / split
    
    if not images_dir.exists():
        return []
    
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    return [str(f) for f in image_files]

def load_data_yaml(dataset_name):
    """Load data.yaml content"""
    if not dataset_name:
        return ""
    
    dataset_dir = Path("datasets") / dataset_name
    yaml_file = dataset_dir / "data.yaml"
    
    if not yaml_file.exists():
        return ""
    
    with open(yaml_file, 'r') as f:
        return f.read()

def save_data_yaml(dataset_name, yaml_content):
    """Save data.yaml content"""
    if not dataset_name:
        return "No dataset selected"
    
    dataset_dir = Path("datasets") / dataset_name
    yaml_file = dataset_dir / "data.yaml"
    
    try:
        # Validate YAML syntax
        yaml.safe_load(yaml_content)
        
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        return "data.yaml saved successfully!"
    except yaml.YAMLError as e:
        return f"Error: Invalid YAML syntax: {str(e)}"

def start_training(model_name, dataset_name, epochs, batch_size, img_size):
    """Start model training"""
    if not model_name or not dataset_name:
        return "Please select both model and dataset"
    
    # Create training directory
    training_dir = Path("training_runs")
    training_dir.mkdir(exist_ok=True)
    
    # Create training command
    model_path = f"models/{model_name}.pt"
    dataset_path = f"datasets/{dataset_name}/data.yaml"
    
    if not Path(model_path).exists():
        return f"Model file {model_path} not found"
    
    if not Path(dataset_path).exists():
        return f"Dataset file {dataset_path} not found"
    
    # Training command using ultralytics
    cmd = [
        "yolo", "train",
        f"model={model_path}",
        f"data={dataset_path}",
        f"epochs={epochs}",
        f"batch={batch_size}",
        f"imgsz={img_size}",
        f"project={training_dir}",
        f"name={model_name}_{dataset_name}_{int(time.time())}"
    ]
    
    try:
        # Run training in background
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return f"Training started! Command: {' '.join(cmd)}\nProcess ID: {process.pid}"
    except Exception as e:
        return f"Error starting training: {str(e)}"

# Example images for static image detection
example_list = [["input/" + example] for example in os.listdir("input")] if os.path.exists("input") else []

# Create tabs for different input types
def create_demo():
    # Validate models and prepare detector choices
    model_validation = model_manager.validate_models()
    detector_choices = []
    
    if model_validation["all_valid"]:
        detector_choices = [(detector["name"], detector["key"]) for detector in model_manager.get_available_detectors()]
    else:
        warning_msg = "Warning: Some models are missing! Only the available detectors will be shown."
        print(warning_msg)
        # Filter out detectors with missing models
        missing_keys = [item["key"] for item in model_validation["missing_models"]]
        detector_choices = [(detector["name"], detector["key"]) for detector in model_manager.get_available_detectors()
                           if detector["key"] not in missing_keys]
    
    with gr.Blocks(title="Multi-Detection AI System") as demo:
        gr.Markdown("""# Multi-Detection AI System""")
        
        if not model_validation["all_valid"]:
            missing_model_names = [item["name"] for item in model_validation["missing_models"]]
            gr.Markdown(f"""⚠️ **Warning:** The following detector models are missing: {', '.join(missing_model_names)}. 
            Please make sure all model files are correctly placed in the models directory.""")
        
        with gr.Tabs():
            with gr.Tab("Static Image"):
                with gr.Row():
                    with gr.Column():
                        # Add detector selector dropdown
                        detector_dropdown = gr.Dropdown(
                            choices=detector_choices,
                            label="Detection Type",
                            value=detector_choices[0][1] if detector_choices else None,
                            info="Select the type of detection to perform"
                        )
                        
                        image_input = gr.Image(type="pil", label="Upload Image")
                        
                        # Get initial config for the first detector
                        initial_detector_key = detector_choices[0][1] if detector_choices else "fire_smoke"
                        initial_config = model_manager.get_detector_config(initial_detector_key)
                        
                        conf_slider = gr.Slider(
                            minimum=0, 
                            maximum=1, 
                            value=initial_config["conf_threshold"], 
                            label="Confidence threshold"
                        )
                        iou_slider = gr.Slider(
                            minimum=0, 
                            maximum=1, 
                            value=initial_config["iou_threshold"], 
                            label="IoU threshold"
                        )
                        img_size_slider = gr.Slider(
                            label="Image Size",
                            minimum=320,
                            maximum=1280,
                            step=32,
                            value=initial_config["image_size"],
                        )
                        image_button = gr.Button("Run Inference")
                    
                    with gr.Column():
                        image_output = gr.Image(type="pil", label="Result")
                        # Add a text box to show the detector description
                        detector_info = gr.Markdown("Detector Information")
                
                # Update the description and sliders when detector changes
                def update_detector_settings(detector_key):
                    detector = model_manager.get_detector(detector_key)
                    description = detector.get_description()
                    config = model_manager.get_detector_config(detector_key)
                    
                    return (
                        f"**{detector.name}**: {description}",
                        config["conf_threshold"],
                        config["iou_threshold"],
                        config["image_size"]
                    )
                
                detector_dropdown.change(
                    fn=update_detector_settings,
                    inputs=[detector_dropdown],
                    outputs=[detector_info, conf_slider, iou_slider, img_size_slider]
                )
                
                # Save settings when sliders change
                def save_detector_settings(detector_key, conf, iou, img_size):
                    model_manager.update_detector_config(
                        detector_key, 
                        conf_threshold=conf, 
                        iou_threshold=iou, 
                        image_size=img_size
                    )
                    
                # Connect slider changes to save function
                conf_slider.change(
                    fn=save_detector_settings,
                    inputs=[detector_dropdown, conf_slider, iou_slider, img_size_slider],
                    outputs=[]
                )
                iou_slider.change(
                    fn=save_detector_settings,
                    inputs=[detector_dropdown, conf_slider, iou_slider, img_size_slider],
                    outputs=[]
                )
                img_size_slider.change(
                    fn=save_detector_settings,
                    inputs=[detector_dropdown, conf_slider, iou_slider, img_size_slider],
                    outputs=[]
                )
                
                # Set initial description
                if detector_choices:
                    initial_detector = model_manager.get_detector(detector_choices[0][1])
                    detector_info.value = f"**{initial_detector.name}**: {initial_detector.get_description()}"
                
                # Example images with different detectors
                if example_list:
                    gr.Examples(
                        examples=[
                            [example_list[0][0], detector_choices[0][1] if len(detector_choices) > 0 else None, 0.25, 0.45, 640],
                            [example_list[0][0], detector_choices[1][1] if len(detector_choices) > 1 else detector_choices[0][1], 0.25, 0.45, 640],
                            [example_list[1][0] if len(example_list) > 1 else example_list[0][0], 
                             detector_choices[0][1] if detector_choices else None, 0.25, 0.45, 960],
                        ],
                        inputs=[image_input, detector_dropdown, conf_slider, iou_slider, img_size_slider],
                        outputs=image_output,
                    )
                
                image_button.click(
                    fn=predict_image,
                    inputs=[image_input, detector_dropdown, conf_slider, iou_slider, img_size_slider],
                    outputs=image_output,
                )
                
            with gr.Tab("Video Upload"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Add detector selector dropdown for video
                        video_detector_dropdown = gr.Dropdown(
                            choices=detector_choices,
                            label="Detection Type",
                            value=detector_choices[0][1] if detector_choices else None,
                            info="Select the type of detection to perform"
                        )
                        
                        video_input = gr.Video(label="Upload Video")
                        
                        # Get initial config for the first detector
                        initial_detector_key = detector_choices[0][1] if detector_choices else "fire_smoke"
                        initial_config = model_manager.get_detector_config(initial_detector_key)
                        
                        video_conf_slider = gr.Slider(
                            minimum=0, 
                            maximum=1, 
                            value=initial_config["conf_threshold"], 
                            label="Confidence threshold"
                        )
                        video_iou_slider = gr.Slider(
                            minimum=0, 
                            maximum=1, 
                            value=initial_config["iou_threshold"], 
                            label="IoU threshold"
                        )
                        video_img_size_slider = gr.Slider(
                            label="Image Size",
                            minimum=320,
                            maximum=1280,
                            step=32,
                            value=initial_config["image_size"],
                        )
                        with gr.Row():
                            video_button = gr.Button("Process Video", variant="primary")
                            stop_button = gr.Button("Stop Processing", variant="stop")
                    
                    with gr.Column(scale=2):
                        # Changed from Video to Image for real-time display of frames
                        video_output = gr.Image(label="Real-time Processing")
                        
                        # Add Status indicator
                        with gr.Row():
                            status_indicator = gr.Textbox(label="Status", value="Ready", interactive=False)
                            fps_indicator = gr.Number(label="FPS", value=0, interactive=False)
                        
                        # Add a text box to show the detector description
                        video_detector_info = gr.Markdown("Detector Information")
                        
                        # Add the detection results table with vertical scrolling
                        gr.Markdown("### Detection Results")
                        detection_table = gr.DataFrame(
                            value=pd.DataFrame(columns=["Time", "Detection Type", "Classes", "Confidence", "Frame Size"]),
                            wrap=True,    # Wrap content in cells
                            interactive=False
                        )
                
                # Update the description and sliders when detector changes
                video_detector_dropdown.change(
                    fn=update_detector_settings,
                    inputs=[video_detector_dropdown],
                    outputs=[video_detector_info, video_conf_slider, video_iou_slider, video_img_size_slider]
                )
                
                # Save settings when sliders change
                video_conf_slider.change(
                    fn=save_detector_settings,
                    inputs=[video_detector_dropdown, video_conf_slider, video_iou_slider, video_img_size_slider],
                    outputs=[]
                )
                video_iou_slider.change(
                    fn=save_detector_settings,
                    inputs=[video_detector_dropdown, video_conf_slider, video_iou_slider, video_img_size_slider],
                    outputs=[]
                )
                video_img_size_slider.change(
                    fn=save_detector_settings,
                    inputs=[video_detector_dropdown, video_conf_slider, video_iou_slider, video_img_size_slider],
                    outputs=[]
                )
                
                # Set initial description
                if detector_choices:
                    initial_detector = model_manager.get_detector(detector_choices[0][1])
                    video_detector_info.value = f"**{initial_detector.name}**: {initial_detector.get_description()}"
                
                # Modified to output both frame and table
                video_button.click(
                    fn=predict_video,
                    inputs=[video_input, video_detector_dropdown, video_conf_slider, video_iou_slider, video_img_size_slider],
                    outputs=[video_output, detection_table],
                    # For real-time outputs, we need to indicate this is a generator function
                    api_name="process_video_realtime",
                    scroll_to_output=True
                )
                
                stop_button.click(
                    fn=stop_video_processing,
                    inputs=[],
                    outputs=status_indicator
                )
            
            # Replace the existing RTSP tab section with this updated version

            with gr.Tab("RTSP (Real-time)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Add detector selector dropdown for webcam
                        webcam_detector_dropdown = gr.Dropdown(
                            choices=detector_choices,
                            label="Detection Type",
                            value=detector_choices[0][1] if detector_choices else None,
                            info="Select the type of detection to perform"
                        )
                        
                        # Get initial config for the first detector
                        initial_detector_key = detector_choices[0][1] if detector_choices else "fire_smoke"
                        initial_config = model_manager.get_detector_config(initial_detector_key)
                        
                        webcam_conf_slider = gr.Slider(
                            minimum=0, 
                            maximum=1, 
                            value=initial_config["conf_threshold"], 
                            label="Confidence threshold", 
                            interactive=True
                        )
                        webcam_iou_slider = gr.Slider(
                            minimum=0, 
                            maximum=1, 
                            value=initial_config["iou_threshold"], 
                            label="IoU threshold", 
                            interactive=True
                        )
                        webcam_img_size_slider = gr.Slider(
                            label="Image Size",
                            minimum=320,
                            maximum=1280,
                            step=32,
                            value=initial_config["image_size"],
                            interactive=True,
                        )
                        
                        # Add FPS indicator
                        fps_indicator = gr.Number(label="Processing FPS", value=0, interactive=False)
                        
                        # Add status indicator
                        status_indicator = gr.Textbox(label="Status", value="Ready", interactive=False)
                        
                        # Add a text box to show the detector description
                        webcam_detector_info = gr.Markdown("Detector Information")
                    
                    with gr.Column(scale=2):
                        # Use gr.Image with webcam source for real-time streaming
                        webcam = gr.Image(sources=["webcam"], streaming=True, label="Live Feed", type="numpy")
                        webcam_output = gr.Image(label="Detection Results")
                
                # Update the description and sliders when detector changes
                def update_webcam_detector_settings(detector_key):
                    detector = model_manager.get_detector(detector_key)
                    description = detector.get_description()
                    config = model_manager.get_detector_config(detector_key)
                    
                    status_msg = f"Detector changed to: {detector.name}"
                    
                    return (
                        f"**{detector.name}**: {description}",
                        config["conf_threshold"],
                        config["iou_threshold"],
                        config["image_size"],
                        status_msg
                    )
                
                webcam_detector_dropdown.change(
                    fn=update_webcam_detector_settings,
                    inputs=[webcam_detector_dropdown],
                    outputs=[webcam_detector_info, webcam_conf_slider, webcam_iou_slider, webcam_img_size_slider, status_indicator]
                )
                
                # Save settings when sliders change
                def save_webcam_detector_settings(detector_key, conf, iou, img_size):
                    model_manager.update_detector_config(
                        detector_key, 
                        conf_threshold=conf, 
                        iou_threshold=iou, 
                        image_size=img_size
                    )
                    return "Settings updated"
                
                # Connect slider changes to save function
                webcam_conf_slider.change(
                    fn=save_webcam_detector_settings,
                    inputs=[webcam_detector_dropdown, webcam_conf_slider, webcam_iou_slider, webcam_img_size_slider],
                    outputs=[status_indicator]
                )
                webcam_iou_slider.change(
                    fn=save_webcam_detector_settings,
                    inputs=[webcam_detector_dropdown, webcam_conf_slider, webcam_iou_slider, webcam_img_size_slider],
                    outputs=[status_indicator]
                )
                webcam_img_size_slider.change(
                    fn=save_webcam_detector_settings,
                    inputs=[webcam_detector_dropdown, webcam_conf_slider, webcam_iou_slider, webcam_img_size_slider],
                    outputs=[status_indicator]
                )
                
                # Set initial description
                if detector_choices:
                    initial_detector = model_manager.get_detector(detector_choices[0][1])
                    webcam_detector_info.value = f"**{initial_detector.name}**: {initial_detector.get_description()}"
                
                # Enhanced webcam processing function that includes status updates
                def process_webcam_with_status(video_input, detector_key, conf_threshold, iou_threshold, image_size):
                    """Processes frames from webcam input in real-time with status feedback"""
                    if video_input is None:
                        return None, 0.0, "No camera input"
                        
                    detector = model_manager.get_detector(detector_key)
                    start_time = time.time()
                    
                    try:
                        # Run prediction on the frame
                        frame_rgb = video_input  # Gradio gives us RGB format already
                        frame_bgr = cv2.cvtColor(video_input, cv2.COLOR_RGB2BGR)
                        
                        annotated_frame, _ = detector.predict_video_frame(frame_bgr, conf_threshold, iou_threshold, image_size)
                        
                        # Calculate processing FPS
                        elapsed_time = time.time() - start_time
                        fps = 1 / elapsed_time if elapsed_time > 0 else 0
                        
                        # Add parameter info to the frame
                        cv2.putText(
                            annotated_frame, 
                            f"FPS: {fps:.1f} | Conf: {conf_threshold:.2f} | IoU: {iou_threshold:.2f} | Detector: {detector.name}", 
                            (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            (0, 255, 0), 
                            2
                        )
                        
                        cv2.putText(
                            annotated_frame, 
                            f"Image Size: {image_size}px", 
                            (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            (0, 255, 0), 
                            2
                        )
                        
                        status_msg = f"Detection active: {detector.name}"
                        return annotated_frame, fps, status_msg
                        
                    except Exception as e:
                        # In case of error, return the original frame with error message
                        error_frame = video_input.copy() if video_input is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(
                            error_frame,
                            f"Error: {str(e)}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )
                        return error_frame, 0.0, f"Error: {str(e)}"
                
                # For real-time processing, we use streaming mode
                webcam.stream(
                    fn=process_webcam_with_status,
                    inputs=[webcam, webcam_detector_dropdown, webcam_conf_slider, webcam_iou_slider, webcam_img_size_slider],
                    outputs=[webcam_output, fps_indicator, status_indicator],
                    show_progress=False,
                    stream_every=0.1  # Capture a frame every 0.1 seconds
                )

            with gr.Tab("Training"):
                gr.Markdown("## Model Training Interface")
                
                with gr.Row():
                    with gr.Column():
                        # Model selection
                        gr.Markdown("### 1. Select Model")
                        model_selector = gr.Dropdown(
                            choices=get_available_models(),
                            label="Select Model to Train",
                            info="Choose a pre-trained model to fine-tune"
                        )
                        
                        # Dataset management
                        gr.Markdown("### 2. Dataset Management")
                        with gr.Row():
                            with gr.Column():
                                dataset_selector = gr.Dropdown(
                                    choices=get_available_datasets(),
                                    label="Select Dataset",
                                    info="Choose an existing dataset or create a new one"
                                )
                                refresh_datasets_btn = gr.Button("Refresh Datasets", size="sm")
                            
                            with gr.Column():
                                new_dataset_name = gr.Textbox(
                                    label="New Dataset Name",
                                    placeholder="Enter name for new dataset"
                                )
                                create_dataset_btn = gr.Button("Create Dataset", variant="primary", size="sm")
                        
                        dataset_info = gr.Textbox(label="Dataset Status", interactive=False)
                        
                        # Image upload
                        gr.Markdown("### 3. Upload Images")
                        image_files = gr.File(
                            label="Upload Images",
                            file_count="multiple",
                            file_types=[".jpg", ".jpeg", ".png"]
                        )
                        upload_btn = gr.Button("Upload to Dataset", variant="secondary")
                        upload_status = gr.Textbox(label="Upload Status", interactive=False)
                        
                        # Data.yaml editor
                        gr.Markdown("### 4. Edit data.yaml")
                        yaml_editor = gr.Textbox(
                            label="data.yaml content",
                            lines=10,
                            placeholder="Select a dataset to edit its data.yaml file",
                            interactive=True
                        )
                        with gr.Row():
                            load_yaml_btn = gr.Button("Load data.yaml", size="sm")
                            save_yaml_btn = gr.Button("Save data.yaml", variant="primary", size="sm")
                        yaml_status = gr.Textbox(label="YAML Status", interactive=False)
                    
                    with gr.Column():
                        # Training parameters
                        gr.Markdown("### 5. Training Parameters")
                        with gr.Row():
                            epochs_slider = gr.Slider(
                                minimum=1,
                                maximum=500,
                                value=100,
                                label="Epochs",
                                step=1
                            )
                            batch_size_slider = gr.Slider(
                                minimum=1,
                                maximum=32,
                                value=16,
                                label="Batch Size",
                                step=1
                            )
                        
                        img_size_training = gr.Dropdown(
                            choices=[320, 416, 512, 608, 640, 736, 832, 1024, 1280],
                            value=640,
                            label="Image Size"
                        )
                        
                        # Training controls
                        gr.Markdown("### 6. Training Control")
                        with gr.Row():
                            start_training_btn = gr.Button("Start Training", variant="primary", size="lg")
                            stop_training_btn = gr.Button("Stop Training", variant="stop", size="lg")
                        
                        training_status = gr.Textbox(
                            label="Training Status",
                            lines=5,
                            interactive=False,
                            value="Ready to start training..."
                        )
                        
                        # Training progress (placeholder for future implementation)
                        gr.Markdown("### 7. Training Progress")
                        progress_placeholder = gr.Markdown("Training progress will be displayed here...")
                
                # Event handlers for training tab
                def refresh_datasets():
                    return gr.Dropdown.update(choices=get_available_datasets())
                
                def on_dataset_created(dataset_name):
                    result = create_new_dataset(dataset_name)
                    datasets = get_available_datasets()
                    return result, gr.Dropdown.update(choices=datasets)
                
                def on_images_uploaded(dataset_name, files):
                    if not dataset_name:
                        return "Please select a dataset first"
                    return upload_images_to_dataset(dataset_name, files)
                
                def on_yaml_loaded(dataset_name):
                    if not dataset_name:
                        return "Please select a dataset first"
                    return load_data_yaml(dataset_name)
                
                def on_yaml_saved(dataset_name, yaml_content):
                    if not dataset_name:
                        return "Please select a dataset first"
                    return save_data_yaml(dataset_name, yaml_content)
                
                def on_training_started(model_name, dataset_name, epochs, batch_size, img_size):
                    return start_training(model_name, dataset_name, epochs, batch_size, img_size)
                
                # Connect event handlers
                refresh_datasets_btn.click(
                    fn=refresh_datasets,
                    outputs=dataset_selector
                )
                
                create_dataset_btn.click(
                    fn=on_dataset_created,
                    inputs=[new_dataset_name],
                    outputs=[dataset_info, dataset_selector]
                )
                
                upload_btn.click(
                    fn=on_images_uploaded,
                    inputs=[dataset_selector, image_files],
                    outputs=upload_status
                )
                
                load_yaml_btn.click(
                    fn=on_yaml_loaded,
                    inputs=[dataset_selector],
                    outputs=yaml_editor
                )
                
                save_yaml_btn.click(
                    fn=on_yaml_saved,
                    inputs=[dataset_selector, yaml_editor],
                    outputs=yaml_status
                )
                
                start_training_btn.click(
                    fn=on_training_started,
                    inputs=[model_selector, dataset_selector, epochs_slider, batch_size_slider, img_size_training],
                    outputs=training_status
                )
        
        return demo

# Create the demo interface
demo = create_demo()

if __name__ == "__main__":
    # For Hugging Face Spaces compatibility
    demo.launch(
        share=False,
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        inbrowser=False  # Don't auto-open browser
    )