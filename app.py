import os
import time
import gradio as gr
import PIL.Image as Image
import numpy as np
import cv2
from utils.model_manager import ModelManager

# Initialize the model manager
model_manager = ModelManager()

def predict_image(img, detector_key, conf_threshold, iou_threshold, image_size):
    """Predicts objects in an image using the selected detector"""
    detector = model_manager.get_detector(detector_key)
    return detector.predict_image(img, conf_threshold, iou_threshold, image_size)

def predict_video(video_path, detector_key, conf_threshold, iou_threshold, image_size):
    """Generator function that processes video frames and yields them in real-time."""
    detector = model_manager.get_detector(detector_key)
    print(f"Processing video with detector: {detector.name}, conf: {conf_threshold}, iou: {iou_threshold}, size: {image_size}", detector_key)
    
    try:
        # Status indicators
        frame_count = 0
        start_time = time.time()
        last_fps_update = start_time
        fps = 0
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            yield np.zeros((300, 400, 3), dtype=np.uint8)  # Return empty frame if video can't be opened
            return
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            # Convert frame to RGB (YOLO expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run prediction on the frame
            annotated_frame, _ = detector.predict_video_frame(frame, conf_threshold, iou_threshold, image_size)
            
            # Calculate FPS every second
            current_time = time.time()
            if current_time - last_fps_update >= 1.0:  # Update FPS every second
                fps = frame_count / (current_time - start_time)
                last_fps_update = current_time
            
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
            
            # Yield the processed frame
            yield annotated_frame
            
            # Small delay to allow UI to update
            time.sleep(0.01)
            
    finally:
        # Ensure resources are released
        if 'cap' in locals():
            cap.release()

def process_webcam(video_input, detector_key, conf_threshold, iou_threshold, image_size):
    """Processes frames from webcam input in real-time."""
    if video_input is None:
        return None, 0.0
        
    detector = model_manager.get_detector(detector_key)
    start_time = time.time()
    
    # Run prediction on the frame
    frame_rgb = video_input  # Gradio gives us RGB format already
    
    annotated_frame, _ = detector.predict_video_frame(frame_rgb, conf_threshold, iou_threshold, image_size)
    
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
    
    # Return both the frame and the calculated FPS
    return annotated_frame, fps

# Example images for static image detection
example_list = [["input/" + example] for example in os.listdir("input")]

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
                    with gr.Column():
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
                        video_button = gr.Button("Process Video")
                    
                    with gr.Column():
                        # Changed from Video to Image for real-time display of frames
                        video_output = gr.Image(label="Real-time Processing")
                        
                        # Add Status indicator
                        with gr.Row():
                            status_indicator = gr.Textbox(label="Status", value="Ready", interactive=False)
                            fps_indicator = gr.Number(label="FPS", value=0, interactive=False)
                        
                        # Add a text box to show the detector description
                        video_detector_info = gr.Markdown("Detector Information")
                
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
                
                video_button.click(
                    fn=predict_video,
                    inputs=[video_input, video_detector_dropdown, video_conf_slider, video_iou_slider, video_img_size_slider],
                    outputs=video_output,
                    # For real-time outputs, we need to indicate this is a generator function
                    api_name="process_video_realtime",
                    scroll_to_output=True
                )
            
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
                        
                        # Add a text box to show the detector description
                        webcam_detector_info = gr.Markdown("Detector Information")
                    
                    with gr.Column(scale=2):
                        # Use gr.Image with webcam source instead of the deprecated Webcam component
                        webcam = gr.Image(sources=["webcam"], streaming=True, label="Live Feed", type="numpy")
                        webcam_output = gr.Image(label="Detection Results")
                
                # Update the description and sliders when detector changes
                webcam_detector_dropdown.change(
                    fn=update_detector_settings,
                    inputs=[webcam_detector_dropdown],
                    outputs=[webcam_detector_info, webcam_conf_slider, webcam_iou_slider, webcam_img_size_slider]
                )
                
                # Save settings when sliders change
                webcam_conf_slider.change(
                    fn=save_detector_settings,
                    inputs=[webcam_detector_dropdown, webcam_conf_slider, webcam_iou_slider, webcam_img_size_slider],
                    outputs=[]
                )
                webcam_iou_slider.change(
                    fn=save_detector_settings,
                    inputs=[webcam_detector_dropdown, webcam_conf_slider, webcam_iou_slider, webcam_img_size_slider],
                    outputs=[]
                )
                webcam_img_size_slider.change(
                    fn=save_detector_settings,
                    inputs=[webcam_detector_dropdown, webcam_conf_slider, webcam_iou_slider, webcam_img_size_slider],
                    outputs=[]
                )
                
                # Set initial description
                if detector_choices:
                    initial_detector = model_manager.get_detector(detector_choices[0][1])
                    webcam_detector_info.value = f"**{initial_detector.name}**: {initial_detector.get_description()}"
                
                # For real-time processing, we use streaming mode
                webcam.stream(
                    fn=process_webcam,
                    inputs=[webcam, webcam_detector_dropdown, webcam_conf_slider, webcam_iou_slider, webcam_img_size_slider],
                    outputs=[webcam_output, fps_indicator],
                    show_progress=False,
                    stream_every=0.1  # Capture a frame every 0.1 seconds
                )
        
        return demo

# Create the demo interface
demo = create_demo()

if __name__ == "__main__":
    # For Hugging Face Spaces compatibility
    demo.launch()