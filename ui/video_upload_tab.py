import gradio as gr
import pandas as pd
from processing.video_processor import VideoProcessor

def create_video_upload_tab(model_manager, global_state, detector_choices):
    """Create the video upload processing tab"""
    video_processor = VideoProcessor(model_manager, global_state)
    
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
        
        video_detector_dropdown.change(
            fn=update_detector_settings,
            inputs=[video_detector_dropdown],
            outputs=[video_detector_info, video_conf_slider, video_iou_slider, video_img_size_slider]
        )
        
        # Save settings when sliders change
        def save_detector_settings(detector_key, conf, iou, img_size):
            model_manager.update_detector_config(
                detector_key, 
                conf_threshold=conf, 
                iou_threshold=iou, 
                image_size=img_size
            )
            
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
            fn=video_processor.predict_video,
            inputs=[video_input, video_detector_dropdown, video_conf_slider, video_iou_slider, video_img_size_slider],
            outputs=[video_output, detection_table],
            # For real-time outputs, we need to indicate this is a generator function
            api_name="process_video_realtime",
            scroll_to_output=True
        )
        
        stop_button.click(
            fn=video_processor.stop_video_processing,
            inputs=[],
            outputs=status_indicator
        )