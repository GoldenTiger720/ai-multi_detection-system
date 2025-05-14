import gradio as gr
from processing.webcam_processor import WebcamProcessor

def create_webcam_tab(model_manager, global_state, detector_choices):
    """Create the webcam/RTSP processing tab"""
    webcam_processor = WebcamProcessor(model_manager, global_state)
    
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
        
        # For real-time processing, we use streaming mode
        webcam.stream(
            fn=webcam_processor.process_webcam_with_status,
            inputs=[webcam, webcam_detector_dropdown, webcam_conf_slider, webcam_iou_slider, webcam_img_size_slider],
            outputs=[webcam_output, fps_indicator, status_indicator],
            show_progress=False,
            stream_every=0.1  # Capture a frame every 0.1 seconds
        )