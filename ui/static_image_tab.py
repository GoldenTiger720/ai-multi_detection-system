import gradio as gr
from processing.image_processor import ImageProcessor

def create_static_image_tab(model_manager, detector_choices, example_list):
    """Create the static image processing tab"""
    image_processor = ImageProcessor(model_manager)
    
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
            fn=image_processor.predict_image,
            inputs=[image_input, detector_dropdown, conf_slider, iou_slider, img_size_slider],
            outputs=image_output,
        )