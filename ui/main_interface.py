import os
import gradio as gr
from ui.static_image_tab import create_static_image_tab
from ui.video_upload_tab import create_video_upload_tab
from ui.webcam_tab import create_webcam_tab
from ui.training_tab import create_training_tab

def create_demo(model_manager, global_state):
    """Create the main demo interface"""
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
    
    # Get example images
    example_list = [["input/" + example] for example in os.listdir("input")] if os.path.exists("input") else []
    
    with gr.Blocks(title="Multi-Detection AI System") as demo:
        gr.Markdown("""# Multi-Detection AI System""")
        
        if not model_validation["all_valid"]:
            missing_model_names = [item["name"] for item in model_validation["missing_models"]]
            gr.Markdown(f"""⚠️ **Warning:** The following detector models are missing: {', '.join(missing_model_names)}. 
            Please make sure all model files are correctly placed in the models directory.""")
        
        with gr.Tabs():
            # Create each tab using separate functions
            create_static_image_tab(model_manager, detector_choices, example_list)
            create_video_upload_tab(model_manager, global_state, detector_choices)
            create_webcam_tab(model_manager, global_state, detector_choices)
            create_training_tab()
    
    return demo