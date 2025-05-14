import os
import gradio as gr
from utils.model_manager import ModelManager
from ui.main_interface import create_demo
from processing.global_state import GlobalState

# Initialize the model manager and global state
model_manager = ModelManager()
global_state = GlobalState()

def main():
    """Main entry point for the application"""
    # Create the demo interface
    demo = create_demo(model_manager, global_state)
    
    # Launch the demo
    demo.launch(
        share=False,
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        inbrowser=False  # Don't auto-open browser
    )

if __name__ == "__main__":
    main()