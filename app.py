import os
import gradio as gr
from utils.model_manager import ModelManager
from ui.main_interface import create_demo
from processing.global_state import GlobalState

# Initialize the model manager and global state
model_manager = ModelManager()
global_state = GlobalState()

def authenticate(username: str, password: str) -> bool:
    """
    Authenticate users with specific credentials
    
    Args:
        username (str): The provided username
        password (str): The provided password
    
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    # Check if the provided credentials match the required ones
    return username == "avad" and password == "danylo2025"

def main():
    """Main entry point for the application"""
    # Create the demo interface
    demo = create_demo(model_manager, global_state)
    
    # Launch the demo with authentication
    demo.launch(
        share=False,
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        inbrowser=False,  # Don't auto-open browser
        auth=authenticate,  # Add authentication function
        auth_message="Please enter your credentials to access the Multi-Detection AI System"
    )

if __name__ == "__main__":
    main()