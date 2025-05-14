import threading
import pandas as pd

class GlobalState:
    """Global state management for the application"""
    
    def __init__(self):
        # Webcam processing control
        self.webcam_processing_flag = threading.Event()
        self.current_detection_type = "fire_smoke"  # Default detection type
        
        # Video processing control
        self.video_processing_flag = threading.Event()
        self.video_processing_flag.set()  # Initially set (not stopped)
        
        # Detection results
        self.detection_results = []
        self.detection_results_lock = threading.Lock()
    
    def clear_detection_results(self):
        """Clear the detection results"""
        with self.detection_results_lock:
            self.detection_results.clear()
    
    def add_detection_result(self, result):
        """Add a detection result to the list"""
        with self.detection_results_lock:
            self.detection_results.append(result)
    
    def get_detection_results_df(self):
        """Get detection results as a pandas DataFrame"""
        with self.detection_results_lock:
            return pd.DataFrame(self.detection_results.copy())
    
    def set_current_detection_type(self, detection_type):
        """Set the current detection type"""
        self.current_detection_type = detection_type
    
    def start_webcam(self):
        """Start webcam processing"""
        self.webcam_processing_flag.set()
    
    def stop_webcam(self):
        """Stop webcam processing"""
        self.webcam_processing_flag.clear()
    
    def start_video_processing(self):
        """Start video processing"""
        self.video_processing_flag.set()
    
    def stop_video_processing(self):
        """Stop video processing"""
        self.video_processing_flag.clear()