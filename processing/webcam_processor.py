import time
import cv2
import numpy as np

class WebcamProcessor:
    """Handles webcam processing operations"""
    
    def __init__(self, model_manager, global_state):
        self.model_manager = model_manager
        self.global_state = global_state
    
    def toggle_webcam(self, is_streaming, detector_key):
        """Toggle webcam on/off and update detection type"""
        if is_streaming:
            # Turn off webcam
            self.global_state.stop_webcam()
            self.global_state.set_current_detection_type(detector_key)
            return False, "Turn On Camera", "Camera turned off"
        else:
            # Turn on webcam
            self.global_state.start_webcam()
            self.global_state.set_current_detection_type(detector_key)
            return True, "Turn Off Camera", "Camera turned on"
    
    def process_webcam_with_status(self, video_input, detector_key, conf_threshold, iou_threshold, image_size):
        """Processes frames from webcam input in real-time with status feedback"""
        if video_input is None:
            return None, 0.0, "No camera input"
            
        detector = self.model_manager.get_detector(detector_key)
        start_time = time.time()
        
        try:
            # Run prediction on the frame
            frame_rgb = video_input  # Gradio gives us RGB format already
            frame_bgr = cv2.cvtColor(video_input, cv2.COLOR_RGB2BGR)
            
            annotated_frame, _ = detector.predict_video_frame(
                frame_bgr, conf_threshold, iou_threshold, image_size
            )
            
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