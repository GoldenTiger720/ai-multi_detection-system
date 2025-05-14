import time
import cv2
import numpy as np
import pandas as pd

class VideoProcessor:
    """Handles video processing operations"""
    
    def __init__(self, model_manager, global_state):
        self.model_manager = model_manager
        self.global_state = global_state
    
    def predict_video(self, video_path, detector_key, conf_threshold, iou_threshold, image_size):
        """Generator function that processes video frames and yields them in real-time."""
        detector = self.model_manager.get_detector(detector_key)
        self.global_state.start_video_processing()
        
        # Clear previous results
        self.global_state.clear_detection_results()
        
        try:
            # Status indicators
            frame_count = 0
            start_time = time.time()
            last_fps_update = start_time
            fps = 0
            
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                yield np.zeros((300, 400, 3), dtype=np.uint8), pd.DataFrame(
                    columns=["Time", "Detection Type", "Classes", "Confidence", "Frame Size"]
                )
                return
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Process frames
            while cap.isOpened() and self.global_state.video_processing_flag.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                # Convert frame to RGB (YOLO expects RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run prediction on the frame
                annotated_frame, results = detector.predict_video_frame(
                    frame, conf_threshold, iou_threshold, image_size
                )
                
                # Extract detection information with special handling for violence detection
                current_time = frame_count / video_fps
                if results and len(results) > 0:
                    for r in results:
                        boxes = r.boxes
                        if boxes is not None and len(boxes) > 0:
                            classes = boxes.cls.tolist()
                            confidences = boxes.conf.tolist()
                            class_names = [
                                detector.class_names[int(cls)] 
                                if int(cls) < len(detector.class_names) 
                                else f"class_{int(cls)}" 
                                for cls in classes
                            ]
                            
                            # Special logic for violence detection - only add if "violence" is detected
                            should_add_to_table = False
                            detected_class_name = f"Detected {detector_key}"
                            
                            if detector_key == "violence":
                                # Only add to table if "violence" class is actually detected
                                violence_detected = any(
                                    "violence" in name.lower() 
                                    for name in class_names
                                )
                                if violence_detected:
                                    should_add_to_table = True
                                    detected_class_name = "Detected violence"
                            else:
                                # For other detectors, always add to table
                                should_add_to_table = True
                            
                            # Add detection to results with better formatting
                            if should_add_to_table:
                                self.global_state.add_detection_result({
                                    "Time": f"{current_time:.2f}s",
                                    "Detection Type": detector.name,
                                    "Classes": detected_class_name,
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
                df = self.global_state.get_detection_results_df()
                
                # Yield the processed frame and updated table
                yield annotated_frame, df
                
                # Small delay to allow UI to update
                time.sleep(0.01)
                
        finally:
            # Ensure resources are released
            if 'cap' in locals():
                cap.release()
    
    def stop_video_processing(self):
        """Stop the video processing"""
        self.global_state.stop_video_processing()
        return "Video processing stopped"