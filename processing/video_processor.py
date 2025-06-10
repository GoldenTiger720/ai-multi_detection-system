import time
import os
import cv2
import numpy as np
import pandas as pd
import socket
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("video_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VideoProcessor")

class VideoProcessor:
    """Handles video processing operations"""
    
    def __init__(self, model_manager, global_state):
        self.model_manager = model_manager
        self.global_state = global_state
        self.output_dir = "output_videos"
        self.current_output_path = None
        self.video_writer = None
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def predict_video(self, video_path, detector_key, conf_threshold, iou_threshold, image_size):
        """Generator function that processes video frames and yields them in real-time."""
        detector = self.model_manager.get_detector(detector_key)
        self.global_state.start_video_processing()
        
        # Clear previous results
        self.global_state.clear_detection_results()
        
        # Initialize the video writer as None to handle early stops
        self.video_writer = None
        
        try:
            # Status indicators
            frame_count = 0
            start_time = time.time()
            last_fps_update = start_time
            fps = 0
            
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video file: {video_path}")
                yield np.zeros((300, 400, 3), dtype=np.uint8), pd.DataFrame(
                    columns=["Time", "Detection Type", "Classes", "Confidence", "Frame Size"]
                )
                return
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Starting video processing: {video_path}")
            logger.info(f"Video properties: {frame_width}x{frame_height}, {total_frames} frames, {video_fps} FPS")
            
            # Create output directory if it doesn't exist (again, just to be safe)
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            # Create output file name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            detector_name = detector.name.replace(" ", "_").lower()
            video_name = os.path.basename(video_path).split('.')[0]
            self.current_output_path = os.path.join(
                self.output_dir, 
                f"{video_name}_{detector_name}_{timestamp}.mp4"
            )
            
            logger.info(f"Output will be saved to: {self.current_output_path}")
            
            # Define codec and create VideoWriter object
            # Use H.264 codec which is more compatible with web browsers
            if os.name == 'nt':  # For Windows
                # Try with the DIVX codec (H.264 variant available on Windows)
                fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                
            self.current_output_path = os.path.join(
                self.output_dir, 
                f"{video_name}_{detector_name}_{timestamp}.mp4"
            )
                
            self.video_writer = cv2.VideoWriter(
                self.current_output_path, 
                fourcc, 
                video_fps, 
                (frame_width, frame_height)
            )
            
            if not self.video_writer.isOpened():
                logger.warning(f"Failed to create video writer with primary codec. Trying fallback options.")
                
                # First fallback: Try with mp4v
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.current_output_path = os.path.join(
                    self.output_dir, 
                    f"{video_name}_{detector_name}_{timestamp}_mp4v.mp4"
                )
                
                self.video_writer = cv2.VideoWriter(
                    self.current_output_path, 
                    fourcc, 
                    video_fps, 
                    (frame_width, frame_height)
                )
                
                # If that also fails, try XVID with AVI
                if not self.video_writer.isOpened():
                    logger.warning("mp4v codec failed too. Trying XVID with AVI format.")
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    self.current_output_path = os.path.join(
                        self.output_dir, 
                        f"{video_name}_{detector_name}_{timestamp}.avi"
                    )
                    
                    self.video_writer = cv2.VideoWriter(
                        self.current_output_path, 
                        fourcc, 
                        video_fps, 
                        (frame_width, frame_height)
                    )
                    
                    # Last resort: MJPG
                    if not self.video_writer.isOpened():
                        logger.warning("XVID codec failed too. Trying MJPG as last resort.")
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        self.current_output_path = os.path.join(
                            self.output_dir, 
                            f"{video_name}_{detector_name}_{timestamp}.avi"
                        )
                        
                        self.video_writer = cv2.VideoWriter(
                            self.current_output_path, 
                            fourcc, 
                            video_fps, 
                            (frame_width, frame_height)
                        )
            
            # Process frames
            while cap.isOpened() and self.global_state.video_processing_flag.is_set():
                try:
                    ret, frame = cap.read()
                    if not ret:
                        logger.info("End of video reached")
                        break
                        
                    frame_count += 1
                    
                    # Run prediction on the frame
                    # Note: we pass the frame in BGR format as is, because predict_video_frame handles the conversion
                    annotated_frame, results = detector.predict_video_frame(
                        frame, conf_threshold, iou_threshold, image_size
                    )
                    
                    # annotated_frame is in BGR format from YOLO's plot() function
                    # Convert to RGB for display in the UI
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
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
                   
                    self.video_writer.write(annotated_frame)
                    
                    # Create a copy of detection results for yielding
                    df = self.global_state.get_detection_results_df()
                    
                    # Yield the RGB processed frame for display and updated table
                    yield annotated_frame_rgb, df
                    
                    # Small delay to allow UI to update
                    time.sleep(0.01)
                        
                except socket.error as e:
                    # Handle network errors gracefully
                    logger.error(f"Network error during processing: {str(e)}")
                    # Continue processing but don't yield
                    continue
                except Exception as e:
                    logger.error(f"Error processing frame {frame_count}: {str(e)}")
                    # Continue with next frame if possible
                    continue
                
        except ConnectionResetError as e:
            logger.error(f"Connection reset error: {str(e)}")
            # Make sure to finalize the video
            if self.video_writer is not None and hasattr(self.video_writer, 'release'):
                self.video_writer.release()
                logger.info(f"Video saved to {self.current_output_path} after connection error")
        except Exception as e:
            logger.error(f"Error in video processing: {str(e)}")
        finally:
            logger.info("Finishing video processing")
            # Ensure resources are released
            if 'cap' in locals() and cap is not None:
                cap.release()
            
            # Make sure the video writer is properly closed
            if self.video_writer is not None:
                try:
                    self.video_writer.release()
                    logger.info(f"Video saved to: {self.current_output_path}")
                    
                    # Convert to web-compatible format
                    web_compatible_path = self.convert_to_web_compatible(self.current_output_path)
                    
                    # Update the current output path if conversion succeeded
                    if web_compatible_path != self.current_output_path:
                        self.current_output_path = web_compatible_path
                        logger.info(f"Web-compatible video created at: {web_compatible_path}")
                    
                    # Add the output path to the detection results for retrieval
                    self.global_state.add_detection_result({
                        "Time": "Complete",
                        "Detection Type": "Output File",
                        "Classes": "Saved Video",
                        "Confidence": "",
                        "Frame Size": self.current_output_path
                    })
                except Exception as e:
                    logger.error(f"Error releasing video writer: {str(e)}")
    
    def stop_video_processing(self):
        """Stop the video processing"""
        logger.info("Stopping video processing")
        self.global_state.stop_video_processing()
        
        # Ensure the video writer is properly closed if it exists
        if self.video_writer is not None:
            try:
                self.video_writer.release()
                self.video_writer = None
                logger.info(f"Video saved to {self.current_output_path} after manual stop")
                
                # Convert to web-compatible format
                web_compatible_path = self.convert_to_web_compatible(self.current_output_path)
                
                # Update the current output path if conversion succeeded
                if web_compatible_path != self.current_output_path:
                    self.current_output_path = web_compatible_path
                    logger.info(f"Web-compatible video created at: {web_compatible_path}")
                
                # Generate message with output path
                output_msg = f"Video processing stopped. Output saved to: {self.current_output_path}"
                return output_msg
            except Exception as e:
                logger.error(f"Error when releasing video writer: {str(e)}")
                return f"Video processing stopped with error: {str(e)}"
        
        return "Video processing stopped (no active video writer)"
    
    def convert_to_web_compatible(self, video_path):
        """
        Converts a video to a web-compatible format using FFmpeg if it's available.
        This is a fallback method to ensure videos can be played in web browsers.
        
        Args:
            video_path: Path to the original video
            
        Returns:
            str: Path to the converted video or original if conversion failed
        """
        try:
            # Check if ffmpeg is available
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            ffmpeg_available = result.returncode == 0
        except:
            ffmpeg_available = False
            logger.warning("FFmpeg not found, skipping web-compatible conversion")
            return video_path
        
        if not ffmpeg_available:
            return video_path
        
        logger.info(f"Converting {video_path} to web-compatible format using FFmpeg")
        
        # Create new filename for the web-compatible video
        base_dir = os.path.dirname(video_path)
        filename = os.path.basename(video_path)
        name, ext = os.path.splitext(filename)
        web_path = os.path.join(base_dir, f"{name}_web.mp4")
        
        try:
            # Use FFmpeg to convert the video to H.264 in MP4 container (web compatible)
            command = [
                'ffmpeg',
                '-i', video_path,                # Input file
                '-c:v', 'libx264',               # H.264 codec
                '-preset', 'fast',               # Encoding speed/compression tradeoff
                '-crf', '23',                    # Quality (lower = better)
                '-pix_fmt', 'yuv420p',           # Pixel format for compatibility
                '-y',                            # Overwrite output file if it exists
                web_path                         # Output file
            ]
            
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if result.returncode == 0 and os.path.exists(web_path):
                logger.info(f"Successfully converted video to web format: {web_path}")
                return web_path
            else:
                logger.error(f"FFmpeg conversion failed: {result.stderr.decode()}")
                return video_path
                
        except Exception as e:
            logger.error(f"Error during FFmpeg conversion: {str(e)}")
            return video_path
    
    def get_last_output_path(self):
        """Get the path to the last generated output video"""
        return self.current_output_path