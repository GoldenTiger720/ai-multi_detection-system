from abc import ABC, abstractmethod
import numpy as np
from ultralytics import YOLO
import PIL.Image as Image
import cv2

class BaseDetector(ABC):
    """Abstract base class for all detectors"""
    
    def __init__(self, model_path, name):
        """Initialize the detector with a model path and name"""
        self.model_path = model_path
        self.name = name
        self.model = None
        self.class_names = []
        
    def load_model(self):
        """Load the YOLO model"""
        if self.model is None:
            self.model = YOLO(self.model_path)
            # Store class names from the model
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
        return self.model
    
    def predict_image(self, img, conf_threshold, iou_threshold, image_size):
        """Predict objects in an image using the detector's model"""
        # Ensure model is loaded
        model = self.load_model()
        
        results = model.predict(
            source=img,
            conf=conf_threshold,
            iou=iou_threshold,
            show_labels=True,
            show_conf=True,
            imgsz=image_size,
        )

        for r in results:
            im_array = r.plot()
            im = Image.fromarray(im_array[..., ::-1])
            
        return im
    
    def predict_video_frame(self, frame, conf_threshold, iou_threshold, image_size):
        """Process a single video frame using the detector's model"""
        # Ensure model is loaded
        model = self.load_model()
        
        # Run prediction on the frame
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            show_labels=True,
            show_conf=True,
            imgsz=image_size,
        )
        
        # Get the annotated frame
        for r in results:
            annotated_frame = r.plot()
            
        return annotated_frame, results
    
    @abstractmethod
    def get_description(self):
        """Return a description of the detector"""
        pass
    
    @abstractmethod
    def get_model_info(self):
        """Return information about the model used by the detector"""
        pass