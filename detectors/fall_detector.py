from detectors.base_detector import BaseDetector

class FallDetector(BaseDetector):
    """Detector specialized for fall detection"""
    
    def __init__(self):
        """Initialize the fall detector"""
        super().__init__(model_path="./models/fall.pt", name="Fall Detection")
    
    def get_description(self):
        """Return a description of the detector"""
        return "Detects people falling, which is especially useful for elderly care and safety monitoring."
    
    def get_model_info(self):
        """Return information about the model used by the detector"""
        return {
            "name": self.name,
            "path": self.model_path,
            "classes": self.class_names,
            "type": "YOLOv8 Object Detection"
        }