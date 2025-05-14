class ImageProcessor:
    """Handles image processing operations"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
    
    def predict_image(self, img, detector_key, conf_threshold, iou_threshold, image_size):
        """Predicts objects in an image using the selected detector"""
        detector = self.model_manager.get_detector(detector_key)
        return detector.predict_image(img, conf_threshold, iou_threshold, image_size)