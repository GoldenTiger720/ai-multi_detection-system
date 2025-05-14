import time
import subprocess
from pathlib import Path

class Trainer:
    """Handles model training operations"""
    
    def __init__(self):
        self.training_dir = Path("training_runs")
        self.training_dir.mkdir(exist_ok=True)
        self.models_dir = Path("models")
    
    def get_available_models(self):
        """Get available models for training"""
        if not self.models_dir.exists():
            return []
        
        model_files = list(self.models_dir.glob("*.pt"))
        return [f.stem for f in model_files]
    
    def start_training(self, model_name, dataset_name, epochs, batch_size, img_size):
        """Start model training"""
        if not model_name or not dataset_name:
            return "Please select both model and dataset"
        
        # Create training command
        model_path = f"models/{model_name}.pt"
        dataset_path = f"datasets/{dataset_name}/data.yaml"
        
        if not Path(model_path).exists():
            return f"Model file {model_path} not found"
        
        if not Path(dataset_path).exists():
            return f"Dataset file {dataset_path} not found"
        
        # Training command using ultralytics
        cmd = [
            "yolo", "train",
            f"model={model_path}",
            f"data={dataset_path}",
            f"epochs={epochs}",
            f"batch={batch_size}",
            f"imgsz={img_size}",
            f"project={self.training_dir}",
            f"name={model_name}_{dataset_name}_{int(time.time())}"
        ]
        
        try:
            # Run training in background
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return f"Training started! Command: {' '.join(cmd)}\nProcess ID: {process.pid}"
        except Exception as e:
            return f"Error starting training: {str(e)}"