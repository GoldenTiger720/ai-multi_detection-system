import os
import shutil
import yaml
from pathlib import Path

class DatasetManager:
    """Handles dataset management operations"""
    
    def __init__(self):
        self.datasets_dir = Path("datasets")
        self.datasets_dir.mkdir(exist_ok=True)
    
    def get_available_datasets(self):
        """Get available datasets"""
        if not self.datasets_dir.exists():
            return []
        
        dataset_dirs = [d for d in self.datasets_dir.iterdir() if d.is_dir()]
        return [d.name for d in dataset_dirs]
    
    def create_new_dataset(self, dataset_name):
        """Create a new dataset directory structure"""
        new_dataset_dir = self.datasets_dir / dataset_name
        
        if new_dataset_dir.exists():
            return f"Dataset '{dataset_name}' already exists!"
        
        # Create dataset directory structure
        (new_dataset_dir / "images" / "train").mkdir(parents=True)
        (new_dataset_dir / "images" / "val").mkdir(parents=True)
        (new_dataset_dir / "labels" / "train").mkdir(parents=True)
        (new_dataset_dir / "labels" / "val").mkdir(parents=True)
        
        # Create data.yaml file
        data_yaml = {
            'path': str(new_dataset_dir),
            'train': 'images/train',
            'val': 'images/val',
            'names': {
                0: 'class1'  # Default class
            }
        }
        
        with open(new_dataset_dir / "data.yaml", 'w') as f:
            yaml.dump(data_yaml, f)
        
        return f"Dataset '{dataset_name}' created successfully!"
    
    def upload_images_to_dataset(self, dataset_name, files):
        """Upload images to dataset"""
        if not files:
            return "No files selected"
        
        dataset_dir = self.datasets_dir / dataset_name
        train_images_dir = dataset_dir / "images" / "train"
        
        if not train_images_dir.exists():
            return f"Dataset '{dataset_name}' not found"
        
        uploaded_count = 0
        for file in files:
            if file is not None:
                # Copy file to train images directory
                filename = os.path.basename(file.name)
                shutil.copy(file.name, train_images_dir / filename)
                uploaded_count += 1
        
        return f"Uploaded {uploaded_count} images to dataset '{dataset_name}'"
    
    def get_dataset_images(self, dataset_name, split="train"):
        """Get images from dataset"""
        if not dataset_name:
            return []
        
        dataset_dir = self.datasets_dir / dataset_name
        images_dir = dataset_dir / "images" / split
        
        if not images_dir.exists():
            return []
        
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        return [str(f) for f in image_files]
    
    def load_data_yaml(self, dataset_name):
        """Load data.yaml content"""
        if not dataset_name:
            return ""
        
        dataset_dir = self.datasets_dir / dataset_name
        yaml_file = dataset_dir / "data.yaml"
        
        if not yaml_file.exists():
            return ""
        
        with open(yaml_file, 'r') as f:
            return f.read()
    
    def save_data_yaml(self, dataset_name, yaml_content):
        """Save data.yaml content"""
        if not dataset_name:
            return "No dataset selected"
        
        dataset_dir = self.datasets_dir / dataset_name
        yaml_file = dataset_dir / "data.yaml"
        
        try:
            # Validate YAML syntax
            yaml.safe_load(yaml_content)
            
            with open(yaml_file, 'w') as f:
                f.write(yaml_content)
            return "data.yaml saved successfully!"
        except yaml.YAMLError as e:
            return f"Error: Invalid YAML syntax: {str(e)}"