import gradio as gr
from training.dataset_manager import DatasetManager
from training.trainer import Trainer

def create_training_tab():
    """Create the training tab"""
    dataset_manager = DatasetManager()
    trainer = Trainer()
    
    with gr.Tab("Training"):
        gr.Markdown("## Model Training Interface")
        
        with gr.Row():
            with gr.Column():
                # Model selection
                gr.Markdown("### 1. Select Model")
                model_selector = gr.Dropdown(
                    choices=trainer.get_available_models(),
                    label="Select Model to Train",
                    info="Choose a pre-trained model to fine-tune"
                )
                
                # Dataset management
                gr.Markdown("### 2. Dataset Management")
                with gr.Row():
                    with gr.Column():
                        dataset_selector = gr.Dropdown(
                            choices=dataset_manager.get_available_datasets(),
                            label="Select Dataset",
                            info="Choose an existing dataset or create a new one"
                        )
                        refresh_datasets_btn = gr.Button("Refresh Datasets", size="sm")
                    
                    with gr.Column():
                        new_dataset_name = gr.Textbox(
                            label="New Dataset Name",
                            placeholder="Enter name for new dataset"
                        )
                        create_dataset_btn = gr.Button("Create Dataset", variant="primary", size="sm")
                
                dataset_info = gr.Textbox(label="Dataset Status", interactive=False)
                
                # Image upload
                gr.Markdown("### 3. Upload Images")
                image_files = gr.File(
                    label="Upload Images",
                    file_count="multiple",
                    file_types=[".jpg", ".jpeg", ".png"]
                )
                upload_btn = gr.Button("Upload to Dataset", variant="secondary")
                upload_status = gr.Textbox(label="Upload Status", interactive=False)
                
                # Data.yaml editor
                gr.Markdown("### 4. Edit data.yaml")
                yaml_editor = gr.Textbox(
                    label="data.yaml content",
                    lines=10,
                    placeholder="Select a dataset to edit its data.yaml file",
                    interactive=True
                )
                with gr.Row():
                    load_yaml_btn = gr.Button("Load data.yaml", size="sm")
                    save_yaml_btn = gr.Button("Save data.yaml", variant="primary", size="sm")
                yaml_status = gr.Textbox(label="YAML Status", interactive=False)
            
            with gr.Column():
                # Training parameters
                gr.Markdown("### 5. Training Parameters")
                with gr.Row():
                    epochs_slider = gr.Slider(
                        minimum=1,
                        maximum=500,
                        value=100,
                        label="Epochs",
                        step=1
                    )
                    batch_size_slider = gr.Slider(
                        minimum=1,
                        maximum=32,
                        value=16,
                        label="Batch Size",
                        step=1
                    )
                
                img_size_training = gr.Dropdown(
                    choices=[320, 416, 512, 608, 640, 736, 832, 1024, 1280],
                    value=640,
                    label="Image Size"
                )
                
                # Training controls
                gr.Markdown("### 6. Training Control")
                with gr.Row():
                    start_training_btn = gr.Button("Start Training", variant="primary", size="lg")
                    stop_training_btn = gr.Button("Stop Training", variant="stop", size="lg")
                
                training_status = gr.Textbox(
                    label="Training Status",
                    lines=5,
                    interactive=False,
                    value="Ready to start training..."
                )
                
                # Training progress (placeholder for future implementation)
                gr.Markdown("### 7. Training Progress")
                progress_placeholder = gr.Markdown("Training progress will be displayed here...")
        
        # Event handlers for training tab
        def refresh_datasets():
            return gr.Dropdown.update(choices=dataset_manager.get_available_datasets())
        
        def on_dataset_created(dataset_name):
            result = dataset_manager.create_new_dataset(dataset_name)
            datasets = dataset_manager.get_available_datasets()
            return result, gr.Dropdown.update(choices=datasets)
        
        def on_images_uploaded(dataset_name, files):
            if not dataset_name:
                return "Please select a dataset first"
            return dataset_manager.upload_images_to_dataset(dataset_name, files)
        
        def on_yaml_loaded(dataset_name):
            if not dataset_name:
                return "Please select a dataset first"
            return dataset_manager.load_data_yaml(dataset_name)
        
        def on_yaml_saved(dataset_name, yaml_content):
            if not dataset_name:
                return "Please select a dataset first"
            return dataset_manager.save_data_yaml(dataset_name, yaml_content)
        
        def on_training_started(model_name, dataset_name, epochs, batch_size, img_size):
            return trainer.start_training(model_name, dataset_name, epochs, batch_size, img_size)
        
        # Connect event handlers
        refresh_datasets_btn.click(
            fn=refresh_datasets,
            outputs=dataset_selector
        )
        
        create_dataset_btn.click(
            fn=on_dataset_created,
            inputs=[new_dataset_name],
            outputs=[dataset_info, dataset_selector]
        )
        
        upload_btn.click(
            fn=on_images_uploaded,
            inputs=[dataset_selector, image_files],
            outputs=upload_status
        )
        
        load_yaml_btn.click(
            fn=on_yaml_loaded,
            inputs=[dataset_selector],
            outputs=yaml_editor
        )
        
        save_yaml_btn.click(
            fn=on_yaml_saved,
            inputs=[dataset_selector, yaml_editor],
            outputs=yaml_status
        )
        
        start_training_btn.click(
            fn=on_training_started,
            inputs=[model_selector, dataset_selector, epochs_slider, batch_size_slider, img_size_training],
            outputs=training_status
        )