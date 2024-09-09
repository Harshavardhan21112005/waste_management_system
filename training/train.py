import ultralytics
from ultralytics import YOLO

# Set up your dataset paths
dataset_dir = r"..\data_set"   # Replace with your dataset path
model_name = "yolov8n-cls.pt"  # You can change the model to a larger one like yolov8m-cls.pt or yolov8l-cls.pt
epochs = 100  # Define how many epochs you want to train for
save_dir = "."  # Directory to save the trained model

# Load the YOLOv8 model for classification
model = YOLO(model_name)

# Train the model on the custom dataset
model.train(data=dataset_dir, epochs=epochs, project=save_dir)

# Save the model after training
model.save(f"{save_dir}/best_model.pt")

print("Model training complete and saved!")