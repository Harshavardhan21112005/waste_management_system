import os
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
# Path to your trained model
model_path = r"..\best_model.pt"  # Replace with your trained model path

my_dict = {
    'is_empty': 0,
    'is_empty is_scattered': 1,
    'is_full': 2,
    'is_full is_scattered': 3,
}

# Load the trained YOLOv8 classification model
model = YOLO(model_path)

# Define the directory where your test images are located
test_img_dir = r"C:\Users\DELL\OneDrive\Documents\GitHub\waste_management_system\training\test\test_data\test\is_full"  # Replace with your test images directory
print(model.names)
# Iterate over test images and make predictions
with open(r'C:\Users\DELL\OneDrive\Documents\GitHub\waste_management_system\results.txt', 'w') as file:
    for img_file in os.listdir(test_img_dir):
        img_path = os.path.join(test_img_dir, img_file)
        img = Image.open(img_path)

        # Perform prediction
        results = model.predict(source=img,verbose=False)
        # Display the resultsfor result in results:
        for result in results:
            print(img_file)
            prediction = str(my_dict[model.names[result.probs.top1]])
            file.write(prediction)