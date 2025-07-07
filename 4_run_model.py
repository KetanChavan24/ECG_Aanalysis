import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Define paths
model_path = "Models/heart_attack_cnn_model_final.h5"
new_data_dir = "C:\\Users\\AYUSH\\OneDrive\\Desktop\\arinp\\ECG Data\\New Data"  # Replace with the path to your new ECG data
output_file = "predictions.txt"

# Check if the new data directory exists
if not os.path.exists(new_data_dir):
    print(f"Error: The directory '{new_data_dir}' does not exist.")
    print("Please create the directory and place your new ECG images in it.")
    exit()

# Parameters
img_size = (100, 100)  # Resize images to 100x100 pixels (same as training data)

# Load the trained model
model = load_model(model_path)

# Function to preprocess a single image
def preprocess_image(image_path):
    # Load image using PIL
    img = Image.open(image_path)
    
    # Resize image
    img = img.resize(img_size)
    
    # Convert image to grayscale (if needed)
    img = img.convert("L")
    
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Normalize pixel values to [0, 1]
    img_array = img_array / 255.0
    
    # Add batch and channel dimensions
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension
    
    return img_array

# Function to map class indices to class names
def get_class_name(class_index):
    class_names = {
        0: "Abnormal Heartbeat",
        1: "Myocardial Infarction",
        2: "Normal",
        3: "History of Myocardial Infarction"
    }
    return class_names.get(class_index, "Unknown")

# Process new ECG data
predictions = []
for file_name in os.listdir(new_data_dir):
    file_path = os.path.join(new_data_dir, file_name)
    
    # Preprocess the image
    img_array = preprocess_image(file_path)
    
    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_class_name = get_class_name(predicted_class)
    
    # Store the result
    predictions.append(f"{file_name}: {predicted_class_name}")

# Save predictions to a file
with open(output_file, "w") as f:
    for prediction in predictions:
        f.write(prediction + "\n")

print(f"Predictions saved to {output_file}")