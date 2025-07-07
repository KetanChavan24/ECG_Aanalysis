import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import logging

# Define paths
data_dir = "./ECG Data"
subfolders = [
    "Abnormal Heartbeat Patients",
    "Myocardial Infarction Patients",
    "Normal Person",
    "Patient that have History of Myocardial Infraction"
]

# Parameters
img_size = (100, 100)  # Resize images to 100x100 pixels
test_size = 0.15       # 15% for test set
val_size = 0.15        # 15% for validation set

# Initialize lists to store data and labels
data = []
labels = []

# Load image data from subfolders
for i, subfolder in enumerate(subfolders):
    folder_path = os.path.join(data_dir, subfolder)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        # Load image using PIL
        img = Image.open(file_path)
        
        # Resize image
        img = img.resize(img_size)
        
        # Convert image to grayscale (if needed)
        img = img.convert("L")
        
        # Convert image to numpy array
        img_array = np.array(img)
        
        # Normalize pixel values to [0, 1]
        img_array = img_array / 255.0
        
        # Append to data and labels
        data.append(img_array)
        labels.append(i)  # Assign label based on subfolder index

# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Encode labels to one-hot vectors
label_encoder = LabelEncoder()
integer_labels = label_encoder.fit_transform(labels)

# Use OneHotEncoder to convert integer labels to one-hot vectors
onehot_encoder = OneHotEncoder(sparse_output=False)  # Use sparse_output=False for scikit-learn >= 1.2.0
integer_labels = integer_labels.reshape(len(integer_labels), 1)
labels = onehot_encoder.fit_transform(integer_labels)

# Convert one-hot encoded labels back to integers for stratification
y_train_int = np.argmax(labels, axis=1)

# Stratified split for training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42, stratify=y_train_int)

# Stratified split for training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42, stratify=np.argmax(y_train, axis=1))

# Add a channel dimension to X_train, X_val, and X_test
X_train = np.expand_dims(X_train, axis=-1)  # Shape: (num_samples, 100, 100, 1)
X_val = np.expand_dims(X_val, axis=-1)      # Shape: (num_samples, 100, 100, 1)
X_test = np.expand_dims(X_test, axis=-1)     # Shape: (num_samples, 100, 100, 1)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,  # Increased from 10
    width_shift_range=0.2,  # Increased from 0.1
    height_shift_range=0.2,  # Increased from 0.1
    shear_range=0.2,  # Increased from 0.1
    zoom_range=0.2,  # Increased from 0.1
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)  # Now X_train has the correct shape (num_samples, 100, 100, 1)

# Compute class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
class_weights = dict(enumerate(class_weights))

# Visualize class distribution
class_counts = np.sum(y_train, axis=0)
class_names = ["Abnormal Heartbeat", "Myocardial Infarction", "Normal", "History of MI"]
plt.bar(class_names, class_counts)
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.show()

# Create the 'Code' directory if it doesn't exist
os.makedirs("Code", exist_ok=True)

# Save preprocessed data
np.save("Code/X_train.npy", X_train)
np.save("Code/X_val.npy", X_val)
np.save("Code/X_test.npy", X_test)
np.save("Code/y_train.npy", y_train)
np.save("Code/y_val.npy", y_val)
np.save("Code/y_test.npy", y_test)
np.save("Code/class_weights.npy", class_weights)  # Save class weights

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info("Data preprocessing completed!")
logging.info(f"Training data shape: {X_train.shape}")
logging.info(f"Validation data shape: {X_val.shape}")
logging.info(f"Test data shape: {X_test.shape}")