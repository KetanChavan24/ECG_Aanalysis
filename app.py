import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Set the page title and icon
st.set_page_config(page_title="ECG Heart Attack Prediction", page_icon="❤️")

# Title of the app
st.title("ECG Heart Attack Prediction")
st.write("Upload an ECG image to check for signs of a heart attack.")

# Load the trained model
model_path = "Models/heart_attack_cnn_model_final.h5"
model = load_model(model_path)

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Resize the image to match the model's input size
    img_size = (100, 100)
    img = image.resize(img_size)
    
    # Convert the image to grayscale
    img = img.convert("L")
    
    # Convert the image to a numpy array
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
        1: "Myocardial Infarction (Heart Attack)",
        2: "Normal",
        3: "History of Myocardial Infarction"
    }
    return class_names.get(class_index, "Unknown")

# File uploader for ECG image
uploaded_file = st.file_uploader("Upload an ECG image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded ECG Image", use_column_width=True)
    
    # Preprocess the image
    img_array = preprocess_image(image)
    
    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_class_name = get_class_name(predicted_class)
    
    # Display the prediction result
    st.subheader("Prediction Result")
    if predicted_class_name == "Myocardial Infarction (Heart Attack)":
        st.error(f"**Warning:** The ECG report indicates a **{predicted_class_name}**.")
    else:
        st.success(f"The ECG report is classified as **{predicted_class_name}**.")
    
    # Display prediction probabilities
    st.write("Prediction Probabilities:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"- {get_class_name(i)}: {prob * 100:.2f}%")