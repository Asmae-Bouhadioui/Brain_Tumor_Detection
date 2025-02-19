import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model
model = load_model('tumor_model.h5')

# Sidebar
with st.sidebar:
    st.title("Brain Tumor Detector App")
    st.info("This app allows you to upload an MRI image and detect if it contains a tumor or not.")

# Main title and image
st.title("MRI Tumor Classifier")
st.image(r"C:\Users\MTNRA\Videos\Brain_tumor_model\pic3.gif")  # Use raw string for the file path

# File uploader
uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "png"])

if uploaded_file:
    # Preprocess the uploaded image
    image = load_img(uploaded_file, target_size=(128, 128))  # Resize to match model input
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image to array and normalize
    image_array = img_to_array(image) / 255.0  # Normalize pixel values to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make predictions
    prediction = model.predict(image_array)
    class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.write(f"#### Result: {predicted_class} with confidence {confidence:.2f}%")
