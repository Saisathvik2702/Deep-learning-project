import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load your trained model
model = load_model('potatoes.keras')

# Define class names (adjust to your actual classes)
class_names = ['Early Blight', 'Healthy', 'Late Blight']

st.title("Potato Leaf Disease Classifier")

# Upload image through the interface
uploaded_file = st.file_uploader("Upload a potato leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    image = image.convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))  # use your model's input size
    img_array = np.array(img) / 255.0  # normalize if your model expects that
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions)
    confidence = predictions[0][pred_index] * 100

    # Display result
    st.write(f"### Prediction: {class_names[pred_index]}")
    st.write(f"### Confidence: {confidence:.2f}%")
