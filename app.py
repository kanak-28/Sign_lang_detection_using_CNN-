import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model('ULNN_model2.h5')

# Define class labels
class_labels = [chr(65 + i) for i in range(model.output_shape[1])]  # A-Z

# App UI
st.set_page_config(page_title="sign language Detection", layout="centered")
st.title("🤟 SIGN LANGUAGE DETECTION")
st.write("Upload an image of a hand showing a sign (A-Z), and the model will predict the letter.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Function to preprocess image
def preprocess_image(image):
    image = image.convert("RGB")  # Convert to grayscale
    image = image.resize((256,256))
    img_array = np.array(image)
    img_array = img_array.reshape(1, 256, 256, 3)
    img_array = img_array.astype('float32') / 255.0
    return img_array

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    processed_image = preprocess_image(image)
    
    # Predict
    prediction = model.predict(processed_image)
    predicted_class = class_labels[np.argmax(prediction)]

    st.success(f"🔤 Predicted Sign: **{predicted_class}**")
