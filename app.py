import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained MNIST model
model = load_model('model/best_model_v2.keras')

def preprocess_image(image):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    # Check if the background is light (white) or dark (black)
    if np.mean(gray) > 127:  # Adjust threshold as necessary
        # Invert the image
        gray = cv2.bitwise_not(gray)

    gray = gray.resize((28, 28))  # Resize to match MNIST dimensions
    gray = np.array(gray)
    gray = gray.reshape(1, 28, 28, 1)  # Add batch dimension
    return gray

# Streamlit app layout
st.title('MNIST Digit Classification: Upload or Take Picture')

# Choice between upload and camera
image_source = st.selectbox("Choose image source:", ("Upload Image", "Take Picture"))

if image_source == "Upload Image":
    uploaded_file = st.file_uploader("Upload a handwritten digit image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess and predict
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]

        st.write(f"Predicted Digit: {predicted_class}")
        st.write("Confidence Scores:")
        st.bar_chart(prediction[0])

elif image_source == "Take Picture":
    # Capture image from the webcam
    image_file = st.camera_input("Take a picture of a handwritten digit")

    if image_file is not None:
        image = Image.open(image_file)
        st.image(image, caption='Captured Image', use_column_width=True)

        # Preprocess and predict
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]

        st.write(f"Predicted Digit: {predicted_class}")
        st.write("Confidence Scores:")
        st.bar_chart(prediction[0])
else:
    st.error("Invalid image source selection.")