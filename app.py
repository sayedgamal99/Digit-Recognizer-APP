import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


model = load_model('model/best_model.keras')

def preprocess_image(image):
    image = image.convert('L')  
    image = image.resize((28, 28))  # Resize to match MNIST dimensions
    image = np.array(image)
    image = image.reshape(28, 28, 1)  # Add batch dimension
    return image

# Streamlit app layout
st.title('MNIST Digit Classification using CNN')

uploaded_file = st.file_uploader("Upload a handwritten digit image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image and predict the class
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]

    st.write(f"Predicted Digit: {predicted_class}")
    
    st.write("Confidence Scores:")
    st.bar_chart(prediction[0])
