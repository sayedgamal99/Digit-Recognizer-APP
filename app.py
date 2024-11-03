import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import random

st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",

)

# Load the pre-trained MNIST model


@st.cache_resource
def load_cached_model():
    return load_model('model/best_model_v2.keras')


model = load_cached_model()


def preprocess_image(image):
    if image.mode != "L":
        image = image.convert("L")

    if np.mean(image) > 127:
        image = ImageOps.invert(image)

    image = np.array(image.resize((28, 28), Image.LANCZOS))
    image = image.reshape(1, 28, 28, 1)
    return image


def load_sample_images(folder):
    image_files = [f for f in os.listdir(
        folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    return [os.path.join(folder, img) for img in image_files]


def make_prediction(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Selected Image', use_column_width=True)
    with col2:
        st.write(f"## Predicted Digit: {predicted_class}")
        st.write("### Confidence Scores:")
        st.bar_chart(prediction[0])


# Streamlit app layout
st.title('MNIST Digit Classification')
st.write('Select a method to input a digit image')

# Choice between upload, camera, and samples with "Use Sample Images" as default
image_source = st.selectbox(
    "Choose image source:",
    ("Use Sample Images", "Upload Image", "Take Picture"),
    index=0
)

if image_source == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload a handwritten digit image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        make_prediction(image)

elif image_source == "Take Picture":
    image_file = st.camera_input("Take a picture of a handwritten digit")
    if image_file is not None:
        image = Image.open(image_file)
        make_prediction(image)

elif image_source == "Use Sample Images":
    # Load all sample images
    sample_images = load_sample_images("samples")

    if not sample_images:
        st.warning(
            "Please ensure the 'samples' folder exists and contains image files.")
    else:
        st.write("### Click on a sample image to classify it:")

        # Create a grid layout for the images
        cols = 5  # Number of columns in the grid
        rows = (len(sample_images) + cols - 1) // cols  # Calculate needed rows

        # Create grid
        grid = [st.columns(cols) for _ in range(rows)]

        # Initialize session state for selected image
        if 'selected_image' not in st.session_state:
            st.session_state.selected_image = None

        # Display images in grid with buttons
        for idx, img_path in enumerate(sample_images):
            row = idx // cols
            col = idx % cols

            with grid[row][col]:
                # Load image
                img = Image.open(img_path)
                # Display image
                st.image(img, width=80, use_column_width=False)
                # Add a button below each image
                if st.button(f"Select", key=f"btn_{idx}"):
                    st.session_state.selected_image = img_path

        # Show prediction if an image is selected
        if st.session_state.selected_image:
            st.write("---")
            selected_image = Image.open(st.session_state.selected_image)
            make_prediction(selected_image)
