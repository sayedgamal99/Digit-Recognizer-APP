# MNIST Digit Classification using CNN with Streamlit

<p align="center">
    <img src="image.png" alt="Project Cover Image" width="600"/>
</p>

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The trained model is deployed as a web application using Streamlit, allowing users to interactively upload images of handwritten digits and receive real-time predictions.

## Project Overview

The MNIST dataset is a widely used benchmark in the field of machine learning, containing 60,000 training images and 10,000 test images of handwritten digits (0-9). This project demonstrates the following key components:

- **Data Preprocessing**: Efficient loading and normalization of the MNIST dataset, along with data augmentation techniques to improve the model's generalization ability.
- **Model Architecture**: Construction of a CNN model incorporating convolutional layers, pooling layers, dropout for regularization, and batch normalization to enhance performance.
- **Training and Evaluation**: Training the CNN on the MNIST dataset, evaluating its performance on unseen data, and achieving high accuracy.

## Streamlit Application

The Streamlit application provides a user-friendly interface that allows users to:

- **Upload Images**: Users can upload their own handwritten digit images in JPG or PNG format.
- **Receive Real-time Predictions**: The app processes the uploaded images and returns predictions for the digits, displaying the predicted class along with confidence scores.
- **Interactive Experience**: Designed to be intuitive and accessible, the application caters to both technical and non-technical users.

## Installation and Usage

- You can use the app directly via <font size=4>[Streamlit Hub](dhttps://digit-recognizer-app-2vttsjj9haryx4mn2bvcrv.streamlit.app/)</font>

<br>

To run this project locally, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/mnist-cnn-streamlit.git
   ```

2. **Navigate into the project directory**:

   ```bash
   cd mnist-cnn-streamlit
   ```

3. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

Once the app is running, open your web browser and go to the URL provided in the terminal to interact with the application.

## Project Structure

- `app.py`: The main Streamlit application script for digit classification.
- `model/`: Contains the pre-trained CNN model file (`mnist_cnn_model.keras`).
- `requirements.txt`: Lists all Python packages required to run the application.
- `README.md`: Documentation for the project.
