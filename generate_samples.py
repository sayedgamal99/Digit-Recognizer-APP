import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os
import random


def create_simple_samples():
    # Create samples directory if it doesn't exist
    if not os.path.exists('samples'):
        os.makedirs('samples')

    # Load MNIST dataset
    print("Loading MNIST dataset...")
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Number of samples per digit (0-9)
    samples_per_digit = 2

    print(f"Generating {samples_per_digit} samples for each digit (0-9)...")

    # For each digit (0-9)
    for digit in range(10):
        # Get indices where y_test equals the current digit
        digit_indices = np.where(y_test == digit)[0]

        # Randomly select samples_per_digit number of images
        selected_indices = random.sample(
            list(digit_indices), samples_per_digit)

        # Save each selected image
        for idx, image_idx in enumerate(selected_indices):
            # Get the image
            image = x_test[image_idx]

            # Convert to PIL Image
            pil_image = Image.fromarray(image)

            # Convert to L mode (grayscale)
            pil_image = pil_image.convert('L')

            # Randomly decide whether to invert the image (50% chance)
            if random.random() > 0.5:
                pil_image = ImageOps.invert(pil_image)

            # Resize slightly larger for better visibility
            pil_image = pil_image.resize((100, 100), Image.LANCZOS)

            # Save with descriptive filename
            filename = f'digit_{digit}_sample_{idx+1}.png'
            filepath = os.path.join('samples', filename)
            pil_image.save(filepath)
            print(f'Saved {filename}')


if __name__ == "__main__":
    create_simple_samples()
    print("\nSample images have been created in the 'samples' directory.")
    print("Total images created:", 20)  # 10 digits * 2 samples per digit
