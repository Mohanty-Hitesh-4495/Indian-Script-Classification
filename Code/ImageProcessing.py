import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Function to preprocess images
def preprocess_image(image_path, target_size=(224, 224)):
    # Read image
    image = cv2.imread(image_path)
    # Resize image
    image = cv2.resize(image, target_size)
    # Normalize pixel values
    image = image.astype("float") / 255.0
    return image

# Define paths to your dataset
dataset_dir = r"D:\CNN Project"
odia_dir = os.path.join(dataset_dir, "Odia Database")
non_odia_dir = os.path.join(dataset_dir, "Non Odia Database")

# Collect all image paths
odia_images = [os.path.join(odia_dir, image) for image in os.listdir(odia_dir)]
non_odia_images = [os.path.join(non_odia_dir, image) for image in os.listdir(non_odia_dir)]

# Preprocess and collect images and labels
# Preprocess and collect images and labels
images = []
labels = []

# Function to load and preprocess images with error handling


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from path {image_path}")
        return None
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary threshold to convert to black and white
    _, bw_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    # Resize image
    bw_image = cv2.resize(bw_image, target_size)
    # Normalize pixel values
    bw_image = bw_image.astype("float") / 255.0
    return bw_image


# Load and preprocess Odia images
for image_path in odia_images:
    image = load_and_preprocess_image(image_path)
    if image is not None:
        images.append(image)
        labels.append(1)  # Odia script label is 1

# Load and preprocess Non-Odia images
for image_path in non_odia_images:
    image = load_and_preprocess_image(image_path)
    if image is not None:
        images.append(image)
        labels.append(0)  # Non-Odia script label is 0


# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split the dataset into training, validation, and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Print the shapes of the datasets
print("Train images shape:", train_images.shape)
print("Train labels shape:", train_labels.shape)
print("Validation images shape:", val_images.shape)
print("Validation labels shape:", val_labels.shape)
print("Test images shape:", test_images.shape)
print("Test labels shape:", test_labels.shape)
