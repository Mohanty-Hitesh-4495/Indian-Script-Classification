import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("binary_classification_model.h5")

# Load and preprocess the new image
img_path = r'D:\hitu.jpg'  # Replace with the path to your new image
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match batch size
img_array = img_array / 255.0  # Normalize pixel values

# Make prediction
predictions = model.predict(img_array)

# Convert the probability score to class labels ("Odia" or "Non-Odia")
predicted_class = np.argmax(predictions, axis=1)[0]  # Get the index of the max probability

if predicted_class == 0:
    print("Predicted class: Non-Odia")
else:
    print("Predicted class: Odia")
