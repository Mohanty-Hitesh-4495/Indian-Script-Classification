import os
import tensorflow as tf
from tensorflow.keras.utils import plot_model

# Specify Graphviz executable path
os.environ["PATH"] += os.pathsep + r'C:\Users\mohan\AppData\Local\Programs\Python\Python312\Lib\site-packages\graphviz\bin'  

# Load the trained model
model = tf.keras.models.load_model("binary_classification_model.h5")

# Display the model summary
model.summary()

# Plot the model architecture
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

