# train_model.py

# Step 1: Import necessary libraries
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

print("Libraries imported successfully!")

# --- Project Setup ---
# Define the path to your dataset folder
dataset_path = 'dataset' 
# Define the image size for the model
IMG_WIDTH, IMG_HEIGHT = 150, 150

# --- Will be filled in the next steps ---

# Step 2: Load and Preprocess Data
# (We will write this code together next)
print("Data loading and preprocessing step is next...")


# Step 3: Split the Data
# (This will come after data loading)
print("Data splitting step is next...")


# Step 4: Build the CNN Model
# (We will define the model architecture here)
print("Model building step is next...")


# Step 5: Compile and Train the Model
# (This is where the training happens)
print("Model training step is next...")


# Step 6: Save the Trained Model
# (The final step of this script)
print("Saving the model will be the final step.")