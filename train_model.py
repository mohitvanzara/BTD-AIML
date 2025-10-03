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
# define the number of classes
NUM_CLASSES = 4
# Step 2: Load and Preprocess Data
# (We will write this code together next)
print("Data loading and preprocessing step is next...")
# ######################################
# ## NEW CODE STARTS HERE - STEP 2    ##
# ######################################
def load_data(dataset_path, img_width, img_height):
    """
    Loads images and labels from the dataset directory.
    """
    images = []
    labels = []
    # The order is important: glioma=0, meningioma = 1, notumor = 2 pituitary=3 as per sorted order
    # We get this order from os.listdir, so let's sort it to be consistent
    class_names = sorted(os.listdir(os.path.join(dataset_path, 'Training')))
    # Correct Mapping based on alphabetical sort:
    # 'glioma' -> 0
    # 'meningioma' -> 1
    # 'notumor' -> 2
    # 'pituitary' -> 3
    
    print(f"Found classes: {class_names}")

    for folder_type in ['Training', 'Testing']:
        folder_path = os.path.join(dataset_path, folder_type)
        print(f"Loading data from: {folder_path}")

        for class_name in class_names:
            class_path = os.path.join(folder_path, class_name)
            
            if os.path.isdir(class_path):
                # Get the numerical label
                label = class_names.index(class_name)
                
                for image_file in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_file)
                    
                    try:
                        # Read the image
                        image = cv2.imread(image_path)
                        # Resize the image
                        image = cv2.resize(image, (img_width, img_height))
                        # Normalize the image
                        image = image / 255.0
                        
                        images.append(image)
                        labels.append(label)
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

# --- Call the function to load data ---
X, y = load_data(dataset_path, IMG_WIDTH, IMG_HEIGHT)

print(f"Data loading complete!")
print(f"Total images loaded: {X.shape[0]}")
print(f"Shape of image data (X): {X.shape}")
print(f"Shape of labels (y): {y.shape}")

# ######################################
# ## NEW CODE ENDS HERE - STEP 2      ##
# ######################################

# Step 3: Split the Data
# (This will come after data loading)
print("Data splitting step is next...")

# ######################################
# ## NEW CODE STARTS HERE - STEP 3    ##
# ######################################

# --- Step 3: Split the Data ---

# First, perform One-Hot Encoding on the labels
y_one_hot = to_categorical(y, num_classes= NUM_CLASSES)

# Now, split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y_one_hot, 
    test_size=0.2, 
    random_state=42 # random_state ensures the split is the same every time
)

print("Data splitting complete!")
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_test: {y_test.shape}")


# ######################################
# ## NEW CODE ENDS HERE - STEP 3      ##
# ######################################

# Step 4: Build the CNN Model
# (We will define the model architecture here)
print("Model building step is next...")


# Step 5: Compile and Train the Model
# (This is where the training happens)
print("Model training step is next...")


# Step 6: Save the Trained Model
# (The final step of this script)
print("Saving the model will be the final step.")