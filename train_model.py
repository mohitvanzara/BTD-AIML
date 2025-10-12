# Step 1: Import necessary libraries
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

print("Libraries imported successfully!")

# --- Project Setup ---
# Define the path to your dataset folder
dataset_path = 'dataset' 
# Define the image size for the model
IMG_WIDTH, IMG_HEIGHT = 150, 150
# define the number of classes
NUM_CLASSES = 4


# Step 2: Load and Preprocess Data

print("Data loading and preprocessing step is next...")

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
    # label pan aaj order ma malshe
    
    print(f"Found classes: {class_names}")

    for folder_type in ['Training', 'Testing']:
        folder_path = os.path.join(dataset_path, folder_type)  # Ex : Folder_path = 'dataset/Training'
        print(f"Loading data from: {folder_path}")

        for class_name in class_names:
            class_path = os.path.join(folder_path, class_name) # Ex: class_path = 'dataset/Training/glioma'
            
            if os.path.isdir(class_path):
                # Get the numerical label
                label = class_names.index(class_name)
                
                for image_file in os.listdir(class_path): 
                    image_path = os.path.join(class_path, image_file) # Ex :image_path = 'dataset/Training/glioma/g1.jpg'
                    
                    try:
                        # Read the image
                        image = cv2.imread(image_path)  # EX :  'g1.jpg' image read thashe
                        # Resize the image
                        image = cv2.resize(image, (img_width, img_height)) # EX :  'g1.jpg' image ni size 150*150 karva 
                        # Normalize the image
                        image = image / 255.0 # renge( 0.0 to 1.0)
                        
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
print(f"Total images loaded: {X.shape[0]}") # 7023 
print(f"Shape of image data (X): {X.shape}") # (7023, 150, 150, 3)
print(f"Shape of labels (y): {y.shape}") # (7023,)


# Step 3: Split the Data

print("Data splitting step is next...")

# First, perform One-Hot Encoding on the labels
# label ko ek binary vector (0s aur 1s ka array) 
y_one_hot = to_categorical(y, num_classes= NUM_CLASSES)
# y_one_hot.shape --> (7023,4)
#0 (glioma) ⟶ [1, 0, 0, 0]

#1 (meningioma) ⟶ [0, 1, 0, 0]

#2 (notumor) ⟶ [0, 0, 1, 0]

#3 (pituitary) ⟶ [0, 0, 0, 1]


# Now, split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y_one_hot, 
    test_size=0.2, 
    random_state=42 # random_state ensures the split is the same every time
)

print("Data splitting complete!")
print(f"Shape of X_train: {X_train.shape}") # X_train.shape (5618, 150, 150, 3)
print(f"Shape of y_train: {y_train.shape}") # y_train.shape (5618,4)
print(f"Shape of X_test: {X_test.shape}") # X_test.shape (1405, 150, 150, 3 )
print(f"Shape of y_test: {y_test.shape}") # y_test.shape (1405,4)



# Step 4: Build the CNN Model

print("Model building step is next...")

def build_model(input_shape, num_classes):
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape), # ex : Output Shape: (148, 148, 32)
        MaxPooling2D((2, 2)), # Output Shape: (74, 74, 32) half thai gayo

        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'), # Output Shape: (72, 72, 64)
        MaxPooling2D((2, 2)), # Output Shape: (36, 36, 64)
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'), # Output Shape: (34, 34, 128)
        MaxPooling2D((2, 2)), ## Output Shape: (17, 17, 128)

        # Flatten the results to feed into a DNN
        Flatten(), # 17 * 17 * 128 = 36992. final shape (36992,)

        # Dense Layer for decision making
        Dense(128, activation='relu'), # aa 36992 features ni lambi line ne 128 "neurons" sathe jodi de chhe 

        # Output Layer
        Dense(num_classes, activation='softmax') # softmax aa 4 neurons na output ne probebility ma badali de chhe 
        # aa 128 features ne aakhri 4 neurons (darek class mate ek) sathe jodi de chhe
    ])
    return model # Final Output Shape: (4,) (Jaise: [0.1, 0.8, 0.05, 0.05])

# Create the model
input_shape = (IMG_WIDTH, IMG_HEIGHT, 3) # 3 for RGB images
model = build_model(input_shape, NUM_CLASSES)

# Print the model summary
print("Model built successfully! Here is the summary:")
model.summary()


# --- Step 5: Compile and Train the Model ---
# we are using adam (adeptive moment Estimation)
print("\nCompiling the model...")
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

print("Starting model training...")
# This is where the training happens. It will take some time.
history = model.fit(X_train, y_train, 
                    epochs=10, 
                    batch_size=32, 
                    validation_data=(X_test, y_test)
                    )

print("Model training complete!")

# --- Step 6: Save the Trained Model ---
print("Saving the trained model...")
model.save('brain_tumor_model.h5')
print("Model saved successfully as brain_tumor_model.h5")

