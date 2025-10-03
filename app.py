#(Final Version with Prediction Logic)

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

# --- App Configuration ---
st.set_page_config(
    page_title="Brain Tumor Detection AI",
    page_icon="🧠",
    layout="wide"
)

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads the trained Keras model."""
    try:
        model = tf.keras.models.load_model('brain_tumor_model.h5')
        print("Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- Class Names (Must be in the same alphabetical order as during training) ---
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --- Prediction Function ---
def preprocess_image_and_predict(image, model):
    """
    Preprocesses the image and returns the prediction and confidence.
    """
    # Convert PIL image to OpenCV format (NumPy array)
    image_np = np.array(image)
    
    # Resize the image to the target size
    resized_image = cv2.resize(image_np, (150, 150))
    
    # Normalize the image
    normalized_image = resized_image / 255.0
    
    # Expand dimensions to create a batch of 1
    batched_image = np.expand_dims(normalized_image, axis=0)
    
    # Make prediction
    prediction = model.predict(batched_image)
    
    # Get the predicted class index and confidence
    predicted_class_index = np.argmax(prediction)
    confidence = np.max(prediction)
    
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    
    return predicted_class_name, confidence

# --- UI Elements ---
st.title("Brain Tumor Detection AI 🧠")
st.write("Upload an MRI scan and the AI will predict the tumor type. This is a demonstration project and should not be used for actual medical diagnosis.")
st.write("---")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded MRI Scan.', use_column_width=True)
    
    with col2:
        with st.spinner('AI is thinking...'):
            # Get prediction
            predicted_class, confidence = preprocess_image_and_predict(image, model)
            
            st.write("### Prediction Result:")
            
            if predicted_class == 'notumor':
                st.success(f"**Result:** The model predicts there is **No Tumor**.")
            else:
                st.warning(f"**Result:** The model predicts a **{predicted_class.capitalize()}** type of tumor.")
            
            st.write(f"**Confidence:** {confidence * 100:.2f}%")
            st.info("Note: This is an AI-generated prediction. Please consult a medical professional for an accurate diagnosis.")

elif model is None:
    st.error("Model could not be loaded. Please check the model file and restart.")