import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import os

# Load model
@st.cache_resource
def load_model():
    model_path = 'models/brain_tumor_model.keras'
    if not os.path.exists(model_path):
        return None
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

st.title("Brain Tumor Detection System")
st.write("Upload a Brain MRI image to detect if it has a tumor.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI scan.', use_column_width=True)
    
    if st.button('Predict'):
        if model is None:
            st.error("Model not found. Please train the model first.")
        else:
            with st.spinner('Analyzing...'):
                # Preprocess
                img_array = np.array(image)
                # Convert RGB if needed
                if len(img_array.shape) == 2:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                
                img_resized = cv2.resize(img_array, (224, 224))
                img_normalized = img_resized / 255.0
                img_batch = np.expand_dims(img_normalized, axis=0)
                
                # Predict
                prediction = model.predict(img_batch)
                
                # Result
                # Since we used sigmoid, output is probability of class 1 (Tumor)
                probability = prediction[0][0]
                
                if probability > 0.5:
                    st.error(f"Prediction: TUMOR DETECTED (Confidence: {probability:.2%})")
                else:
                    st.success(f"Prediction: NO TUMOR (Confidence: {1-probability:.2%})")
