import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
import os

# Load the model (relative path)
extension_model = load_model('model/extension_weights.keras')

# Labels
labels = ['DR', 'MH', 'Normal', 'ODC']

# Prediction function
def predict(image_path):
    image = cv2.imread(image_path)
    img = cv2.resize(image, (32,32))
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 32, 32, 3)
    prediction = extension_model.predict(img)
    return np.argmax(prediction)

# UI
st.title("👁 EyeDeepNet Prediction")
st.subheader("A deep learning framework for early detection of multi-retinal diseases")
st.markdown("Upload a retina image (jpg, jpeg, png) and get a prediction.")

# Upload image
image_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])

if st.button("EyeDeepNet Prediction"):
    if image_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(image_file.getbuffer())
            tmp_path = tmp_file.name

        prediction = predict(tmp_path)

        image = cv2.imread(tmp_path)
        image = cv2.resize(image, (400, 300))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.image(image, caption=f"Predicted: {labels[prediction]}", use_column_width=True)
        st.success(f"Prediction: {labels[prediction]}")
    else:
        st.warning("Please upload an image first.")
