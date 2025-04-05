import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os




# Load the model
extension_model = load_model(r'C:\Users\Adil shaik\PycharmProjects\PythonProject\.venv\EyeNetDeepNeuralNetwork\model\extension_weights.keras')

# Define the predict function
def predict(image_path):
    image = cv2.imread(image_path)
    img = cv2.resize(image, (32,32))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    predict = extension_model.predict(img)
    predict = np.argmax(predict)
    return predict

# Define the labels
labels = ['DR', 'MH', 'Normal', 'ODC']

# Create the Streamlit app
st.title("👁 EyeDeepNet Prediction")
st.subheader("A deep learning framework for the early detection of multi-retinal diseases")
st.markdown("Welcome to the **EyeDeepNet Prediction** -user should upload an image [formate jpg ,png, jpeg ]of a retina and this application  predicts whether it has a retinal disease or not.")


         

# Add a file uploader for the input image
image_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])

# Add a prediction button
if st.button("EyeDeepNet Prediction"):
    if image_file is not None:
        # Read the image file
        image_path = image_file.name
        with open(image_path, "wb") as f:
            f.write(image_file.getbuffer())
        # Make the prediction
        prediction = predict(image_path)
        # Display the prediction
        st.write("Prediction:", labels[prediction])
        # Display the image with the predicted output
        image = cv2.imread(image_path)
        image = cv2.resize(image, (400,300))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.putText(image, 'Predicted As : '+labels[prediction], (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
        st.image(image)
    else:
        st.write("Please choose an image file")