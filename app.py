import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf


model = tf.keras.models.load_model('CNN/tumor_detection/results/model/cnn_tumor.h5')

st.title('Tumor Detection App')
st.text("Tumor detection using CNN.")

def make_prediction(img, model):
  
    img = img.resize((128, 128))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.utils.normalize(img, axis=1)  
    
    res = model.predict(img)
    if res > 0.5:
        return "Tumor Detected"
    else:
        return "No Tumor Detected"

uploaded_image = st.file_uploader('Choose an image', type='jpg')

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    prediction = make_prediction(image, model)
    st.write(prediction)
    if uploaded_image is  None:
        st.write('Upload a picture')
