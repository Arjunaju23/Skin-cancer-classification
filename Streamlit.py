import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import tempfile

def modelPrediction(testImagePath):
    model = tf.keras.models.load_model("EfficientNet.keras")
    img = tf.keras.utils.load_img(testImagePath, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = model.predict(x)
    resultIndex = np.argmax(predictions)
    confidence = float(np.max(predictions))
    return resultIndex, confidence

st.set_page_config(page_title="SKin cancer lesion analysis", layout="centered")

hideStreamlitStyle = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hideStreamlitStyle, unsafe_allow_html=True)

# --- Main Disease Identification Page ---
st.header("Welcome to the SKin cancer lesion analysis Platform")
st.markdown("Please upload a **SKin cancer lesion images** only. The model is trained to detect: `Basal Cell Carcinoma`, `Melonama`,`Nevus`.")

testImage = st.file_uploader("Upload your Image:")
if testImage is not None:
    # Save to a temporary file and get its path
    with tempfile.NamedTemporaryFile(delete=False, suffix=testImage.name) as tmpFile:
        tmpFile.write(testImage.read())
        testImagePath = tmpFile.name

# Predict button
if st.button("Predict") and testImage is not None:
    with st.spinner("Please Wait...Loading"):
        resultIndex, confidence = modelPrediction(testImagePath)
        className = ['Basal Cell Carcinoma', 'Melonama', 'Nevus']
        predictedLabel = className[resultIndex]

    if confidence < 0.75:
        st.warning(f"⚠️ The model is unsure about this prediction. Please consult your doctor")
    else:
        st.success(f"✅ Model predicts as: **{predictedLabel}**")

