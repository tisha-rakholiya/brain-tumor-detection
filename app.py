import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠")
st.title("Brain Tumor Detection from MRI")
st.warning("Educational project only. Not a medical diagnosis tool.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_model.keras")

model = load_model()

file = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded image")

    x = np.array(img.resize((150, 150)), dtype="float32")
    x = np.expand_dims(x, axis=0)

    prob = float(model.predict(x, verbose=0)[0][0])
    if prob > 0.5:
        st.error(f"Tumor likely detected (probability {prob:.1%})")
    else:
        st.success(f"No tumor detected (tumor probability {prob:.1%})")