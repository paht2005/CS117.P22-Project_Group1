
import streamlit as st
from PIL import Image
import numpy as np
from fire_classifier import classify_fire
from fire_segmenter import segment_fire
from utils.metadata import generate_gps, estimate_fire_area
from utils.image_overlay import overlay_mask

st.title("Drone-based Wildfire Detection System")
uploaded_file = st.file_uploader("Upload an RGB frame", type=["jpg", "png"])

if uploaded_file:
    img_path = "temp.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded Frame")

    if classify_fire(img_path):
        st.success("Fire detected!")
        mask = segment_fire(img_path)
        gps = generate_gps()
        area = estimate_fire_area(mask)
        overlay_mask(img_path, mask, gps, area, "output.jpg")
        st.image("output.jpg", caption=f"Fire Area: {area:.2f} m²\nGPS: {gps}")
    else:
        st.warning("No fire detected.")
