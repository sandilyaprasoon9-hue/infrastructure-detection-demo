import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import numpy as np

# ----------- Roboflow API -----------
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="YOUR_API_KEY"   # paste your roboflow key here
)

MODEL_ID = "wall-infrastructure-detection/2"

# ----------- Streamlit UI -----------
st.title("Wall Infrastructure Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run inference
    result = CLIENT.infer(uploaded_file, model_id=MODEL_ID)

    st.subheader("Detections")

    # ----------- Measurement Logic -----------
    PIXEL_TO_CM = 0.1   # calibration factor (can adjust later)

    for pred in result["predictions"]:
        label = pred["class"]
        height_px = pred["height"]
        width_px = pred["width"]

        height_cm = height_px * PIXEL_TO_CM
        width_cm = width_px * PIXEL_TO_CM

        st.write(f"Object: {label}")
        st.write(f"Approx Height: {height_cm:.2f} cm")
        st.write(f"Approx Width: {width_cm:.2f} cm")
        st.write("---")

