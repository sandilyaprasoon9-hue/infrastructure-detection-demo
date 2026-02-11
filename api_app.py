import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import numpy as np
import tempfile
import cv2

# ----------- Roboflow API -----------
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="YOUR_API_KEY"
)

MODEL_ID = "wall-infrastructure-detection/2"

# ----------- Streamlit UI -----------
st.title("Wall Infrastructure Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # Run inference
    result = CLIENT.infer(temp_path, model_id=MODEL_ID)

    # Convert image to numpy
    img_np = np.array(image)

    PIXEL_TO_CM = 0.1
    total_length = 0
    pipe_count = 0

    st.subheader("Detections")

    for pred in result["predictions"]:
        label = pred["class"]
        x = int(pred["x"])
        y = int(pred["y"])
        w = int(pred["width"])
        h = int(pred["height"])

        # Draw bounding box
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0,255,0), 2)

        # Measurement
        height_cm = h * PIXEL_TO_CM
        width_cm = w * PIXEL_TO_CM

        pipe_count += 1
        total_length += height_cm

        st.write(f"Object: {label}")
        st.write(f"Length: {height_cm:.2f} cm")
        st.write(f"Width: {width_cm:.2f} cm")
        st.write("---")

    st.image(img_np, caption="Detected Objects")

    st.subheader("Summary")
    st.write(f"Total Pipes Detected: {pipe_count}")
    st.write(f"Estimated Total Pipe Length: {total_length:.2f} cm")
