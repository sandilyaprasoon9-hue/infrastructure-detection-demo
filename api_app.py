



import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import tempfile

# ----------- Roboflow API -----------
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="6xjiriCPTWTkyix8KnVO"
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

    st.subheader("Detections")

    PIXEL_TO_CM = 0.1

    total_length = 0
    pipe_count = 0

    for pred in result["predictions"]:
        label = pred["class"]
        height_px = pred["height"]
        width_px = pred["width"]

        height_cm = height_px * PIXEL_TO_CM
        width_cm = width_px * PIXEL_TO_CM

        pipe_count += 1
        total_length += height_cm

        st.write(f"Object: {label}")
        st.write(f"Length: {height_cm:.2f} cm")
        st.write(f"Width: {width_cm:.2f} cm")
        st.write("---")

    st.subheader("Summary")
    st.write(f"Total Pipes Detected: {pipe_count}")
    st.write(f"Estimated Total Pipe Length: {total_length:.2f} cm")

