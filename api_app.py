import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import numpy as np
import tempfile
import cv2

# ----------- Roboflow API -----------
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="6xjiriCPTWTkyix8KnVO"
)

MODEL_ID = "wall-infrastructure-detection/2"

# ----------- UI -----------
st.title("Wall Infrastructure Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_np = np.array(image)
    img_h, img_w = img_np.shape[:2]

    # ======================================================
    # SCALE MODE
    # ======================================================
    st.subheader("Measurement Settings")

    mode = st.radio(
        "Measurement Mode",
        ["Manual Wall Dimensions", "Auto Estimate using Brick Module"]
    )

    PIXEL_TO_CM_X = 0.1
    PIXEL_TO_CM_Y = 0.1

    # ---------------- MANUAL MODE ----------------
    if mode == "Manual Wall Dimensions":

        wall_width_cm = st.number_input("Wall Width (cm)", min_value=1.0)
        wall_height_cm = st.number_input("Wall Height (cm)", min_value=1.0)

        if wall_width_cm > 0 and wall_height_cm > 0:
            PIXEL_TO_CM_X = wall_width_cm / img_w
            PIXEL_TO_CM_Y = wall_height_cm / img_h

    # ---------------- AUTO BRICK MODE ----------------
    else:

        brick_mode = st.selectbox(
            "Brick measurement",
            ["Brick only (19×9 cm)", "Brick with mortar (20×10 cm)"]
        )

        orientation = st.selectbox(
            "Brick Orientation",
            ["Stretcher (Length visible)", "Header (Width visible)"]
        )

        if brick_mode == "Brick only (19×9 cm)":
            brick_length = 19
            brick_height = 9
        else:
            brick_length = 20
            brick_height = 10

        if orientation == "Header (Width visible)":
            brick_length = brick_height

        st.info("Enter detected brick pixel dimensions")

        brick_pixel_w = st.number_input("Brick pixel width", min_value=1.0)
        brick_pixel_h = st.number_input("Brick pixel height", min_value=1.0)

        if brick_pixel_w > 0 and brick_pixel_h > 0:
            PIXEL_TO_CM_X = brick_length / brick_pixel_w
            PIXEL_TO_CM_Y = brick_height / brick_pixel_h

    # ======================================================
    # Run inference
    # ======================================================
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    result = CLIENT.infer(temp_path, model_id=MODEL_ID)

    total_length = 0
    pipe_count = 0

    st.subheader("Detections")

    for pred in result["predictions"]:
        label = pred["class"]
        x = int(pred["x"])
        y = int(pred["y"])
        w = int(pred["width"])
        h = int(pred["height"])

        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)

        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0,255,0), 2)

        # Measurement using dynamic scale
        height_cm = h * PIXEL_TO_CM_Y
        width_cm = w * PIXEL_TO_CM_X

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


