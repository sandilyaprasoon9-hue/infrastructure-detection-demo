import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import numpy as np
import cv2
import base64

# ----------- Roboflow API -----------
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="YOUR_API_KEY"
)

MODEL_ID = "wall-infrastructure-detection/2"

# ----------- Brick Auto Detection -----------
def detect_brick_size(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    widths, heights = [], []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w < 30 or h < 15:
            continue

        ratio = w / float(h)
        if 1.5 < ratio < 3.5:
            widths.append(w)
            heights.append(h)

    if len(widths) == 0:
        return None, None

    return int(np.mean(widths)), int(np.mean(heights))

# ----------- UI -----------
st.title("Wall Infrastructure Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    uploaded_file.seek(0)
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_np = np.array(image)
    img_h, img_w = img_np.shape[:2]

    # ======================================================
    # Measurement Mode
    # ======================================================
    mode = st.radio(
        "Measurement Mode",
        ["Manual Wall Dimensions", "Brick Calibration (Manual)", "Brick Calibration (Auto Detect)"]
    )

    scale_ready = False

    # ---------- Manual Wall ----------
    if mode == "Manual Wall Dimensions":
        wall_width_cm = st.number_input("Wall Width (cm)", min_value=0.0)
        wall_height_cm = st.number_input("Wall Height (cm)", min_value=0.0)

        if wall_width_cm > 0 and wall_height_cm > 0:
            PIXEL_TO_CM_X = wall_width_cm / img_w
            PIXEL_TO_CM_Y = wall_height_cm / img_h
            scale_ready = True

    # ---------- Manual Brick ----------
    elif mode == "Brick Calibration (Manual)":

        brick_pixel_w = st.number_input("Brick pixel width", min_value=0.0)
        brick_pixel_h = st.number_input("Brick pixel height", min_value=0.0)

        if brick_pixel_w > 5 and brick_pixel_h > 5:
            PIXEL_TO_CM_X = 20 / brick_pixel_w
            PIXEL_TO_CM_Y = 10 / brick_pixel_h
            scale_ready = True

    # ---------- Auto Brick ----------
    else:

        brick_w_px, brick_h_px = detect_brick_size(img_np)

        if brick_w_px is not None:
            st.success("Brick auto detected")
            st.write(f"Brick pixel width: {brick_w_px}")
            st.write(f"Brick pixel height: {brick_h_px}")

            PIXEL_TO_CM_X = 20 / brick_w_px
            PIXEL_TO_CM_Y = 10 / brick_h_px
            scale_ready = True
        else:
            st.warning("Bricks not detected clearly")

    if not scale_ready:
        st.stop()

    st.write(f"Pixel→CM X: {PIXEL_TO_CM_X:.4f}")
    st.write(f"Pixel→CM Y: {PIXEL_TO_CM_Y:.4f}")

    # ======================================================
    # Inference using BASE64 (fix for Streamlit cloud)
    # ======================================================
    uploaded_file.seek(0)
    img_bytes = uploaded_file.read()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    result = CLIENT.infer(img_base64, model_id=MODEL_ID)

    total_length = 0
    pipe_count = 0

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

        height_cm = h * PIXEL_TO_CM_Y
        width_cm = w * PIXEL_TO_CM_X

        pipe_count += 1
        total_length += height_cm

        st.write(f"{label} | Length: {height_cm:.2f} cm | Width: {width_cm:.2f} cm")

    st.image(img_np, caption="Detected Objects")

    st.subheader("Summary")
    st.write(f"Total Pipes: {pipe_count}")
    st.write(f"Total Pipe Length: {total_length:.2f} cm")



