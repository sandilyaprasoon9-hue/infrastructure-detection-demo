import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import numpy as np
import cv2
import base64

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

# ----------- Roboflow API -----------
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="YOUR_API_KEY"
)

MODEL_ID = "wall-infrastructure-detection/2"

# ----------- PDF Generator -----------
def generate_a4_pipe_layout(img_np, predictions, PIXEL_TO_CM_X,
                            filename="wall_pipe_layout.pdf"):

    c = canvas.Canvas(filename, pagesize=A4)
    page_w, page_h = A4

    border_margin = 1*cm
    c.rect(border_margin, border_margin,
           page_w - 2*border_margin,
           page_h - 2*border_margin)

    img_h, img_w = img_np.shape[:2]

    max_draw_w = page_w - 4*cm
    max_draw_h = page_h - 6*cm

    scale = min(max_draw_w/img_w, max_draw_h/img_h)

    draw_w = img_w * scale
    draw_h = img_h * scale

    offset_x = (page_w - draw_w) / 2
    offset_y = (page_h - draw_h) / 2

    wall_img = ImageReader(img_np)
    c.drawImage(wall_img, offset_x, offset_y,
                width=draw_w, height=draw_h)

    c.setLineWidth(2)

    for pred in predictions:

        x = pred["x"]
        y = pred["y"]
        w = pred["width"]
        h = pred["height"]

        length_pixels = max(w, h)
        length_cm = length_pixels * PIXEL_TO_CM_X

        px = offset_x + (x - w/2) * scale
        py = offset_y + (img_h - y) * scale

        pipe_len = length_pixels * scale

        c.line(px, py, px + pipe_len, py)

        c.setFont("Helvetica", 8)
        c.drawString(px, py + 5, f"{length_cm:.1f} cm")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, page_h - 2*cm,
                 "Wall Pipe Architectural Layout")

    c.save()
    return filename

# ----------- UI -----------
st.title("Wall Infrastructure Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    uploaded_file.seek(0)
    image = Image.open(uploaded_file)
    st.image(image)

    img_np = np.array(image)
    img_h, img_w = img_np.shape[:2]

    wall_width_cm = st.number_input("Wall Width (cm)", min_value=1.0)
    wall_height_cm = st.number_input("Wall Height (cm)", min_value=1.0)

    if wall_width_cm <= 0 or wall_height_cm <= 0:
        st.stop()

    PIXEL_TO_CM_X = wall_width_cm / img_w

    uploaded_file.seek(0)
    img_bytes = uploaded_file.read()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    result = CLIENT.infer(img_base64, model_id=MODEL_ID)

    total_length = 0

    for pred in result["predictions"]:
        w = pred["width"]
        h = pred["height"]

        length_pixels = max(w, h)
        length_cm = length_pixels * PIXEL_TO_CM_X
        total_length += length_cm

    st.write(f"Total Pipe Length: {total_length:.2f} cm")

    if st.button("Generate Architectural A4 Blueprint"):
        pdf_file = generate_a4_pipe_layout(
            img_np,
            result["predictions"],
            PIXEL_TO_CM_X
        )

        with open(pdf_file, "rb") as f:
            st.download_button("Download Blueprint", f,
                               file_name=pdf_file)












