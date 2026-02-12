from architectural_layout import generate_architecture_diagram



import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import numpy as np
import cv2
import base64

# PDF generation
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

# ----------- Roboflow API -----------
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="6xjiriCPTWTkyix8KnVO"
)

MODEL_ID = "wall-infrastructure-detection/2"

# ----------- A4 Layout Generator -----------
def generate_a4_pipe_layout(predictions, PIXEL_TO_CM_X, filename="wall_pipe_layout.pdf"):
    c = canvas.Canvas(filename, pagesize=A4)
    page_w, page_h = A4

    margin_x = 2*cm
    margin_y = 2*cm
    draw_scale = 0.3

    c.setFont("Helvetica", 10)
    c.drawString(2*cm, page_h - 2*cm, "Wall Pipe Layout")

    for pred in predictions:
        x = pred["x"]
        y = pred["y"]
        w = pred["width"]
        h = pred["height"]

        length_pixels = max(w, h)
        length_cm = length_pixels * PIXEL_TO_CM_X

        draw_x = margin_x + x * draw_scale
        draw_y = margin_y + y * draw_scale
        draw_length = length_pixels * draw_scale

        c.line(draw_x, draw_y, draw_x + draw_length, draw_y)
        c.drawString(draw_x, draw_y + 5, f"{length_cm:.1f} cm")

    c.save()
    return filename

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



# ----------- Clone Engineering Diagram Generator -----------
def generate_clone_diagram(img_w, img_h, predictions, PIXEL_TO_CM_X, filename="clone_layout.png"):

    canvas_img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

    for pred in predictions:
        x = int(pred["x"])
        y = int(pred["y"])
        w = int(pred["width"])
        h = int(pred["height"])

        # compute rectangle corners (true pipe box)
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)

        # draw filled pipe box (engineering clone)
        cv2.rectangle(canvas_img, (x1, y1), (x2, y2), (0,0,0), -1)

        # label length using longest dimension
        length_px = max(w, h)
        length_cm = length_px * PIXEL_TO_CM_X

        cv2.putText(
            canvas_img,
            f"{length_cm:.1f}cm",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,0,0),
            1
        )

    cv2.imwrite(filename, canvas_img)
    return filename





from streamlit_cropper import st_cropper

def crop_wall_image(image):
    st.subheader("Crop Wall Area")

    cropped_img = st_cropper(
        image.convert("RGB"),
        realtime_update=True,
        box_color="blue",
        aspect_ratio=None
    )

    return cropped_img.convert("RGB")




from streamlit_drawable_canvas import st_canvas




from streamlit_drawable_canvas import st_canvas

def door_window_box_tool(image, PIXEL_TO_CM_X, PIXEL_TO_CM_Y):
    st.subheader("Draw rectangle over Door / Window / Gate")

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 255, 0.2)",
        stroke_width=2,
        stroke_color="blue",
        background_image=image,
        update_streamlit=True,
        height=image.height,
        width=image.width,
        drawing_mode="rect",
        key="door_canvas",
    )

    items = []

    if canvas_result.json_data is not None:
        for obj in canvas_result.json_data["objects"]:
            w_px = obj["width"]
            h_px = obj["height"]

            w_cm = w_px * PIXEL_TO_CM_X
            h_cm = h_px * PIXEL_TO_CM_Y

            st.write(f"Door/Window → Width: {w_cm:.2f} cm | Height: {h_cm:.2f} cm")

            items.append({
                "width_cm": w_cm,
                "height_cm": h_cm
            })

    return items







def generate_full_clone(img_w, img_h, predictions, door_items, PIXEL_TO_CM_X, filename="final_clone.png"):
    canvas_img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

    # draw pipes
    for pred in predictions:
        x = int(pred["x"])
        y = int(pred["y"])
        w = int(pred["width"])
        h = int(pred["height"])

        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)

        cv2.rectangle(canvas_img, (x1,y1),(x2,y2),(0,0,0),-1)

    cv2.imwrite(filename, canvas_img)
    return filename



# ----------- UI -----------
st.title("Wall Infrastructure Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    uploaded_file.seek(0)
    original_image = Image.open(uploaded_file).convert("RGB")

    # --- Step 1: Crop wall ---
    

    st.subheader("Step 1: Crop Wall")

    cropped_image = crop_wall_image(original_image)

    if cropped_image is None:
        st.info("Adjust the crop box to continue")
        st.stop()

    cropped_image = cropped_image.convert("RGB")

    st.success("Wall cropped successfully")
    st.image(cropped_image, caption="Cropped Wall", use_column_width=True)

    # Convert to numpy AFTER cropping
    img_np = np.array(cropped_image)

    img_h, img_w = img_np.shape[:2]

    mode = st.radio(
        "Measurement Mode",
        ["Manual Wall Dimensions", "Brick Calibration (Manual)", "Brick Calibration (Auto Detect)"]
    )

    scale_ready = False

    if mode == "Manual Wall Dimensions":
        wall_width_cm = st.number_input("Wall Width (cm)", min_value=0.0)
        wall_height_cm = st.number_input("Wall Height (cm)", min_value=0.0)

        if wall_width_cm > 0 and wall_height_cm > 0:
            PIXEL_TO_CM_X = wall_width_cm / img_w
            PIXEL_TO_CM_Y = wall_height_cm / img_h
            scale_ready = True

    elif mode == "Brick Calibration (Manual)":
        brick_pixel_w = st.number_input("Brick pixel width", min_value=0.0)
        brick_pixel_h = st.number_input("Brick pixel height", min_value=0.0)

        if brick_pixel_w > 5 and brick_pixel_h > 5:
            PIXEL_TO_CM_X = 20 / brick_pixel_w
            PIXEL_TO_CM_Y = 10 / brick_pixel_h
            scale_ready = True

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
    st.subheader("Draw rectangle over Doors / Windows")

    door_items = door_window_box_tool(
        cropped_image.convert("RGB"),
        PIXEL_TO_CM_X,
        PIXEL_TO_CM_Y
    )

    


    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(buffer).decode("utf-8")


    result = CLIENT.infer(img_base64, model_id=MODEL_ID)

    STANDARD_PIPE_CM = 3.0

    total_length = 0
    pipe_count = 0
    pipes_for_architecture = []


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

        length_pixels = max(w, h)
        length_cm = length_pixels * PIXEL_TO_CM_X
        pipes_for_architecture.append({
        "x_cm": x * PIXEL_TO_CM_X,
        "y_cm": y * PIXEL_TO_CM_Y,
        "length_cm": length_cm
        })


        


        pipe_count += 1
        total_length += length_cm

        st.write(f"{label} | Length: {length_cm:.2f} cm | Diameter: {STANDARD_PIPE_CM:.2f} cm")

    st.image(img_np, caption="Detected Objects")

    st.subheader("Summary")
    st.write(f"Total Pipes: {pipe_count}")

    st.write(f"Total Pipe Length: {total_length:.2f} cm")

    # ----------- Architectural Layout Button -----------
    if st.button("Generate Architectural Layout"):
        pdf_file = generate_architecture_diagram(
            wall_width_cm if mode == "Manual Wall Dimensions" else img_w * PIXEL_TO_CM_X,
            wall_height_cm if mode == "Manual Wall Dimensions" else img_h * PIXEL_TO_CM_Y,
            pipes_for_architecture
        )
        with open(pdf_file, "rb") as f:
            st.download_button("Download Architectural Layout", f, file_name=pdf_file)

    # ----------- Clone Layout Button -----------
    if st.button("Generate Clone Diagram"):
        clone_file = generate_clone_diagram(
            img_w,
            img_h,
            result["predictions"],
            PIXEL_TO_CM_X
        )

        with open(clone_file, "rb") as f:
            st.download_button("Download Clone Diagram", f, file_name=clone_file)

    # ----------- Final Engineering Drawing -----------
    if st.button("Generate Final Engineering Drawing"):
        final_file = generate_full_clone(
            img_w,
            img_h,
            result["predictions"],
            door_items,
            PIXEL_TO_CM_X
        )

        with open(final_file,"rb") as f:
            st.download_button("Download Final Drawing", f, file_name=final_file)








