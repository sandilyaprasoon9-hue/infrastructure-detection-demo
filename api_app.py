import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import tempfile

st.title("Wall Infrastructure Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    image.save(temp_file.name)

    CLIENT = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="YOUR_API_KEY"
    )

    result = CLIENT.infer(temp_file.name, model_id="wall-infrastructure-detection/2")

    st.write(result)
