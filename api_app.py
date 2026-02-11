import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image

st.title("Wall Infrastructure Detection")

uploaded_file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    CLIENT = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="YOUR_API_KEY"
    )

    result = CLIENT.infer(uploaded_file, model_id="wall-infrastructure-detection/2")

    st.write(result)


