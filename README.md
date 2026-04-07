#  AI-Based Wall Infrastructure Layout Detection

## Overview

This project is a Computer Vision-based system that detects wall infrastructure components such as pipelines and layouts from images using a YOLO (You Only Look Once) object detection model.

It provides a real-time interface where users can upload images and visualize detected components.

---

## Features

* Detects infrastructure elements from wall images
* Real-time object detection using YOLO
* Interactive web interface using Streamlit
* Supports image upload and instant visualization
* Optimized for efficient and fast predictions

---

## 🛠️ Tech Stack

* Python
* YOLO (Object Detection)
* OpenCV
* Streamlit

---

##  Project Structure

wall-detection-system/
│── data/
│── models/
│── app/
│── main_app.py
│── requirements.txt
│── README.md

---

## Installation & Setup

### 1. Clone the repository

git clone https://github.com/your-username/wall-detection-system.git
cd wall-detection-system

### 2. Install dependencies

pip install -r requirements.txt

### 3. Run the application

streamlit run main_app.py

---

##  How It Works

1. Upload an image through the web interface
2. The YOLO model processes the image
3. Detected components are highlighted with bounding boxes


---

##  Model Details

* Model: YOLO (You Only Look Once)
* Task: Object Detection
* Input: Wall/Infrastructure images
* Output: Detected components with bounding boxes

---

##  Results

* Successfully detects infrastructure components from images
* Provides fast and efficient real-time predictions

*(You can add accuracy or sample results later)*

---

## Future Improvements

* Improve detection accuracy with larger dataset
* Add support for video input
* Deploy on cloud platforms
* Enhance UI for better user experience

---

##  Author

Prasoon Jha
Email: [sandilyaprasoon9@gmail.com](mailto:sandilyaprasoon9@gmail.com)


---

## Support

If you found this project useful, give it a star ⭐
