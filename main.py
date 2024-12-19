import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

# Directory paths
UPLOAD_FOLDER = "images/"
OUTPUT_FOLDER = "outputs/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize YOLO model
model = YOLO('yolov8n.pt')  # Lightweight YOLO model

def detect_cars(image_path, output_path):
    # Load and process the image
    results = model(image_path)

    # Filter results for cars (COCO class ID: 2)
    car_class_id = 2
    car_detections = [r for r in results[0].boxes.data if int(r[-1]) == car_class_id]

    # Load image using OpenCV
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw bounding boxes for all detected cars
    for idx, box in enumerate(car_detections):
        x1, y1, x2, y2, conf, class_id = box.tolist()
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"Car {idx+1}: {conf:.2f}"
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the processed image
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    return image

# Streamlit App
st.set_page_config(page_title="Car Detection App", layout="wide")

st.title("Car Detection Application 🚗")
st.write("Upload an image, and the app will detect cars using YOLOv8.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file
    input_image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(input_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(input_image_path, caption="Uploaded Image", use_column_width=True)

    # Perform detection
    output_image_path = os.path.join(OUTPUT_FOLDER, "detected_image.jpg")
    with st.spinner("Detecting cars..."):
        processed_image = detect_cars(input_image_path, output_image_path)

    st.success("Detection Complete!")
    st.image(processed_image, caption="Detected Cars", use_column_width=True)
