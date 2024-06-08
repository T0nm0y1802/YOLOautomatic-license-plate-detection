import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import pytesseract
st.set_page_config(page_title="YOLO Image and Video Processing", page_icon="🚗")

# Define custom CSS styles
custom_css = """
<style>
h1 {
    color: #FFFFFF; /* Set font color to white */
    text-align: center;
}
.upload-section {
    margin-top: 20px;
}
.output-section {
    margin-top: 20px;
    text-align: center;
}
</style>
"""
# Inject custom CSS into Streamlit
st.markdown(custom_css, unsafe_allow_html=True)

pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\tesseract.exe'
# Streamlit title
st.title("Automatic License Plate Detection")

# File uploader in Streamlit
uploaded_file = st.file_uploader("Upload an image or video file", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"])

# Load the YOLO model
model = YOLO('best.pt')

def predict_and_save_image(path_test_car, output_image_path):
    """
    Predict and annotate the image with bounding boxes and confidence scores,
    then save the processed image.

    Args:
        path_test_car (str): Path to the input image.
        output_image_path (str): Path to save the processed image.

    Returns:
        str: Path to the saved processed image.
    """
    try:
        results = model.predict(path_test_car, device='cpu')
        image = cv2.imread(path_test_car)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detected_plate=[]
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f'{confidence * 100:.2f}%', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                roi = image[y1:y2, x1:x2]

            # Perform OCR on the cropped image
                text = pytesseract.image_to_string(roi, config='--psm 6')
                detected_plate.append(text)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_image_path, image)
        if detected_plate:
            st.write("Detected License No.:")
            for idx, text in enumerate(detected_plate, 1):
                st.write(f"License:- {idx}: {text.strip()}")
        return output_image_path
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def predict_and_plot_video(video_path, output_path):
    """
    Predict and annotate the video with bounding boxes and confidence scores,
    then save the processed video.

    Args:
        video_path (str): Path to the input video.
        output_path (str): Path to save the processed video.

    Returns:
        str: Path to the saved processed video.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Error opening video file: {video_path}")
            return None
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb_frame, device='cpu')
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{confidence * 100:.2f}%', (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    # Crop the bounding box from the image for OCR
                    roi = frame[y1:y2, x1:x2]

            # Perform OCR on the cropped image
                    text = pytesseract.image_to_string(roi, config='--psm 6')
                    cv2.putText(frame, text, (x1, y2 + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            out.write(frame)
                    
        
        cap.release()
        out.release()
        return output_path
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None

def process_media(input_path, output_path):
    """
    Determine the type of media (image or video) and process accordingly.

    Args:
        input_path (str): Path to the input media file.
        output_path (str): Path to save the processed media file.

    Returns:
        str: Path to the saved processed media file.
    """
    try:
        file_extension = os.path.splitext(input_path)[1].lower()
        if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
            return predict_and_plot_video(input_path, output_path)
        elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
            return predict_and_save_image(input_path, output_path)
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return None
    except Exception as e:
        st.error(f"Error processing media: {e}")
        return None

if uploaded_file is not None:
    # Save uploaded file to a temporary directory
    input_path = os.path.join("temp", uploaded_file.name)
    output_path = os.path.join("temp", f"output_{uploaded_file.name}")
    
    try:
        with open(input_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.write("Processing ......")
        result_path = process_media(input_path, output_path)

        if result_path:
            if input_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                # Display processed video
                video_file = open(result_path, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
            else:
                # Display processed image
                st.image(result_path)
    except Exception as e:
        st.error(f"Error handling uploaded file: {e}")
