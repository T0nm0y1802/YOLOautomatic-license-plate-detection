
# YOLO License Plate Detection
This project demonstrates automatic license plate detection and text extraction using YOLO (You Only Look Once) and Tesseract OCR (Optical Character Recognition) on a YOLOv8 model. The application is deployed using Streamlit, providing an interactive web interface for users to upload and process images and videos.

## Tech Stack
YOLO (You Only Look Once): For real-time object detection.
PyTesseract: For optical character recognition to extract text from detected license plates.
YOLOv8 Model: The specific version of YOLO used in this project.
Streamlit: For deploying the application as a web service.
## Steps Involved
### 1. Model Training
Dataset Collection:

Collect a dataset containing images of vehicles with clearly visible license plates.
Label the license plates in the images using a tool like LabelImg to create bounding box annotations.
Prepare Dataset:

Split the dataset into training, validation, and test sets.
Ensure the dataset is in a format compatible with YOLOv8.
Train YOLOv8 Model:

Use the YOLOv8 repository or a pre-built framework to train the model on the prepared dataset.
Fine-tune the model parameters (like learning rate, batch size, number of epochs) for optimal performance.
Save the trained model weights (best.pt).
### 2. Application Development
Environment Setup:

Install required dependencies: OpenCV, PyTesseract, YOLOv8, Streamlit, and other necessary libraries.
Ensure Tesseract is installed and added to the system's PATH.
Model Integration:

Load the trained YOLOv8 model in the application.
Integrate PyTesseract for text extraction from detected license plate regions.
Develop Streamlit App:

Create a user interface using Streamlit for users to upload images or videos.
Process the uploaded media to detect license plates and extract text.
Display the results (annotated images/videos and extracted text) on the Streamlit interface.
### 3. Deployment
Prepare for Deployment:

Ensure all necessary files (trained model, scripts, etc.) are in place.
Test the application locally to ensure everything works as expected.
Deploy with Streamlit:

Use Streamlit's sharing platform or any other cloud provider to deploy the application.
Make the application accessible via a web URL.
