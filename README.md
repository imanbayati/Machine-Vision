Machine Vision Project: Mask and Object Detection
This project is a deep learning-based system designed to detect face masks and objects in real-time video streams from a webcam or IP cameras (via RTSP protocol). It is implemented using OpenCV, TensorFlow, and Flask, with a modular architecture for easy maintenance and scalability. The system features a web interface built with Tailwind CSS and JavaScript, allowing users to select either the mask detection or object detection module via checkboxes. Results are displayed as bounding boxes and labels in real-time video streams.
Features

Mask Detection Module: Detects faces and classifies them as "With Mask" (green bounding box) or "No Mask" (red bounding box).
Object Detection Module: Detects objects (e.g., chair, car) using the SSD MobileNet V2 FPNLite 640x640 model, displaying yellow bounding boxes with class labels and confidence scores.
Web Interface:
Two checkboxes for selecting the active module (only one can be active at a time).
Alerts: Blue (loading, 2 seconds), Green (success, 3 seconds), Red (previous module canceled).
Video streams displayed in two sections (top for mask detection, bottom for object detection).
Cache prevention for video streams to ensure real-time updates using Cache-Control headers and dynamic src clearing in the frontend.


Modular Code: Organized into separate modules for face detection, mask detection, object detection, video processing, and Flask server management.

Project Architecture and Execution
The project is structured in a modular fashion to ensure maintainability and scalability. It consists of two primary modules: Mask Detection and Object Detection, both integrated into a Flask-based web application. The codebase is organized into separate Python files for each functionality, with a web interface for user interaction. To run either module (mask detection or object detection), simply execute the main.py file, which serves as the entry point for the application. This file initializes the Flask server, loads the appropriate detection models, and streams the processed video to the web interface.

Key Files:
main.py: The core Flask application that handles module selection, initializes detectors, and streams video via the /video_feed endpoint.
video_source.py: Manages video input (webcam or RTSP) and processes frames by applying detection results (bounding boxes and labels).
face_detection.py: Implements face detection logic for the mask detection module.
mask_detection.py: Classifies detected faces as "With Mask" or "No Mask".
object_detection.py: Handles object detection using the SSD MobileNet V2 model.
index.html: The web interface (located in the templates/ directory) built with Tailwind CSS and JavaScript for module selection and real-time video display.



To execute the project:

Ensure all dependencies and models are installed (see Installation section).
Run python main.py from the project root directory.
Open http://localhost:5000 in a browser and select the desired module via checkboxes.

This modular design allows for easy extension, such as adding new detection modules (e.g., motion detection or tracking) by integrating them into the main.py and video_source.py files.
Prerequisites

Python 3.9 or higher
Webcam or video source (RTSP supported for IP cameras)
FFmpeg (for RTSP support, if needed)
Internet connection (to download models)

Installation

Clone the Repository:
git clone https://github.com/imanbayati/Machine-Vision.git
cd Machine_Vision


Create and Activate Virtual Environment:
python -m venv env
source env/bin/activate  # On Windows: .\env\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Download Models:

Mask Detection Model:cd mask_detector
wget https://github.com/ikigai-aa/Face-Mask-Detector-using-MobileNetV2/raw/master/mask_detector_model.model


Object Detection Model:cd object_detector
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz
tar -xvzf ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz
wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt -O mscoco_label_map.pbtxt




Run the Server:
python main.py


Access the Web Interface:

Open http://localhost:5000 in a browser.
Select a module ("Mask Detection" or "Object Detection") via checkboxes.
Video streams will appear in the corresponding section (top for mask, bottom for object).
To stop the server, visit http://localhost:5000/shutdown or press Ctrl+C.



Technical Details
Mask Detection Module

Model: Custom-trained Keras model (mask_detector_model.model) based on MobileNetV2.
Functionality: Detects faces using Haar Cascade or DNN-based methods and classifies them as "With Mask" or "No Mask".
Output: Green bounding box for "With Mask", red for "No Mask".
Files:
face_detection.py: Detects faces in video frames using OpenCV.
mask_detection.py: Classifies faces for mask presence using the Keras model.


Performance: Approximately 20 FPS on CPU (i5/i7, recent generations).

Object Detection Module

Model: SSD MobileNet V2 FPNLite 640x640 from TensorFlow 2 Object Detection API.
Functionality: Detects 80 COCO classes (e.g., person, car, chair, dog) with bounding boxes and confidence scores.
Output: Yellow bounding boxes with class labels and confidence scores.
Files:
object_detection.py: Handles object detection logic using TensorFlow.


Performance: Approximately 10-15 FPS on CPU (i5/i7, recent generations). Can be optimized with TFLite for embedded devices.

Web Interface

Framework: Tailwind CSS (via CDN) and vanilla JavaScript.
Features:
Checkbox-based module selection (exclusive selection, only one module active at a time).
Alerts: Blue (loading, 2 seconds), Green (success, 3 seconds), Red (cancellation of previous module).
Real-time video streaming with cache prevention using Cache-Control headers and dynamic clearing of src attributes in the frontend.


File: index.html (located in the templates/ directory).

Video Processing

File: video_source.py
Functionality: Processes video frames, applies detection results (from either mask or object detection), and renders bounding boxes and labels.
Input: Webcam (default) or RTSP stream (configurable in main.py via the VIDEO_SOURCE variable).

Flask Server

File: main.py
Functionality: Manages module selection, initializes detectors, and streams processed video via the /video_feed endpoint.

Work Report Summary
Mask Detection Module

Development Period: July 9-10, 2025 (4 hours 15 minutes)
Activities:
July 9: Research on MobileNetV2 and OpenCV, downloading initial model, designing modular structure (1h 30m).
July 10: Implementing modular code, resolving version conflicts (Python 3.9, TensorFlow 2.10.0, NumPy 1.23.5, Flask 2.2.2), debugging model, replacing initial model with mask_detector_model.model for better accuracy, and optimizing video source for webcam and RTSP support (2h 45m).


Challenges Resolved:
Version incompatibilities with TensorFlow, NumPy, and Flask.
Model debugging to ensure accurate mask detection.
Adding support for both RTSP and webcam inputs using OpenCV and FFmpeg.



Object Detection Module

Development Period: July 10-11, 2025 (5 hours 35 minutes)
Activities:
July 10: Research on TensorFlow Object Detection API, evaluation of SSD MobileNet V2 models, and selection of the 640x640 model for higher accuracy (3h).
July 11: Downloading ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8, structuring project for modularity, writing and testing code in object_detection.py, setting up virtual environment with dependencies from requirements.txt, and developing client-side web interface with Tailwind CSS and JavaScript for module selection (2h 35m).


Challenges Resolved:
Switching to the 640x640 model for improved accuracy over the 320x320 version.
Configuring correct model paths in object_detection.py.
Preventing video stream caching by adding Cache-Control headers in main.py and dynamically clearing src attributes in index.html.
Optimizing performance for CPU execution (10-15 FPS).



Notes

The project is designed to be modular, allowing easy addition of new modules (e.g., motion detection, tracking) by extending main.py and video_source.py.
For improved performance on low-power devices (e.g., embedded systems), consider converting the object detection model to TFLite.
To use an RTSP stream (e.g., IP camera), configure the VIDEO_SOURCE variable in main.py.
The web interface prevents video stream caching to ensure real-time updates, using HTTP headers and frontend logic.
The repository excludes heavy model files (via .gitignore), with download links provided for both models.

Contact
For questions or issues, contact the developer at bayati.pro@gmail.com.