Machine Vision Project: Mask and Object Detection
This project is a deep learning-based system designed to detect face masks and objects in real-time video streams from a webcam or IP cameras (via RTSP protocol). It is implemented using OpenCV, TensorFlow, and Flask, with a modular architecture for easy maintenance and scalability. The system features a web interface built with Tailwind CSS and JavaScript, allowing users to select either the mask detection or object detection module via checkboxes. Results are displayed as bounding boxes and labels in real-time video streams.
Features

Mask Detection Module: Detects faces and classifies them as "With Mask" (green bounding box) or "No Mask" (red bounding box).
Object Detection Module: Detects objects (e.g., chair, car) using the SSD MobileNet V2 FPNLite 640x640 model, displaying yellow bounding boxes with class labels and confidence scores.
Web Interface:
Two checkboxes for selecting the active module (only one can be active at a time).
Alerts: Blue (loading, 2 seconds), Green (success, 3 seconds), Red (previous module canceled).
Video streams displayed in two sections (top for mask detection, bottom for object detection).
Cache prevention for video streams to ensure real-time updates.


Modular Code: Organized into separate files for face detection, mask detection, object detection, video processing, and Flask server.

Project Structure
Root/
├── templates/
│   └── index.html                  # Web interface with Tailwind CSS and JavaScript
├── main.py                         # Flask server for module management and video streaming
├── video_source.py                 # Video processing and bounding box rendering
├── requirements.txt                # Project dependencies
├── README.md                      # Project documentation
├── .gitignore                     # Git ignore file to exclude models and virtual environment
├── mask-detector/
│   ├── face_detection.py           # Face detection module
│   ├── mask_detection.py           # Mask detection module
│   ├── temp/                      # Temporary storage for images
│   └── mask-detector-model.model   # Mask detection model (not included in repo, see download link)
├── object-detector/
│   ├── object_detection.py         # Object detection module
│   ├── mscoco_label_map.pbtxt     # COCO dataset label map
│   └── ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/  # Object detection model (not included in repo, see download link)

Prerequisites

Python 3.9 or higher
Webcam or video source (RTSP supported for IP cameras)
FFmpeg (for RTSP support, if needed)
Internet connection (to download models)

Installation

Clone the Repository:
git clone https://github.com/imanbayati/Machine-Vision.git
cd Machine-Vision-2


Create and Activate Virtual Environment:
python -m venv env
source env/bin/activate  # On Windows: .\env\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Download Models:

Mask Detection Model:cd mask-detector
wget https://github.com/ikigai-aa/Face-Mask-Detector-using-MobileNetV2/raw/master/mask-detector-model.model


Object Detection Model:cd object-detector
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

Model: Custom-trained Keras model (mask-detector-model.model) based on MobileNetV2.
Functionality: Detects faces using Haar Cascade or DNN-based methods and classifies them as "With Mask" or "No Mask".
Output: Green bounding box for "With Mask", red for "No Mask".
Files:
face_detection.py: Detects faces in video frames.
mask_detection.py: Classifies faces for mask presence.


Performance: ~20 FPS on CPU (i5/i7, recent generations).

Object Detection Module

Model: SSD MobileNet V2 FPNLite 640x640 from TensorFlow 2 Object Detection API.
Functionality: Detects 80 COCO classes (e.g., person, car, chair) with bounding boxes and confidence scores.
Output: Yellow bounding boxes with class labels and confidence scores.
Files:
object_detection.py: Handles object detection logic.


Performance: ~10-15 FPS on CPU (i5/i7, recent generations). Can be optimized with TFLite for embedded devices.

Web Interface

Framework: Tailwind CSS (via CDN) and vanilla JavaScript.
Features:
Checkbox-based module selection (exclusive selection).
Alerts: Blue (loading, 2s), Green (success, 3s), Red (cancellation).
Real-time video streaming with cache prevention (Cache-Control headers and dynamic src clearing).


File: index.html (in templates/).

Video Processing

File: video_source.py
Functionality: Processes video frames, applies detection results, and renders bounding boxes/labels.
Input: Webcam (default) or RTSP stream (configurable in main.py via VIDEO_SOURCE).

Flask Server

File: main.py
Functionality: Manages module selection, initializes detectors, and streams video via /video_feed endpoint.

Work Report Summary
Mask Detection Module

Development Period: July 9-10, 2025 (4 hours 15 minutes)
Activities:
July 9: Research on MobileNetV2 and OpenCV, downloading initial model, designing modular structure (1h 30m).
July 10: Implementing modular code, resolving version conflicts (Python 3.9, TensorFlow 2.10.0), debugging model, and optimizing video source (2h 45m).


Challenges Resolved:
Version incompatibilities (TensorFlow, NumPy, Flask).
Model debugging and replacement with mask-detector-model.model for better accuracy.
Support for RTSP and webcam inputs.



Object Detection Module

Development Period: July 10-11, 2025 (5 hours 35 minutes)
Activities:
July 10: Research on TensorFlow Object Detection API and model selection (3h).
July 11: Downloading ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8, structuring project, writing and testing code, setting up virtual environment, and developing client-side web interface (2h 35m).


Challenges Resolved:
Switching to 640x640 model for higher accuracy.
Configuring model paths and preventing video stream caching.
Optimizing performance for CPU execution.



Notes

The project is designed to be modular, allowing easy addition of new modules (e.g., motion detection, tracking).
For improved performance on low-power devices, consider converting the object detection model to TFLite.
Configure VIDEO_SOURCE in main.py for RTSP streams if using IP cameras.
The web interface prevents video stream caching to ensure real-time updates.

Contact
For questions or issues, contact the developer at bayati.pro@gmail.com.