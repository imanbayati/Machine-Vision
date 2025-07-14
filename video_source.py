import cv2
from mask_detector.face_detection import FaceDetector
from mask_detector.mask_detection import MaskDetector
from object_detector.object_detection import ObjectDetector

class VideoProcessor:
    def __init__(self, face_detector=None, mask_detector=None, object_detector=None, source=0):
        self.face_detector = face_detector
        self.mask_detector = mask_detector
        self.object_detector = object_detector
        self.mode = None
        if isinstance(source, str) and source.startswith('rtsp://'):
            self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {source}")

    def set_mode(self, mode):
        if mode in ['mask', 'object']:
            self.mode = mode
        else:
            raise ValueError("Mode must be 'mask' or 'object'")

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        if self.mode == 'mask' and self.face_detector and self.mask_detector:
            faces = self.face_detector.detect_faces(frame)
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                label = self.mask_detector.detect_mask(face_img)
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                display_label = "With Mask" if label == "Mask" else "No Mask"
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, display_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        elif self.mode == 'object' and self.object_detector:
            objects = self.object_detector.detect_objects(frame, confidence_threshold=0.5)
            for obj in objects:
                class_name = obj['class']
                confidence = obj['confidence']
                (startX, startY, endX, endY) = obj['box']
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 0), 2)
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (startX, startY-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def release(self):
        self.cap.release()