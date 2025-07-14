import cv2

class FaceDetector:
    def __init__(self, cascade_path='haarcascade_frontalface_default.xml'):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)
        if self.face_cascade.empty():
            raise ValueError("Failed to load Haar Cascade classifier")

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=6,
            minSize=(30, 30)
        )
        return faces