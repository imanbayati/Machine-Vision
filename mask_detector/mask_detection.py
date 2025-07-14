import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class MaskDetector:
    def __init__(self, model_path='mask_detector/mask-detector-model.model'):
        self.model = load_model(model_path)

    def detect_mask(self, face_img):
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # تبدیل به RGB
        face_img = cv2.resize(face_img, (224, 224))
        face_img = img_to_array(face_img)
        face_img = preprocess_input(face_img)
        face_img = np.expand_dims(face_img, axis=0)
        (mask, without_mask) = self.model.predict(face_img)[0]
        print(f"Mask Probability: {mask}, No Mask Probability: {without_mask}")  # برای دیباگ
        return "Mask" if mask > 0.7 else "No Mask"