import cv2
import numpy as np
import tensorflow as tf
import os

class ObjectDetector:
    def __init__(self, model_path='object_detector/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model', 
                 labelmap_path='object_detector/mscoco_label_map.pbtxt'):
        # چک کردن وجود مسیر مدل
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        print(f"Loading model from: {model_path}")  # برای دیباگ
        
        # لود مدل TensorFlow SavedModel
        try:
            self.model = tf.saved_model.load(model_path)
            self.detect_fn = self.model.signatures['serving_default']
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")
        
        # لود کلاس‌ها از labelmap
        try:
            self.classes = self.load_label_map(labelmap_path)
        except Exception as e:
            raise ValueError(f"Error loading labelmap: {str(e)}")
        
    def load_label_map(self, labelmap_path):
        classes = {}
        try:
            with open(labelmap_path, 'r') as f:
                lines = f.readlines()
                current_id = None
                for line in lines:
                    line = line.strip()
                    if 'id:' in line:
                        current_id = int(line.split(':')[1].strip())
                    elif 'name:' in line and current_id is not None:
                        class_name = line.split('"')[1]
                        classes[current_id] = class_name
                        current_id = None
            if not classes:
                raise ValueError("No classes found in labelmap file")
            return classes
        except FileNotFoundError:
            raise ValueError(f"Labelmap file not found: {labelmap_path}")

    def detect_objects(self, frame, confidence_threshold=0.5):
        height, width = frame.shape[:2]
        # آماده‌سازی فریم برای ورودی مدل
        input_tensor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = cv2.resize(input_tensor, (320, 320))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.uint8)

        # اجرای تشخیص
        detections = self.detect_fn(input_tensor)
        
        # پردازش خروجی‌ها
        objects = []
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(np.int32)

        for i in range(len(scores)):
            if scores[i] > confidence_threshold:
                class_id = classes[i]
                if class_id in self.classes:
                    class_name = self.classes[class_id]
                    box = boxes[i] * np.array([height, width, height, width])
                    (startY, startX, endY, endX) = box.astype("int")
                    objects.append({
                        'class': class_name,
                        'confidence': float(scores[i]),
                        'box': (startX, startY, endX, endY)
                    })
        return objects