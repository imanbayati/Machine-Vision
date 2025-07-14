import os
import time
from flask import Flask, Response, render_template, request, jsonify
from mask_detector.face_detection import FaceDetector
from mask_detector.mask_detection import MaskDetector
from object_detector.object_detection import ObjectDetector
from video_source import VideoProcessor

# برنامه اصلی Flask
app = Flask(__name__)

# مسیر ذخیره موقت تصاویر
TEMP_DIR = "temp"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# تنظیم منبع ویدیویی
VIDEO_SOURCE = 0  # برای وب‌کم محلی
# VIDEO_SOURCE = "rtsp://192.168.1.100:554/stream"  # مثال برای دوربین IP

# متغیرهای جهانی برای مدیریت ماژول‌ها
face_detector = None
mask_detector = None
object_detector = None
video_processor = None

def initialize_processor(mode):
    global face_detector, mask_detector, object_detector, video_processor
    # آزاد کردن منابع قبلی
    if video_processor:
        video_processor.release()
    
    # نمونه‌سازی ماژول‌ها بر اساس حالت
    if mode == 'mask':
        face_detector = FaceDetector()
        mask_detector = MaskDetector()
        object_detector = None
    elif mode == 'object':
        face_detector = None
        mask_detector = None
        object_detector = ObjectDetector()
    else:
        raise ValueError("Invalid mode. Use 'mask' or 'object'")

    # نمونه‌سازی VideoProcessor
    video_processor = VideoProcessor(face_detector, mask_detector, object_detector, source=VIDEO_SOURCE)
    video_processor.set_mode(mode)
    # شبیه‌سازی زمان لود سرویس
    time.sleep(2)  # تأخیر 2 ثانیه برای لودینگ
    return video_processor

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_mode', methods=['POST'])
def set_mode():
    data = request.get_json()
    mode = data.get('mode')
    if mode not in ['mask', 'object']:
        return jsonify({'status': 'error', 'message': 'Invalid mode'}), 400
    try:
        initialize_processor(mode)
        return jsonify({'status': 'success', 'message': f'Mode set to {mode}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    mode = request.args.get('mode', 'mask')
    if not video_processor or video_processor.mode != mode:
        initialize_processor(mode)
    def gen_frames():
        while True:
            frame = video_processor.process_frame()
            if frame is None:
                break
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(gen_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={
                        'Cache-Control': 'no-store, no-cache, must-revalidate, proxy-revalidate',
                        'Pragma': 'no-cache',
                        'Expires': '0'
                    })

@app.route('/shutdown')
def shutdown():
    if video_processor:
        video_processor.release()
    return 'Server shutting down...'

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)