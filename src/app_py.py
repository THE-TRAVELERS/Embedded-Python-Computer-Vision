from flask import Flask, Response
from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import time

app = Flask(__name__)
model = YOLO("yolov8n.pt")

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 640)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

def gen_frames():
    prev_time = time.time()

    while True:
        frame = picam2.capture_array()

        # Inference
        results = model(frame, imgsz=640, conf=0.4)
        annotated = results[0].plot()

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        # Overlay FPS on image
        text = f"FPS: {fps:.1f}"
        cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', annotated)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return '<h1>YOLOv8 PyTorch</h1><img src="/video">'

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
