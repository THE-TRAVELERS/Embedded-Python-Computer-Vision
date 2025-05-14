import cv2
import time
from flask import Flask, Response
from picamera2 import Picamera2
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)
model = YOLO("yolov8n_ncnn_model")  

picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 1280)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

INTERESTED_CLASSES = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    11, 12, 13, 14, 15, 16,
    24, 26, 27, 28,
    31, 32, 33,
    36, 39, 40,
    56, 57, 59, 60,
    62, 63, 64, 65, 66, 67,
    72, 73, 74, 75, 76
}

def gen_frames():
    target_fps = 30
    min_frame_interval = 1.0 / target_fps
    last_time = time.time()

    while True:
        current_time = time.time()
        elapsed = current_time - last_time
        if elapsed < min_frame_interval:
            continue
        last_time = current_time

        frame = picam2.capture_array()

        results = model(frame, imgsz = 360)
        result = results[0]

        kept_indices = [
            i for i, cls in enumerate(result.boxes.cls.cpu().numpy())
            if int(cls) in INTERESTED_CLASSES
        ]
        if kept_indices:
            result.boxes = result.boxes[kept_indices]
        else:
            result.boxes = result.boxes[:0]  # vider les boxes

        # Annoter l'image
        annotated_frame = result.plot()

        # Affichage du FPS
        fps = 1.0 / elapsed if elapsed > 0 else 0
        text = f'FPS: {fps:.1f}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = annotated_frame.shape[1] - text_size[0] - 10
        text_y = text_size[1] + 10
        cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2)

        # Encodage et envoi MJPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '<h1>YOLOv8 Live Stream</h1><img src="/video">'

if __name__ == '__main__':
    import socket
    ip_address = socket.gethostbyname(socket.gethostname())
    print(f"🚀 Serveur en ligne : http://{ip_address}:5000")
    app.run(host='0.0.0.0', port=5000)
