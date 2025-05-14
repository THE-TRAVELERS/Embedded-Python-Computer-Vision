import os
os.environ["OMP_NUM_THREADS"] = "1"  # ✅ Limite l'utilisation CPU

from flask import Flask, Response
from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import time
import socket

app = Flask(__name__)
model = YOLO("yolov8n_ncnn_model")  # ✅ Modèle NCNN exporté

# Liste des classes à afficher
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

# Configure la Picamera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 640)  # ✅ Résolution fixe
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

def gen_frames():
    while True:
        frame = picam2.capture_array()
        results = model(frame)

        # Filtrage des classes d’intérêt
        boxes = results[0].boxes
        kept = [i for i, c in enumerate(boxes.cls.cpu().tolist()) if int(c) in INTERESTED_CLASSES]
        boxes = boxes[kept] if kept else boxes[:0]
        results[0].boxes = boxes

        # Optimisation .plot() : plus léger (moins d’effets visuels)
        results[0].names = {i: model.names[i] for i in INTERESTED_CLASSES if i in model.names}
        annotated_frame = results[0].plot(boxes=True, labels=True, conf=False, line_width=1)

        # Calcul FPS
        inference_time = results[0].speed['inference']
        fps = 1000 / inference_time if inference_time > 0 else 0
        cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Compression JPEG (qualité 60)
        ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        time.sleep(0.005)  # ✅ pause pour fluidité
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return '''
    <html>
    <head>
        <title>YOLOv8n NCNN Stream</title>
        <style>
            body { margin: 0; background: black; }
            img { width: 100vw; height: 100vh; object-fit: contain; }
        </style>
    </head>
    <body>
        <img src="/video">
    </body>
    </html>
    '''

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    ip_address = socket.gethostbyname(socket.gethostname())
    print(f"🚀 Accès au flux : http://{ip_address}:5000")
    app.run(host='0.0.0.0', port=5000)
