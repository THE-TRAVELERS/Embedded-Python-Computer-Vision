from ultralytics import YOLO

# load the YOLOv8n PyTorch model
model = YOLO("yolov8n.pt")

# Eeport the model to NCNN format
model.export(format="ncnn", imgsz=320) # value can be 640 for instance but we search a viable number of fps
