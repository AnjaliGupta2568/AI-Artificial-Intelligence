from ultralytics import YOLO
import numpy

# load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt", "v8")

# predict on an image
detections_output = model.predict(source=r"C:\AVSCODE\19.YOLO\zebra.jpg", conf=0.25, save=True)

# Display tensor array
print(detections_output)

# Display numpy array
