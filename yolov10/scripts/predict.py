from ultralytics import YOLO

# Load a pre-trained YOLOv10n model
model = YOLO("weights/yolov10n.pt")

# Perform object detection on an image
results = model.val()

# Display the results
results[0].show()

# model = YOLO('yolov8n.pt')
# >>> results = model.val(data='coco128.yaml', imgsz=640)
# >>> print(results.box.map)  