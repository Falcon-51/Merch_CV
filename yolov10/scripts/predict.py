from ultralytics import YOLO

# Load a pre-trained YOLOv10n model
model = YOLO("weights/yolov10n.pt")

# Perform object detection on an image
results = model("1.jpg")

# Display the results
results[0].show()