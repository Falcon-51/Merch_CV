from ultralytics import YOLOv10

# Load a pre-trained YOLOv10n model
model = YOLOv10("../../inference/weights/YOLOV10_Karelia.pt")

# Perform object detection on an image
results = model.val(data='../v1_test.yaml', imgsz=640,conf=0.1, iou=0.4)

# Display the results
print(results.box.map50)  