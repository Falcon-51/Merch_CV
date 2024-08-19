from ultralytics import YOLOv10

model = YOLOv10("../../inference/weights/YOLOV10_Karelia.pt")

results = model.val(data='../v1_test.yaml', imgsz=640,conf=0.1, iou=0.4)

print(results.box.map50)  