from ultralytics import YOLO
import torch


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    # Load YOLOv10n model from scratch
    model = YOLO("../weights/yolov10m.pt")
    params = {
        "data": "../v1.yaml",
        "epochs": 100,
        "imgsz": 640,
        "device": 0,
        "batch": 16,
        "optimizer": "AdamW",
        "lr0": 1e-4
    }

    model.train(**params)