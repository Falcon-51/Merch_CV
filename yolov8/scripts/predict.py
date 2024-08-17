from ultralytics import YOLO

# Загрузка предварительно обученной модели
model = YOLO('runs/train/exp/weights/best.pt')  # Замените на путь к вашей модели

def predict(image):
    results = model(image)  # Получение результатов предсказания
    return results[0].plot()  # Возвращаем изображение с аннотациями


# #yolo task=detect mode=val model=runs/detect/yolov8s_v8_50e/weights/best.pt name=yolov8s_eval data=pothole_v8.yaml imgsz=1280