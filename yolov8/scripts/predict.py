from ultralytics import YOLO
from PIL import Image
# Загрузка предварительно обученной модели
model = YOLO('../weights/yolov8s')  # Замените на путь к вашей модели

def predict(image):
    results = model(image)  # Получение результатов предсказания
    return results[0].plot()  # Возвращаем изображение с аннотациями


# #yolo task=detect mode=val model=runs/detect/yolov8s_v8_50e/weights/best.pt name=yolov8s_eval data=pothole_v8.yaml imgsz=1280
img = Image.open("1.jpg")
predict(img)