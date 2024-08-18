#yolo task=detect mode=train model=yolov8n.pt imgsz=640 data=custom_data.yaml epochs=10 batch=8 name=yolov8n_custom


    # task: Хотим ли мы detect, segmentили classify на выбранном нами наборе данных.

    # mode: Режим может быть либо train, valили predict. Поскольку мы проводим обучение, оно должно быть train.

    # model: Модель, которую мы хотим использовать. Здесь мы используем модель YOLOv8 Nano, предварительно обученную на наборе данных COCO.

    # imgsz: Размер изображения. Разрешение по умолчанию - 640.

    # data: Путь к файлу YAML dataset.

    # epochs: Количество эпох, для которых мы хотим обучаться.

    # batch: Размер пакета для загрузчика данных. Вы можете увеличить или уменьшить его в зависимости от доступности памяти вашего графического процессора.

    # name: Имя каталога результатов для runs/detect.


from ultralytics import YOLO
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

#Load the model.
model = YOLO('../weights/yolov8s.pt')
 
# Training.
results = model.train(
   data='../v1.yaml',
   imgsz=640,
   epochs=60,
   batch=4,
   name='all')

