import gradio as gr  
import PIL.Image as Image 

from ultralytics import ASSETS, YOLO  


model_choices = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]

def predict_image(img, conf_threshold, iou_threshold, model_choice):
    """Функция для предсказания объектов на изображении с использованием модели YOLOv8.
    Аргументы:
    - img: изображение для обработки.
    - conf_threshold: порог уверенности для детекции объектов.
    - iou_threshold: порог Intersection Over Union для подавления ненужных перекрывающихся объектов.
    """
    model = YOLO(model_choice)
    
    detected_objects = []
    # Прогоняем изображение через модель YOLO для детекции объектов.
    results = model.predict(
        source=img,  # Передаем изображение, которое нужно обработать.
        conf=conf_threshold,  # Устанавливаем порог уверенности для предсказания (например, объекты с уверенностью ниже этого значения игнорируются).
        iou=iou_threshold,  # Устанавливаем порог для подавления перекрывающихся предсказаний.
        show_labels=True,  # Указываем, что нужно показывать метки (классы объектов) на изображении.
        show_conf=True,  # Указываем, что нужно показывать уровень уверенности для каждого объекта на изображении.
        imgsz=640,  # Устанавливаем размер изображения (можно изменять для ускорения обработки, но с потерей точности).
    )

    # Цикл по результатам предсказаний.
    for r in results:
        im_array = r.plot()  # Визуализируем результаты предсказания
        im = Image.fromarray(im_array[..., ::-1])  # Преобразуем массив обратно в изображение формата PIL
        
        # Заполняем список объектами
        for box in r.boxes:
            cls = r.names[int(box.cls)]  # Извлекаем название класса объекта
            conf = box.conf  # Извлекаем уверенность объекта
            detected_objects.append(f"{cls}: {conf.item()*100:.2f}%")

    # Возвращаем как изображение, так и список найденных объектов
    return im, "\n".join(detected_objects)



# Определяем веб-интерфейс Gradio для взаимодействия с моделью YOLO.
iface = gr.Interface(
    fn=predict_image,  # Указываем функцию предсказания, которая будет вызываться при загрузке изображения.
    inputs=[
        gr.Image(type="pil", label="Загруженное изображение"),  # Задаем тип входных данных (изображение формата PIL).
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),  # Добавляем слайдер для регулировки порога уверенности.
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),  # Добавляем слайдер для регулировки порога IoU.
        gr.Dropdown(model_choices, label="Выбор модели YOLO"),
    ],
    
     outputs=[
        gr.Image(type="pil", label="Результат"),  # Первое выходное значение - это изображение
        gr.Textbox(label="Найденные объекты")  # Второе выходное значение - список объектов
    ], 
    title="Карельская продукция",  # Устанавливаем заголовок веб-интерфейса.
    description="ОПИСАНИЕ",  # Устанавливаем описание интерфейса (можно указать, что делает приложение).
    examples=[
        [ASSETS / "bus.jpg", 0.25, 0.45],  # Пример изображения автобуса с предустановленными значениями порогов для демонстрации.
        [ASSETS / "zidane.jpg", 0.5, 0.5],  # Пример изображения Зидана для демонстрации.
    ],
     allow_flagging="never"
)

if __name__ == "__main__":  
    iface.launch()  
