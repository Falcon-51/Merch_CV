import gradio as gr  
import PIL.Image as Image 
from ultralytics import ASSETS, YOLO  
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO




model_choices = ["weights/YOLOV10_Karelia.pt", "weights/yolov10n.pt", "weights/yolov10s.pt", "weights/yolov10m.pt"]


def get_shelfs(img, polka_conf, polka_iou) -> BytesIO: 

        # Задаем необходимые параметры
    URL = "https://detect.roboflow.com/shelves-ugxt3/3"
    API_KEY = "z5gSjUxoC2gzAYUByax6"
    CONFIDENSE = polka_conf  # Укажите нужный порог уверенности
    IOU = polka_iou

    # Формируем параметры запроса
    PARAMS = {
        'api_key': API_KEY,
        'confidence': CONFIDENSE,  # Добавляем параметр confidence
        'iou': IOU
    }

    # Создаем буфер для изображения
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='JPEG')  # Или 'PNG', в зависимости от вашего формата
    img_byte_arr.seek(0)  # Возвращаемся в начало потока

    response = requests.post(URL, params=PARAMS, files={'file': img_byte_arr})

    # Проверяем статус ответа
    if response.status_code == 200:
        data = response.json()  # Получаем JSON ответ

        # Получаем размеры изображения
        _, height = img.size
        
        # Создаем объект для рисования
        draw = ImageDraw.Draw(img)

        # Проходим по всем предсказанным объектам
        for prediction in data['predictions']:
            if prediction['confidence'] >= CONFIDENSE:
                # Получаем координаты ограничивающей рамки
                x1 = prediction['x'] - prediction['width'] / 2
                y1 = prediction['y'] - prediction['height'] / 2
                x2 = prediction['x'] + prediction['width'] / 2
                y2 = prediction['y'] + prediction['height'] / 2
                
                # Рисуем рамку
                draw.rectangle([x1, y1, x2, y2], outline="blue", width=8)

                # Подготавливаем текст для отображения
                class_name = prediction['class']  # Название класса
                confidence = prediction['confidence']  # Уровень уверенности
                text = f"{class_name} ({confidence:.2f})"  # Форматируем текст

                            # Определяем размер текста
                font_size = int(height / 50)   # Размер шрифта
                font = ImageFont.load_default(font_size)  # Используем стандартный шрифт

                # Рисуем фон для текста
                draw.rectangle([x1, y1 , x1+100, y1+15], fill="red")

                # Рисуем текст
                draw.text((x1, y1), text, fill="white", font=font)


    else:
        print(f"Ошибка: {response.status_code}, {response.text}")

    return img




def predict_image(img, conf_threshold:float, iou_threshold:float, model_choice:list[str]) -> Image:
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


def infer() -> None:

    # Определяем веб-интерфейс Gradio для взаимодействия с моделью YOLO.
    iface = gr.Interface(
        fn=lambda img, conf_threshold, iou_threshold, polka_conf, polka_iou, model_choice: predict_image(get_shelfs(img, polka_conf, polka_iou), conf_threshold, iou_threshold, model_choice),  # Указываем функцию предсказания, которая будет вызываться при загрузке изображения.
        inputs=[
            gr.Image(type="pil", label="Загруженное изображение"),  # Задаем тип входных данных (изображение формата PIL).
            gr.Slider(minimum=0, maximum=1, value=0.25, label="Products onfidence threshold"),  # Добавляем слайдер для регулировки порога уверенности.
            gr.Slider(minimum=0, maximum=1, value=0.45, label="Products IoU threshold"),  # Добавляем слайдер для регулировки порога IoU.
            gr.Slider(minimum=0, maximum=1, value=0.25, label="Shelf Confidence threshold"),  # Добавляем слайдер для регулировки порога уверенности.
            gr.Slider(minimum=0, maximum=1, value=0.45, label="Shelf IoU polka threshold"),
            gr.Dropdown(model_choices, label="Выбор модели YOLO"),
        ],
        
        outputs=[
            gr.Image(type="pil", label="Результат"),  # Первое выходное значение - это изображение
            gr.Textbox(label="Найденные объекты")  # Второе выходное значение - список объектов
        ], 
        title="Карельская продукция",  
        description="Детекция товаров карельских производителей в розничных магазинах на основе фото", 
        examples=[
            [ASSETS / "bus.jpg", 0.25, 0.45],  
            [ASSETS / "zidane.jpg", 0.5, 0.5],  
        ],
        allow_flagging="never"
    )

    iface.launch()  


if __name__ == "__main__":  
    infer()
