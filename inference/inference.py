import gradio as gr  
import PIL.Image as Image 
import requests
import os
from ultralytics import ASSETS, YOLOv10
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from typing import Optional


# Список для выбора весов модели
model_choices = ["weights/YOLOV10_Karelia_v2.pt","weights/YOLOV10_Karelia.pt", "weights/yolov10n.pt", "weights/yolov10s.pt", "weights/yolov10m.pt"]

# Переменные для расчёта площади и положения продукта и полок
PRODUCTS_AREA = 0
SHELFES_AREA = 0
CORD_SHELFS = []


# names = {
#   "sampo_waffle_classic": "sampo_waffle_classic",
#   "sampo_waffle_lemon": "sampo_waffle_lemon",
#   "sl_pm_2_5_1": "slavmo_milk_2.5_1" ,
#   "sl_pm_3_5_1": "slavmo_milk_3.5_1" ,
#   "sl_pm_2_5_1_5": "slavmo_milk_2.5_1.5" ,
#   "sl_pm_3_5_1_5": "slavmo_milk_3.5_1.5",
#   "sl_pbm_0_5": "slavmo_milk_melted_3.5_0.5",
#   "ol_pm_2_5_1": "olonia_milk_2.5_1",
#   "ol_pm_3_5_1": "olonia_milk_3.5_1",
#   "ol_pbm_0_5": "olonia_milk_melted_4_0.5",
# }


def calculate_area(x_min:float, y_min:float, x_max:float, y_max:float) -> float:
    """
    Рассчитывает площадь прямоугольника

    Параметры:
    - x_min: координата x левого верхнего угла
    - y_min: координата y левого верхнего угла
    - x_max: координата x правого верхнего угла
    - y_max: координата y правого верхнего угла

    Возвращает: float
    
    """

    area = (x_max - x_min) * (y_max - y_min)

    return area


def get_box_center(x_min:float, y_min:float, x_max:float, y_max:float) -> tuple[float]:
    """
    Находит координаты центра прямоугольника

    Параметры:
    - x_min: координата x левого верхнего угла
    - y_min: координата y левого верхнего угла
    - x_max: координата x правого верхнего угла
    - y_max: координата y правого верхнего угла

    Возвращает: tuple[float]
    """

    center_y = (y_min + y_max) / 2
    center_x = (x_min + x_max) / 2

    return center_x, center_y


def point_in_shelves(data:tuple[float], point:tuple[float])-> Optional[int]:
    """
    Проверяем, находится ли точка внутри прямоугольника

    Параметры:
    - data: Список координат полок
    - point: координаты точки на проверку

    Возвращает: tuple[float]
    """
    
    x, y = point
    for (x1, y1, x2, y2, number_shelf) in data:
        # Проверяем, находится ли точка внутри прямоугольника
        if x1 <= x <= x2 and y1 <= y <= y2:
            return number_shelf
    return None  # Если точка не находится ни в одном прямоугольнике


def get_shelfs(img:Image, polka_conf:float, polka_iou:float) -> BytesIO: 
    """
    Обращение к модели на Roboflow для детектирования полок. \n 
    Получает координаты и в соовтествии с ними ставит BB на исходное изображение

    Параметры:
    - img: изображение
    - polka_conf: порог уверенности
    - polka_iou: порого IOU
    
    Возвращает: BytesIO

    """

    # Задаем необходимые параметры
    URL = "https://detect.roboflow.com/shelves-ugxt3/3"
    API_KEY = ""
    CONFIDENSE = polka_conf  # Указываем порог уверенности
    IOU = polka_iou

    global SHELFES_AREA 
    global CORD_SHELFS

    # Формируем параметры запроса
    PARAMS = {
        'api_key': API_KEY,
        'confidence': CONFIDENSE, 
        'iou': IOU
    }

    min_cord = []

    # Создаем буфер для изображения
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)  # Возвращаемся в начало потока

    response = requests.post(URL, params=PARAMS, files={'file': img_byte_arr})

    # Проверяем статус ответа
    if response.status_code == 200:
        data = response.json()  # Получаем JSON ответ

        # Получаем размеры изображения
        _, height = img.size
        
        # Создаем объект для рисования
        draw = ImageDraw.Draw(img)

        # Проходим по всем предсказаниям и собираем координаты y левого верхнего угла каждого BB
        for prediction in data['predictions']:
            x1 = prediction['x'] - prediction['width'] / 2
            y1 = prediction['y'] - prediction['height'] / 2
            x2 = prediction['x'] + prediction['width'] / 2
            y2 = prediction['y'] + prediction['height'] / 2
            min_cord.append(y1)
        # Сортируем
        min_cord.sort()

        # Проходим по всем предсказанным объектам
        for prediction in data['predictions']:
            if prediction['confidence'] >= CONFIDENSE:
                # Получаем координаты ограничивающей рамки
                x1 = prediction['x'] - prediction['width'] / 2
                y1 = prediction['y'] - prediction['height'] / 2
                x2 = prediction['x'] + prediction['width'] / 2
                y2 = prediction['y'] + prediction['height'] / 2
                
                # Рассчитывем номер полки и сохраняем координаты с номером полки
                number_shelf = abs(min_cord.index(y1) - len(min_cord))
                CORD_SHELFS.append((x1, y1, x2, y2,number_shelf))

                # Рисуем рамку
                draw.rectangle([x1, y1, x2, y2], outline="blue", width=6)

                # Считаем площадь каждой полки
                area = calculate_area(x1,y1,x2,y2)
                SHELFES_AREA += area

                # Подготавливаем текст для отображения
                class_name = prediction['class']  # Название класса
                confidence = prediction['confidence']  # Уровень уверенности
                text = f"({confidence:.2f}, {class_name}), Level {number_shelf}, Area:{area}"  # Форматируем текст

                # Определяем размер текста
                font_size = int(height / 35)   # Размер шрифта
                font = ImageFont.load_default(font_size)  # Используем стандартный шрифт

                # Рисуем текст
                draw.text((x1, y1), text, fill="white", font=font)
        

    else:
        print(f"Ошибка: {response.status_code}, {response.text}")

    return img





def predict_image(img:Image, conf_threshold:float, iou_threshold:float, model_choice:list[str]) -> Image:
    """
    Функция для предсказания объектов на изображении с использованием модели YOLOv8.

    Параметры:
    - img: изображение для обработки.
    - conf_threshold: порог уверенности для детекции объектов.
    - iou_threshold: порог Intersection Over Union для подавления ненужных перекрывающихся объектов.

    Возвращает: Image
    """
    model = YOLOv10(model_choice)
    global PRODUCTS_AREA
    detected_objects = []
    center_box = 0

    # Прогоняем изображение через модель YOLO для детекции объектов.
    results = model.predict(
        source=img,  # Передаем изображение, которое нужно обработать.
        conf=conf_threshold,  # Устанавливаем порог уверенности для предсказания.
        iou=iou_threshold,  # Устанавливаем порог для подавления перекрывающихся предсказаний.
        show_labels=True,  # Указываем, что нужно показывать метки (классы объектов) на изображении.
        show_conf=True,  # Указываем, что нужно показывать уровень уверенности для каждого объекта на изображении.
        imgsz=640,  # Устанавливаем размер изображения.
    )

    # Цикл по результатам предсказаний.
    for r in results:
        im_array = r.plot()  # Визуализируем результаты предсказания
        im = Image.fromarray(im_array[..., ::-1])  # Преобразуем массив обратно в изображение формата PIL
        
        # Заполняем список объектами
        for box in r.boxes:
            cls = r.names[int(box.cls)]  # Извлекаем название класса объекта
            conf = box.conf  # Извлекаем уверенность объекта

            # Извлекаем координаты ограничивающего прямоугольника
            x_min, y_min, x_max, y_max = box.xyxy[0]  # Координаты: [x1, y1, x2, y2]

            # Считаем площадь каждого BB продукта
            area = calculate_area(x_min, y_min, x_max, y_max)
            PRODUCTS_AREA += area

            # Находим центр BB
            center_box = get_box_center(x_min, y_min, x_max, y_max)
            #Проверяем, находится ли продукт на полке
            shelf = point_in_shelves(CORD_SHELFS, center_box)

            detected_objects.append(f"Area: {area}, cls: {cls} - {conf.item()*100:.2f}%, shelf: {shelf}")
    detected_objects.append(f"Square (PRODUCTS_AREA/SHELFES_AREA): {PRODUCTS_AREA/SHELFES_AREA*100:.2f}%")
    # Возвращаем как изображение, так и список найденных объектов
    return im, "\n".join(detected_objects)



def infer() -> None:
    """
    Запускает веб-приложение

    Параметры: None

    Возвращает: None
    """

    # Определяем веб-интерфейс Gradio для взаимодействия с моделью YOLO.
    iface = gr.Interface(
        fn=lambda img, conf_threshold, iou_threshold, polka_conf, polka_iou, model_choice: predict_image(get_shelfs(img, polka_conf, polka_iou), conf_threshold, iou_threshold, model_choice),  # Указываем функцию предсказания, которая будет вызываться при загрузке изображения.
        inputs=[
            gr.Image(type="pil", label="Загруженное изображение"),  # Задаем тип входных данных (изображение формата PIL).
            gr.Slider(minimum=0, maximum=1, value=0.25, label="Products Confidence threshold"),  # Добавляем слайдер для регулировки порога уверенности.
            gr.Slider(minimum=0, maximum=1, value=0.45, label="Products IoU threshold"),  # Добавляем слайдер для регулировки порога IoU.
            gr.Slider(minimum=0, maximum=1, value=0.25, label="Shelf Confidence threshold"),  # Добавляем слайдер для регулировки порога уверенности.
            gr.Slider(minimum=0, maximum=1, value=0.45, label="Shelf IoU threshold"),
            gr.Dropdown(model_choices, value="weights/YOLOV10_Karelia.pt", label="Выбор модели YOLO"),
        ],
        
        outputs=[
            gr.Image(type="pil", label="Результат"),  # Первое выходное значение - это изображение
            gr.Textbox(label="Найденные объекты")  # Второе выходное значение - список объектов
        ], 
        title="Карельская продукция",  
        description="Детекция товаров карельских производителей в розничных магазинах на основе фото", 
        examples=[
            [ASSETS / os.path.abspath("images/EXAM.jpg"), 0.21, 0.45, 0.12, 0.21,"weights/YOLOV10_Karelia.pt"],  
            [ASSETS / os.path.abspath("images/EXAM2.jpg"), 0.21, 0.41, 0.18, 0.22,"weights/YOLOV10_Karelia.pt"],
            [ASSETS / os.path.abspath("images/EXAM3.jpg"), 0.064, 0.25, 0.17, 0.21,"weights/YOLOV10_Karelia.pt"],  
        ],
        allow_flagging="never"
    )

    iface.launch(share=True)  


if __name__ == "__main__":  
    
    infer()
