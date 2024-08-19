import requests
from PIL import Image, ImageDraw, ImageFont



# Путь к изображению
image_path = "1.jpg"

# Задаем необходимые параметры
URL = "https://detect.roboflow.com/shelves-ugxt3/3"
API_KEY = "z5gSjUxoC2gzAYUByax6"
CONFIDENSE = 0.2  # Указываем нужный порог уверенности

# Формируем параметры запроса
PARAMS = {
    'api_key': API_KEY,
    'confidence': CONFIDENSE  # Добавляем параметр confidence
}



def get_shelfs(url:str, params:dict, confidense:float) -> Image: 
    """
    Отправляет запрос к Roboflow для детектирования полок и возвращает картинку с BB

    Параметры:
    - url: endpoint
    - params: список параметров для модели
    - confidense: порог уверенности

    Возвращает: Image
    """

    # Открываем изображение и отправляем POST запрос
    with open(image_path, 'rb') as image_file:
        response = requests.post(url, params=params, files={'file': image_file})

    # Проверяем статус ответа
    if response.status_code == 200:
        data = response.json()  # Получаем JSON ответ

        img = Image.open('1.jpg')
        # Получаем размеры изображения
        _, height = img.size
        
        # Создаем объект для рисования
        draw = ImageDraw.Draw(img)

        # Проходим по всем предсказанным объектам
        for prediction in data['predictions']:
            if prediction['confidence'] >= confidense:
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

