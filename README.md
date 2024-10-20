# Merch_CV (🥇 Хакатон IT-КЕМП 4.0)

## Предметная область:  
#### Мерчендайзинг  
Если задача маркетинга состоит в том, чтобы привлечь покупателя в магазин, то задача мерчендайзинга состоит в том, чтобы покупатель купил как можно больше, когда он зашел в магазин.
Мерчендайзинг – это умение повысить свой оборот с торгового пространства за счет:

* правильного выбора места для товара в торговом зале;
* правильного выбора места для товара на полке; правильной экспозиции товара на полке;
* эффективной информации в месте расположения товара;
* своевременного пополнения товаром полок.

## Задачи проекта:
Разработать сервис для детекции товаров карельских производителей в розничных магазинах на основе фото.  
На предоставленных изображениях торговых стеллажей распознать продукты, произведённые в Карелии (10 конкретных наименований), определить их долю на конкретном стеллаже и уровень размещения и на какой полке.

## Описание проекта:  
* Для детектирования полок на изображениях использовали модель обращаясь к ней через [ROBOFLOW API](https://universe.roboflow.com/shelfdetect-yzkro/shelves-ugxt3).
* Для классификации Карельских продуктов обучили классификатор на базе модели YOLOv10s.
* В качестве инференcа всей системы использовади Gradio. При загрузке изображения происходила его обработка через сервис roboflow (детектирование полок) и возврат результата на клиент. Далее полученное изображение обрабатывалось через, обученный на Карельских продуктах классификатор, результат, которого подавался на постпроцессинг (Рассчет площади, поиск продукта на полке и определения уровня размещения на полке).
    
![image](https://github.com/user-attachments/assets/42684944-0886-4936-96ab-4fe5ff8bf02b)

## Инструменты:
* Python
* YoloV10
* Pytorch
* CocoAnnotator
* Gradio
* Самописный парсес для перевода из формата Coco в Yolo
* Roboflow API
* GIT
* Docker

## Классы детектируемых объектов.

| class                | view |
|----------------------|------|
| sampo_waffle_classic |<image src="face_images/vafles.png" width="110" height="130">|
| sampo_waffle_lemon   |<image src="face_images/vafles2.png" width="110" height="130">|
| sl_pm_2_5_1          |<image src="face_images/milk1.png" width="110" height="130">|
| sl_pm_3_5_1          |<image src="face_images/milk2.png" width="110" height="130">|
| sl_pm_2_5_1_5        |<image src="face_images/milk3.png" width="110" height="130">|
| sl_pm_3_5_1_5        |<image src="face_images/milk4.png" width="110" height="130">|
| sl_pbm_0_5           |<image src="face_images/milk5.png" width="110" height="130">|
| ol_pm_2_5_1          |<image src="face_images/milk6.png" width="110" height="130">|
| ol_pm_3_5_1          |<image src="face_images/milk7.png" width="110" height="130">|
| ol_pbm_0_5           |<image src="face_images/milk8.png" width="110" height="130">|

## Результат работы.  
<image src="face_images/itog2.jpg"> <image src="face_images/itog1.jpg"> <image src="face_images/test1.jpg"> <image src="face_images/test2.jpg"> <image src="face_images/test4й6.jpg"> <image src="face_images/test46.jpg">

