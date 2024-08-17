import gradio as gr
import PIL.Image as Image
from ultralytics import ASSETS, YOLO

model = YOLO("yolov8n.pt")

def calculate_area(box):
    x_min, y_min, x_max, y_max = box
    area = (x_max - x_min) * (y_max - y_min)
    return area

def get_box_center(box):
    x_min, y_min, x_max, y_max = box
    center_y = (y_min + y_max) / 2
    return center_y

def calculate_height(product_box, shelf_bottom_y):
    product_center_y = get_box_center(product_box)
    height = shelf_bottom_y - product_center_y
    return height

def predict_image(img, conf_threshold, iou_threshold):
    """Predicts objects in an image using a YOLOv8 model with adjustable confidence and IOU thresholds."""
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
    )

    total_milk_product_area = 0
    shelf_area = 0
    shelf_bottom_y = None
    heights = []

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

        # Получаем координаты ограничивающих рамок и классы
        boxes = r.boxes.xyxy.numpy()  # Координаты ограничивающих рамок
        labels = r.boxes.cls.numpy()   # Метки классов

        for box, label in zip(boxes, labels):
            area = calculate_area(box)

            if label == 0:  # Предположим, что класс 0 - это стеллаж
                shelf_area = area
                shelf_bottom_y = box[3]  # y_max стеллажа
            else:  # Остальные объекты - молочные продукты
                total_milk_product_area += area
                if shelf_bottom_y is not None:
                    height = calculate_height(box, shelf_bottom_y)
                    heights.append(height)

    area_ratio = total_milk_product_area / shelf_area if shelf_area > 0 else 0
    avg_height = sum(heights) / len(heights) if heights else 0

    result_text = f"Отношение площадей: {area_ratio:.2f}\nСредняя высота продукции: {avg_height:.2f}"

    return im, result_text


iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Загруженное изображение"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
    ],
    outputs=[
        gr.Image(type="pil", label="Результат"),
        gr.Textbox(label="Результаты анализа")
    ],
    title="Карельская продукция",
    description="ОПИСАНИЕ",
    examples=[
        [ASSETS / "bus.jpg", 0.25, 0.45],
        [ASSETS / "zidane.jpg", 0.25, 0.45],
    ],
)

if __name__ == "__main__":
    iface.launch()
