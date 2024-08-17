import json
import os

def convert_coco_to_yolo(coco_file, output_dir, image_dir):
    # Чтение файла COCO
    with open(coco_file) as f:
        coco_data = json.load(f)

    # Создание выходного каталога, если он не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Словарь для соответствия классов и индексов
    class_mapping = {category['id']: category['name'] for category in coco_data['categories']}
    
    for image in coco_data['images']:
        image_id = image['id']
        width = image['width']
        height = image['height']

        # Путь к изображению
        image_filename = os.path.join(image_dir, image['file_name'])
        
        # Получение аннотаций для текущего изображения
        annotations = [annotation for annotation in coco_data['annotations'] if annotation['image_id'] == image_id]

        # Запись аннотаций в файл формата YOLO
        yolo_file_path = os.path.join(output_dir, f"{os.path.splitext(image['file_name'])[0]}.txt")
        with open(yolo_file_path, 'w') as yolo_file:
            for annotation in annotations:
                # Получение координат ограничивающего прямоугольника
                x_center = (annotation['bbox'][0] + annotation['bbox'][2] / 2) / width
                y_center = (annotation['bbox'][1] + annotation['bbox'][3] / 2) / height
                bbox_width = annotation['bbox'][2] / width
                bbox_height = annotation['bbox'][3] / height
                
                # Индекс класса
                class_id = annotation['category_id']
                
                # Запись строки в файл YOLO
                yolo_file.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

if __name__ == "__main__":
    coco_file = "path/to/your/coco_annotations.json"  # Замените на путь к вашему файлу COCO
    output_dir = "path/to/output/yolo/labels"         # Замените на желаемую директорию для вывода
    image_dir = "path/to/your/images"                  # Замените на путь к вашим изображениям
    
    convert_coco_to_yolo(coco_file, output_dir, image_dir)