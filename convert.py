import json
import os

if not os.path.exists('annotations/instances_val2017.json'):
    print("Error: annotations/instances_val2017.json not found. Please run the download script first.")
    exit(1)

if not os.path.exists('data/labels'):
    os.makedirs('data/labels')

with open('annotations/instances_val2017.json', 'r') as f:
    coco = json.load(f)

for img in coco['images']:
    img_id = img['id']
    filename = img['file_name'].replace('.jpg', '.txt')
    w, h = img['width'], img['height']
    
    labels = []
    for ann in coco['annotations']:
        if ann['image_id'] == img_id:
            bbox = ann['bbox']
            x_center = (bbox[0] + bbox[2]/2) / w
            y_center = (bbox[1] + bbox[3]/2) / h
            width = bbox[2] / w
            height = bbox[3] / h
            class_id = ann['category_id'] - 1
            labels.append(f'{class_id} {x_center} {y_center} {width} {height}')
    
    if labels:
        with open(f'data/labels/{filename}', 'w') as f:
            f.write('\n'.join(labels))

print("Conversion completed successfully")