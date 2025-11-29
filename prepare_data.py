import os
import json
import random
import yaml

# --- Configuration ---
IMAGE_DIR = 'data/images'
ANNOTATION_DIR = 'data/annotations'
YOLO_DATA_DIR = 'data/yolo_data'
TRAIN_RATIO = 0.8

def convert_to_yolo_format(image_width, image_height, rect_mask):
    """Converts a single bounding box to the YOLOv8 format."""
    x_center = (rect_mask['xMin'] + rect_mask['width'] / 2) / image_width
    y_center = (rect_mask['yMin'] + rect_mask['height'] / 2) / image_height
    width = rect_mask['width'] / image_width
    height = rect_mask['height'] / image_height
    return f"0 {x_center} {y_center} {width} {height}"

def main():
    # Create directories
    yolo_images_train_dir = os.path.join(YOLO_DATA_DIR, 'images', 'train')
    yolo_labels_train_dir = os.path.join(YOLO_DATA_DIR, 'labels', 'train')
    yolo_images_val_dir = os.path.join(YOLO_DATA_DIR, 'images', 'val')
    yolo_labels_val_dir = os.path.join(YOLO_DATA_DIR, 'labels', 'val')

    os.makedirs(yolo_images_train_dir, exist_ok=True)
    os.makedirs(yolo_labels_train_dir, exist_ok=True)
    os.makedirs(yolo_images_val_dir, exist_ok=True)
    os.makedirs(yolo_labels_val_dir, exist_ok=True)

    # Get list of images
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]
    random.shuffle(image_files)

    # Split into training and validation sets
    split_index = int(len(image_files) * TRAIN_RATIO)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    # Process files
    for file_list, img_dest, lbl_dest in [
        (train_files, yolo_images_train_dir, yolo_labels_train_dir),
        (val_files, yolo_images_val_dir, yolo_labels_val_dir)
    ]:
        for image_file in file_list:
            base_name = os.path.splitext(image_file)[0]
            json_file = base_name + '.json'

            # Copy image
            os.link(os.path.join(IMAGE_DIR, image_file), os.path.join(img_dest, image_file))

            # Convert annotation
            with open(os.path.join(ANNOTATION_DIR, json_file), 'r') as f:
                annotations = json.load(f)

            # For YOLO, we need image dimensions
            from PIL import Image
            with Image.open(os.path.join(IMAGE_DIR, image_file)) as img:
                img_width, img_height = img.size

            yolo_annotations = []
            for ann in annotations:
                yolo_annotations.append(convert_to_yolo_format(img_width, img_height, ann['rectMask']))

            with open(os.path.join(lbl_dest, base_name + '.txt'), 'w') as f:
                f.write('\n'.join(yolo_annotations))

    # Create data.yaml file
    data_yaml = {
        'train': os.path.abspath(os.path.join(YOLO_DATA_DIR, 'images', 'train')),
        'val': os.path.abspath(os.path.join(YOLO_DATA_DIR, 'images', 'val')),
        'nc': 1,
        'names': ['balloon']
    }
    with open(os.path.join(YOLO_DATA_DIR, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f)

    print("Data preparation complete.")

if __name__ == '__main__':
    main()
