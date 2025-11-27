
import json
import os
import cv2
import shutil
from sklearn.model_selection import train_test_split

def convert_to_yolo(image_width, image_height, xmin, ymin, width, height):
    x_center = (xmin + width / 2) / image_width
    y_center = (ymin + height / 2) / image_height
    w_norm = width / image_width
    h_norm = height / image_height
    return x_center, y_center, w_norm, h_norm

def main():
    # Create directories
    os.makedirs('dataset/train/images', exist_ok=True)
    os.makedirs('dataset/train/labels', exist_ok=True)
    os.makedirs('dataset/valid/images', exist_ok=True)
    os.makedirs('dataset/valid/labels', exist_ok=True)

    # Get all image files
    image_files = [f for f in os.listdir('dataset') if f.endswith('.JPG')]

    # Split the data into training and validation sets
    train_files, valid_files = train_test_split(image_files, test_size=0.2, random_state=42)

    for image_file in image_files:
        # Determine destination
        if image_file in train_files:
            img_dest_folder = 'dataset/train/images'
            lbl_dest_folder = 'dataset/train/labels'
        else:
            img_dest_folder = 'dataset/valid/images'
            lbl_dest_folder = 'dataset/valid/labels'

        # Get image dimensions
        image_path = os.path.join('dataset', image_file)
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape

        # Read JSON file
        json_file = image_file.replace('.JPG', '.json')
        json_path = os.path.join('dataset', json_file)
        with open(json_path, 'r') as f:
            annotations = json.load(f)

        # Create YOLO label file
        yolo_labels = []
        for ann in annotations:
            rect_mask = ann['rectMask']
            xmin = rect_mask['xMin']
            ymin = rect_mask['yMin']
            width = rect_mask['width']
            height = rect_mask['height']

            x_center, y_center, w_norm, h_norm = convert_to_yolo(image_width, image_height, xmin, ymin, width, height)
            yolo_labels.append(f"0 {x_center} {y_center} {w_norm} {h_norm}")

        # Write YOLO label file
        label_file = image_file.replace('.JPG', '.txt')
        label_path = os.path.join(lbl_dest_folder, label_file)
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_labels))

        # Copy image file
        shutil.copy(image_path, os.path.join(img_dest_folder, image_file))

    # Create data.yaml
    data_yaml = f"""
    path: {os.path.abspath('dataset')}
    train: train/images
    val: valid/images

    nc: 1
    names: ['balloon']
    """
    with open('dataset/data.yaml', 'w') as f:
        f.write(data_yaml)

if __name__ == '__main__':
    main()
