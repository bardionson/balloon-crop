
import cv2
import json
import os
from crop_balloons import crop_and_square

def main():
    # Load the image
    image_path = 'dataset/valid/images/IMG_4575.JPG'
    image = cv2.imread(image_path)

    # Load the annotations
    json_path = 'dataset/IMG_4575.json'
    with open(json_path, 'r') as f:
        annotations = json.load(f)

    # Create output directory
    os.makedirs('output', exist_ok=True)

    # Process each annotation
    for i, ann in enumerate(annotations):
        rect_mask = ann['rectMask']
        x1 = int(rect_mask['xMin'])
        y1 = int(rect_mask['yMin'])
        x2 = int(rect_mask['xMin'] + rect_mask['width'])
        y2 = int(rect_mask['yMin'] + rect_mask['height'])

        # Crop and save the image
        output_path = f'output/manual_crop_{i}.jpg'
        crop_and_square(image, x1, y1, x2, y2, output_path)
        print(f"Saved cropped image to {output_path}")

if __name__ == '__main__':
    main()
