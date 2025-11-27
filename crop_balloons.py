
import cv2
import os
import argparse
from ultralytics import YOLO

def crop_and_square(image, x1, y1, x2, y2, output_path):
    """
    Crops the image to the given bounding box and makes it a square.
    """
    # Crop the image
    cropped_image = image[y1:y2, x1:x2]

    # Get the dimensions of the cropped image
    height, width, _ = cropped_image.shape

    # Determine the size of the square
    size = max(width, height)

    # Create a new square image with a black background
    square_image = cv2.copyMakeBorder(
        cropped_image,
        (size - height) // 2,
        (size - height + 1) // 2,
        (size - width) // 2,
        (size - width + 1) // 2,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    # Save the image
    cv2.imwrite(output_path, square_image)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Detect and crop hot air balloons in an image.")
    parser.add_argument("image_path", help="Path to the input image.")
    args = parser.parse_args()

    # Load the trained model
    model = YOLO('runs/train/weights/best.pt')

    # Image to process
    image = cv2.imread(args.image_path)

    # Run inference
    results = model(image, conf=0.1)

    # Create output directory
    os.makedirs('output', exist_ok=True)

    # Process results
    for i, r in enumerate(results):
        for j, box in enumerate(r.boxes):
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()

            # Crop and save the image
            output_path = f'output/balloon_{i}_{j}.jpg'
            crop_and_square(image, x1, y1, x2, y2, output_path)
            print(f"Saved cropped image to {output_path}")

if __name__ == '__main__':
    main()
