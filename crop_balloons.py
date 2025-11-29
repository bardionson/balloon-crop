import cv2
import os
import argparse
from ultralytics import YOLO

# --- Configuration ---
OUTPUT_DIR = 'output_crops'
DEFAULT_PADDING_PERCENT = 0.2
DEFAULT_CONFIDENCE = 0.1

def crop_balloon_with_padding(image, x1, y1, x2, y2, padding_percent, output_path):
    """
    Crops a padded, square region around the bounding box, handling image edges.
    """
    # Get original image dimensions
    img_height, img_width, _ = image.shape

    # Calculate bounding box dimensions
    box_width = x2 - x1
    box_height = y2 - y1

    # Apply padding
    padding_w = int(box_width * padding_percent)
    padding_h = int(box_height * padding_percent)

    # Calculate padded box coordinates
    x1_padded = x1 - padding_w
    y1_padded = y1 - padding_h
    x2_padded = x2 + padding_w
    y2_padded = y2 + padding_h

    # Determine the size of the square crop based on the largest padded dimension
    padded_width = x2_padded - x1_padded
    padded_height = y2_padded - y1_padded
    crop_size = max(padded_width, padded_height)

    # Calculate the center of the padded box
    center_x = (x1_padded + x2_padded) / 2
    center_y = (y1_padded + y2_padded) / 2

    # Calculate initial crop coordinates for the square
    crop_x1 = int(center_x - crop_size / 2)
    crop_y1 = int(center_y - crop_size / 2)
    crop_x2 = crop_x1 + crop_size
    crop_y2 = crop_y1 + crop_size

    # Shift the crop box to stay within image boundaries
    if crop_x1 < 0:
        crop_x2 -= crop_x1  # Shift right
        crop_x1 = 0
    if crop_y1 < 0:
        crop_y2 -= crop_y1  # Shift down
        crop_y1 = 0
    if crop_x2 > img_width:
        crop_x1 -= (crop_x2 - img_width)  # Shift left
        crop_x2 = img_width
    if crop_y2 > img_height:
        crop_y1 -= (crop_y2 - img_height)  # Shift up
        crop_y2 = img_height

    # Final clamp to ensure coordinates are within bounds after shifting
    final_x1 = max(0, crop_x1)
    final_y1 = max(0, crop_y1)
    final_x2 = min(img_width, crop_x2)
    final_y2 = min(img_height, crop_y2)

    # Crop the image
    final_crop = image[final_y1:final_y2, final_x1:final_x2]

    # Save the cropped image
    cv2.imwrite(output_path, final_crop)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Detect and crop hot air balloons in an image.")
    parser.add_argument("image_path", help="Path to the input image.")
    parser.add_argument(
        "--model",
        default='runs/detect/train/weights/best.pt',
        help="Path to the trained YOLOv8 model file."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"Confidence threshold for detection (default: {DEFAULT_CONFIDENCE})."
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=DEFAULT_PADDING_PERCENT,
        help=f"Percentage of padding to add around the bounding box (default: {DEFAULT_PADDING_PERCENT})."
    )
    args = parser.parse_args()

    # Load the trained model
    model = YOLO(args.model)

    # Image to process
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Error: Could not read image from {args.image_path}")
        return

    # Run inference
    results = model(image, conf=args.conf)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get the base name of the input image for the output filename
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]

    # Process results
    if len(results[0].boxes) == 0:
        print("No balloons detected.")
    else:
        for j, box in enumerate(results[0].boxes):
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()

            # Define the output path for the cropped image
            output_path = os.path.join(OUTPUT_DIR, f"{base_name}_crop_{j}.jpg")

            # Crop and save the image
            crop_balloon_with_padding(image, x1, y1, x2, y2, args.padding, output_path)
            print(f"Saved cropped image to {output_path}")

if __name__ == '__main__':
    main()
