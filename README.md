# Hot Air Balloon Detection and Cropping

This project provides a complete pipeline to train a custom YOLOv8 model to detect hot air balloons in images and then crop them out into padded, square tiles.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Step 1: Training Your Own Model

For the best results, you should train a model on your own dataset of images and annotations.

### 1. Organize Your Data

The scripts are designed to work with a specific folder structure. Before you begin, make sure your data is organized as follows:

-   **Images:** Place all of your image files (e.g., `.jpg`, `.png`) inside the `data/images/` directory.
-   **Annotations:** Place all of your corresponding JSON annotation files inside the `data/annotations/` directory.

**Important:** Each image in `data/images/` must have a corresponding JSON file in `data/annotations/` with the same base filename (e.g., `photo1.jpg` and `photo1.json`).

The JSON annotation format should be an array of objects, where each object has a `rectMask` with `xMin`, `yMin`, `width`, and `height` for the bounding box.

Example `photo1.json`:
```json
[
  {
    "rectMask": { "xMin": 100, "yMin": 100, "width": 200, "height": 200 },
    "labels": { "labelName": "balloon" }
  }
]
```

### 2. Prepare Data for YOLOv8

Once your data is organized correctly, run the data preparation script. This script converts your JSON annotations into the YOLOv8 format required for training and splits your data into training and validation sets.

Open your terminal and run the following command:
```bash
python prepare_data.py
```
This will create a `data/yolo_data` directory containing the processed data and a `data.yaml` configuration file.

### 3. Run the Training

Now you are ready to train the model. Run the training script:
```bash
python train.py
```
This process may take a significant amount of time, depending on the size of your dataset and the power of your computer.

Once training is complete, the best-performing model will be saved to a file, typically located at `runs/detect/train/weights/best.pt`. Note this path for the next step.

## Step 2: Evaluating Your Model (Optional)

You can evaluate your new model's performance on the validation dataset by running:
```bash
python evaluate.py
```
This will print performance metrics like mAP (mean Average Precision) and save detailed results in a `runs/detect/val` directory.

## Step 3: Cropping Balloons from New Images

After training, you can use your custom model to detect and crop balloons from new images.

The `crop_balloons.py` script uses your trained model to find balloons and save a padded, square crop of each one. The cropping logic is designed to avoid adding black bars, even if the balloon is near the edge of the image.

### Basic Usage
To run the script on a single image, provide the path to the image. It will automatically use the default model trained in the previous step.

```bash
python crop_balloons.py path/to/your/image.jpg
```

### Advanced Usage (Command-Line Arguments)
You can customize the behavior of the cropping script with the following options:

-   `--model`: Specify the path to your trained model file.
-   `--conf`: Set the confidence threshold for detection (a value between 0.0 and 1.0).
-   `--padding`: Set the percentage of padding to add around the detected balloon.

**Example:**
This command runs detection on an image using a specific model, a confidence threshold of 25%, and 15% padding around the detected balloons.
```bash
python crop_balloons.py path/to/your/image.jpg --model runs/detect/train/weights/best.pt --conf 0.25 --padding 0.15
```

All cropped images will be saved in the `output_crops/` directory.
