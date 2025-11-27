# Hot Air Balloon Detection and Cropping

This project provides a set of scripts to train a YOLOv8 model to detect hot air balloons in images and then crop them out into square tiles.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Training Your Own Model

If you have your own dataset of images and annotations, you can train a custom model for the best results.

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

Once your data is organized correctly, run the data preparation script. This script will convert your JSON annotations into the YOLOv8 format required for training and split your data into training and validation sets.

Open your terminal and run the following command:

```bash
python prepare_data.py
```

This will create a `yolo_data` directory containing the processed data and a `data.yaml` configuration file.

### 3. Run the Training

Now you are ready to train the model. Run the training script:

```bash
python train.py
```

This process may take a significant amount of time, depending on the size of your dataset and the power of your computer.

Once training is complete, the best-performing model will be saved to a file, typically located at `runs/detect/train/weights/best.pt`.

## Using Your Trained Model for Cropping

After training, you can use your custom model to detect and crop balloons from new images.

### 1. Update the Model Path

Open the `crop_balloons.py` script and find the `MODEL_PATH` variable. Update it to point to your newly trained model file.

For example:
```python
# In crop_balloons.py
MODEL_PATH = 'runs/detect/train/weights/best.pt'
```

### 2. Run the Cropping Script

You can now run the cropping script on any image:

```bash
python crop_balloons.py <path_to_your_image>
```

The cropped balloon images will be saved in the `output_crops/` directory.

## Other Scripts

### `evaluate.py`

Use this script to evaluate the performance of your trained model on the validation set.

```bash
python evaluate.py
```
