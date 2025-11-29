from ultralytics import YOLO

# --- Configuration ---
# Path to the best trained model weights
MODEL_PATH = 'runs/detect/train/weights/best.pt'
# Path to the data configuration file
DATA_YAML_PATH = 'data/yolo_data/data.yaml'

def main():
    # Load the trained YOLO model
    model = YOLO(MODEL_PATH)

    # Evaluate the model's performance on the validation set
    metrics = model.val(data=DATA_YAML_PATH)

    print("Evaluation complete.")
    # The metrics object contains various performance indicators,
    # such as mAP50-95, precision, and recall.
    # For example, to access the mAP50-95:
    print(f"mAP50-95: {metrics.box.map}")
    print("For a detailed breakdown, check the results saved in the 'runs/detect/val' directory.")


if __name__ == '__main__':
    main()
