from ultralytics import YOLO

# --- Configuration ---
# You can adjust these parameters as needed
DATA_YAML_PATH = 'data/yolo_data/data.yaml'
EPOCHS = 50
IMAGE_SIZE = 640
MODEL_NAME = 'yolov8n.pt'  # Base model to start from

def main():
    # Load a pre-trained YOLO model
    model = YOLO(MODEL_NAME)

    # Train the model
    # The training results, including the best model ('best.pt'), will be saved
    # in a 'runs/detect/train' directory.
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE
    )

    print("Training complete. Model saved to the 'runs' directory.")
    print(f"Best model path: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    main()
