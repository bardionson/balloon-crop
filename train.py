
from ultralytics import YOLO

def main():
    # Load a pre-trained YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Train the model
    model.train(data='dataset/data.yaml', epochs=10, imgsz=640, project='runs', name='train', exist_ok=True)

if __name__ == '__main__':
    main()
