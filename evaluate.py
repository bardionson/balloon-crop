
from ultralytics import YOLO

def main():
    # Load the trained model
    model = YOLO('runs/train/weights/best.pt')

    # Evaluate the model
    metrics = model.val()
    print(metrics)

if __name__ == '__main__':
    main()
