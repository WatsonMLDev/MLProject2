from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO('yolov8x-cls.pt')

model.tune(data='cifar10', epochs=30, iterations=50, batch=96,workers=16, optimizer='AdamW', plots=False, save=False, val=False)
