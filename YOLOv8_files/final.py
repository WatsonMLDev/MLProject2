from ultralytics import YOLO
import os



#region train
model = YOLO('yolov8x-cls.pt')

results = model.train(data='cifar10', epochs=50, imgsz=32, batch=128, workers=16,
                      lr0= 0.00859,
                      lrf= 0.01068,
                      momentum= 0.92692,
                      weight_decay= 0.00046,
                      warmup_epochs= 3.06646,
                      warmup_momentum= 0.8081,
                      box= 6.46683,
                      cls= 0.55668,
                      dfl= 1.53146,
                      hsv_h= 0.01546,
                      hsv_s= 0.85974,
                      hsv_v= 0.44395,
                      degrees= 0.0,
                      translate= 0.06773,
                      scale= 0.49418,
                      shear= 0.0,
                      perspective= 0.0,
                      flipud= 0.0,
                      fliplr= 0.44357,
                      mosaic= 0.9805,
                      mixup= 0.0,
                      copy_paste= 0.0)
#endregion

#region test locally

accuracy = []

for model in os.listdir('models'):
    if model.endswith(".pt"):
        model = YOLO("models/" + model)

        metrics = model.val(data='./datasets/cifar10/')
        accuracy.append(metrics.top1)

for i in range(len(accuracy)):
    print(f"Model {i} has accuracy {accuracy[i]}")

#endregion

