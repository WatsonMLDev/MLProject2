import matplotlib.pyplot as plt
import torch
import cv2
import yaml
from matplotlib import colors
from torchvision import transforms
import numpy as np

from utils.datasets import letterbox
from utils.general import non_max_suppression
from utils.plots import plot_one_box
import torchvision
import os



# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
#
#
# import os
# import torchvision
#
# # Create directory structure inside 'yolov7' but without the 'yolov7' prefix in paths
# base_path = "cifar10"
# if not os.path.exists(base_path):
#     os.makedirs(f"{base_path}/images/train")
#     os.makedirs(f"{base_path}/images/val")
#     os.makedirs(f"{base_path}/labels/train")
#     os.makedirs(f"{base_path}/labels/val")
#
# # Split dataset into training and validation sets
# train_size = int(0.9 * len(trainset))
# val_size = len(trainset) - train_size
# train_subset, val_subset = torch.utils.data.random_split(trainset, [train_size, val_size])
#
# def convert_to_yolo_format(dataset, subset, mode):
#     image_paths = []
#
#     for idx in range(len(subset)):
#         # Get image and label
#         image, label = subset[idx]
#
#         # Save image
#         image_path = f"{base_path}/images/{mode}/{idx}.jpg"
#         torchvision.utils.save_image(image, image_path)
#         image_paths.append(f"cifar10/images/{mode}/{idx}.jpg")  # Relative path for train.txt and val.txt
#
#         # Create annotation in YOLO format
#         height, width = image.shape[1:3]
#         x_center = width / 2.0
#         y_center = height / 2.0
#         annotation = f"{label} {x_center/width} {y_center/height} {width/width} {height/height}\n"
#
#         # Save annotation
#         with open(f"{base_path}/labels/{mode}/{idx}.txt", "w") as f:
#             f.write(annotation)
#
#     return image_paths
#
# # Convert datasets to YOLO format
# train_image_paths = convert_to_yolo_format(trainset, train_subset, "train")
# val_image_paths = convert_to_yolo_format(trainset, val_subset, "val")
#
# # Generate .txt files listing the paths of training and validation images
# with open(f"{base_path}/train.txt", "w") as f:
#     f.write("\n".join(train_image_paths))
#
# with open(f"{base_path}/val.txt", "w") as f:
#     f.write("\n".join(val_image_paths))
#
# print("Data preparation completed!")




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
half = device.type != 'cpu'  # half precision only supported on CUDA
print(device)

# Load hyperparameters
with open('data/hyp.yaml', 'r') as f:
    hyp = yaml.load(f, Loader=yaml.FullLoader)

weights = 'best.pt'
model = torch.load(weights)['model'].to(device)
if half:
    model.half()  # to FP16

# Get class names (for displaying results)
names = model.module.names if hasattr(model, 'module') else model.names

def get_actual_class(annotation_path):
    """Extract the actual class from the YOLO annotation file."""
    with open(annotation_path, 'r') as f:
        # The class is the first value on the first (and only) line of the YOLO annotation
        return int(f.readline().split()[0])

base_path = "cifar10"
with open(f"{base_path}/val.txt", "r") as f:
    image_paths = [line.strip() for line in f.readlines()]

all_predictions = []
correct_count = 0

for img_path in image_paths:
    img = cv2.imread(img_path)  # Adjust path for reading
    img = letterbox(img, 640, stride=64, auto=True)[0]
    img = transforms.ToTensor()(img).unsqueeze(0).to(device)
    if half:
        img = img.half()

    with torch.no_grad():
        outputs = model(img)

    # Extract the main output tensor
    main_output = outputs[0]

    # Aggregate across the spatial dimensions (i.e., take the mean across the 25500 predictions)
    class_predictions = torch.mean(main_output, dim=1)
    # Average over the spatial dimensions
    class_predictions_avg = torch.mean(class_predictions, dim=[1, 2])

    # Get the predicted class
    predicted_class = torch.argmax(class_predictions_avg, dim=1).item()


    # Get the actual class
    annotation_path = os.path.join(img_path.replace('images', 'labels').replace('.jpg', '.txt'))
    actual_class = get_actual_class(annotation_path)

    # Check if the prediction is correct
    if predicted_class == actual_class:
        correct_count += 1

    print(f"Image: {img_path} | Predicted Class: {names[predicted_class]} | Actual Class: {names[actual_class]}")

accuracy = (correct_count / len(image_paths)) * 100
print(f"Accuracy: {accuracy:.2f}%")






