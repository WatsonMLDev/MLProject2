from ultralytics import YOLO
import os
import glob
from ultralytics.utils.benchmarks import benchmark

# Load a model
# model = YOLO('yolov8-cls.yaml')
#
# # Train the model
# results = model.train(data='cifar10', epochs=100, imgsz=32)

#-----------------------------------------------------------------------------------------------------------------------

# model = YOLO('runs/classify/train/weights/last.pt')  # initialize


# # Directory containing test set
# test_dir = 'datasets/cifar10/test'
#
# # Get list of all classes
# classes = [d.name for d in os.scandir(test_dir) if d.is_dir()]
#
# # Placeholder for predictions and ground truth
# predictions = []
# ground_truth = []
#
# # Loop through each class directory
# for class_name in classes:
#     # Path to current class directory
#     class_dir = os.path.join(test_dir, class_name)
#
#     # List all images in current class directory
#     image_paths = glob.glob(os.path.join(class_dir, '*.png'))
#
#     for image_path in image_paths:
#         # Run inference
#         result = model(image_path)
#
#         # Extract prediction (modify based on how YOLOv8 returns results)
#         predicted_class = result[0].probs.top1  # assuming result.pred[0] contains class prediction
#
#         # Append to lists
#         predictions.append(predicted_class)
#         ground_truth.append(class_name)  # true label is the directory name
#
#
# from sklearn.metrics import average_precision_score
# import numpy as np
#
# # Convert class names in ground_truth to integer indices
# ground_truth_indices = [classes.index(name) for name in ground_truth]
#
# # Convert ground truth indices to one-hot encoded format
# y_true = np.eye(10)[ground_truth_indices]
#
# # Convert predictions to one-hot encoded format
# y_pred = np.eye(10)[predictions]
#
# # Compute average precision for each class
# average_precisions = []
# for i in range(10):
#     ap = average_precision_score(y_true[:, i], y_pred[:, i])
#     average_precisions.append(ap)
#
# # Compute mAP
# mAP = np.mean(average_precisions)
# print(f"mAP: {mAP * 100:.2f}%")


#-----------------------------------------------------------------------------------------------------------------------

# import torch
# import torchvision
# import torchvision.transforms as transforms
# import pandas as pd
# from ultralytics import YOLO
# import os
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Load and normalize the CIFAR-10 dataset for testing
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
# test_images = torch.load('test_image.pt')
#
# # Load the pretrained YOLOv8 model
# model = YOLO('runs/classify/train/weights/last.pt')
#
# # Classes for CIFAR-10
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
# # Make predictions on the test dataset
# predictions = []
#
# for image in test_images:
#     # Note: Adjust the following line if the model's prediction format is different
#     result = model.predict(image.unsqueeze(0), show=False )  # Add batch dimension
#     predicted_class = result[0].probs.top1  # Assuming this gives the class index
#     predictions.append(classes[predicted_class])
#
# # Visualize some of the test images along with their predicted labels
# def imshow(img):
#     img = img / 2 + 0.5  # Unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# # Display first 4 test images
# imshow(torchvision.utils.make_grid(test_images[:4]))
# print('Predicted: ', ' '.join(f'{predictions[j]:5s}' for j in range(4)))
#
# # Create a CSV submission file
# submission = pd.DataFrame()
# submission['label'] = predictions
# submission.to_csv("submission.csv", index=True, index_label='id')




