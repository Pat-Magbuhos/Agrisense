from ultralytics import YOLO
import torch
import cv2
import matplotlib.pyplot as plt

# Load the PyTorch-trained YOLO model
model = YOLO("/home/Agrisense/Thesis/best.pt")  # Use .pt instead of .onnx

# Load an unseen test image
image_path = "/home/Agrisense/Thesis/test1.jpg"  
results = model(image_path)  # Run inference

# Plot results with bounding boxes
res_plotted = results[0].plot()
cv2.imwrite("/home/Agrisense/Thesis/detected_image.jpg", res_plotted)

print("✅ Detection complete! Image saved at: /home/Agrisense/Thesis/detected_image.jpg")
