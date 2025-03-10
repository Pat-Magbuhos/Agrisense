import onnxruntime as ort
import numpy as np
import cv2

# Load ONNX model
model_path = "/home/Agrisense/Thesis/best.onnx"
session = ort.InferenceSession(model_path)

# Load test image
image_path = "/home/Agrisense/Thesis/test_image.jpg"  # Update this path
image = cv2.imread(image_path)

# Preprocess Image (Resize to 640x640 and normalize)
image = cv2.resize(image, (640, 640))
image = image.astype(np.float32) / 255.0  # Normalize pixel values
image = np.transpose(image, (2, 0, 1))  # Convert HWC to CHW
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
outputs = session.run([output_name], {input_name: image})

# Print the raw output
print("✅ Inference completed!")
print("Raw Output:", outputs)

