import os
import base64
import cv2
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, db

from ultralytics import YOLO  # YOLO model for inference

# Load environment variables from .env
dotenv_path = os.path.join(os.path.dirname(__file__), "venv/.env")
load_dotenv(dotenv_path)

# Retrieve Firebase credentials from .env
FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL")
SERVICE_ACCOUNT_PATH = os.getenv("SERVICE_ACCOUNT_PATH", "venv/serviceAccountKey.json")

# Validate environment variables
if not FIREBASE_DB_URL:
    raise ValueError("❌ ERROR: FIREBASE_DB_URL is missing from .env!")

if not os.path.exists(SERVICE_ACCOUNT_PATH):
    raise ValueError(f"❌ ERROR: Service account key not found at {SERVICE_ACCOUNT_PATH}")

# Initialize Firebase
cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
print("✅ Firebase Initialized Successfully!")

# Load trained model
model = YOLO("/home/Agrisense/Thesis/best.pt")  # Update with correct model path

# Load an image from Raspberry Pi
input_image_path = "/home/Agrisense/Thesis/test_image.jpg"  # Update with actual image path
image = cv2.imread(input_image_path)

if image is None:
    raise FileNotFoundError(f"❌ ERROR: Image file not found at {input_image_path}")

# Run YOLO inference
results = model.predict(image, conf=0.5)
output_image = results[0].plot()  # Apply bounding boxes

# Generate a unique filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_image_path = f"/home/Agrisense/Thesis/detected_{timestamp}.jpg"

# Save the detected image
cv2.imwrite(output_image_path, output_image)
print(f"✅ Detection complete! Image saved at {output_image_path}")

# Convert image to base64 string
with open(output_image_path, "rb") as img_file:
    base64_image = base64.b64encode(img_file.read()).decode("utf-8")

# Store image data in Firebase Realtime Database
ref = db.reference(f"/detections/{timestamp}")
ref.set({
    "timestamp": timestamp,
    "image_data": base64_image,
    "message": "Detection result uploaded successfully!"
})
print("✅ Image data successfully sent to Firebase Realtime Database!")

# ---- TEST RETRIEVING IMAGE FROM FIREBASE ---- #
print("🔄 Testing retrieval of stored image from Firebase...")

# Fetch the image data from Firebase
retrieved_data = db.reference(f"/detections/{timestamp}").get()

if retrieved_data:
    # Decode the base64 string back to image
    decoded_image = base64.b64decode(retrieved_data["image_data"])
    
    # Save retrieved image
    retrieved_image_path = f"/home/Agrisense/Thesis/retrieved_{timestamp}.jpg"
    with open(retrieved_image_path, "wb") as file:
        file.write(decoded_image)

    print(f"✅ Retrieved image successfully saved at {retrieved_image_path}")
else:
    print("❌ ERROR: Image retrieval failed!")
