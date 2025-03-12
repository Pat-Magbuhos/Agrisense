import os
import cv2
import torch
from datetime import datetime
from ultralytics import YOLO
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, storage

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), "venv", ".env")  # Ensure correct .env path
load_dotenv(dotenv_path)

FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET")
SERVICE_ACCOUNT_PATH = os.getenv("SERVICE_ACCOUNT_PATH", "venv/serviceAccountKey.json")

# Validate environment variables
if not FIREBASE_STORAGE_BUCKET:
    raise ValueError("❌ ERROR: FIREBASE_STORAGE_BUCKET is missing from .env!")
if not os.path.exists(SERVICE_ACCOUNT_PATH):
    raise ValueError(f"❌ ERROR: Service account key not found at {SERVICE_ACCOUNT_PATH}")

# Initialize Firebase
cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_STORAGE_BUCKET})
bucket = storage.bucket()

# Load trained YOLO model
model = YOLO("/home/Agrisense/Thesis/best.pt")  # Update with your model path

# Define input image path (Modify this to the image you want to test)
input_image_path = "/home/Agrisense/Thesis/test1.jpg"  # Place test image here

# Load the image
image = cv2.imread(input_image_path)

if image is None:
    raise ValueError(f"❌ ERROR: Could not load image from {input_image_path}")

# Run inference
results = model.predict(image, imgsz=640, conf=0.5)

# Annotate the image with bounding boxes
annotated_image = results[0].plot()

# Generate timestamp-based filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_image_path = f"/home/Agrisense/Thesis/detected_{timestamp}.jpg"

# Save the annotated image locally
cv2.imwrite(output_image_path, annotated_image)
print(f"✅ Detection complete! Image saved at {output_image_path}")

# Upload the detected image to Firebase Storage
blob = bucket.blob(f"detections/detected_{timestamp}.jpg")
blob.upload_from_filename(output_image_path)
blob.make_public()
firebase_image_url = blob.public_url

print(f"✅ Image uploaded to Firebase: {firebase_image_url}")
