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
input_image_path = "/home/Agrisense/Thesis/Detected/Detected/20250315_181349.jpg"  # Update with actual image path
image = cv2.imread(input_image_path)

if image is None:
    raise FileNotFoundError(f"❌ ERROR: Image file not found at {input_image_path}")

# Generate a unique filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save raw image for reference
raw_image_path = f"/home/Agrisense/Thesis/raw_{timestamp}.jpg"
cv2.imwrite(raw_image_path, image)

# Run YOLO inference
results = model.predict(image, conf=0.5)
output_image = results[0].plot()  # Apply bounding boxes

# Save the detected image
detected_image_path = f"/home/Agrisense/Thesis/detected_{timestamp}.jpg"
cv2.imwrite(detected_image_path, output_image)
print(f"✅ Detection complete! Images saved: {raw_image_path}, {detected_image_path}")

# Convert images to base64 string
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

base64_raw_image = encode_image(raw_image_path)
base64_detected_image = encode_image(detected_image_path)

# Store both images in Firebase under separate nodes
ref = db.reference(f"/detections/{timestamp}")
ref.set({
    "timestamp": timestamp,
    "raw_image": base64_raw_image,  # Stores raw image data
    "detected_image": base64_detected_image,  # Stores detected image data
    "message": "Both raw and detected images uploaded successfully!"
})
print("✅ Raw and detected images successfully sent to Firebase Realtime Database!")

# ---- TEST RETRIEVING IMAGES FROM FIREBASE ---- #
print("🔄 Testing retrieval of stored images from Firebase...")

# Fetch the image data from Firebase
retrieved_data = db.reference(f"/detections/{timestamp}").get()

if retrieved_data:
    # Decode raw image from Firebase
    decoded_raw_image = base64.b64decode(retrieved_data["raw_image"])
    retrieved_raw_image_path = f"/home/Agrisense/Thesis/retrieved_raw_{timestamp}.jpg"
    with open(retrieved_raw_image_path, "wb") as file:
        file.write(decoded_raw_image)
    
    # Decode detected image from Firebase
    decoded_detected_image = base64.b64decode(retrieved_data["detected_image"])
    retrieved_detected_image_path = f"/home/Agrisense/Thesis/retrieved_detected_{timestamp}.jpg"
    with open(retrieved_detected_image_path, "wb") as file:
        file.write(decoded_detected_image)

    print(f"✅ Retrieved raw image saved at {retrieved_raw_image_path}")
    print(f"✅ Retrieved detected image saved at {retrieved_detected_image_path}")
else:
    print("❌ ERROR: Image retrieval failed!")
