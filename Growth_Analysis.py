import time
import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, storage, db
from datetime import datetime
from ultralytics import YOLO

# Firebase Initialization
cred = credentials.Certificate("/home/Agrisense/Thesis/venv/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://agrisense-6a089-default-rtdb.asia-southeast1.firebasedatabase.app/',
    'storageBucket': 'agrisense-6a089.appspot.com'
})

# Load YOLO Model
model = YOLO("/home/Agrisense/Thesis/best.pt")

# Camera Setup
camera = cv2.VideoCapture(0)

# Trigonometry Constants
CAMERA_ANGLE = 45  # Degrees
CAMERA_HEIGHT = 30  # cm (Height from the ground)
FOCAL_LENGTH = 800  # Pixels (Calibrated for estimation)

# Growth Stage Thresholds (Adjustable)
GROWTH_THRESHOLDS = {
    "seedling": {"height": 5, "leaves": 4, "leaf_area": 15},
    "vegetative": {"height": 15, "leaves": 8, "leaf_area": 50},
    "mature": {"height": 25, "leaves": 12, "leaf_area": 100},
}

# Function to estimate height using trigonometry
def estimate_height(bbox):
    pixel_height = bbox[3] - bbox[1]
    real_height = (CAMERA_HEIGHT * pixel_height) / FOCAL_LENGTH
    real_height /= np.tan(np.radians(CAMERA_ANGLE))
    return round(real_height, 2)

# Function to estimate leaf area
def estimate_leaf_area(bbox):
    pixel_width = bbox[2] - bbox[0]
    pixel_height = bbox[3] - bbox[1]
    pixel_area = pixel_width * pixel_height  # Approximate area in pixels

    # Convert pixel area to cm² (Adjust scaling factor based on calibration)
    scale_factor = 0.05  
    real_area = pixel_area * scale_factor
    return round(real_area, 2)

# Function to classify growth stage
def classify_growth(height, leaf_count, leaf_area):
    if height < GROWTH_THRESHOLDS["seedling"]["height"] and leaf_count < GROWTH_THRESHOLDS["seedling"]["leaves"] and leaf_area < GROWTH_THRESHOLDS["seedling"]["leaf_area"]:
        return "Seedling"
    elif height < GROWTH_THRESHOLDS["vegetative"]["height"] and leaf_count < GROWTH_THRESHOLDS["vegetative"]["leaves"] and leaf_area < GROWTH_THRESHOLDS["vegetative"]["leaf_area"]:
        return "Vegetative"
    else:
        return "Mature"

# Function to capture, process, and upload images
def capture_and_upload():
    ret, frame = camera.read()
    if not ret:
        print("❌ Failed to capture image")
        return
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save raw image
    raw_image_path = f"/home/Agrisense/Thesis/raw_{timestamp}.jpg"
    cv2.imwrite(raw_image_path, frame)

    # Run YOLO Object Detection
    results = model.predict(frame)
    detected_frame = frame.copy()  # Copy original image to draw on

    # Extract growth parameters
    leaf_count = 0
    total_leaf_area = 0
    estimated_height = 0

    for result in results:
        for bbox in result.boxes.xyxy:
            bbox = bbox.cpu().numpy().astype(int)  # Convert to integer

            estimated_height = estimate_height(bbox)
            leaf_area = estimate_leaf_area(bbox)
            total_leaf_area += leaf_area
            leaf_count += 1

            # Classify Growth Stage
            growth_stage = classify_growth(estimated_height, leaf_count, total_leaf_area)

            # Draw bounding box
            cv2.rectangle(detected_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            # Label the bounding box with Growth Stage
            label = f"{growth_stage} ({estimated_height}cm)"
            cv2.putText(detected_frame, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save detected image
    detected_image_path = f"/home/Agrisense/Thesis/detected_{timestamp}.jpg"
    cv2.imwrite(detected_image_path, detected_frame)

    # Upload images to Firebase Storage
    bucket = storage.bucket()

    raw_blob = bucket.blob(f"raw_images/raw_{timestamp}.jpg")
    raw_blob.upload_from_filename(raw_image_path)
    raw_blob.make_public()
    raw_image_url = raw_blob.public_url

    detected_blob = bucket.blob(f"detected_images/detected_{timestamp}.jpg")
    detected_blob.upload_from_filename(detected_image_path)
    detected_blob.make_public()
    detected_image_url = detected_blob.public_url

    print(f"📤 Uploaded to Firebase: {raw_image_url} & {detected_image_url}")

    # Send Data to Firebase Database
    data = {
        "timestamp": timestamp,
        "raw_image_url": raw_image_url,
        "detected_image_url": detected_image_url,
        "growth_stage": growth_stage,
        "estimated_height_cm": estimated_height,
        "leaf_count": leaf_count,
        "total_leaf_area_cm2": total_leaf_area
    }
    ref = db.reference("/plant_analysis")
    ref.push(data)
    print("📡 Data successfully sent to Firebase!")

# Run process every 1 minute
try:
    while True:
        capture_and_upload()
        time.sleep(60)
except KeyboardInterrupt:
    print("🛑 Stopping capture process")
    camera.release()
