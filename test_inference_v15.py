import os
import base64
import numpy as np
import subprocess
from datetime import datetime
import time
import sys
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, db
from ultralytics import YOLO  # YOLO model for inference
import cv2  # OpenCV for processing

from AgrisenseSensors import (
    read_ds18b20_temp,
    read_light,
    dhtDevice,
    upload_sensor_data_to_firebase
)

# Load environment variables from .env
dotenv_path = os.path.join(os.path.dirname(__file__), "venv/.env")
load_dotenv(dotenv_path)

# Retrieve Firebase credentials from .env
FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL")
SERVICE_ACCOUNT_PATH = os.getenv("SERVICE_ACCOUNT_PATH", "venv/serviceAccountKey.json")

# Validate environment variables
if not FIREBASE_DB_URL:
    raise ValueError("ERROR: FIREBASE_DB_URL is missing from .env!")
if not os.path.exists(SERVICE_ACCOUNT_PATH):
    raise ValueError(f"ERROR: Service account key not found at {SERVICE_ACCOUNT_PATH}")

# Initialize Firebase
try:
    firebase_admin.delete_app(firebase_admin.get_app())
except ValueError:
    pass  # No app was initialized

cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})

# Ensure directory structure exists
BASE_DIR = "/home/Agrisense/Thesis"
CAPTURED_RAW_DIR = os.path.join(BASE_DIR, "Captured", "Raw")
CAPTURED_RETRIEVED_DIR = os.path.join(BASE_DIR, "Captured", "Retrieved")
DETECTED_DIR = os.path.join(BASE_DIR, "Detected", "Detected")
DETECTED_RETRIEVED_DIR = os.path.join(BASE_DIR, "Detected", "Retrieved")

for directory in [CAPTURED_RAW_DIR, CAPTURED_RETRIEVED_DIR, DETECTED_DIR, DETECTED_RETRIEVED_DIR]:
    os.makedirs(directory, exist_ok=True)

# Load trained model
model = YOLO("/home/Agrisense/Thesis/bestv2.pt")

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

GROWTH_STAGE_CLASSES = ["Seedling", "Vegetative", "Mature"]

# Function to estimate height using trigonometry
def estimate_height(bbox):
    pixel_height = bbox[3] - bbox[1]
    real_height = (CAMERA_HEIGHT * pixel_height) / FOCAL_LENGTH
    real_height /= np.tan(np.radians(CAMERA_ANGLE))
    return round(real_height, 2)

# Function to estimate leaf area
def estimate_leaf_area(bbox, cm_per_pixel):
    pixel_width = bbox[2] - bbox[0]
    pixel_height = bbox[3] - bbox[1]
    pixel_area = pixel_width * pixel_height

    # Convert pixel area to cm² using calibrated scale
    pixel_area_to_cm2 = cm_per_pixel ** 2
    real_area = pixel_area * pixel_area_to_cm2
    return round(real_area, 2)

# Improved Leaf Counting using Contours
def count_leaves(image_path):
    # Load image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding to enhance leaf edges
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Apply morphological operations to remove noise
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to remove small noise
    min_leaf_area = 500  # Adjust based on plant image
    leaf_contours = [c for c in contours if cv2.contourArea(c) > min_leaf_area]

    # Draw contours for visualization
    output = image.copy()
    cv2.drawContours(output, leaf_contours, -1, (0, 255, 0), 2)

    # Save the processed image with contours
    processed_image_path = image_path.replace(".jpg", "_contours.jpg")
    cv2.imwrite(processed_image_path, output)

    leaf_count = len(leaf_contours)
    print(f"Detected Leaves: {leaf_count} (Saved processed image: {processed_image_path})")
    
    return leaf_count, processed_image_path

# Function to classify growth stage
def classify_growth(height, leaf_count, leaf_area):
    if height < GROWTH_THRESHOLDS["seedling"]["height"] and leaf_count < GROWTH_THRESHOLDS["seedling"]["leaves"] and leaf_area < GROWTH_THRESHOLDS["seedling"]["leaf_area"]:
        return "Seedling"
    elif height < GROWTH_THRESHOLDS["vegetative"]["height"] and leaf_count < GROWTH_THRESHOLDS["vegetative"]["leaves"] and leaf_area < GROWTH_THRESHOLDS["vegetative"]["leaf_area"]:
        return "Vegetative"
    else:
        return "Mature"
    
def trigger_pump():
    try:
        print("Pest detected! Activating pump...")
        import RPi.GPIO as GPIO
        RELAY_PIN = 23
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(RELAY_PIN, GPIO.OUT)
        GPIO.output(RELAY_PIN, GPIO.LOW)  # Turn pump ON
        time.sleep(5)
        GPIO.output(RELAY_PIN, GPIO.HIGH)  # Turn pump OFF
        GPIO.cleanup()
        print("Pump deactivated.")
    except Exception as e:
        print(f"Error triggering pump: {e}")


# Function to capture image
def capture_image():
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        image_path = os.path.join(CAPTURED_RAW_DIR, f"{timestamp}.jpg")

        print("Capturing image...")
        os.system(f"libcamera-jpeg -o {image_path} --width 1280 --height 1280 --quality 90 --framerate 30")
        

	# Resize the image for model processing
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (640, 640))  # Resize to a smaller size for faster detection
        
        resized_image_path = image_path.replace(".jpg", "_resized.jpg")
        cv2.imwrite(resized_image_path, resized_image)  # Save resized image

        return image_path, timestamp

	#resizing the image

    except Exception as e:
        print(f"Error capturing image: {e}")
        return None, None

# Function to upload images to Firebase
def upload_image(image_path, image_type, timestamp):
    try:
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')

        firebase_path = f"detections/{timestamp}/{image_type}"
        ref = db.reference(firebase_path)
        ref.set(image_data)
        print(f"Uploaded {image_path} to Firebase under {firebase_path}")
    except Exception as e:
        print(f"Error uploading {image_path}: {e}")

# Function to process captured image
def process_image(raw_image_path, timestamp, cm_per_pixel):
    start_time = time.perf_counter()
    detected_image_path = os.path.join(DETECTED_DIR, f"{timestamp}.jpg")
    pest_name = "None"
    growth_stage = "None"

    try:
        image = cv2.imread(raw_image_path)
        if image is None:
            raise FileNotFoundError(f"ERROR: Image file not found at {raw_image_path}")

        # Resize the image
        resized_image = cv2.resize(image, (1280, 1280))
        inference_start = time.perf_counter()
        results = model.predict(resized_image, conf=0.3)
        output_image = results[0].plot()
        print(f"🧠 YOLO inference took {round(time.perf_counter() - inference_start, 2)} sec")
        output_image = results[0].plot()

        total_leaf_count = 0
        total_leaf_area = 0
        estimated_height = 0

        bbox_start = time.perf_counter()
        for result in results:
            if len(result.boxes) > 0:
                print(f"Detected {len(result.boxes)} bounding boxes")

                for i, (bbox, class_id) in enumerate(zip(result.boxes.xyxy, result.boxes.cls)):
                    class_name = result.names[int(class_id)]
                    bbox = bbox.cpu().numpy().astype(int)
                    x1, y1, x2, y2 = bbox
                    print(f"Detected class: {class_name}, Bounding box: {bbox}")

                    if class_name not in GROWTH_STAGE_CLASSES:
                        pest_name = class_name
                        trigger_pump()  # 🚨 Trigger pump for pest
                        break  # Exit after detecting a pest

                    # Crop image to bounding box for leaf counting
                    cropped = image[y1:y2, x1:x2]
                    crop_path = f"/tmp/crop_{timestamp}_{i}.jpg"
                    cv2.imwrite(crop_path, cropped)

                    # Count leaves inside this bounding box
                    leaf_count, _ = count_leaves(crop_path)
                    total_leaf_count += leaf_count

                    # Estimate height and leaf area for this plant box
                    est_height = estimate_height([x1, y1, x2, y2])
                    estimated_height = max(estimated_height, est_height)
                    leaf_area = estimate_leaf_area([x1, y1, x2, y2], cm_per_pixel)
                    total_leaf_area += leaf_area

        print(f"🌿 Leaf detection and measurements took {round(time.perf_counter() - bbox_start, 2)} sec")
        # Classify growth stage after checking all boxes
        growth_stage = classify_growth(estimated_height, total_leaf_count, total_leaf_area)

        firebase_start = time.perf_counter()
        # Upload parameters to Firebase
        firebase_path = f"detections/{timestamp}/growth_parameters"
        ref = db.reference(firebase_path)
        ref.set({
            "height_cm": estimated_height,
            "leaf_count": leaf_count,
            "leaf_area_cm2": total_leaf_area,
            "growth_stage": growth_stage,
            "pest_detected": pest_name
        })
        print(f"✅ Uploaded to Firebase: {firebase_path}")
        print(f"☁️ Firebase upload took {round(time.perf_counter() - firebase_start, 2)} sec")
        end_time = time.perf_counter()
        print(f"🕒 Total processing time: {round(end_time - start_time, 2)} seconds")

        # Save and upload images
        cv2.imwrite(detected_image_path, output_image)
        upload_image(raw_image_path, "Raw", timestamp)
        upload_image(detected_image_path, "Detected", timestamp)

        return detected_image_path, processed_image_path, pest_name, growth_stage

    except Exception as e:
        print(f"Error processing image: {e}")
        upload_image(raw_image_path, "Raw", timestamp)
        upload_image(detected_image_path, "Detected", timestamp)
        return None, None, pest_name, growth_stage


# Main loop for continuous image capture and processing
while True:
    loop_start = time.perf_counter()
    raw_image_path, timestamp = capture_image()
    if raw_image_path:
        # Upload sensor data at the moment of image capture
        sensor_start = time.perf_counter()
        water_temp = read_ds18b20_temp()
        light = read_light()
        air_temp = dhtDevice.temperature
        humidity = dhtDevice.humidity
        upload_sensor_data_to_firebase(timestamp, water_temp, light, air_temp, humidity)
    print(f"🌡️ Sensor read + upload: {round(time.perf_counter() - sensor_start, 2)} sec")

        # Process image
    cm_per_pixel = 0.2736
    detected_image_path, processed_image_path, pest_name, growth_stage = process_image(
    raw_image_path, timestamp, cm_per_pixel
        )

    print(f"🔁 Loop total duration: {round(time.perf_counter() - loop_start, 2)} seconds\n")

