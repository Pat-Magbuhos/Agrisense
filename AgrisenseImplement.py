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
import socket
import pytz
import schedule

#Internet Diagnostic
def is_connected():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

#Sensors Diagnostic
def diagnostics_check():
    try:
        print("Running diagnostics...")
        temp = read_ds18b20_temp()
        assert -10 < temp < 100, "Invalid water temp reading"
        assert 0 < dhtDevice.temperature < 60, "Invalid air temp"
        assert 0 < dhtDevice.humidity <= 100, "Invalid humidity"
        light_val = read_light()
        assert 0 <= light_val <= 10000, "Invalid light reading"

        import RPi.GPIO as GPIO
        test_pin = 23
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(test_pin, GPIO.OUT)
        GPIO.output(test_pin, GPIO.LOW)
        time.sleep(1)
        GPIO.output(test_pin, GPIO.HIGH)
        GPIO.cleanup()

        print("✅ All sensors and pump working properly!")
    except AssertionError as e:
        print(f"❌ Sensor check failed: {e}")
    except Exception as e:
        print(f"❌ Diagnostics failed: {e}")

#Time Diagnostic
def is_time_synced():
    try:
        result = subprocess.run(["chronyc", "tracking"], stdout=subprocess.PIPE)
        output = result.stdout.decode()
        return "Leap status     : Normal" in output
    except Exception as e:
        print(f"[TIME ERROR] Failed to check time sync: {e}")
        return False

def get_local_time():
    tz = pytz.timezone("Asia/Manila")
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

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

# Improved Leaf Counting using Contours + Threshold Refinement
def count_leaves(image_path):
    # Load image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use bilateral filter to preserve edges
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)

    # Use Otsu's thresholding for better separation
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations to close gaps
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area
    min_leaf_area = 200  # tuned threshold
    leaf_contours = [c for c in contours if cv2.contourArea(c) > min_leaf_area]

    # Draw and save result for visual inspection
    output = image.copy()
    cv2.drawContours(output, leaf_contours, -1, (0, 255, 0), 2)
    processed_image_path = image_path.replace(".jpg", "_contours.jpg")
    cv2.imwrite(processed_image_path, output)

    leaf_count = len(leaf_contours)
    print(f"Refined Leaf Count: {leaf_count} (Saved to {processed_image_path})")
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

        return image_path, timestamp

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
    processed_image_path = None  # Fix for undefined variable
    detected_image_path = os.path.join(DETECTED_DIR, f"{timestamp}.jpg")
    pest_name = "None"
    growth_stage = "None"

    try:
        image = cv2.imread(raw_image_path)
        if image is None:
            raise FileNotFoundError(f"ERROR: Image file not found at {raw_image_path}")

        # Resize the image
        resized_image = cv2.resize(image, (1280, 1280))
        results = model.predict(resized_image, conf=0.3)
        output_image = results[0].plot()

        total_leaf_count = 0
        leaf_data_per_box = []
        total_leaf_area = 0
        estimated_height = 0

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
                        trigger_pump()  #Trigger pump for pest
                        break  #Exit after detecting a pest

                    # Crop image to bounding box for leaf counting
                    cropped = image[y1:y2, x1:x2]

                    # Apply mask to isolate green regions inside the crop
                    hsv_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
                    lower_green = np.array([35, 40, 40])
                    upper_green = np.array([85, 255, 255])
                    mask = cv2.inRange(hsv_crop, lower_green, upper_green)
                    green_only = cv2.bitwise_and(cropped, cropped, mask=mask)

                    crop_path = f"/tmp/crop_{timestamp}_{i}.jpg"
                    cv2.imwrite(crop_path, green_only)

                    # Count leaves inside this masked bounding box
                    leaf_count, _ = count_leaves(crop_path)
                    total_leaf_count += leaf_count

                    # Visualize leaf count on the lower-left of the bounding box with background highlight
                    label = f"Leaves: {leaf_count}"
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    cv2.rectangle(output_image, (x1, y2 - text_height - baseline - 4), (x1 + text_width + 6, y2), (0, 255, 0), -1)
                    cv2.putText(output_image, label, (x1 + 3, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

                    # Store per-box info
                    leaf_data_per_box.append({
                        "bounding_box": [int(x1), int(y1), int(x2), int(y2)],
                        "leaf_count": leaf_count
                    })
                    # Estimate height and leaf area for this plant box
                    est_height = estimate_height([x1, y1, x2, y2])
                    estimated_height = max(estimated_height, est_height)
                    leaf_area = estimate_leaf_area([x1, y1, x2, y2], cm_per_pixel)
                    total_leaf_area += leaf_area

        # Classify growth stage after checking all boxes
        growth_stage = classify_growth(estimated_height, total_leaf_count, total_leaf_area)

        # Upload parameters to Firebase
        firebase_path = f"detections/{timestamp}/growth_parameters"
        ref = db.reference(firebase_path)
        ref.set({
            "height_cm": estimated_height,
            "leaf_count": total_leaf_count,
            "leaf_area_cm2": total_leaf_area,
            "growth_stage": growth_stage,
            "pest_detected": pest_name,
            "num_bounding_boxes": len(leaf_data_per_box),
            "leaf_data_per_box": leaf_data_per_box
        })
        print(f"Uploaded to Firebase: {firebase_path}")

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

OFFLINE_DIR = os.path.join(BASE_DIR, "Offline")
os.makedirs(OFFLINE_DIR, exist_ok=True)

def save_offline(image_path, timestamp):
    new_path = os.path.join(OFFLINE_DIR, f"{timestamp}.jpg")
    os.rename(image_path, new_path)
    print(f"📁 Saved image offline: {new_path}")

def try_upload_offline_images():
    if not is_connected():
        return
    for filename in os.listdir(OFFLINE_DIR):
        if filename.endswith(".jpg"):
            filepath = os.path.join(OFFLINE_DIR, filename)
            timestamp = filename.replace(".jpg", "")
            try:
                upload_image(filepath, "Raw", timestamp)
                os.remove(filepath)
                print(f"☁️ Uploaded and removed: {filename}")
            except Exception as e:
                print(f"⚠️ Failed to upload {filename}: {e}")

def main_capture_loop():
    raw_image_path, timestamp = capture_image()
    if raw_image_path:
        water_temp = read_ds18b20_temp()
        light = read_light()
        air_temp = dhtDevice.temperature
        humidity = dhtDevice.humidity

        if is_connected():
            upload_sensor_data_to_firebase(timestamp, water_temp, light, air_temp, humidity)
            cm_per_pixel = 0.2736
            process_image(raw_image_path, timestamp, cm_per_pixel)
            try_upload_offline_images()
        else:
            save_sensor_data_offline(timestamp, water_temp, light, air_temp, humidity)
            save_offline(raw_image_path, timestamp)

if not is_time_synced():
    print("❌ Time not synced!")
else:
    print(f"✅ Time is synced: {get_local_time()}")

diagnostics_check()


# ✅ Save sensor data offline if no internet
def save_sensor_data_offline(timestamp, water_temp, light, air_temp, humidity):
    data = {
        "height_cm": None,
        "leaf_count": None,
        "leaf_area_cm2": None,
        "growth_stage": "Unknown",
        "pest_detected": "Unknown",
        "water_temp": water_temp,
        "light": light,
        "air_temp": air_temp,
        "humidity": humidity
    }
    log_path = os.path.join(OFFLINE_DIR, f"{timestamp}.json")
    with open(log_path, "w") as f:
        import json
        json.dump(data, f)
    print(f"📄 Saved sensor log offline: {log_path}")

# ✅ Try to reupload offline sensor data and images
def try_upload_offline_data():
    if not is_connected():
        return
    for filename in os.listdir(OFFLINE_DIR):
        if filename.endswith(".jpg"):
            timestamp = filename.replace(".jpg", "")
            image_path = os.path.join(OFFLINE_DIR, filename)
            json_path = os.path.join(OFFLINE_DIR, f"{timestamp}.json")
            try:
                upload_image(image_path, "Raw", timestamp)
                if os.path.exists(json_path):
                    with open(json_path, "r") as f:
                        import json
                        data = json.load(f)
                    firebase_path = f"detections/{timestamp}/growth_parameters"
                    ref = db.reference(firebase_path)
                    ref.set(data)
                    os.remove(json_path)
                os.remove(image_path)
                print(f"☁️ Uploaded offline image and log: {timestamp}")
            except Exception as e:
                print(f"⚠️ Failed to upload offline log/image for {timestamp}: {e}")
schedule.every(10).minutes.do(main_capture_loop)
print("⏳ System ready. Capturing every 10 minutes.")

while True:
    schedule.run_pending()
    time.sleep(1)
