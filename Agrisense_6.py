import os
import base64
import numpy as np
import subprocess
from datetime import datetime
import time
import sys
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, db, storage
from ultralytics import YOLO  # YOLO model for inference
import cv2  # OpenCV for processing
import socket
import pytz
import schedule
from AgrisenseSensors import (
    read_ds18b20_temp,
    read_light,
    dhtDevice,
    upload_sensor_data_to_firebase
)

# Load cm_per_pixel and focal_length from calibration_result.txt
def load_calibration():
    try:
        with open('calibration_result.txt', 'r') as f:
            lines = f.readlines()
            cm_per_pixel = float(lines[0].split('=')[1].strip())
            focal_length = float(lines[1].split('=')[1].strip())
            print(f"✅ Loaded calibration: {cm_per_pixel:.6f} cm/pixel, Focal Length: {focal_length:.2f} pixels")
            return cm_per_pixel, focal_length
    except Exception as e:
        print(f"⚠️ Failed to load calibration: {e}")
        return 0.038666, 800  # fallback default

cm_per_pixel, focal_length = load_calibration()

# Sensor Diagnostic
# Improved diagnostics check
def diagnostics_check():
    try:
        print("Running diagnostics...")
        temp = read_ds18b20_temp()
        if temp is None:
            raise ValueError("Water temperature reading is None")
        assert -10 < temp < 100, "Invalid water temp reading"
        
        air_temp = dhtDevice.temperature
        if air_temp is None:
            raise ValueError("Air temperature reading is None")
        assert 0 < air_temp < 60, "Invalid air temp"
        
        humidity = dhtDevice.humidity
        if humidity is None:
            raise ValueError("Humidity reading is None")
        assert 0 < humidity <= 100, "Invalid humidity"
        
        light_val = read_light()
        if light_val is None:
            raise ValueError("Light intensity reading is None")
        assert 0 <= light_val <= 10000, "Invalid light reading"

        # Test pump GPIO pin
        import RPi.GPIO as GPIO
        test_pin = 23
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(test_pin, GPIO.OUT)
        GPIO.output(test_pin, GPIO.LOW)
        time.sleep(1)
        GPIO.output(test_pin, GPIO.HIGH)
        GPIO.cleanup()

        print("✅ All sensors and pump working properly!")
    except ValueError as e:
        print(f"❌ Sensor reading failed: {e}")
    except AssertionError as e:
        print(f"❌ Sensor check failed: {e}")
    except Exception as e:
        print(f"❌ Diagnostics failed: {e}")

# Internet Connectivity Check
def is_connected():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

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

# Logs to verify the code is working
def log(message):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    full_message = f"{timestamp} {message}"
    print(full_message)
    with open("/home/Agrisense/Thesis/log.txt", "a") as log_file:
        log_file.write(full_message + "\n")

print("Running startup checks...\n")

# Checking internet connection
if is_connected():
    log("✅ Internet connection established.")
else:
    log("❌ No internet connection detected.")

# Checking if Time is synched
if is_time_synced():
    log(f"✅ Time is synced: {get_local_time()}")
else:
    log("❌ Time not synced! Attempting to continue...")

# Run Sensor diagnostics
log("🔍 Running sensor and pump diagnostics...")
diagnostics_check()
log("✅ Startup checks complete.\n")

# Sensor diagnostics
diagnostics_check()

print("\nStartup complete. Preparing image capture loop...")

log("===== Booting Agrisense System... =====")

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

# Initialize Firebase Admin SDK only if it's not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred, {
        'databaseURL': FIREBASE_DB_URL,  # Initialize Realtime Database URL
        'storageBucket': 'agrisense-6a089.firebasestorage.app'  # Firebase Storage bucket URL
    })
else:
    print("Firebase has already been initialized.")

# Initialize Firebase Storage explicitly with the bucket name
bucket = firebase_admin.storage.bucket('agrisense-6a089.firebasestorage.app')

print("Firebase Storage initialized successfully.")

# Ensure directory structure exists
BASE_DIR = "/home/Agrisense/Thesis"
CAPTURED_RAW_DIR = os.path.join(BASE_DIR, "Captured", "Raw")
CAPTURED_RETRIEVED_DIR = os.path.join(BASE_DIR, "Captured", "Retrieved")
DETECTED_DIR = os.path.join(BASE_DIR, "Detected", "Detected")
DETECTED_RETRIEVED_DIR = os.path.join(BASE_DIR, "Detected", "Retrieved")
OFFLINE_DIR = os.path.join(BASE_DIR, "Offline")
os.makedirs(OFFLINE_DIR, exist_ok=True)

for directory in [CAPTURED_RAW_DIR, CAPTURED_RETRIEVED_DIR, DETECTED_DIR, DETECTED_RETRIEVED_DIR]:
    os.makedirs(directory, exist_ok=True)

# Load trained model
plant_model = YOLO("/home/Agrisense/Thesis/bestv2.pt")  # Plant Detector
leaf_model = YOLO("/home/Agrisense/Thesis/leaf_detector_final.pt")  # Leaf Detector

# Trigonometry Constants
CAMERA_ANGLE = 60  # Degrees
CAMERA_HEIGHT = 43  # cm (Height from the ground)

# Growth Stage Thresholds (Adjustable)
GROWTH_THRESHOLDS = {
    "seedling": {"height": 5, "leaves": 3, "leaf_area": 10},
    "vegetative": {"height": 15, "leaves": 8, "leaf_area": 100},
    "mature": {"height": 30, "leaves": 12, "leaf_area": 300},
}


# Function to estimate height using trigonometry
def estimate_height(bbox):
    pixel_height = bbox[3] - bbox[1]
    real_height = (CAMERA_HEIGHT * pixel_height) / focal_length
    real_height /= np.tan(np.radians(CAMERA_ANGLE))
    return round(real_height, 2)

# Function to estimate the leaf area
def estimate_largest_leaf_area(leaf_bboxes, cm_per_pixel):
    """
    Detect the largest leaf area from the detected leaf bounding boxes.

    Arguments:
    leaf_bboxes -- List of bounding boxes for each detected leaf.
    cm_per_pixel -- Calibration factor (cm per pixel) for conversion.

    Returns:
    largest_leaf_area_cm2 -- The area of the largest detected leaf in cm².
    """
    if not leaf_bboxes:
        return 0.0  # No leaves found, return 0 area
    
    # List to store the area of each detected leaf in pixels
    leaf_areas = []

    for bbox in leaf_bboxes:
        x1, y1, x2, y2 = bbox
        pixel_width = x2 - x1  # Width of the leaf in pixels
        pixel_height = y2 - y1  # Height of the leaf in pixels
        pixel_area = pixel_width * pixel_height  # Area in pixels (px²)

        # Append the area of this leaf to the list
        leaf_areas.append(pixel_area)
    
    # Find the largest leaf area in pixels
    largest_pixel_area = max(leaf_areas)
    
    # Convert pixel area to cm² using the cm_per_pixel calibration factor
    pixel_area_to_cm2 = cm_per_pixel ** 2  # Area conversion factor from pixels to cm²
    largest_leaf_area_cm2 = largest_pixel_area * pixel_area_to_cm2
    
    # Return the largest leaf area in cm², rounded to two decimal places
    return round(largest_leaf_area_cm2, 2)


# Function to classify growth stage
def classify_growth(height_cm, leaf_count, leaf_area_cm2):
    thresholds = GROWTH_THRESHOLDS
    scores = {"seedling": 0, "vegetative": 0, "mature": 0}

    # Height
    if height_cm >= thresholds["mature"]["height"]:
        scores["mature"] += 1
    elif height_cm >= thresholds["vegetative"]["height"]:
        scores["vegetative"] += 1
    elif height_cm >= thresholds["seedling"]["height"]:
        scores["seedling"] += 1

    # Leaves
    if leaf_count >= thresholds["mature"]["leaves"]:
        scores["mature"] += 1
    elif leaf_count >= thresholds["vegetative"]["leaves"]:
        scores["vegetative"] += 1
    elif leaf_count >= thresholds["seedling"]["leaves"]:
        scores["seedling"] += 1

    # Leaf Area
    if leaf_area_cm2 >= thresholds["mature"]["leaf_area"]:
        scores["mature"] += 1
    elif leaf_area_cm2 >= thresholds["vegetative"]["leaf_area"]:
        scores["vegetative"] += 1
    elif leaf_area_cm2 >= thresholds["seedling"]["leaf_area"]:
        scores["seedling"] += 1

    # Final Decision
    if scores["mature"] >= 2:
        return "Mature"
    elif scores["vegetative"] >= 2:
        return "Vegetative"
    else:
        return "Seedling"

# Function to trigger pump    
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

# Function to upload sensor data to Firebase and handle error in sensors reading
def upload_sensor_data_to_firebase(timestamp, water_temp, light, air_temp, humidity):
    try:
        # Check if water_temp, air_temp, etc. are not None before calling round()
        if water_temp is not None:
            water_temp_c = round(water_temp, 2)
        else:
            water_temp_c = None

        if air_temp is not None:
            air_temp_c = round(air_temp, 2)
        else:
            air_temp_c = None

        if humidity is not None:
            humidity_percent = round(humidity, 2)
        else:
            humidity_percent = None

        if light is not None:
            light_intensity = round(light, 2)
        else:
            light_intensity = None

        # Save sensor data to Firebase
        sensor_data = {
            "timestamp": timestamp,
            "water_temperature_c": water_temp_c,
            "air_temperature_c": air_temp_c,
            "humidity_percent": humidity_percent,
            "light_intensity": light_intensity,
        }

        firebase_path = f"detections/{timestamp}/sensor_data"
        ref = db.reference(firebase_path)
        ref.set(sensor_data)
        print(f"Uploaded sensor data to Firebase for timestamp: {timestamp}")

    except Exception as e:
        print(f"Error uploading sensor data: {e}")


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

# Function to upload images to Firebase Storage
def upload_image(image_path, image_type, timestamp):
    try:
        # Get the Firebase storage bucket
        bucket = firebase_admin.storage.bucket('agrisense-6a089.firebasestorage.app')

        # Create a blob object with the desired path and filename
        blob = bucket.blob(f'detections/{timestamp}/{image_type}.jpg')

        # Upload image file to Firebase Storage
        blob.upload_from_filename(image_path)

        print(f"Uploaded {image_path} to Firebase Storage under {blob.name}")
    except Exception as e:
        print(f"Error uploading {image_path} to Firebase Storage: {e}")

# Upload growth parameters as JSON to Firebase Storage
def upload_json(data, data_type, timestamp):
    try:
        # Convert data to JSON string
        import json
        json_data = json.dumps(data)

        # Get the Firebase storage bucket
        bucket = firebase_admin.storage.bucket('agrisense-6a089.firebasestorage.app')

        # Define the blob object with the desired path and filename (storing JSON as a file)
        blob = bucket.blob(f'detections/{timestamp}/{data_type}.json')

        # Upload JSON string to Firebase Storage
        blob.upload_from_string(json_data, content_type='application/json')

        print(f"Uploaded {data_type} JSON data to Firebase Storage under {blob.name}")
    except Exception as e:
        print(f"Error uploading {data_type} JSON data: {e}")

# Save offline if no internet
def save_offline(image_path, timestamp):
    new_path = os.path.join(OFFLINE_DIR, f"{timestamp}.jpg")
    os.rename(image_path, new_path)
    print(f"📁 Saved image offline: {new_path}")

# Try uploading offline data
def try_upload_offline_data():
    if not is_connected():
        return

    for filename in os.listdir(OFFLINE_DIR):
        if filename.endswith(".jpg") and "_detected" not in filename and "_error" not in filename:
            timestamp = filename.replace(".jpg", "")
            image_path = os.path.join(OFFLINE_DIR, filename)
            detected_path = os.path.join(OFFLINE_DIR, f"{timestamp}_detected.jpg")
            error_detected_path = os.path.join(OFFLINE_DIR, f"{timestamp}_detected_error.jpg")
            growth_path = os.path.join(OFFLINE_DIR, f"{timestamp}_growth.json")
            env_path = os.path.join(OFFLINE_DIR, f"{timestamp}_env.json")

            try:
                # Upload raw image
                upload_image(image_path, "Raw", timestamp)

                # Upload detected image (or error version)
                if os.path.exists(detected_path):
                    upload_image(detected_path, "Detected", timestamp)
                    os.remove(detected_path)
                elif os.path.exists(error_detected_path):
                    upload_image(error_detected_path, "Detected", timestamp)
                    os.remove(error_detected_path)

                # Upload growth_parameters
                if os.path.exists(growth_path):
                    with open(growth_path, "r") as f:
                        import json
                        growth_data = json.load(f)
                    ref = db.reference(f"detections/{timestamp}/growth_parameters")
                    ref.set(growth_data)
                    os.remove(growth_path)
                    log(f"☁️ Uploaded growth_parameters for: {timestamp}")

                # Upload environment_data
                if os.path.exists(env_path):
                    with open(env_path, "r") as f:
                        import json
                        env_data = json.load(f)
                    ref = db.reference(f"detections/{timestamp}/environment_data")
                    ref.set(env_data)
                    os.remove(env_path)
                    log(f"☁️ Uploaded environment_data for: {timestamp}")

                # Clean up raw image
                os.remove(image_path)
                log(f"✅ Synced offline data for: {timestamp}")

            except Exception as e:
                log(f"⚠️ Failed to upload offline data for {timestamp}: {e}")

def process_image(raw_image_path, timestamp, cm_per_pixel):
    detected_image_path = os.path.join('Detected', f"{timestamp}.jpg")
    pest_name = "None"
    growth_stage = "None"
    
    # Attempt to read sensors twice
    water_temp, air_temp, humidity, light = None, None, None, None
    for attempt in range(2):
        try:
            water_temp = read_ds18b20_temp()  # Get water temperature from the sensor
            air_temp = dhtDevice.temperature  # Get air temperature from the DHT sensor
            humidity = dhtDevice.humidity     # Get humidity from the DHT sensor
            light = read_light()              # Get light intensity from the sensor
            break
        except RuntimeError as e:
            log(f"Sensor read failed (attempt {attempt + 1}): {e}")
            time.sleep(2)
            if attempt == 1:
                # Set sensor data to None if still failing
                water_temp = None
                air_temp = None
                humidity = None
                light = None

    try:
        image = cv2.imread(raw_image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {raw_image_path}")

        resized_image = cv2.resize(image, (1280, 1280))
        plant_results = plant_model.predict(resized_image, conf=0.3)[0]
        leaf_results = leaf_model.predict(resized_image, conf=0.3)[0]

        output_image = resized_image.copy()
        plant_boxes = plant_results.boxes.xyxy.cpu().numpy().astype(int) if plant_results.boxes else []
        leaf_boxes = leaf_results.boxes.xyxy.cpu().numpy().astype(int) if leaf_results.boxes else []

        total_leaf_count = 0
        total_leaf_area = 0
        estimated_height = 0
        leaf_data_per_box = []

        assigned_leaves = set()

        def is_inside(plant_bbox, leaf_bbox):
            x1p, y1p, x2p, y2p = plant_bbox
            x1l, y1l, x2l, y2l = leaf_bbox
            center_x = (x1l + x2l) / 2
            center_y = (y1l + y2l) / 2
            return x1p <= center_x <= x2p and y1p <= center_y <= y2p

        # Step 1: Analyze plant detections
        for i, plant_bbox in enumerate(plant_boxes):
            x1p, y1p, x2p, y2p = plant_bbox

            leaves_in_plant = []
            for j, leaf_bbox in enumerate(leaf_boxes):
                if j in assigned_leaves:
                    continue
                if is_inside(plant_bbox, leaf_bbox):
                    leaves_in_plant.append(leaf_bbox)
                    assigned_leaves.add(j)

            leaf_count = len(leaves_in_plant)
            total_leaf_count += leaf_count

            est_height = estimate_height(plant_bbox)
            estimated_height = max(estimated_height, est_height)

            # Calculate the largest leaf area for each plant (using the bounding box)
            largest_leaf_area = estimate_largest_leaf_area(resized_image[y1p:y2p, x1p:x2p], cm_per_pixel)
            total_leaf_area += largest_leaf_area

            # Classify growth stage based on the estimated metrics
            growth_stage = classify_growth(est_height, leaf_count, largest_leaf_area)

            # Annotate output image
            label = f"{growth_stage} | Leaves: {leaf_count} | Largest Leaf Area: {largest_leaf_area} cm²"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(output_image, (x1p, y2p - th - baseline - 4), (x1p + tw + 6, y2p), (0, 255, 0), -1)
            cv2.putText(output_image, label, (x1p + 3, y2p - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.rectangle(output_image, (x1p, y1p), (x2p, y2p), (0, 255, 0), 2)

            leaf_data_per_box.append({
                "bounding_box": [int(x1p), int(y1p), int(x2p), int(y2p)],
                "leaf_count": leaf_count,
                "largest_leaf_area": largest_leaf_area  # Add largest leaf area to the data
            })

        # Handle unassigned leaves (manually detected plants)
        unassigned_leaves = [leaf_boxes[i] for i in range(len(leaf_boxes)) if i not in assigned_leaves]

        if unassigned_leaves:
            x1 = min([box[0] for box in unassigned_leaves])
            y1 = min([box[1] for box in unassigned_leaves])
            x2 = max([box[2] for box in unassigned_leaves])
            y2 = max([box[3] for box in unassigned_leaves])

            leaf_count = len(unassigned_leaves)
            total_leaf_count += leaf_count

            est_height = estimate_height([x1, y1, x2, y2])
            estimated_height = max(estimated_height, est_height)

            # Calculate the largest leaf area for unassigned leaves (using the bounding box)
            largest_leaf_area = estimate_largest_leaf_area(resized_image[y1:y2, x1:x2], cm_per_pixel)
            total_leaf_area += largest_leaf_area

            growth_stage = classify_growth(est_height, leaf_count, largest_leaf_area)

            # Annotate ungrouped leaves
            label = f"{growth_stage} | Leaves: {leaf_count} | Largest Leaf Area: {largest_leaf_area} cm²"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(output_image, (x1, y2 - th - baseline - 4), (x1 + tw + 6, y2), (0, 0, 255), -1)
            cv2.putText(output_image, label, (x1 + 3, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            leaf_data_per_box.append({
                "bounding_box": [int(x1), int(y1), int(x2), int(y2)],
                "leaf_count": leaf_count,
                "largest_leaf_area": largest_leaf_area  # Add largest leaf area to the data
            })

        # Upload growth parameters to Firebase (Include sensor data)
        growth_data = {
            "height_cm": estimated_height,
            "leaf_count": total_leaf_count,
            "leaf_area_cm2": total_leaf_area,
            "growth_stage": growth_stage,
            "pest_detected": pest_name,
            "num_bounding_boxes": len(leaf_data_per_box),
            "leaf_data_per_box": leaf_data_per_box,
            "water_temp": water_temp,
            "air_temp": air_temp,
            "humidity": humidity,
            "light": light
        }

        upload_json(growth_data, "growth_parameters", timestamp)
        print(f"✅ Uploaded growth parameters to Firebase Storage.")

        # Upload images to Firebase
        upload_image(raw_image_path, "Raw", timestamp)
        temp_detected_path = f"/tmp/{timestamp}_detected.jpg"
        cv2.imwrite(temp_detected_path, output_image)
        upload_image(temp_detected_path, "Detected", timestamp)
        os.remove(temp_detected_path)

        # Trigger pump if pest detected
        if pest_name != "None":
            trigger_pump()

        return None, None, pest_name, growth_stage

    except Exception as e:
        print(f"Error processing image: {e}")
        # Save offline if error
        save_offline(raw_image_path, timestamp)
        return None, None, pest_name, growth_stage

# ✅ Save sensor data offline if no internet
def save_sensor_data_offline(timestamp, water_temp, light, air_temp, humidity):
    env_data = {
        "water_temp": water_temp,
        "light": light,
        "air_temp": air_temp,
        "humidity": humidity
    }
    env_log_path = os.path.join(OFFLINE_DIR, f"{timestamp}_env.json")
    with open(env_log_path, "w") as f:
        import json
        json.dump(env_data, f)
    print(f"📄 Saved sensor log offline: {env_log_path}")
    log(f"✅ Saved environment_data for {timestamp} offline")

# Remove default merged save if exists
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


def main_capture_loop():
    raw_image_path, timestamp = capture_image()
    if raw_image_path:
        # Retry sensor readings up to 2 times
        for attempt in range(2):
            try:
                water_temp = read_ds18b20_temp()
                light = read_light()
                air_temp = dhtDevice.temperature
                humidity = dhtDevice.humidity
                break
            except RuntimeError as e:
                log(f"Sensor read failed (attempt {attempt + 1}): {e}")
                time.sleep(2)
                if attempt == 1:
                    water_temp = None
                    light = None
                    air_temp = None
                    humidity = None

        cm_per_pixel = 0.2736  # Always defined regardless of connection

        # Run image through model either way
        process_image(raw_image_path, timestamp, cm_per_pixel)

        if is_connected():
            # Upload sensor data to Firebase (this handles the rounding and checks for None)
            upload_sensor_data_to_firebase(timestamp, water_temp, light, air_temp, humidity)
            try_upload_offline_data()
        else:
            save_sensor_data_offline(timestamp, water_temp, light, air_temp, humidity)


# Scheduling image capture every 10 minutes
schedule.every(1).minutes.do(main_capture_loop)
print("⏳ System ready. Capturing every 10 minutes.")

while True:
    schedule.run_pending()
    time.sleep(1)
