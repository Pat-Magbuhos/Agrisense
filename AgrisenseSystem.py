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
import serial
import cv2  # OpenCV for processing
import socket
import pytz
import schedule
import json
import tempfile


# Import Sensor reading functions
from AgrisenseSensors import (
    read_ds18b20_temp,
    dhtDevice,
    upload_sensor_data_to_firebase
)

# System Diagnostics Functions (Internet/Date&Time/Sensors)
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

# Local Time Diagnostic
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

# Internet Connection & Time Sync Logic
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

print("\nStartup complete. Preparing image capture loop...")

log("===== Booting Agrisense System... =====")

# Load environment variables from .env to access Firebase credentials
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

# Load the YOLO models (Update this with recently trained model)
plant_model = YOLO("/home/Agrisense/Thesis/PlantModel.pt") # Plant Detector
pest_model = YOLO("/home/Agrisense/Thesis/PestModel.pt")  # Pest Detector
leaf_model = YOLO("/home/Agrisense/Thesis/LeafModel.pt") # Leaf Detector

# Load Calibration Values for Object Detection
try:
    with open('calibration_result.txt', 'r') as f:
        lines = f.readlines()
        cm_per_pixel = float(lines[0].split('=')[1].strip())
        focal_length = float(lines[1].split('=')[1].strip())
    print(f"✅ Loaded calibration: {cm_per_pixel:.6f} cm/pixel, Focal Length: {focal_length:.2f} pixels")
except Exception as e:
    print(f"Failed to load calibration: {e}")
    cm_per_pixel = 0.038666  
    focal_length = 800        

# Growth Stage Thresholds for maturity prediction
GROWTH_THRESHOLDS = {
    "seedling": {"height": 5, "leaves": 3, "leaf_area": 10},
    "vegetative": {"height": 15, "leaves": 8, "leaf_area": 100},
    "mature": {"height": 30, "leaves": 12, "leaf_area": 300},
}

# Estimate Largest Leaf Area (1st Classifier)
def estimate_largest_leaf_area(leaf_boxes):
    if not leaf_boxes:
        return 0.0
    areas = []
    for bbox in leaf_boxes:
        x1, y1, x2, y2 = bbox
        pixel_area = (x2 - x1) * (y2 - y1)
        areas.append(pixel_area)
    largest_pixel_area = max(areas)
    cm2_area = (cm_per_pixel ** 2) * largest_pixel_area
    return round(cm2_area, 2)

# Classify Growth Stage (Needs to be updated removing height as a classifier)
def classify_growth(leaf_count, leaf_area_cm2, days_since_transplant):
    """Classify growth stage based on leaf metrics and days since transplant"""
    if days_since_transplant < 7:
        return "Seedling"
    elif days_since_transplant < 21:
        return "Vegetative"
    else:
        # For mature stage, consider leaf metrics
        if leaf_count >= 8 and leaf_area_cm2 >= 100:
            return "Mature"
        else:
            return "Vegetative"

def classify_growth_with_age_hydroponics(leaf_count, leaf_area_cm2, days_old):
    # Hydroponic stage thresholds by age
    if days_old >= 20:
        return "Mature"
    elif days_old >= 7:
        growth_stage_by_measurement = classify_growth(leaf_count, leaf_area_cm2, days_old)
        if growth_stage_by_measurement == "Seedling":
            return "Vegetative"
        else:
            return growth_stage_by_measurement
    else:
        return classify_growth(leaf_count, leaf_area_cm2, days_old)


# Check all leaf inside bounding box from Plant Model (2nd Classifier)
def is_inside(outer_bbox, inner_bbox):
    x1, y1, x2, y2 = outer_bbox
    cx = (inner_bbox[0] + inner_bbox[2]) / 2
    cy = (inner_bbox[1] + inner_bbox[3]) / 2
    return x1 <= cx <= x2 and y1 <= cy <= y2

# Calculate days since transplanting (3rd Classifier)
def calculate_days_since_transplant(transplant_date_str, current_timestamp_str):
    """Calculate days since transplanting given transplant date and current detection timestamp."""
    transplant_date = datetime.strptime(transplant_date_str, "%Y-%m-%d")
    current_date = datetime.strptime(current_timestamp_str.split('_')[0], "%Y-%m-%d")
    delta = current_date - transplant_date
    return max(delta.days, 0)

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
def upload_sensor_data_to_firebase(timestamp, water_temp, air_temp, humidity):

    try:
        water_temp_c = round(water_temp, 2) if water_temp is not None else None
        air_temp_c = round(air_temp, 2) if air_temp is not None else None
        humidity_percent = round(humidity, 2) if humidity is not None else None

        sensor_data = {
            "timestamp": timestamp,
            "water_temperature_c": water_temp_c,
            "air_temperature_c": air_temp_c,
            "humidity_percent": humidity_percent,
        }

        firebase_path = f"detections/{timestamp}/sensor_data"
        upload_json(sensor_data, "environment_data", timestamp)
        print(f"📤 Uploaded sensor data (with pH) to Firebase for timestamp: {timestamp}")

    except Exception as e:
        print(f"Error uploading sensor data: {e}")


# Function to capture image
def capture_image():
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        image_path = os.path.join(CAPTURED_RAW_DIR, f"{timestamp}.jpg")

        print("Capturing image...")
        os.system(f"libcamera-jpeg -o {image_path} --width 3280 --height 2464 --quality 90 --framerate 15 --shutter 5000 --awb auto")

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
# Upload growth parameters as JSON to Firebase Storage
def upload_json(data, data_type, timestamp):
    try:
        # Convert data to JSON string
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

# Function to upload offline data
def try_upload_offline_data():
    """Attempt to upload all offline stored data when internet connection is restored"""
    if not is_connected():
        return

    log("Attempting to upload offline data...")
    
    # Process error images first
    for filename in os.listdir(OFFLINE_DIR):
        if filename.endswith("_error.jpg"):
            timestamp = filename.replace("_error.jpg", "")
            error_path = os.path.join(OFFLINE_DIR, filename)
            try:
                upload_image(error_path, "Error", timestamp)
                os.remove(error_path)
                log(f"Uploaded error image for {timestamp}")
            except Exception as e:
                log(f"Failed to upload error image {filename}: {e}")

    # Process regular data files
    for filename in os.listdir(OFFLINE_DIR):
        if filename.endswith(".jpg") and not filename.endswith(("_detected.jpg", "_error.jpg")):
            timestamp = filename.replace(".jpg", "")
            raw_path = os.path.join(OFFLINE_DIR, filename)
            detected_path = os.path.join(OFFLINE_DIR, f"{timestamp}_detected.jpg")
            error_detected_path = os.path.join(OFFLINE_DIR, f"{timestamp}_detected_error.jpg")
            growth_path = os.path.join(OFFLINE_DIR, f"{timestamp}_growth.json")
            env_path = os.path.join(OFFLINE_DIR, f"{timestamp}_env.json")

            try:
                # Upload raw image
                upload_image(raw_path, "Raw", timestamp)

                # Upload detected image if exists
                detected_uploaded = False
                if os.path.exists(detected_path):
                    upload_image(detected_path, "Detected", timestamp)
                    os.remove(detected_path)
                    detected_uploaded = True
                elif os.path.exists(error_detected_path):
                    upload_image(error_detected_path, "Detected", timestamp)
                    os.remove(error_detected_path)
                    detected_uploaded = True

                # Upload growth data if exists
                if os.path.exists(growth_path):
                    with open(growth_path, "r") as f:
                        growth_data = json.load(f)
                    upload_json(growth_data, "growth_parameters", timestamp)
                    os.remove(growth_path)
                    log(f"Uploaded growth parameters for {timestamp}")

                # Upload environment data if exists
                if os.path.exists(env_path):
                    with open(env_path, "r") as f:
                        env_data = json.load(f)
                    upload_json(env_data, "environment_data", timestamp)
                    os.remove(env_path)
                    log(f"Uploaded environment data for {timestamp}")

                # Only remove raw image if everything succeeded
                if detected_uploaded or (not os.path.exists(detected_path) and not os.path.exists(error_detected_path)):
                    os.remove(raw_path)
                    log(f"Successfully synced all data for {timestamp}")
                else:
                    log(f"⚠️ Detected image missing for {timestamp}, keeping raw image")

            except Exception as e:
                log(f"⚠️ Failed to upload offline data for {timestamp}: {e}")
                # Don't delete files if upload failed

def get_detection_timestamps():
    """Get list of timestamps from Firebase Storage detection folders"""
    try:
        bucket = firebase_admin.storage.bucket('agrisense-6a089.firebasestorage.app')
        blobs = bucket.list_blobs(prefix='detections/')
        
        # Extract unique timestamps from existing detection folders
        timestamps = set()
        for blob in blobs:
            # Parse timestamp from paths like 'detections/2024-03-01_10-00-00/...'
            parts = blob.name.split('/')
            if len(parts) > 1:
                potential_timestamp = parts[1]
                if '_' in potential_timestamp and '-' in potential_timestamp:
                    timestamps.add(potential_timestamp)
        
        return sorted(list(timestamps))
    except Exception as e:
        print(f"Error getting detection timestamps: {e}")
        return []

def log_to_firebase(message, event_type, details=None):
    """Log events directly to Firebase Realtime Database"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_entry = {
            "timestamp": timestamp,
            "message": message,
            "event_type": event_type,
            "details": details or {}
        }
        db.reference(f'logs/{timestamp.replace("-", "_")}').set(log_entry)
    except Exception as e:
        print(f"Error logging to Firebase: {e}")

def verify_detection_upload(timestamp, max_retries=3, retry_delay=2):
    """Verify that all required files for a detection are uploaded with retries"""
    for attempt in range(max_retries):
        try:
            bucket = firebase_admin.storage.bucket('agrisense-6a089.firebasestorage.app')
            prefix = f'detections/{timestamp}/'
            blobs = list(bucket.list_blobs(prefix=prefix))
            
            # Check for required files
            required_files = {
                f"{prefix}Raw.jpg": False,
                f"{prefix}Detected.jpg": False,
                f"{prefix}growth_parameters.json": False,
                f"{prefix}environment_data.json": False
            }
            
            for blob in blobs:
                if blob.name in required_files:
                    required_files[blob.name] = True
            
            # If all files are present, return True immediately
            if all(required_files.values()):
                print(f"✅ All files verified for {timestamp}")
                return True
                
            # If not all files are present and we have retries left
            if attempt < max_retries - 1:
                missing_files = [file.split('/')[-1] for file, exists in required_files.items() if not exists]
                print(f"Attempt {attempt + 1}/{max_retries}: Waiting for files: {', '.join(missing_files)}")
                time.sleep(retry_delay)
                continue
            
            # On last attempt, print missing files
            missing_files = [file for file, exists in required_files.items() if not exists]
            if missing_files:
                print(f"Missing files for {timestamp} after {max_retries} attempts:")
                for file in missing_files:
                    print(f"  - {file.split('/')[-1]}")
            
            return False
            
        except Exception as e:
            print(f"Error verifying detection upload (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return False
    
    return False

def update_detection_index(new_timestamp):
    """Update detection_index.json with a new timestamp"""
    try:
        # First verify that all detection files are uploaded
        if not verify_detection_upload(new_timestamp):
            print(f"Warning: Detection {new_timestamp} not fully uploaded yet")
            return
            
        bucket = firebase_admin.storage.bucket('agrisense-6a089.firebasestorage.app')
        blob = bucket.blob('detections/detection_index.json')
        
        # Get current timestamps as direct array
        if blob.exists():
            current_timestamps = json.loads(blob.download_as_string())
            if isinstance(current_timestamps, dict) and 'timestamps' in current_timestamps:
                # Convert old format to new format if necessary
                current_timestamps = current_timestamps['timestamps']
        else:
            current_timestamps = []
            
        # Add new timestamp if not already present
        if new_timestamp not in current_timestamps:
            current_timestamps.append(new_timestamp)
            current_timestamps.sort()
            
            # Upload updated array directly
            json_str = json.dumps(current_timestamps, indent=2)
            blob.upload_from_string(json_str, content_type='application/json')
            print(f"✅ Updated detection index with timestamp: {new_timestamp}")
            
            # Verify the update
            updated_blob = bucket.blob('detections/detection_index.json')
            if updated_blob.exists():
                updated_content = json.loads(updated_blob.download_as_string())
                if new_timestamp in updated_content:
                    print(f"✅ Verified index update for {new_timestamp}")
                else:
                    print(f"⚠️ Index update verification failed for {new_timestamp}")
            
    except Exception as e:
        print(f"Error updating detection index: {e}")
        print(f"Details: {str(e)}")

def append_to_firebase_log(new_log_entries):
    """Append new log entries to the existing log file in Firebase"""
    try:
        bucket = firebase_admin.storage.bucket('agrisense-6a089.firebasestorage.app')
        log_blob = bucket.blob('logs/pest_detection_log.txt')
        
        # Get the last modified time of local log file
        local_log_mtime = os.path.getmtime(pest_tracker.log_file)
        
        # Read local log file
        with open(pest_tracker.log_file, 'r') as f:
            local_lines = f.readlines()
            
        if log_blob.exists():
            # Download existing log content
            existing_log = log_blob.download_as_string().decode('utf-8').splitlines()
            
            # Find new entries by comparing timestamps
            new_entries = []
            for line in local_lines:
                # Parse timestamp from log line
                try:
                    log_time = datetime.strptime(line[1:20], "[%Y-%m-%d %H:%M:%S]")
                    if log_time.timestamp() > local_log_mtime:
                        new_entries.append(line)
                except:
                    continue
                    
            # Append new entries to existing log
            updated_log = existing_log + new_entries
        else:
            # If no existing log, use all local entries
            updated_log = local_lines
            
        # Upload updated log
        log_blob.upload_from_string('\n'.join(updated_log), content_type='text/plain')
        print("✅ Appended new entries to Firebase log")
        
    except Exception as e:
        print(f"Error appending to Firebase log: {e}")

# Pest Detection Functions and Pump Triggering
# Class mapping for pest detection (Update this with the mapping of pest model)
PEST_CLASSES = {
    0: "Pest Type 1",
    1: "Pest Type 2",
    2: "Pest Type 3",
    # Add more pest types based on your pest model classes
}

class PestTracker:
    def __init__(self):
        self.pest_history = {}  # Format: {plant_id: {'bbox': bbox, 'pest_type': type, 'last_treatment': timestamp, 'needs_inspection': bool}}
        self.log_file = "/home/Agrisense/Thesis/log.txt"
        self.offline_log_dir = "/home/Agrisense/Thesis/Offline/logs"
        self.SPRAY_COOLDOWN = 72 * 3600  # 72 hours in seconds
        self.INSPECTION_THRESHOLD = 24 * 3600  # 24 hours in seconds
        os.makedirs(self.offline_log_dir, exist_ok=True)
        self.load_history_from_log()

    def load_history_from_log(self):
        """Load pest detection history from log file"""
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if "Pump triggered for" in line:
                        timestamp_str = line[1:20]  # Format: [YYYY-MM-DD HH:MM:SS]
                        timestamp = datetime.strptime(timestamp_str, "[%Y-%m-%d %H:%M:%S]")
                        if "at plant location" in line:
                            pest_type = line.split("Pump triggered for")[1].split("at plant location")[0].strip()
                            plant_id = line.split("at plant location")[1].strip()

                            self.pest_history[plant_id] = {
                                'pest_type': pest_type,
                                'last_treatment': timestamp,
                                'bbox': None,  # Historical entries won't have bbox
                                'needs_inspection': False,
                                'persistent_detection': False
                            }
        except FileNotFoundError:
            print("No existing log file found. Starting fresh.")
        except Exception as e:
            print(f"Error loading pest history from log: {e}")

    def check_persistent_pests(self, current_time):
        """Check for persistent pest issues that need manual inspection"""
        for plant_id, data in self.pest_history.items():
            if data['last_treatment']:
                time_since_treatment = (current_time - data['last_treatment']).total_seconds()

                if time_since_treatment < self.INSPECTION_THRESHOLD and data.get('persistent_detection', False):
                    data['needs_inspection'] = True
                    log(f"⚠️ Persistent pest detection at location {plant_id}. Manual inspection required!")

    def should_trigger_pump(self, bbox, pest_type, current_time):
        """Determine if pump should be triggered for this pest detection"""
        plant_id = self.get_plant_id(bbox)

        if plant_id in self.pest_history:
            last_record = self.pest_history[plant_id]
            time_since_last = (current_time - last_record['last_treatment']).total_seconds()

            if time_since_last < self.SPRAY_COOLDOWN:
                if pest_type == last_record['pest_type']:
                    self.pest_history[plant_id]['persistent_detection'] = True
                    if time_since_last < self.INSPECTION_THRESHOLD:
                        self.pest_history[plant_id]['needs_inspection'] = True
                        log(f"⚠️ Pest still detected after treatment at location {plant_id}")
                return False

            self.pest_history[plant_id] = {
                'bbox': bbox,
                'pest_type': pest_type,
                'last_treatment': current_time,
                'needs_inspection': False,
                'persistent_detection': False
            }
            log(f"🕒 New treatment cycle started for {pest_type} at location {plant_id}")
            return True

        self.pest_history[plant_id] = {
            'bbox': bbox,
            'pest_type': pest_type,
            'last_treatment': current_time,
            'needs_inspection': False,
            'persistent_detection': False
        }
        log(f"🆕 First treatment for {pest_type} at location {plant_id}")
        return True

    def get_plant_id(self, bbox):
        """Generate a unique ID for a plant based on its bounding box center"""
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        return f"{int(center_x)}_{int(center_y)}"

    def is_same_location(self, bbox1, bbox2, threshold=50):
        """Check if two bounding boxes are in approximately the same location"""
        if bbox1 is None or bbox2 is None:
            return False
        center1_x = (bbox1[0] + bbox1[2]) / 2
        center1_y = (bbox1[1] + bbox1[3]) / 2
        center2_x = (bbox2[0] + bbox2[2]) / 2
        center2_y = (bbox2[1] + bbox2[3]) / 2

        distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
        return distance < threshold

# Initialize PestTracker
pest_tracker = PestTracker()

# New pest detection function
def detect_pests(image, timestamp):
    """Detect pests in the image using pest_model and return pest detections."""
    pest_results = pest_model.predict(image, conf=0.3)[0]
    pest_boxes = pest_results.boxes.xyxy.cpu().numpy().astype(int) if pest_results.boxes else []
    pest_classes = pest_results.boxes.cls.cpu().numpy().astype(int) if pest_results.boxes else []

    pest_detections = []

    # Detect and store pests
    for pest_box, pest_cls in zip(pest_boxes, pest_classes):
        pest_type = PEST_CLASSES.get(int(pest_cls), "Unknown")
        pest_detections.append({
            "pest_type": pest_type,
            "bbox": pest_box
        })

    return pest_detections

# Modify other sensor readings similarly


# Function to handle pest detections and trigger actions (like spraying or inspection)
def handle_pests(pest_detections, current_time, timestamp):
    """Manage pest detections, trigger actions like spraying or inspection."""
    for pest_detection in pest_detections:
        pest_type = pest_detection["pest_type"]
        bbox = pest_detection["bbox"]

        # Check if a pest needs to trigger an action (e.g., spray, inspection)
        if pest_tracker.should_trigger_pump(bbox, pest_type, current_time):
            trigger_pump()  # Trigger the pump action
            log(f"💧 Pump triggered for {pest_type} at plant location {pest_tracker.get_plant_id(bbox)}")

    # Check for persistent pests needing inspection
    pest_tracker.check_persistent_pests(current_time)

    # Add inspection flags to the growth data
    inspection_needed = []
    for plant_id, data in pest_tracker.pest_history.items():
        if data['needs_inspection']:
            inspection_needed.append({
                'location': plant_id,
                'pest_type': data['pest_type'],
                'last_treatment': data['last_treatment'].strftime("%Y-%m-%d %H:%M:%S")
            })

    return inspection_needed

# Existing process_image function with new pest handling integrated
def adjust_maturity_prediction(days, env_factors):
    """Adjust maturity prediction based on environmental conditions"""
    if days is None or env_factors is None:
        return days
    
    adjustment = 1.0
    avg_temp = env_factors.get('avg_temp')
    humidity = env_factors.get('humidity')
    
    # Simple adjustment rules - customize based on your plants' needs
    if avg_temp is not None:
        if avg_temp < 18:  # Cold slows growth
            adjustment *= 1.2
        elif avg_temp > 28:  # Heat can stress plants
            adjustment *= 0.9
        
    if humidity is not None and humidity > 80:  # High humidity can promote disease
        adjustment *= 1.05
        
    return max(1, round(days * adjustment))

# Assuming you have already loaded the regression model and it's working

transplant_date_str = "2025-06-01"  # Example transplant date, replace with actual date

def process_image(raw_image_path, timestamp, transplant_date_str):
    try:
        # Load the raw image and perform plant detection
        image = cv2.imread(raw_image_path)
        if image is None:
            raise ValueError(f"Failed to load image {raw_image_path}")

        # Run YOLO models for plant and leaf detection
        plant_results = plant_model.predict(image, conf=0.3)[0]
        leaf_results = leaf_model.predict(image, conf=0.3)[0]

        plant_boxes = plant_results.boxes.xyxy.cpu().numpy().astype(int) if plant_results.boxes else []
        leaf_boxes = leaf_results.boxes.xyxy.cpu().numpy().astype(int) if leaf_results.boxes else []

        output_image = image.copy()
        leaf_data_per_plant = []
        assigned_leaves = set()

        days_since_transplant = calculate_days_since_transplant(transplant_date_str, timestamp)

        for idx, plant_bbox in enumerate(plant_boxes, start=1):
            leaves_in_plant = []
            for leaf_idx, leaf_bbox in enumerate(leaf_boxes):
                if leaf_idx in assigned_leaves:
                    continue
                if is_inside(plant_bbox, leaf_bbox):
                    leaves_in_plant.append(leaf_bbox)
                    assigned_leaves.add(leaf_idx)

            leaf_count = len(leaves_in_plant)
            largest_leaf_area = estimate_largest_leaf_area(leaves_in_plant)


# Get the number of days since transplant
            days_since_transplant = calculate_days_since_transplant(transplant_date_str, timestamp)

            # Now pass it to classify_growth_with_age_hydroponics
            growth_stage = classify_growth_with_age_hydroponics(leaf_count, largest_leaf_area, days_since_transplant)

            # Predict the number of days to maturity using the regression model

            # Annotate the plant's bounding box
            x1, y1, x2, y2 = plant_bbox
            label = f"Plant {idx}: {growth_stage} | Leaves: {leaf_count} | Age: {days_since_transplant}d"
            box_color = (0, 255, 0)

            cv2.rectangle(output_image, (x1, y1), (x2, y2), box_color, 2)
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            text_x = x1
            text_y = max(y1 - 10, th + 10)
            cv2.rectangle(output_image, (text_x, text_y - th - baseline), (text_x + tw, text_y + baseline), box_color, thickness=cv2.FILLED)
            cv2.putText(output_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            leaf_data_per_plant.append({
                "plant_number": idx,
                "bounding_box": [int(x1), int(y1), int(x2), int(y2)],
                "leaf_count": leaf_count,
                "largest_leaf_area": largest_leaf_area,
                "growth_stage": growth_stage,
                "days_since_transplant": days_since_transplant,
            })

        # Save annotated image
        detected_image_path = os.path.join(tempfile.gettempdir(), f"{timestamp}_Detected.jpg")
        cv2.imwrite(detected_image_path, output_image)

        # Upload detected image to Firebase
        upload_image(detected_image_path, "Detected", timestamp)

        # Upload growth data (including predicted days to maturity) to Firebase
        json_data = {
            "timestamp": timestamp,
            "leaf_data_per_box": leaf_data_per_plant
        }
        json_blob = bucket.blob(f'detections/{timestamp}/growth_parameters.json')
        json_blob.upload_from_string(json.dumps(json_data), content_type='application/json')
        print(f"Uploaded growth parameters JSON for timestamp {timestamp}")

        # Clean up temporary files
        os.remove(raw_image_path)
        os.remove(detected_image_path)

    except Exception as e:
        log(f"Error processing image {raw_image_path}: {e}")


# ✅ Save sensor data offline if no internet
def save_sensor_data_offline(timestamp, water_temp, air_temp, humidity):
    env_data = {
        "timestamp": timestamp,
        "water_temp": water_temp,
        "air_temp": air_temp,
        "humidity": humidity,
    }
    
    env_log_path = os.path.join(OFFLINE_DIR, f"{timestamp}_env.json")
    with open(env_log_path, "w") as f:
        json.dump(env_data, f)
    print(f"📄 Saved sensor log offline (with pH): {env_log_path}")
    log(f"✅ Saved environment_data for {timestamp} offline (with pH)")

    # Optionally also save combined default log for fallback
    data = {
        "timestamp": timestamp,
        "height_cm": None,
        "leaf_count": None,
        "leaf_area_cm2": None,
        "growth_stage": "Unknown",
        "pest_detected": "Unknown",
        "water_temp": water_temp,
        "air_temp": air_temp,
        "humidity": humidity,
    }
    log_path = os.path.join(OFFLINE_DIR, f"{timestamp}.json")
    with open(log_path, "w") as f:
        json.dump(data, f)
    print(f"📄 Saved merged log offline (with pH): {log_path}")


def main_capture_loop():
    raw_image_path, timestamp = capture_image()
    if raw_image_path:
        for attempt in range(2):
            try:
                water_temp = read_ds18b20_temp()
                air_temp = dhtDevice.temperature  # Read air temperature
                humidity = dhtDevice.humidity    # Read humidity
                        
                # Control the humidifier based on humidity
                break
            except RuntimeError as e:
                log(f"Sensor read failed (attempt {attempt + 1}): {e}")
                time.sleep(2)
                if attempt == 1:
                    water_temp = None
                    air_temp = None
                    humidity = None
                

        process_image(raw_image_path, timestamp, transplant_date_str)

        if is_connected():
            upload_sensor_data_to_firebase(timestamp, water_temp, air_temp, humidity)
            try_upload_offline_data()
        else:
            save_sensor_data_offline(timestamp, water_temp, air_temp, humidity)  # You can also modify this to include pH offline


# Scheduling image capture every 10 minutes
schedule.every(10).minutes.do(main_capture_loop)
print("⏳ System ready. Capturing every 10 minutes.")

while True:
    schedule.run_pending()
    time.sleep(1)
