import os
import base64
import numpy as np
import subprocess
from datetime import datetime
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, db
from ultralytics import YOLO  # YOLO model for inference
import cv2  # OpenCV for processing

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
    pass  

cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})

# Ensure directory structure exists
BASE_DIR = "/home/Agrisense/Thesis"
CAPTURED_RAW_DIR = os.path.join(BASE_DIR, "Captured", "Raw")
DETECTED_DIR = os.path.join(BASE_DIR, "Detected", "Detected")

for directory in [CAPTURED_RAW_DIR, DETECTED_DIR]:
    os.makedirs(directory, exist_ok=True)

# Load trained model
model = YOLO("/home/Agrisense/Thesis/best.pt")

# Trigonometry Constants
CAMERA_ANGLE = 45  
CAMERA_HEIGHT = 30  
FOCAL_LENGTH = 800  

# Growth Stage Thresholds
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
    pixel_area = pixel_width * pixel_height  
    scale_factor = 0.05  
    real_area = pixel_area * scale_factor
    return round(real_area, 2)

# Improved Leaf Counting using Contours
def count_leaves(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_leaf_area = 500  
    leaf_contours = [c for c in contours if cv2.contourArea(c) > min_leaf_area]

    output = image.copy()
    cv2.drawContours(output, leaf_contours, -1, (0, 255, 0), 2)

    processed_image_path = image_path.replace(".jpg", "_contours.jpg")
    cv2.imwrite(processed_image_path, output)

    leaf_count = len(leaf_contours)
    print(f"Detected Leaves: {leaf_count} (Processed image saved at {processed_image_path})")
    
    return leaf_count, processed_image_path

# Function to classify growth stage
def classify_growth(height, leaf_count, leaf_area):
    if height < GROWTH_THRESHOLDS["seedling"]["height"] and leaf_count < GROWTH_THRESHOLDS["seedling"]["leaves"] and leaf_area < GROWTH_THRESHOLDS["seedling"]["leaf_area"]:
        return "Seedling"
    elif height < GROWTH_THRESHOLDS["vegetative"]["height"] and leaf_count < GROWTH_THRESHOLDS["vegetative"]["leaves"] and leaf_area < GROWTH_THRESHOLDS["vegetative"]["leaf_area"]:
        return "Vegetative"
    else:
        return "Mature"


# Function to Show Terminal-Based Preview Before Capturing
def show_preview():
    print("\n🔍 Adjust the sample in front of the camera! Press [Enter] to capture or [q] to quit.")
    while True:
        user_input = input("📸 Ready? Press [Enter] to capture or [q] to quit: ")
        if user_input == '':
            return True  # Capture the image
        elif user_input.lower() == 'q':
            print("❌ Capture canceled.")
            return False  # Cancel capture



# Function to capture image
def capture_image():
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(CAPTURED_RAW_DIR, f"{timestamp}.jpg")

        print("\nPress Enter to capture or type 'q' to quit:")
        while True:
            user_input = input("Ready? Press Enter to capture or 'q' to quit: ")
            if user_input == '':
                break
            elif user_input.lower() == 'q':
                print("Capture canceled.")
                return None, None

        os.system(f"libcamera-jpeg -o {image_path} --width 1024 --height 768 --quality 85 --nopreview")

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

# Function to process image and upload growth parameters
def process_image(raw_image_path, timestamp):
    detected_image_path = os.path.join(DETECTED_DIR, f"{timestamp}.jpg")

    try:
        image = cv2.imread(raw_image_path)
        if image is None:
            raise FileNotFoundError(f"ERROR: Image file not found at {raw_image_path}")

        results = model.predict(image, conf=0.5)
        output_image = results[0].plot()

        leaf_count, processed_image_path = count_leaves(raw_image_path)
        total_leaf_area = 0
        estimated_height = 0

        for result in results:
            for bbox in result.boxes.xyxy:
                bbox = bbox.cpu().numpy().astype(int)

                estimated_height = estimate_height(bbox)
                leaf_area = estimate_leaf_area(bbox)
                total_leaf_area += leaf_area

                growth_stage = classify_growth(estimated_height, leaf_count, total_leaf_area)

        firebase_path = f"detections/{timestamp}/growth_parameters"
        ref = db.reference(firebase_path)
        ref.set({
            "height_cm": estimated_height,
            "leaf_count": leaf_count,
            "leaf_area_cm2": total_leaf_area,
            "growth_stage": growth_stage
        })
        print(f"Growth parameters uploaded to {firebase_path}")

        cv2.imwrite(detected_image_path, output_image)
        upload_image(detected_image_path, "Detected", timestamp)

        return detected_image_path, processed_image_path

    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

# Main loop for continuous image capture
while True:
    raw_image_path, timestamp = capture_image()
    if raw_image_path:
        detected_image_path, processed_image_path = process_image(raw_image_path, timestamp)

    cont = input("\nPress Enter to capture again or type 'q' to quit: ")
    if cont.lower() == 'q':
        break
