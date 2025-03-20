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

# ✅ Load environment variables from .env
dotenv_path = os.path.join(os.path.dirname(__file__), "venv/.env")
load_dotenv(dotenv_path)

# ✅ Retrieve Firebase credentials from .env
FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL")
SERVICE_ACCOUNT_PATH = os.getenv("SERVICE_ACCOUNT_PATH", "venv/serviceAccountKey.json")

# ✅ Validate environment variables
if not FIREBASE_DB_URL:
    raise ValueError("❌ ERROR: FIREBASE_DB_URL is missing from .env!")
if not os.path.exists(SERVICE_ACCOUNT_PATH):
    raise ValueError(f"❌ ERROR: Service account key not found at {SERVICE_ACCOUNT_PATH}")

# ✅ Initialize Firebase
try:
    firebase_admin.delete_app(firebase_admin.get_app())
except ValueError:
    pass  # No app was initialized

cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})


# ✅ Ensure directory structure exists
BASE_DIR = "/home/Agrisense/Thesis"
CAPTURED_RAW_DIR = os.path.join(BASE_DIR, "Captured", "Raw")
CAPTURED_RETRIEVED_DIR = os.path.join(BASE_DIR, "Captured", "Retrieved")
DETECTED_DIR = os.path.join(BASE_DIR, "Detected", "Detected")
DETECTED_RETRIEVED_DIR = os.path.join(BASE_DIR, "Detected", "Retrieved")

for directory in [CAPTURED_RAW_DIR, CAPTURED_RETRIEVED_DIR, DETECTED_DIR, DETECTED_RETRIEVED_DIR]:
    os.makedirs(directory, exist_ok=True)

# ✅ Load trained model
model = YOLO("/home/Agrisense/Thesis/best.pt")

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


# ✅ Function to Show Terminal-Based Preview Before Capturing
def show_preview():
    print("\n🔍 Adjust the sample in front of the camera! Press [Enter] to capture or [q] to quit.")
    while True:
        user_input = input("📸 Ready? Press [Enter] to capture or [q] to quit: ")
        if user_input == '':
            return True  # Capture the image
        elif user_input.lower() == 'q':
            print("❌ Capture canceled.")
            return False  # Cancel capture


# ✅ Capture Image Using libcamera
def capture_image():
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(CAPTURED_RAW_DIR, f"{timestamp}.jpg")

        if show_preview():
            print("📸 Capturing image...")
            os.system(f"libcamera-jpeg -o {image_path} --width 1280 --height 1280 --quality 90")

            # Ensure the image is properly formatted for YOLO
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"❌ ERROR: Image file not found at {image_path}")

            # Convert to RGB format
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(image_path, image_rgb)

            print(f"✅ Image captured and saved to {image_path}")
            return image_path, timestamp
        else:
            print("❌ Capture canceled.")
            return None, None

    except Exception as e:
        print(f"❌ Error capturing image: {e}")
        return None, None


# ✅ Function to Run Inference
def process_image(raw_image_path, timestamp):
    detected_image_path = os.path.join(DETECTED_DIR, f"{timestamp}.jpg")

    try:
        image = cv2.imread(raw_image_path)
        if image is None:
            raise FileNotFoundError(f"❌ ERROR: Image file not found at {raw_image_path}")

        results = model.predict(image, conf=0.5)
        output_image = results[0].plot()

        # Extract growth parameters
        leaf_count = 0
        total_leaf_area = 0
        estimated_height = 0

        detected_frame = image.copy()  # Copy image for drawing

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

        # Save the detected image with bounding boxes and labels
        cv2.imwrite(detected_image_path, detected_frame)
        print(f"✅ Detection complete! Processed image saved at {detected_image_path}")

        return detected_image_path

    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None


# ✅ Function to Upload Images to Firebase
def upload_image(image_path, image_type, timestamp):
    try:
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')

        firebase_path = f"detections/{timestamp}/{image_type}"
        ref = db.reference(firebase_path)
        ref.set(image_data)
        print(f"✅ Uploaded {image_path} to Firebase under {firebase_path}")
    except Exception as e:
        print(f"❌ Error uploading {image_path}: {e}")


# ✅ Function to Download Images from Firebase
def download_image(image_type, timestamp, save_path):
    try:
        firebase_path = f"detections/{timestamp}/{image_type}"
        ref = db.reference(firebase_path)
        image_data = ref.get()

        if image_data:
            with open(save_path, "wb") as image_file:
                image_file.write(base64.b64decode(image_data))
            print(f"✅ Image downloaded from {firebase_path} and saved as {save_path}")
        else:
            print(f"❌ No data found for {firebase_path}")
    except Exception as e:
        print(f"❌ Error downloading image: {e}")


# ✅ Main Loop for Continuous Image Capture and Processing
while True:
    print("\n📸 Capturing new image...")
    raw_image_path, timestamp = capture_image()

    if raw_image_path:
        print("🔍 Running inference...")
        detected_image_path = process_image(raw_image_path, timestamp)

        if detected_image_path:
            print("⬆️ Uploading images to Firebase...")
            upload_image(raw_image_path, "Raw", timestamp)
            upload_image(detected_image_path, "Detected", timestamp)

            print("⬇️ Testing download of images...")
            retrieved_raw_path = os.path.join(CAPTURED_RETRIEVED_DIR, f"retrieved_{timestamp}.jpg")
            retrieved_detected_path = os.path.join(DETECTED_RETRIEVED_DIR, f"retrieved_{timestamp}.jpg")

            download_image("Raw", timestamp, retrieved_raw_path)
            download_image("Detected", timestamp, retrieved_detected_path)

    cont = input("\nPress Enter to capture again or type 'q' to quit: ")
    if cont.lower() == 'q':
        break
