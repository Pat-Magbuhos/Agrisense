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
cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
print("✅ Firebase Initialized Successfully!")

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


# ✅ Function to Capture Image using `libcamera-still`
def capture_image():
    """ Captures an image using libcamera-jpeg and saves it in the correct format. """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_image_path = os.path.join(CAPTURED_RAW_DIR, f"{timestamp}.jpg")

    try:
        # Capture the image using libcamera
        subprocess.run(["libcamera-jpeg", "-o", raw_image_path, "-n", "--width", "640", "--height", "640"],
                       check=True)

        # Ensure the image is properly formatted for YOLO
        image = cv2.imread(raw_image_path)
        if image is None:
            raise FileNotFoundError(f"❌ ERROR: Image file not found at {raw_image_path}")

        # Convert to RGB format to match the training data
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(raw_image_path, image_rgb)

        print(f"📸 Image captured and saved to {raw_image_path}")
        return raw_image_path, timestamp
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

        cv2.imwrite(detected_image_path, output_image)
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


# ✅ Main Loop to Capture, Process, Upload, and Download Images
while True:
    print("📸 Capturing new image...")
    raw_image_path, timestamp = capture_image()

    if raw_image_path and timestamp:
        print("🔍 Running inference...")
        detected_image_path = process_image(raw_image_path, timestamp)

        if detected_image_path:
            print("⬆️ Uploading images to Firebase...")
            upload_image(raw_image_path, "raw", timestamp)
            upload_image(detected_image_path, "detected", timestamp)

            # ✅ Paths for downloading images
            retrieved_raw_path = os.path.join(CAPTURED_RETRIEVED_DIR, f"retrieved_{timestamp}.jpg")
            retrieved_detected_path = os.path.join(DETECTED_RETRIEVED_DIR, f"retrieved_{timestamp}.jpg")

            print("⬇️ Testing download of images...")
            download_image("raw", timestamp, retrieved_raw_path)
            download_image("detected", timestamp, retrieved_detected_path)

    cont = input("Press Enter to capture again or type 'q' to quit: ")
    if cont.lower() == 'q':
        break
