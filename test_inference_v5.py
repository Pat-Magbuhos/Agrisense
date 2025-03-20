import os
import base64
from datetime import datetime
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, db
from ultralytics import YOLO  # YOLO model for inference

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

# ✅ Define Paths for Captured and Detected Images
BASE_DIR = "/home/Agrisense/Thesis"
CAPTURED_DIR = os.path.join(BASE_DIR, "Captured")
CAPTURED_RAW_DIR = os.path.join(CAPTURED_DIR, "Raw")
CAPTURED_RETRIEVED_DIR = os.path.join(CAPTURED_DIR, "Retrieved")

DETECTED_DIR = os.path.join(BASE_DIR, "Detected")
DETECTED_PROCESSED_DIR = os.path.join(DETECTED_DIR, "Detected")
DETECTED_RETRIEVED_DIR = os.path.join(DETECTED_DIR, "Retrieved")

# ✅ Ensure Directories Exist
os.makedirs(CAPTURED_RAW_DIR, exist_ok=True)
os.makedirs(CAPTURED_RETRIEVED_DIR, exist_ok=True)
os.makedirs(DETECTED_PROCESSED_DIR, exist_ok=True)
os.makedirs(DETECTED_RETRIEVED_DIR, exist_ok=True)

# ✅ Load YOLO Model
model = YOLO("/home/Agrisense/Thesis/best.pt")  # Update with your trained model

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(CAPTURED_RAW_DIR, f"{timestamp}.jpg")

    if show_preview():
        print("📸 Capturing image...")
        os.system(f"libcamera-jpeg -o {image_path} --width 1280 --height 960 --quality 90")
        print(f"✅ Image saved to {image_path}")
        return image_path, timestamp
    else:
        print("❌ Capture canceled.")
        return None, None

# ✅ Run YOLO Model for Detection
def process_image(raw_image_path, timestamp):
    try:
        results = model.predict(raw_image_path, conf=0.5)
        output_image = results[0].plot()  # Apply bounding boxes

        detected_image_path = os.path.join(DETECTED_PROCESSED_DIR, f"{timestamp}.jpg")
        with open(detected_image_path, "wb") as f:
            f.write(output_image)
        
        print(f"✅ Detection complete! Processed image saved at {detected_image_path}")
        return detected_image_path
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None

# ✅ Upload Image to Firebase Realtime Database
def upload_image(image_path, firebase_path):
    try:
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')

        ref = db.reference(firebase_path)
        ref.set(image_data)
        print(f"✅ Uploaded {image_path} to Firebase as {firebase_path}")
    except Exception as e:
        print(f"❌ Error uploading {image_path}: {e}")

# ✅ Download Image from Firebase and Save Locally
def download_image(firebase_path, save_path):
    try:
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
            upload_image(raw_image_path, f"Detections/Raw/{timestamp}.jpg")
            upload_image(detected_image_path, f"Detections/Detected/{timestamp}.jpg")

            print("⬇️ Testing download of images...")
            retrieved_raw_path = os.path.join(CAPTURED_RETRIEVED_DIR, f"retrieved_{timestamp}.jpg")
            retrieved_detected_path = os.path.join(DETECTED_RETRIEVED_DIR, f"retrieved_{timestamp}.jpg")

            download_image(f"Detections/Raw/{timestamp}.jpg", retrieved_raw_path)
            download_image(f"Detections/Detected/{timestamp}.jpg", retrieved_detected_path)

    cont = input("\nPress Enter to capture again or type 'q' to quit: ")
    if cont.lower() == 'q':
        break
