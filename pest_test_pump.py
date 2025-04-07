import cv2
import time
import RPi.GPIO as GPIO
from ultralytics import YOLO

# Relay setup
RELAY_PIN = 23
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.HIGH)  # Relay off (pump off)

# Load model
model = YOLO("/home/Agrisense/Thesis/bestv2.pt")

# Load image for testing
image_path = "/home/Agrisense/Thesis/sample.jpeg"  # Change this path to your test image
image = cv2.imread(image_path)
if image is None:
    print("Error: Unable to load image at", image_path)
    GPIO.cleanup()
    exit()

# Run YOLO inference
results = model.predict(image, conf=0.5)
output_image = results[0].plot()

# Define growth stage labels that should not trigger the pump
GROWTH_STAGE_CLASSES = ["Seedling", "Vegetative", "Mature"]

# Check if pest (non-growth-stage) is detected
pest_detected = False
for result in results:
    if len(result.boxes) > 0:
        for class_id in result.boxes.cls:
            class_name = result.names[int(class_id)]
            if class_name not in GROWTH_STAGE_CLASSES:
                pest_detected = True
                print(f"Pest detected: {class_name}")
                break
        if pest_detected:
            break

# Activate pump if pest was detected
if pest_detected:
    print("Turning ON pump for 5 seconds...")
    GPIO.output(RELAY_PIN, GPIO.LOW)  # Pump ON
    time.sleep(5)
    GPIO.output(RELAY_PIN, GPIO.HIGH)  # Pump OFF
    print("Pump OFF")
else:
    print("No pest detected.")

# Clean up GPIO
GPIO.cleanup()
