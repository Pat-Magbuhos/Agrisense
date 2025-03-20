import subprocess
import cv2
import numpy as np

def capture_image():
    image_path = "/home/Agrisense/Thesis/live_feed.jpg"

    # Capture an image using libcamera
    subprocess.run(["libcamera-still", "-o", image_path, "--nopreview", "-q", "95"])

    return image_path

def display_image(image_path):
    # Read the captured image
    image = cv2.imread(image_path)

    if image is None:
        print("❌ Error: Failed to read image.")
        return

    # Show the image
    cv2.imshow("Live Camera Feed", image)

    # Press 'q' to exit
    cv2.waitKey(0)
    cv2.destroyAllWindows()

while True:
    print("📸 Capturing Image...")
    image_path = capture_image()
    
    print("✅ Displaying Image...")
    display_image(image_path)
    
    cont = input("Press Enter to capture again or type 'q' to quit: ")
    if cont.lower() == 'q':
        break
