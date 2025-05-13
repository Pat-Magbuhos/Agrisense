import os
from datetime import datetime

def capture_test_photo():
    try:
        # Generate a timestamped file name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        image_path = os.path.join("/home/Agrisense/Thesis/", f"test_photo_{timestamp}.jpg")
        
        # Capture the image with adjusted shutter speed for less light exposure
        print(f"Capturing photo and saving it to {image_path}...")
        os.system(f"libcamera-jpeg -o {image_path} --width 2592 --height 1944 --quality 90 --framerate 15 --shutter 5000 --awb auto")

        print(f"Photo captured and saved as {image_path}")
        return image_path
    except Exception as e:
        print(f"Error capturing photo: {e}")
        return None

if __name__ == '__main__':
    capture_test_photo()  # Capture and save test photo
