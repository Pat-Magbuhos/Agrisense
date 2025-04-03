import RPi.GPIO as GPIO
import time

SENSOR_PIN = 17  # Use any GPIO pin you connected OUT to

GPIO.setmode(GPIO.BCM)
GPIO.setup(SENSOR_PIN, GPIO.IN)

print("Monitoring water level...")

try:
    while True:
        state = GPIO.input(SENSOR_PIN)
        if state == GPIO.HIGH:
            print("💧 Water Detected")
        else:
            print("🚫 No Water Detected")
        time.sleep(1)

except KeyboardInterrupt:
    print("Exiting...")
    GPIO.cleanup()

last_state = None

while True:
    state = GPIO.input(SENSOR_PIN)
    if state != last_state:
        if state == GPIO.HIGH:
            print("💧 Water Detected")
        else:
            print("🚫 No Water")
        last_state = state
    time.sleep(0.5)
