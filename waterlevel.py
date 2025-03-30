import RPi.GPIO as GPIO
import time

SENSOR_PIN = 27

GPIO.setmode(GPIO.BCM)
GPIO.setup(SENSOR_PIN, GPIO.IN)

try:
    print("Debugging water sensor output. Press Ctrl+C to exit.")
    while True:
        state = GPIO.input(SENSOR_PIN)
        if state:
            print("GPIO HIGH - Sensor reports WATER detected.")
        else:
            print("GPIO LOW - Sensor reports NO water detected.")
        time.sleep(1)

except KeyboardInterrupt:
    print("Exited by user.")

finally:
    GPIO.cleanup()
