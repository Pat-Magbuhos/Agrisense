import RPi.GPIO as GPIO
import time

SENSOR_PIN = 22

GPIO.setmode(GPIO.BCM)
GPIO.setup(SENSOR_PIN, GPIO.IN)

def read_sensor():
    readings = []
    for _ in range(5):
        readings.append(GPIO.input(SENSOR_PIN))
        time.sleep(0.05)
    # Return majority vote
    return 1 if readings.count(1) > 2 else 0

try:
    print("Monitoring water level...")
    last_state = None
    while True:
        state = read_sensor()
        if state != last_state:
            if state:
                print("✅ Water detected!")
            else:
                print("❌ No water detected!")
            last_state = state
        time.sleep(1)

except KeyboardInterrupt:
    GPIO.cleanup()
