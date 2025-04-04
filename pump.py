import RPi.GPIO as GPIO
import time

RELAY_PIN = 23  # Using GPIO23

GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)

# Turn off the relay initially
GPIO.output(RELAY_PIN, GPIO.HIGH)

try:
    while True:
        print("Pump ON")
        GPIO.output(RELAY_PIN, GPIO.LOW)
        time.sleep(5)

        print("Pump OFF")
        GPIO.output(RELAY_PIN, GPIO.HIGH)
        time.sleep(5)

except KeyboardInterrupt:
    print("Stopped by user")
finally:
    GPIO.cleanup()
