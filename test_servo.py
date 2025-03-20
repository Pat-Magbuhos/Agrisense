import RPi.GPIO as GPIO
import time

SERVO_PIN = 18  # Update this if needed

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)

pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz PWM
pwm.start(0)

def set_angle(angle):
    """Move the servo to a specific angle."""
    duty = 2 + (angle / 18)  # Convert angle to duty cycle
    pwm.ChangeDutyCycle(duty)
    time.sleep(1)
    pwm.ChangeDutyCycle(0)  # Stop signal to avoid jitter

try:
    set_angle(45)  # Move servo to 315 degrees
    print("Servo set to 315 degrees")

finally:
    pwm.stop()
    GPIO.cleanup()
