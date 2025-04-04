import time
import glob
import board
import adafruit_dht
from smbus2 import SMBus
import RPi.GPIO as GPIO

# --- GPIO & Relay Setup ---
RELAY_PIN = 23  # GPIO23
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.HIGH)  # OFF initially

# --- DS18B20 Setup ---
base_dir = '/sys/bus/w1/devices/'
try:
    device_folder = glob.glob(base_dir + '28*')[0]
    device_file = device_folder + '/w1_slave'
except IndexError:
    device_file = None
    print("⚠️ DS18B20 sensor not detected.")

def read_temp_raw():
    with open(device_file, 'r') as f:
        return f.readlines()

def read_ds18b20_temp():
    lines = read_temp_raw()
    while lines[0].strip()[-3:] != 'YES':
        time.sleep(0.2)
        lines = read_temp_raw()
    temp_pos = lines[1].find('t=')
    if temp_pos != -1:
        temp_string = lines[1][temp_pos+2:]
        temp_c = float(temp_string) / 1000.0
        return temp_c
    return None

# --- BH1750 Setup ---
DEVICE_ADDR = 0x23
CONTINUOUS_HIGH_RES_MODE = 0x10

def read_light():
    try:
        with SMBus(1) as bus:
            data = bus.read_i2c_block_data(DEVICE_ADDR, CONTINUOUS_HIGH_RES_MODE, 2)
            lux = ((data[0] << 8) + data[1]) / 1.2
            return lux
    except Exception as e:
        print(f"⚠️ BH1750 error: {e}")
        return None

# --- DHT22 Setup ---
dhtDevice = adafruit_dht.DHT22(board.D27)  # GPIO27

# --- Main Loop ---
try:
    while True:
        # 💧 Turn pump ON for 1 second
        GPIO.output(RELAY_PIN, GPIO.LOW)
        print("Pump ON")
        time.sleep(1)

        # 💧 Turn pump OFF
        GPIO.output(RELAY_PIN, GPIO.HIGH)
        print("Pump OFF")

        # 🌡️ Read sensors
        water_temp = read_ds18b20_temp() if device_file else None
        light = read_light()
        try:
            air_temp = dhtDevice.temperature
            humidity = dhtDevice.humidity
        except RuntimeError as error:
            print(f"⚠️ DHT22 error: {error.args[0]}")
            air_temp, humidity = None, None

        # 📋 Display sensor data
        print(f"💧 Water Temp: {water_temp:.2f}°C" if water_temp else "❌ Water Temp: Error")
        print(f"💡 Light: {light:.2f} lx" if light else "❌ Light: Error")
        if air_temp is not None and humidity is not None:
            print(f"🌡️ Air Temp: {air_temp:.2f}°C | 💦 Humidity: {humidity:.2f}%")
        else:
            print("❌ DHT22: Error reading data")

        print("-" * 50)
        
        # ⏱️ Wait for 2 seconds before repeating the loop
        time.sleep(2)

except KeyboardInterrupt:
    print("\nProgram stopped by user.")

finally:
    dhtDevice.exit()
    GPIO.cleanup()
